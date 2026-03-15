"""
Live Agent - Proxies browser audio/video to Gemini Live API
Fixes applied:
  1. Audio batching — buffers 300ms of PCM before sending to Gemini
     (AudioWorklet sends every 8ms; Gemini cannot handle that rate)
  2. Keepalive — sends a silent audio ping every 10s when mic is idle
     (prevents the "keepalive ping timeout" crash)
  3. Reconnect — if the Gemini session drops, reconnects up to 3 times
     without dropping the browser WebSocket
"""

import asyncio
import base64
import json
import logging
import os
import re

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)
router = APIRouter()

LIVE_MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"

# How many PCM bytes to accumulate before sending to Gemini.
# 16000 Hz × 2 bytes × 0.3 s = 9600 bytes ≈ 300ms of audio.
# This rate is well within Gemini's limits while still feeling real-time.
AUDIO_BATCH_BYTES = 9600

# Send a silent heartbeat to Gemini if no real audio arrives for this long.
# Keeps the underlying WebSocket alive during pauses in speech.
KEEPALIVE_INTERVAL_S = 10.0

# 80ms of silence (16000 Hz, mono, int16) = 1280 samples = 2560 bytes
SILENCE_BYTES = bytes(2560)

# Max times to transparently reconnect to Gemini before giving up
MAX_RECONNECTS = 3

SYSTEM_PROMPT = """You are an expert, encouraging AI tutor helping a student with their homework.
You can see the student's camera and hear their voice in real time.

When the student shows you homework:
- Immediately identify the subject and problem
- Explain step by step in a warm, patient voice
- Use simple language appropriate for a student
- If they interrupt you, stop and answer their question naturally
- Encourage the student throughout

Keep responses concise. Speak naturally — no markdown, no bullet points."""

LIVE_CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                voice_name="Zephyr"
            )
        )
    ),
    system_instruction=SYSTEM_PROMPT,
)


def _filter_thinking(text: str) -> str:
    text = re.sub(r'\*\*[^*]+\*\*\s*', '', text)
    lines = [l for l in text.split('\n') if l.strip()]
    return ' '.join(lines).strip()


@router.websocket("/ws/live")
async def live_agent_websocket(websocket: WebSocket):
    await websocket.accept()
    logger.info("Live agent WebSocket accepted")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        await websocket.send_json({"type": "error", "message": "GEMINI_API_KEY not set"})
        await websocket.close()
        return

    client = genai.Client(api_key=api_key)

    async def send(data: dict):
        try:
            await websocket.send_json(data)
        except Exception:
            pass

    # ── Shared state between tasks ────────────────────────────────────────
    # Incoming messages from the browser are queued here so browser_to_gemini
    # can be restarted on reconnect without losing messages.
    msg_queue: asyncio.Queue = asyncio.Queue()
    stop_event = asyncio.Event()

    # ── Browser receiver — runs once for the whole session ────────────────
    async def receive_from_browser():
        """Read browser WebSocket and push messages onto msg_queue."""
        try:
            while not stop_event.is_set():
                try:
                    raw = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                    msg = json.loads(raw)
                    if msg.get("type") == "disconnect":
                        stop_event.set()
                        break
                    await msg_queue.put(msg)
                except asyncio.TimeoutError:
                    # No message for 30s — put a keepalive marker so the
                    # Gemini sender loop can send a silence ping
                    await msg_queue.put({"type": "_keepalive"})
        except WebSocketDisconnect:
            logger.info("Browser disconnected")
            stop_event.set()
        except Exception as e:
            logger.error(f"receive_from_browser error: {e}")
            stop_event.set()

    # ── Gemini session runner — reconnects on failure ─────────────────────
    async def run_gemini_session():
        reconnects = 0
        while not stop_event.is_set():
            try:
                async with client.aio.live.connect(
                    model=LIVE_MODEL,
                    config=LIVE_CONFIG,
                ) as session:
                    logger.info(f"Gemini Live session opened ✓ (attempt {reconnects + 1})")
                    if reconnects == 0:
                        await send({
                            "type": "connected",
                            "message": "Live session ready — turn on mic and speak!"
                        })
                    else:
                        # Silent reconnect — don't interrupt the student
                        logger.info("Reconnected to Gemini transparently")

                    reconnects = 0  # Reset counter on successful connect

                    await asyncio.gather(
                        browser_to_gemini(session),
                        gemini_to_browser(session),
                    )

            except Exception as e:
                if stop_event.is_set():
                    break
                reconnects += 1
                logger.error(f"Gemini session error (attempt {reconnects}): {e}", exc_info=True)
                if reconnects >= MAX_RECONNECTS:
                    logger.error("Max reconnects reached, giving up")
                    await send({"type": "error", "message": "Lost connection to Gemini — please refresh."})
                    stop_event.set()
                    break
                wait = 2 ** reconnects  # exponential backoff: 2s, 4s, 8s
                logger.info(f"Reconnecting in {wait}s...")
                await asyncio.sleep(wait)

    # ── Send browser messages → Gemini, with batching + keepalive ─────────
    async def browser_to_gemini(session):
        audio_buf = bytearray()
        last_audio_time = asyncio.get_event_loop().time()

        try:
            while not stop_event.is_set():
                # Drain the queue with a short timeout so we can flush
                # the audio buffer even if no new messages arrive
                try:
                    msg = await asyncio.wait_for(msg_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    msg = None

                now = asyncio.get_event_loop().time()

                if msg is not None:
                    msg_type = msg.get("type")

                    if msg_type == "audio":
                        pcm_bytes = base64.b64decode(msg.get("data", ""))
                        audio_buf.extend(pcm_bytes)
                        last_audio_time = now

                        # FIX 1: Only send when we have 300ms of audio batched.
                        # AudioWorklet fires every ~8ms; sending every chunk
                        # floods Gemini and triggers the keepalive timeout.
                        if len(audio_buf) >= AUDIO_BATCH_BYTES:
                            await session.send_realtime_input(
                                audio=types.Blob(
                                    mime_type="audio/pcm;rate=16000",
                                    data=bytes(audio_buf),
                                )
                            )
                            audio_buf.clear()

                    elif msg_type == "image":
                        # Flush any pending audio first so ordering is correct
                        if audio_buf:
                            await session.send_realtime_input(
                                audio=types.Blob(
                                    mime_type="audio/pcm;rate=16000",
                                    data=bytes(audio_buf),
                                )
                            )
                            audio_buf.clear()

                        img_b64 = msg.get("data", "")
                        if img_b64:
                            if "," in img_b64:
                                img_b64 = img_b64.split(",", 1)[1]
                            await session.send_realtime_input(
                                video=types.Blob(
                                    mime_type="image/jpeg",
                                    data=base64.b64decode(img_b64),
                                )
                            )

                    elif msg_type == "text":
                        text = msg.get("text", "")
                        if text:
                            await session.send_client_content(
                                turns=types.Content(
                                    role="user",
                                    parts=[types.Part(text=text)]
                                ),
                                turn_complete=True,
                            )

                    elif msg_type == "_keepalive":
                        pass  # handled below

                # FIX 2: Keepalive — if no real audio for KEEPALIVE_INTERVAL_S,
                # send a tiny silence blob so Gemini's WS ping doesn't time out.
                if now - last_audio_time >= KEEPALIVE_INTERVAL_S:
                    logger.debug("Sending keepalive silence to Gemini")
                    await session.send_realtime_input(
                        audio=types.Blob(
                            mime_type="audio/pcm;rate=16000",
                            data=SILENCE_BYTES,
                        )
                    )
                    last_audio_time = now

        except Exception as e:
            if not stop_event.is_set():
                logger.error(f"browser_to_gemini error: {e}", exc_info=True)
            raise  # Let run_gemini_session handle reconnect

    # ── Receive Gemini responses → browser ────────────────────────────────
    async def gemini_to_browser(session):
        try:
            async for response in session.receive():
                if stop_event.is_set():
                    break

                if response.data:
                    audio_b64 = base64.b64encode(response.data).decode()
                    await send({"type": "audio", "data": audio_b64})

                if response.server_content:
                    sc = response.server_content

                    if sc.interrupted:
                        await send({"type": "interrupted"})
                        logger.info("Gemini interrupted by student")

                    if sc.turn_complete:
                        await send({"type": "turn_complete"})
                        logger.info("Gemini turn complete")

                    if sc.model_turn and sc.model_turn.parts:
                        for part in sc.model_turn.parts:
                            if hasattr(part, "text") and part.text:
                                clean = _filter_thinking(part.text)
                                if clean:
                                    await send({"type": "text", "text": clean})

        except Exception as e:
            if not stop_event.is_set():
                logger.error(f"gemini_to_browser error: {e}", exc_info=True)
                await send({"type": "error", "message": str(e)})
            raise  # Let run_gemini_session handle reconnect

    # ── Run everything ────────────────────────────────────────────────────
    try:
        await asyncio.gather(
            receive_from_browser(),
            run_gemini_session(),
        )
    except WebSocketDisconnect:
        logger.info("Live WebSocket disconnected cleanly")
    except Exception as e:
        logger.error(f"Live agent top-level error: {e}", exc_info=True)
        try:
            await send({"type": "error", "message": f"Session error: {str(e)}"})
        except Exception:
            pass
    finally:
        stop_event.set()
        logger.info("Live agent session ended")