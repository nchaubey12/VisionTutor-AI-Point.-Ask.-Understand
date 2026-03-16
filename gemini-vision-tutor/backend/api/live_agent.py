"""
Live Agent - Proxies browser audio/video to Gemini Live API

Fixes in this version:
  1. All writes to Gemini go through a single gemini_send_queue handled
     by one gemini_sender task — eliminates concurrent send races that
     caused keepalive to silently fail and the session to timeout.
  2. keepalive_producer is a lightweight task that only puts messages
     onto the queue — it never touches the session directly.
  3. msg_queue is drained on every reconnect so stale audio from before
     the drop doesn't flood the new Gemini session (was causing the long
     delay on second unmute).
  4. Reconnect signals frontend via "reconnected" so mic restarts cleanly.
"""

import asyncio
import base64
import json
import logging
import os
import re
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)
router = APIRouter()

LIVE_MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"

AUDIO_BATCH_BYTES    = 3200        # 100ms @ 16kHz int16
# Gemini's infrastructure closes the WS after ~40s of no audio.
# We prevent this by streaming silence continuously at 100ms intervals
# whenever the mic is idle. This is the only reliable keepalive method
# with the Google GenAI SDK (we can't set ping_interval directly).
SILENCE_INTERVAL_S   = 0.1         # send silence every 100ms when mic idle
SILENCE_CHUNK        = bytes(3200) # 100ms of silence @ 16kHz int16
MAX_RECONNECTS       = 3

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
    realtime_input_config=types.RealtimeInputConfig(
        automatic_activity_detection=types.AutomaticActivityDetection(
            disabled=True,
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

    # ── Shared queues and state ───────────────────────────────────────────
    # msg_queue   : raw messages from the browser (audio, image, text)
    # gemini_send_queue : serialised outbound sends to Gemini
    # Both are recreated on reconnect to avoid stale data.
    msg_queue:         asyncio.Queue = asyncio.Queue()
    gemini_send_queue: asyncio.Queue = asyncio.Queue()
    stop_event = asyncio.Event()
    last_audio_time = [time.monotonic()]
    interrupt_cooldown   = [0.0]   # timestamp of last interrupt
    reset_speech_active  = [False] # signal browser_to_queue to reset speech state
    waiting_for_response = [False] # True after activity_end, False after turn_complete

    # ── Browser receiver — single long-lived task ─────────────────────────
    async def receive_from_browser():
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
                    pass  # normal during silence, keepalive handles Gemini side
        except WebSocketDisconnect:
            logger.info("Browser disconnected")
            stop_event.set()
        except Exception as e:
            logger.error(f"receive_from_browser error: {e}")
            stop_event.set()

    # ── Drain stale messages before each new Gemini session ───────────────
    def drain_queues():
        """
        FIX 3: Clear both queues when reconnecting.
        Without this, audio buffered during the dropout floods the new
        Gemini session all at once, causing a long silence before it
        can process the backlog and respond.
        """
        drained = 0
        while not msg_queue.empty():
            try:
                msg_queue.get_nowait()
                drained += 1
            except asyncio.QueueEmpty:
                break
        while not gemini_send_queue.empty():
            try:
                gemini_send_queue.get_nowait()
                drained += 1
            except asyncio.QueueEmpty:
                break
        if drained:
            logger.info(f"Drained {drained} stale messages before reconnect")

    # ── Gemini session runner ─────────────────────────────────────────────
    async def run_gemini_session():
        reconnects = 0
        while not stop_event.is_set():
            # FIX 3: Always drain before connecting so the new session
            # starts clean regardless of what was buffered.
            drain_queues()
            last_audio_time[0] = time.monotonic()

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
                        logger.info("Reconnected to Gemini — notifying frontend")
                        await send({"type": "reconnected"})

                    reconnects = 0

                    # FIX 1: All four tasks share the session but only
                    # gemini_sender ever calls session.send_* — no races.
                    await asyncio.gather(
                        browser_to_queue(),
                        keepalive_producer(),
                        gemini_sender(session),
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
                wait = 2 ** reconnects
                logger.info(f"Reconnecting in {wait}s...")
                await asyncio.sleep(wait)

    # ── browser_to_queue — reads msg_queue, pushes to gemini_send_queue ───
    async def browser_to_queue():
        # Auto speech detection — sends activity_start/end to Gemini
        # based on audio energy, no user button clicking needed.
        import struct as _struct
        speech_active    = False
        last_speech_time = [time.monotonic()]
        SPEECH_RMS       = 600   # int16 RMS above this = speech
        SILENCE_SECS     = 1.2   # silence after speech = end of turn

        async def silence_watcher():
            nonlocal speech_active
            while not stop_event.is_set():
                await asyncio.sleep(0.15)
                # Handle interrupt reset — abandon current activity window
                if reset_speech_active[0]:
                    reset_speech_active[0] = False
                    if speech_active:
                        speech_active = False
                        logger.info("Speech window abandoned due to interrupt — not sending activity_end")
                    continue
                if speech_active:
                    if time.monotonic() - last_speech_time[0] > SILENCE_SECS:
                        speech_active = False
                        await gemini_send_queue.put(("activity_end", None))
                        logger.info("activity_end — user stopped talking")

        asyncio.ensure_future(silence_watcher())

        try:
            while not stop_event.is_set():
                try:
                    msg = await asyncio.wait_for(msg_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue

                msg_type = msg.get("type")

                if msg_type == "audio":
                    pcm_bytes = base64.b64decode(msg.get("data", ""))
                    last_audio_time[0] = time.monotonic()

                    # Compute RMS from int16 samples
                    n = len(pcm_bytes) // 2
                    if n > 0:
                        samples = _struct.unpack(f"{n}h", pcm_bytes[:n*2])
                        rms = (sum(s*s for s in samples) / n) ** 0.5
                        if rms > SPEECH_RMS:
                            last_speech_time[0] = time.monotonic()
                            if not speech_active:
                                cooldown_elapsed = time.monotonic() - interrupt_cooldown[0]
                                if cooldown_elapsed >= 1.5 and not waiting_for_response[0]:
                                    speech_active = True
                                    await gemini_send_queue.put(("activity_start", None))
                                    logger.info(f"activity_start — rms={rms:.0f}")
                                elif waiting_for_response[0]:
                                    logger.debug("activity_start blocked — waiting for Gemini response")

                    # Only send audio to Gemini inside an active speech window.
                    # Audio sent outside activity_start/end confuses Gemini
                    # and causes it to silently ignore subsequent turns.
                    if speech_active:
                        await gemini_send_queue.put(("audio", pcm_bytes))

                elif msg_type == "image":
                    img_b64 = msg.get("data", "")
                    if img_b64:
                        if "," in img_b64:
                            img_b64 = img_b64.split(",", 1)[1]
                        await gemini_send_queue.put(("image", base64.b64decode(img_b64)))

                elif msg_type == "text":
                    text = msg.get("text", "")
                    if text:
                        await gemini_send_queue.put(("text", text))

        except Exception as e:
            if not stop_event.is_set():
                logger.error(f"browser_to_queue error: {e}", exc_info=True)
            raise

    # ── silence_streamer — sends silence continuously when mic is idle ──────
    async def keepalive_producer():
        """
        Streams silence to Gemini every 100ms whenever no real audio is
        flowing. This is the only reliable way to prevent Gemini's
        infrastructure-level ping timeout (~40s) with the GenAI SDK,
        since we cannot set ping_interval on the underlying websocket.
        The silence is low-energy enough that Gemini's VAD ignores it.
        """
        try:
            while not stop_event.is_set():
                await asyncio.sleep(SILENCE_INTERVAL_S)
                # Send silence for keepalive — but only when NOT in an
                # active speech window. During speech, real audio is flowing.
                # We check gemini_send_queue size as a proxy — if it has
                # many items, speech is active and silence is not needed.
                if gemini_send_queue.qsize() < 5:
                    await gemini_send_queue.put(("silence", None))
        except Exception as e:
            if not stop_event.is_set():
                logger.error(f"keepalive_producer error: {e}")
            raise

    # ── gemini_sender — the ONLY task that writes to session ─────────────
    async def gemini_sender(session):
        """
        FIX 1: Single owner of all session.send_* calls.
        Batches audio to 300ms chunks before sending.
        Handles silence, image, and text inline.
        """
        audio_buf = bytearray()
        try:
            while not stop_event.is_set():
                try:
                    kind, data = await asyncio.wait_for(
                        gemini_send_queue.get(), timeout=0.3
                    )
                except asyncio.TimeoutError:
                    # Flush any partial audio buffer that's been waiting
                    if audio_buf:
                        await session.send_realtime_input(
                            audio=types.Blob(
                                mime_type="audio/pcm;rate=16000",
                                data=bytes(audio_buf),
                            )
                        )
                        audio_buf.clear()
                    continue

                if kind == "audio":
                    audio_buf.extend(data)

                    if len(audio_buf) >= AUDIO_BATCH_BYTES:
                        await session.send_realtime_input(
                            audio=types.Blob(
                                mime_type="audio/pcm;rate=16000",
                                data=bytes(audio_buf),
                            )
                        )
                        audio_buf.clear()

                elif kind == "activity_start":
                    await session.send_realtime_input(
                        activity_start=types.ActivityStart()
                    )
                    logger.info("activity_start sent to Gemini")

                elif kind == "activity_end":
                    if audio_buf:
                        await session.send_realtime_input(
                            audio=types.Blob(
                                mime_type="audio/pcm;rate=16000",
                                data=bytes(audio_buf),
                            )
                        )
                        audio_buf.clear()
                    await session.send_realtime_input(
                        activity_end=types.ActivityEnd()
                    )
                    waiting_for_response[0] = True
                    logger.info("activity_end sent to Gemini — waiting for response")

                elif kind == "end_of_speech":
                    if audio_buf:
                        await session.send_realtime_input(
                            audio=types.Blob(
                                mime_type="audio/pcm;rate=16000",
                                data=bytes(audio_buf),
                            )
                        )
                        audio_buf.clear()



                elif kind == "silence":
                    # Send silence only if no real audio is buffered
                    if not audio_buf:
                        await session.send_realtime_input(
                            audio=types.Blob(
                                mime_type="audio/pcm;rate=16000",
                                data=SILENCE_CHUNK,
                            )
                        )

                elif kind == "image":
                    # Flush audio before image for correct ordering
                    if audio_buf:
                        await session.send_realtime_input(
                            audio=types.Blob(
                                mime_type="audio/pcm;rate=16000",
                                data=bytes(audio_buf),
                            )
                        )
                        audio_buf.clear()
                    await session.send_realtime_input(
                        video=types.Blob(
                            mime_type="image/jpeg",
                            data=data,
                        )
                    )

                elif kind == "text":
                    await session.send_client_content(
                        turns=types.Content(
                            role="user",
                            parts=[types.Part(text=data)]
                        ),
                        turn_complete=True,
                    )



        except Exception as e:
            if not stop_event.is_set():
                logger.error(f"gemini_sender error: {e}", exc_info=True)
            raise

    # ── Gemini → Browser ──────────────────────────────────────────────────
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
                        await send({"type": "gate_reset"})
                        interrupt_cooldown[0]   = time.monotonic()
                        reset_speech_active[0]  = True
                        waiting_for_response[0] = False
                        logger.info("Gemini interrupted — resetting speech state, 1.5s cooldown")

                    if sc.turn_complete:
                        await send({"type": "turn_complete"})
                        waiting_for_response[0] = False
                        interrupt_cooldown[0]   = time.monotonic()  # full 1.5s cooldown
                        logger.info("Gemini turn complete — ready for next question")

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
            raise

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