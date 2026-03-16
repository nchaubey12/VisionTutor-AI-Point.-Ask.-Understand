"""
Live Agent - Proxies browser audio/video to Gemini Live API
"""

import asyncio
import json
import logging
import os
import re

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from websockets import connect as ws_connect

logger = logging.getLogger(__name__)
router = APIRouter()

GEMINI_WS_URI = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
    "?key={api_key}"
)

# ✅ Fixed: correct model name for free-tier API keys
GEMINI_MODEL   = "gemini-2.5-flash-native-audio-latest"
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

SETUP_MESSAGE = {
    "setup": {
        "model": f"models/{GEMINI_MODEL}",
        "generation_config": {
            "response_modalities": ["AUDIO"],
            "speech_config": {
                "voice_config": {
                    "prebuilt_voice_config": {"voice_name": "Zephyr"}
                }
            }
        },
        "system_instruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        }
    }
}


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

    gemini_uri = GEMINI_WS_URI.format(api_key=api_key)

    async def send(data: dict):
        try:
            await websocket.send_json(data)
        except Exception:
            pass

    msg_queue  = asyncio.Queue()
    stop_event = asyncio.Event()

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
                    pass
        except WebSocketDisconnect:
            logger.info("Browser disconnected")
            stop_event.set()
        except Exception as e:
            logger.error(f"receive_from_browser error: {e}")
            stop_event.set()

    def drain_queue():
        drained = 0
        while not msg_queue.empty():
            try:
                msg_queue.get_nowait()
                drained += 1
            except asyncio.QueueEmpty:
                break
        if drained:
            logger.info(f"Drained {drained} stale messages before reconnect")

    async def run_gemini_session():
        reconnects = 0
        while not stop_event.is_set():
            drain_queue()
            try:
                async with ws_connect(
                    gemini_uri,
                    extra_headers={"Content-Type": "application/json"},
                ) as gemini_ws:
                    logger.info(f"Gemini WS connected (attempt {reconnects + 1})")

                    await gemini_ws.send(json.dumps(SETUP_MESSAGE))
                    setup_resp = await gemini_ws.recv()
                    logger.info(f"Gemini setup ack: {str(setup_resp)[:120]}")

                    if reconnects == 0:
                        await send({"type": "connected", "message": "Live session ready — speak!"})
                    else:
                        await send({"type": "reconnected"})
                        logger.info("Reconnected to Gemini")

                    reconnects = 0

                    await asyncio.gather(
                        browser_to_gemini(gemini_ws),
                        gemini_to_browser(gemini_ws),
                    )

            except Exception as e:
                if stop_event.is_set():
                    break
                reconnects += 1
                logger.error(f"Gemini session error (attempt {reconnects}): {e}", exc_info=True)
                if reconnects >= MAX_RECONNECTS:
                    await send({"type": "error", "message": "Lost connection to Gemini — please refresh."})
                    stop_event.set()
                    break
                wait = 2 ** reconnects
                logger.info(f"Reconnecting in {wait}s...")
                await asyncio.sleep(wait)

    async def browser_to_gemini(gemini_ws):
        try:
            while not stop_event.is_set():
                try:
                    msg = await asyncio.wait_for(msg_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                msg_type = msg.get("type")

                if msg_type == "audio":
                    payload = {
                        "realtime_input": {
                            "media_chunks": [{
                                "data": msg.get("data", ""),
                                "mime_type": "audio/pcm;rate=16000"
                            }]
                        }
                    }
                    await gemini_ws.send(json.dumps(payload))

                elif msg_type == "image":
                    img_b64 = msg.get("data", "")
                    if img_b64 and "," in img_b64:
                        img_b64 = img_b64.split(",", 1)[1]
                    if img_b64:
                        payload = {
                            "realtime_input": {
                                "media_chunks": [{
                                    "data": img_b64,
                                    "mime_type": "image/jpeg"
                                }]
                            }
                        }
                        await gemini_ws.send(json.dumps(payload))

                elif msg_type == "text":
                    text = msg.get("text", "")
                    if text:
                        payload = {
                            "client_content": {
                                "turns": [{"role": "user", "parts": [{"text": text}]}],
                                "turn_complete": True
                            }
                        }
                        await gemini_ws.send(json.dumps(payload))

                # ✅ Handle stopAudio — signals end of user turn so Gemini responds
                elif msg_type == "stopAudio":
                    payload = {
                        "realtime_input": {
                            "activity_end": {}
                        }
                    }
                    try:
                        await gemini_ws.send(json.dumps(payload))
                        logger.info("Sent activityEnd to Gemini")
                    except Exception as e:
                        logger.warning(f"activityEnd not supported: {e}")

        except Exception as e:
            if not stop_event.is_set():
                logger.error(f"browser_to_gemini error: {e}", exc_info=True)
            raise

    async def gemini_to_browser(gemini_ws):
        try:
            async for raw in gemini_ws:
                if stop_event.is_set():
                    break

                try:
                    response = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                # Audio / text parts
                try:
                    parts = response["serverContent"]["modelTurn"]["parts"]
                    for part in parts:
                        if "inlineData" in part:
                            await send({"type": "audio", "data": part["inlineData"]["data"]})
                        elif "text" in part:
                            clean = _filter_thinking(part["text"])
                            if clean:
                                await send({"type": "text", "text": clean})
                except (KeyError, TypeError):
                    pass

                # Input transcription (what user said)
                try:
                    input_text = response["serverContent"]["inputTranscription"]["text"]
                    if input_text:
                        await send({"type": "input_transcription", "text": input_text})
                except (KeyError, TypeError):
                    pass

                # Output transcription (what Gemini is saying)
                try:
                    output_text = response["serverContent"]["outputTranscription"]["text"]
                    if output_text:
                        await send({"type": "output_transcription", "text": output_text})
                except (KeyError, TypeError):
                    pass

                # Turn complete
                try:
                    if response["serverContent"].get("turnComplete"):
                        await send({"type": "turn_complete"})
                        logger.info("Gemini turn complete")
                except (KeyError, TypeError):
                    pass

                # Interrupted
                try:
                    if response["serverContent"].get("interrupted"):
                        await send({"type": "interrupted"})
                        await send({"type": "gate_reset"})
                        logger.info("Gemini interrupted")
                except (KeyError, TypeError):
                    pass

        except Exception as e:
            if not stop_event.is_set():
                logger.error(f"gemini_to_browser error: {e}", exc_info=True)
                await send({"type": "error", "message": str(e)})
            raise

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