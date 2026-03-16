"""
WebSocket Handler - with debug logging to trace exactly where execution stops
"""

import json
import uuid
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)
router = APIRouter()

_gemini    = None
_firestore = None
_storage   = None

# ── Interrupt flags keyed by session_id ──────────────────────────────────────
_interrupted: dict[str, bool] = {}


def init_services(gemini, firestore):
    global _gemini, _firestore, _storage
    _gemini    = gemini
    _firestore = firestore

    from services.storage_service import StorageService
    _storage = StorageService()

    logger.info("WebSocket handler: all services ready")


@router.websocket("/ws/tutor")
async def tutor_websocket(websocket: WebSocket):
    logger.info("DEBUG STEP 1: websocket endpoint hit, calling accept()")
    await websocket.accept()
    logger.info("DEBUG STEP 2: accept() done")

    session_id = str(uuid.uuid4())
    logger.info(f"DEBUG STEP 3: session_id={session_id[:8]}")

    logger.info(f"DEBUG STEP 4: gemini={_gemini is not None} firestore={_firestore is not None}")
    if _gemini is None or _firestore is None:
        await websocket.send_json({"type": "error", "message": "Server still initializing"})
        await websocket.close()
        return

    logger.info("DEBUG STEP 5: importing agents...")
    try:
        from agents.vision_agent    import VisionAgent
        from agents.reasoning_agent import ReasoningAgent
        from agents.teaching_agent  import TeachingAgent
        from agents.dialogue_agent  import DialogueAgent

        vision_agent    = VisionAgent(_gemini, _firestore)
        reasoning_agent = ReasoningAgent(_gemini, _firestore)
        teaching_agent  = TeachingAgent(_gemini, _firestore, _storage)
        dialogue_agent  = DialogueAgent(_gemini, _firestore)
        logger.info("DEBUG STEP 5: agents created OK")
    except Exception as e:
        logger.error(f"DEBUG STEP 5 FAILED: {e}", exc_info=True)
        await websocket.send_json({"type": "error", "message": f"Agent init error: {e}"})
        await websocket.close()
        return

    logger.info("DEBUG STEP 6: creating firestore session...")
    try:
        await _firestore.create_session(session_id)
        logger.info("DEBUG STEP 6: firestore session created OK")
    except Exception as e:
        logger.warning(f"DEBUG STEP 6: Firestore failed (continuing): {e}")

    logger.info("DEBUG STEP 7: sending connected message...")
    await websocket.send_json({
        "type":       "connected",
        "session_id": session_id,
        "message":    "Connected! Point your camera at homework and click Analyze."
    })
    logger.info("DEBUG STEP 7: connected message sent ✓")

    _interrupted[session_id] = False

    async def send(data: dict):
        try:
            await websocket.send_json(data)
        except Exception as e:
            logger.debug(f"send() skipped (connection gone): {e}")

    problem_info:  dict = {}
    teaching_plan: dict = {}
    step:          int  = 0

    logger.info("DEBUG STEP 8: entering message loop...")
    try:
        while True:
            logger.info("DEBUG LOOP: waiting for receive_text()...")
            try:
                raw = await websocket.receive_text()
                logger.info(f"DEBUG LOOP: got message — first 80 chars: {raw[:80]}")
            except WebSocketDisconnect:
                logger.info(f"DEBUG LOOP: WebSocketDisconnect for {session_id[:8]}")
                break
            except Exception as e:
                logger.info(f"DEBUG LOOP: receive error: {type(e).__name__}: {e}")
                break

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await send({"type": "error", "message": "Invalid JSON received"})
                continue

            msg_type = msg.get("type", "")
            logger.info(f"DEBUG LOOP: msg_type={msg_type} session={session_id[:8]}")

            # ── INTERRUPT ─────────────────────────────────────────────
            if msg_type == "interrupt":
                _interrupted[session_id] = True
                dialogue_agent.clear_active_explanation(session_id)
                await send({"type": "interrupted"})
                logger.info(f"Interrupt received — stopping stream for {session_id[:8]}")

            # ── FRAME ─────────────────────────────────────────────────
            elif msg_type == "frame":
                b64 = msg.get("image", "")
                logger.info(f"DEBUG FRAME: image present={bool(b64)}, length={len(b64)}")
                if not b64:
                    await send({"type": "error", "message": "No image in frame message"})
                    continue

                await send({"type": "status", "text": "Analyzing your homework..."})

                try:
                    logger.info("DEBUG FRAME: calling vision_agent.process_frame()...")
                    problem_info = await vision_agent.process_frame(
                        session_id, b64,
                        force_reanalyze=msg.get("force_reanalyze", False)
                    )
                    logger.info(f"DEBUG FRAME: vision done — problem={problem_info.get('problem','')[:60]}")

                    err_desc = problem_info.get("error_description", "")
                    if err_desc.startswith("API_ERROR:"):
                        human_error = err_desc[len("API_ERROR:"):].strip()
                        logger.error(f"DEBUG FRAME: API_ERROR: {human_error}")
                        await send({"type": "error", "message": f"Analysis failed: {human_error}."})
                        continue

                    logger.info("DEBUG FRAME: sending frame_analyzed (placeholder)...")
                    await send({
                        "type":        "frame_analyzed",
                        "problem":     problem_info.get("problem", ""),
                        "subject":     problem_info.get("subject", ""),
                        "difficulty":  problem_info.get("difficulty_level", ""),
                        "has_errors":  problem_info.get("has_errors", False),
                        "total_steps": 3,
                    })
                    logger.info("DEBUG FRAME: frame_analyzed sent ✓")

                    logger.info("DEBUG FRAME: calling reasoning_agent.create_teaching_plan()...")
                    if not teaching_plan or msg.get("force_reanalyze", False):
                        teaching_plan = await reasoning_agent.create_teaching_plan(problem_info)
                        step = 0
                    logger.info(f"DEBUG FRAME: teaching plan done — steps={teaching_plan.get('total_steps')}")

                    await send({
                        "type":        "frame_analyzed",
                        "problem":     problem_info.get("problem", ""),
                        "subject":     problem_info.get("subject", ""),
                        "difficulty":  problem_info.get("difficulty_level", ""),
                        "has_errors":  problem_info.get("has_errors", False),
                        "total_steps": teaching_plan.get("total_steps", 1),
                    })

                    if problem_info.get("problem"):
                        logger.info("DEBUG FRAME: calling _do_explain()...")
                        await _do_explain(
                            send, teaching_agent, dialogue_agent,
                            session_id, problem_info, teaching_plan, step
                        )
                        step += 1
                        logger.info("DEBUG FRAME: _do_explain() done ✓")

                except Exception as e:
                    logger.error(f"DEBUG FRAME: exception: {type(e).__name__}: {e}", exc_info=True)
                    await send({"type": "error", "message": f"Analysis failed: {str(e)}"})

            # ── VOICE INPUT ───────────────────────────────────────────
            elif msg_type == "voice_input":
                text = msg.get("text", "").strip()
                if not text:
                    continue
                _interrupted[session_id] = False  # reset so answer streams fully
                prev = dialogue_agent.get_active_explanation(session_id)
                dialogue_agent.clear_active_explanation(session_id)
                await send({"type": "response_start"})
                full = ""
                try:
                    async for chunk in dialogue_agent.handle_user_input(
                        session_id, text,
                        current_explanation=prev,
                        problem_info=problem_info
                    ):
                        full += chunk
                        await send({"type": "text_chunk", "text": chunk})
                except Exception as e:
                    logger.error(f"Voice error: {e}", exc_info=True)
                    full = "Sorry, I had trouble processing that. Could you repeat?"
                    await send({"type": "text_chunk", "text": full})
                await send({
                    "type": "explanation_complete",
                    "full_text": full,
                    "step": step,
                    "follow_up": "Does that make sense?"
                })

            # ── DIAGRAM ───────────────────────────────────────────────
            elif msg_type == "request_diagram":
                concept = msg.get("concept", "") or problem_info.get("problem", "")
                if not concept:
                    continue
                await send({"type": "generating_diagram"})
                try:
                    svg = await teaching_agent.generate_diagram_for_concept(session_id, concept, problem_info=problem_info)
                    logger.info(f"DEBUG DIAGRAM: svg length={len(svg) if svg else 0}")
                    logger.info(f"DEBUG DIAGRAM: svg first 300 chars: {svg[:300] if svg else 'NONE'}")
                    if svg:
                        await send({"type": "diagram", "svg": svg, "concept": concept})
                    else:
                        await send({"type": "error", "message": "Could not generate diagram"})
                except Exception as e:
                    logger.error(f"Diagram error: {e}", exc_info=True)
                    await send({"type": "error", "message": "Diagram generation failed"})

            # ── PRACTICE ──────────────────────────────────────────────
            elif msg_type == "request_practice":
                if not problem_info:
                    continue
                await send({"type": "generating_practice"})
                try:
                    p = await teaching_agent.generate_practice_question(problem_info)
                    await send({
                        "type":     "practice_question",
                        "question": p.get("question", ""),
                        "hint":     p.get("hint", ""),
                        "answer":   p.get("answer", ""),
                    })
                except Exception as e:
                    logger.error(f"Practice error: {e}", exc_info=True)

            # ── NEXT STEP ─────────────────────────────────────────────
            elif msg_type == "next_step":
                if problem_info:
                    try:
                        await _do_explain(
                            send, teaching_agent, dialogue_agent,
                            session_id, problem_info, teaching_plan, step
                        )
                        step = min(step + 1, teaching_plan.get("total_steps", 1) - 1)
                    except Exception as e:
                        logger.error(f"Next step error: {e}", exc_info=True)

            # ── RESET ─────────────────────────────────────────────────
            elif msg_type == "new_session":
                problem_info  = {}
                teaching_plan = {}
                step          = 0
                _interrupted[session_id] = False
                vision_agent.clear_cache(session_id)
                dialogue_agent.clear_active_explanation(session_id)
                try:
                    await _firestore.create_session(session_id)
                except Exception:
                    pass
                await send({"type": "session_reset", "message": "Session reset!"})

            else:
                logger.warning(f"DEBUG LOOP: unknown msg_type={msg_type}")

    except Exception as e:
        logger.error(f"DEBUG: Unhandled loop error [{session_id[:8]}]: {type(e).__name__}: {e}", exc_info=True)

    finally:
        _interrupted.pop(session_id, None)

    logger.info(f"DEBUG: WS session ended: {session_id[:8]}")


async def _do_explain(send, teaching_agent, dialogue_agent,
                      session_id, problem_info, teaching_plan, step):
    logger.info(f"DEBUG _do_explain: step={step}")

    # Reset interrupt flag at start of every new explanation
    _interrupted[session_id] = False

    steps     = teaching_plan.get("steps", [])
    step_data = steps[step] if step < len(steps) else {}

    await send({
        "type":        "explanation_start",
        "step":        step,
        "step_title":  step_data.get("title", f"Step {step + 1}"),
        "total_steps": teaching_plan.get("total_steps", 1),
    })

    logger.info("DEBUG _do_explain: calling generate_step_response()...")
    try:
        response = await teaching_agent.generate_step_response(
            session_id, problem_info, step_data, step
        )
        logger.info(f"DEBUG _do_explain: done, text={len(response.get('text',''))} diagram={bool(response.get('diagram_svg'))}")
    except Exception as e:
        logger.error(f"DEBUG _do_explain: FAILED: {type(e).__name__}: {e}", exc_info=True)
        await send({"type": "text_chunk", "text": "I had trouble generating the explanation. Please try again."})
        await send({"type": "explanation_complete", "full_text": "", "step": step, "follow_up": ""})
        return

    text = response.get("text", "")
    dialogue_agent.set_active_explanation(session_id, text)

    words = text.split()
    for i in range(0, len(words), 8):
        if _interrupted.get(session_id):
            logger.info(f"_do_explain: interrupted mid-stream for {session_id[:8]}")
            return
        chunk = " ".join(words[i:i + 8]) + " "
        await send({"type": "text_chunk", "text": chunk})

    if _interrupted.get(session_id):
        return

    # ── Send diagram simultaneously with explanation if generated ──
    if response.get("diagram_svg"):
        logger.info(f"DEBUG _do_explain: sending inline diagram, length={len(response['diagram_svg'])}")
        await send({
            "type":    "diagram",
            "svg":     response["diagram_svg"],
            "concept": step_data.get("title", ""),
        })

    await send({
        "type":         "explanation_complete",
        "full_text":    text,
        "step":         step,
        "is_last_step": response.get("is_last_step", False),
        "follow_up":    teaching_plan.get("follow_up_question", "Does that make sense?"),
    })
    logger.info("DEBUG _do_explain: complete ✓")