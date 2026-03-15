"""
Tutor REST API Routes - HTTP endpoints for non-realtime operations
Session management, history retrieval, diagram requests
"""

import logging
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import Optional

logger = logging.getLogger(__name__)
router = APIRouter()


class AnalyzeRequest(BaseModel):
    image: str  # base64 encoded image
    session_id: Optional[str] = None


class DiagramRequest(BaseModel):
    concept: str
    session_id: str


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, request: Request):
    """Retrieve a tutoring session with full conversation history."""
    firestore = request.app.state.firestore
    session = await firestore.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return session


@router.get("/sessions/{session_id}/history")
async def get_conversation_history(session_id: str, request: Request):
    """Get conversation history for a session."""
    firestore = request.app.state.firestore
    history = await firestore.get_conversation_history(session_id)
    return {"session_id": session_id, "history": history, "count": len(history)}


@router.post("/analyze")
async def analyze_frame(body: AnalyzeRequest, request: Request):
    """
    One-shot frame analysis endpoint (for non-WebSocket clients).
    Returns problem info from a base64 image.
    """
    import uuid
    gemini = request.app.state.gemini
    firestore = request.app.state.firestore

    session_id = body.session_id or str(uuid.uuid4())

    from agents.vision_agent import VisionAgent
    vision_agent = VisionAgent(gemini, firestore)

    problem_info = await vision_agent.process_frame(session_id, body.image)
    return {
        "session_id": session_id,
        "problem_info": problem_info
    }


@router.post("/diagram")
async def generate_diagram(body: DiagramRequest, request: Request):
    """Generate an SVG diagram for a concept."""
    gemini = request.app.state.gemini

    svg = await gemini.generate_diagram_code(body.concept)
    return {"svg": svg, "concept": body.concept}


@router.get("/models")
async def list_models():
    """List available Gemini models."""
    return {
        "models": [
            {"id": "gemini-1.5-pro", "use": "vision, reasoning, complex tasks"},
            {"id": "gemini-1.5-flash", "use": "fast dialogue, simple queries"},
        ]
    }
