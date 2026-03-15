"""
Vision Agent - Processes camera frames and extracts problem information
First agent in the pipeline: sees → understands → hands off to Reasoning Agent
"""

import logging
from typing import Optional
from services.gemini_service import GeminiService
from services.firestore_service import FirestoreService

logger = logging.getLogger(__name__)


class VisionAgent:
    """
    Analyzes webcam frames to extract homework/problem information.
    Uses Gemini Vision to understand what the student is showing.
    """

    def __init__(self, gemini: GeminiService, firestore: FirestoreService):
        self.gemini = gemini
        self.firestore = firestore
        self._frame_buffer = {}  # Buffer frames per session to avoid re-analysis

    async def process_frame(
        self,
        session_id: str,
        image_base64: str,
        force_reanalyze: bool = False
    ) -> dict:
        """
        Process a camera frame and extract problem information.

        Args:
            session_id: Current tutoring session ID
            image_base64: Base64-encoded JPEG frame
            force_reanalyze: Force re-analysis even if cached

        Returns:
            Problem information dict
        """
        # Check if we recently analyzed this session (avoid redundant API calls)
        if not force_reanalyze and session_id in self._frame_buffer:
            cached = self._frame_buffer[session_id]
            logger.debug(f"Using cached frame analysis for session {session_id}")
            return cached

        # Get conversation context to help with interpretation
        history = await self.firestore.get_conversation_history(session_id)
        context = ""
        if history:
            last_turns = history[-4:]
            context = " | ".join([f"{t['role']}: {t['content'][:100]}" for t in last_turns])

        # Analyze the frame with Gemini Vision
        logger.info(f"Analyzing frame for session {session_id}")
        problem_info = await self.gemini.analyze_frame(image_base64, context)

        # Store the analysis
        await self.firestore.update_problem_info(session_id, problem_info)

        # Cache for this session
        self._frame_buffer[session_id] = problem_info

        logger.info(f"Frame analyzed: {problem_info.get('subject')} - {problem_info.get('problem', '')[:50]}")
        return problem_info

    def clear_cache(self, session_id: str):
        """Clear cached frame analysis for a session."""
        self._frame_buffer.pop(session_id, None)

    async def is_problem_visible(self, image_base64: str) -> bool:
        """Quick check if the frame contains readable content."""
        try:
            info = await self.gemini.analyze_frame(image_base64)
            return info.get("problem", "").lower() not in [
                "", "unknown", "could not analyze image clearly"
            ]
        except Exception:
            return False
