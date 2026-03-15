"""
Teaching Agent - Converts reasoning into clear explanations and diagrams
"""

import logging
from typing import Optional
from services.gemini_service import GeminiService
from services.firestore_service import FirestoreService
from services.storage_service import StorageService

logger = logging.getLogger(__name__)


class TeachingAgent:
    def __init__(self, gemini: GeminiService, firestore: FirestoreService, storage: StorageService):
        self.gemini = gemini
        self.firestore = firestore
        self.storage = storage

    async def generate_step_response(
        self,
        session_id: str,
        problem_info: dict,
        step: dict,
        step_number: int
    ) -> dict:
        """
        Generate a complete response for one teaching step.
        Returns dict with text, diagram_svg, step_number.
        """
        response = {
            "text": "",
            "diagram_svg": None,
            "diagram_url": None,
            "step_number": step_number,
            "is_last_step": False
        }

        # Collect streamed explanation
        history = await self.firestore.get_conversation_history(session_id)
        parts = []
        async for chunk in self.gemini.generate_explanation(problem_info, history, step=step_number):
            parts.append(chunk)

        full_text = "".join(parts)

        # Check for diagram request embedded in explanation
        if "[DIAGRAM]:" in full_text:
            text_part, _, diagram_desc = full_text.partition("[DIAGRAM]:")
            response["text"] = text_part.strip()
            if diagram_desc.strip():
                try:
                    svg = await self.gemini.generate_diagram_code(diagram_desc.strip())
                    response["diagram_svg"] = svg
                    url = await self.storage.upload_diagram(session_id, svg)
                    response["diagram_url"] = url
                except Exception as e:
                    logger.error(f"Diagram generation failed: {e}")
        else:
            response["text"] = full_text

        # Save to conversation history
        await self.firestore.add_message(
            session_id, "assistant", response["text"],
            {"step": step_number, "has_diagram": bool(response["diagram_svg"])}
        )

        return response

    async def generate_diagram_for_concept(self, session_id: str, concept: str) -> Optional[str]:
        """Generate a standalone diagram for a concept. Returns SVG string."""
        try:
            svg = await self.gemini.generate_diagram_code(concept)
            await self.storage.upload_diagram(session_id, svg)
            return svg
        except Exception as e:
            logger.error(f"Diagram generation failed: {e}")
            return None

    async def generate_practice_question(self, problem_info: dict) -> dict:
        return await self.gemini.generate_practice_question(problem_info)
