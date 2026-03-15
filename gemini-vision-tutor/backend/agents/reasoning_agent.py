"""
Reasoning Agent - Solves problems and creates structured explanation steps
"""

import asyncio
import json
import logging
import google.generativeai as genai
from services.gemini_service import GeminiService
from services.firestore_service import FirestoreService

logger = logging.getLogger(__name__)


class ReasoningAgent:
    def __init__(self, gemini: GeminiService, firestore: FirestoreService):
        self.gemini = gemini
        self.firestore = firestore

    async def create_teaching_plan(self, problem_info: dict) -> dict:
        """Create a structured teaching plan. Returns dict with steps."""

        prompt = (
            "Create a short teaching plan as JSON.\n"
            "Rules:\n"
            "- Respond ONLY with raw JSON\n"
            "- No markdown, no code fences, no explanation outside JSON\n"
            "- Keep ALL string values under 80 characters\n"
            "- Maximum 3 steps\n\n"
            f"Subject: {problem_info.get('subject', 'unknown')}\n"
            f"Problem: {problem_info.get('problem', 'unknown')}\n"
            f"Level: {problem_info.get('difficulty_level', 'unknown')}\n"
        )

        if problem_info.get('has_errors'):
            prompt += f"Error: {problem_info.get('error_description', '')[:80]}\n"

        prompt += (
            "\nJSON structure to return:\n"
            '{"total_steps":2,'
            '"steps":['
            '{"step_number":1,"title":"Step title","explanation":"Brief what to cover",'
            '"needs_diagram":false,"diagram_description":""},'
            '{"step_number":2,"title":"Step title","explanation":"Brief what to cover",'
            '"needs_diagram":false,"diagram_description":""}'
            '],'
            '"final_answer":"the answer",'
            '"common_mistakes":["mistake1"],'
            '"follow_up_question":"short question"}'
        )

        raw = ""
        try:
            response = await asyncio.wait_for(
                self.gemini.vision_model.generate_content_async(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.2,
                        max_output_tokens=2048,  # raised from 1024
                    )
                ),
                timeout=10.0
            )

            raw = response.text.strip()
            logger.info(f"Teaching plan raw (first 300 chars): {raw[:300]}")

            text = self.gemini._strip_fences(raw)
            plan = json.loads(text)
            logger.info(f"Teaching plan created: {plan.get('total_steps')} steps")
            return plan

        except asyncio.TimeoutError:
            logger.error("create_teaching_plan timed out — using fallback")
            return self._fallback_plan(problem_info)

        except json.JSONDecodeError as e:
            logger.error(f"Teaching plan JSON parse error: {e}")
            logger.error(f"Full raw response: {raw}")
            return self._fallback_plan(problem_info)

        except Exception as e:
            logger.error(f"Teaching plan error: {e}", exc_info=True)
            return self._fallback_plan(problem_info)

    def _fallback_plan(self, problem_info: dict) -> dict:
        approach = problem_info.get("suggested_approach", "Let me explain step by step.")[:80]
        return {
            "total_steps": 3,
            "steps": [
                {
                    "step_number": 1,
                    "title": "Understanding the problem",
                    "explanation": approach,
                    "needs_diagram": False,
                    "diagram_description": ""
                },
                {
                    "step_number": 2,
                    "title": "Working through it",
                    "explanation": "Let's solve this step by step.",
                    "needs_diagram": False,
                    "diagram_description": ""
                },
                {
                    "step_number": 3,
                    "title": "Final answer",
                    "explanation": "Let's verify our answer.",
                    "needs_diagram": False,
                    "diagram_description": ""
                }
            ],
            "final_answer": "Let's work through this together.",
            "common_mistakes": [],
            "follow_up_question": "Does this make sense so far?"
        }

    async def get_step_explanation(
        self,
        session_id: str,
        problem_info: dict,
        step_number: int,
        teaching_plan: dict
    ):
        history = await self.firestore.get_conversation_history(session_id)
        async for chunk in self.gemini.generate_explanation(
            problem_info, history, step=step_number
        ):
            yield chunk