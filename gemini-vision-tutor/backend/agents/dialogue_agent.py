"""
Dialogue Agent - Manages conversation flow, interruptions, and natural dialogue
"""

import logging
from typing import AsyncIterator
import google.generativeai as genai
from services.gemini_service import GeminiService
from services.firestore_service import FirestoreService

logger = logging.getLogger(__name__)

INTENTS = {
    "interruption": ["wait", "stop", "hold on", "why", "what do you mean", "i don't understand", "confused", "huh"],
    "confirmation":  ["yes", "ok", "got it", "i see", "makes sense", "understand", "sure", "okay"],
    "next_step":     ["next", "continue", "go on", "what's next", "and then", "keep going"],
    "question":      ["?", "how", "what", "why", "when", "where", "explain", "can you"],
    "answer":        ["the answer is", "i think", "i got", "my answer", "is it", "equals"],
}


class DialogueAgent:
    def __init__(self, gemini: GeminiService, firestore: FirestoreService):
        self.gemini = gemini
        self.firestore = firestore
        self._active_explanations: dict[str, str] = {}

    def classify_intent(self, message: str) -> str:
        msg = message.lower()
        for intent, keywords in INTENTS.items():
            if any(kw in msg for kw in keywords):
                return intent
        return "general"

    def _is_related_to_problem(self, question: str, problem_info: dict) -> bool:
        """
        Check if the student's question is related to the current problem.
        If not, we should answer it as a standalone question without
        dragging the math problem context into the response.
        """
        if not problem_info:
            return False

        subject  = (problem_info.get('subject', '') or '').lower()
        problem  = (problem_info.get('problem', '') or '').lower()
        concepts = [c.lower() for c in problem_info.get('key_concepts', [])]
        q        = question.lower()

        # Check if any key term from the current problem appears in the question
        subject_words = [w for w in subject.split() if len(w) > 3]
        problem_words = [w for w in problem.split() if len(w) > 3]
        all_terms     = subject_words + problem_words + concepts

        return any(term in q for term in all_terms)

    async def handle_user_input(
        self,
        session_id: str,
        user_message: str,
        current_explanation: str = "",
        problem_info: dict = None
    ) -> AsyncIterator[str]:
        """Route user input to appropriate handler. Async generator."""
        await self.firestore.add_message(session_id, "user", user_message)

        intent = self.classify_intent(user_message)
        logger.info(f"Intent: {intent} | '{user_message[:60]}'")

        if intent == "interruption" and current_explanation:
            async for chunk in self.gemini.handle_interruption(
                user_message, current_explanation, problem_info or {}
            ):
                yield chunk

        elif intent == "confirmation":
            async for chunk in self._handle_confirmation(session_id, problem_info):
                yield chunk

        elif intent in ("question", "general", "interruption"):
            async for chunk in self._handle_question(session_id, user_message, problem_info):
                yield chunk

        elif intent == "answer":
            async for chunk in self._evaluate_answer(session_id, user_message, problem_info):
                yield chunk

        elif intent == "next_step":
            yield "Great! Let's move on. "
            async for chunk in self._continue_explanation(session_id, problem_info):
                yield chunk

        else:
            async for chunk in self._handle_question(session_id, user_message, problem_info):
                yield chunk

    async def _handle_confirmation(self, session_id: str, problem_info: dict) -> AsyncIterator[str]:
        history = await self.firestore.get_conversation_history(session_id)
        prompt = (
            f"The student confirmed they understand.\n"
            f"Problem: {problem_info.get('problem', '') if problem_info else ''}\n"
            f"Recent: {str(history[-2:])[:200]}\n\n"
            "Give a brief encouraging response (1-2 sentences) and ask if they're ready for the next step."
        )
        response = await self.gemini.dialogue_model.generate_content_async(
            prompt,
            generation_config=genai.GenerationConfig(temperature=0.8, max_output_tokens=150),
            stream=True
        )
        async for chunk in response:
            if chunk.text:
                yield chunk.text

    async def _handle_question(self, session_id: str, question: str, problem_info: dict) -> AsyncIterator[str]:
        related = self._is_related_to_problem(question, problem_info)
        logger.info(f"_handle_question: related_to_problem={related} | '{question[:60]}'")

        if related and problem_info:
            # Question is about the current problem — include context
            history = await self.firestore.get_conversation_history(session_id)
            history_text = "\n".join(
                f"{'Student' if t['role']=='user' else 'Tutor'}: {t['content'][:100]}"
                for t in history[-4:]
            )
            prompt = (
                f'Student question: "{question}"\n\n'
                f"Current problem: {problem_info.get('problem', '')}\n"
                f"Subject: {problem_info.get('subject', '')}\n\n"
                f"Recent conversation:\n{history_text}\n\n"
                "Answer the student's question directly and clearly in 2-4 sentences. "
                "Plain text only, no markdown. Do NOT redirect them back to the problem unless it is directly relevant to their question."
            )
        else:
            # Question is unrelated to the current problem — answer it standalone
            # Do NOT inject the math problem into the prompt at all
            prompt = (
                f'Student question: "{question}"\n\n'
                "Answer this question directly and clearly in 2-4 sentences. "
                "Plain text only, no markdown. "
                "Do NOT mention any previous math problems or redirect to other topics. "
                "Just answer what was asked."
            )

        response = await self.gemini.dialogue_model.generate_content_async(
            prompt,
            generation_config=genai.GenerationConfig(temperature=0.7, max_output_tokens=300),
            stream=True
        )
        async for chunk in response:
            if chunk.text:
                yield chunk.text

    async def _evaluate_answer(self, session_id: str, student_answer: str, problem_info: dict) -> AsyncIterator[str]:
        prompt = (
            f'Student answered: "{student_answer}"\n'
            f"Problem: {problem_info.get('problem', '') if problem_info else ''}\n"
            f"Approach: {problem_info.get('suggested_approach', '') if problem_info else ''}\n\n"
            "Evaluate their answer:\n"
            "- Correct: celebrate and confirm\n"
            "- Partial: acknowledge what's right, gently fix what's wrong\n"
            "- Wrong: be encouraging, explain the error, guide them\n\n"
            "Keep it to 2-4 sentences, plain text."
        )
        response = await self.gemini.dialogue_model.generate_content_async(
            prompt,
            generation_config=genai.GenerationConfig(temperature=0.7, max_output_tokens=250),
            stream=True
        )
        async for chunk in response:
            if chunk.text:
                yield chunk.text

    async def _continue_explanation(self, session_id: str, problem_info: dict) -> AsyncIterator[str]:
        history = await self.firestore.get_conversation_history(session_id)
        async for chunk in self.gemini.generate_explanation(problem_info or {}, history, step=1):
            yield chunk

    def set_active_explanation(self, session_id: str, text: str):
        self._active_explanations[session_id] = text

    def get_active_explanation(self, session_id: str) -> str:
        return self._active_explanations.get(session_id, "")

    def clear_active_explanation(self, session_id: str):
        self._active_explanations.pop(session_id, None)