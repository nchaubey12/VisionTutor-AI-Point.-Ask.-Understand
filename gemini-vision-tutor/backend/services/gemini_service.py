"""
Gemini Service - Google Gemini API integration
Compatible with google-generativeai >= 0.7.0
"""

import os
import base64
import json
import logging
from typing import AsyncIterator

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.0-flash"


def _load_genai():
    try:
        import google.generativeai as genai
        return genai
    except ImportError:
        raise ImportError("google-generativeai not installed. Run: pip install google-generativeai==0.8.3")


class GeminiService:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")

        model_name = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)

        self.genai = _load_genai()
        self.genai.configure(api_key=api_key)

        tutor_system = (
            "You are an expert AI tutor that analyzes student homework. "
            "Be encouraging, patient, and clear. "
            "When you see homework, identify the subject, extract the problem, "
            "and give structured step-by-step explanations."
        )

        dialogue_system = (
            "You are a patient, encouraging AI tutor. "
            "Keep responses concise (2-4 sentences). Be warm and supportive."
        )

        self.vision_model = self.genai.GenerativeModel(
            model_name=model_name,
            system_instruction=tutor_system
        )

        self.dialogue_model = self.genai.GenerativeModel(
            model_name=model_name,
            system_instruction=dialogue_system
        )

        # Dedicated model for diagram solving — strict JSON/equation only output
        self.solver_model = self.genai.GenerativeModel(
            model_name=model_name,
            system_instruction=(
                "You are a math solver that outputs ONLY raw JSON. "
                "Never output prose, markdown, backticks, or explanations. "
                "Every label and value field must contain only equations or numbers — "
                "never sentences or descriptions. "
                "Your entire response must start with { and end with }."
            )
        )

        logger.info(f"GeminiService initialized with model: {model_name}")

    def _make_config(self, temperature: float, max_tokens: int):
        return self.genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    def _strip_fences(self, text: str) -> str:
        """Remove markdown code fences — handles ```json, ```svg, ```xml etc."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]  # remove opening fence line
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text

    def _fallback_info(self, error_msg: str = "") -> dict:
        problem_text = "Could not read the image. Please ensure good lighting and hold camera steady."
        err_field = f"API_ERROR: {error_msg}" if error_msg else ""
        if error_msg:
            problem_text = (
                f"Analysis failed ({error_msg}). "
                "Check the server logs and verify your API key / model name."
            )
        return {
            "subject": "unknown",
            "problem": problem_text,
            "current_work": "",
            "difficulty_level": "unknown",
            "suggested_approach": "Try again with better lighting" if not error_msg else "Check API key and model name",
            "key_concepts": [],
            "has_errors": bool(error_msg),
            "error_description": err_field,
        }

    async def analyze_frame(self, image_base64: str, context: str = "") -> dict:
        raw_text = ""
        try:
            if "," in image_base64:
                image_base64 = image_base64.split(",", 1)[1]
            image_bytes = base64.b64decode(image_base64)

            prompt = (
                "Analyze this homework image carefully.\n"
                "Respond ONLY with raw JSON, no markdown, no explanation, no code fences.\n"
                "Use exactly this structure:\n"
                '{"subject":"math","problem":"exact problem text",'
                '"current_work":"what student wrote so far",'
                '"difficulty_level":"elementary or middle or high or college",'
                '"suggested_approach":"brief solution outline",'
                '"key_concepts":["concept1","concept2"],'
                '"has_errors":false,"error_description":""}\n\n'
                "If no homework visible, set problem to: "
                '"No homework visible - please point camera at your work"'
            )
            if context:
                prompt += f"\n\nPrevious context: {context}"

            response = await self.vision_model.generate_content_async(
                [prompt, {"mime_type": "image/jpeg", "data": image_bytes}],
                generation_config=self._make_config(0.1, 1024)
            )

            raw_text = response.text.strip()
            logger.info(f"analyze_frame raw response (first 200 chars): {raw_text[:200]}")

            cleaned = self._strip_fences(raw_text)
            return json.loads(cleaned)

        except json.JSONDecodeError as e:
            logger.error(f"analyze_frame: JSON decode error — {e}\nRaw: {raw_text[:500]}")
            return self._fallback_info()

        except Exception as e:
            error_summary = f"{type(e).__name__}: {e}"
            logger.error(f"analyze_frame: API error — {error_summary}", exc_info=True)
            return self._fallback_info(error_msg=error_summary)

    async def generate_explanation(
        self,
        problem_info: dict,
        conversation_history: list,
        step: int = 0
    ) -> AsyncIterator[str]:
        """Async generator - streams a rich detailed explanation."""
        ctx = (
            f"Subject: {problem_info.get('subject', 'unknown')}\n"
            f"Problem: {problem_info.get('problem', 'unknown')}\n"
            f"Student work: {problem_info.get('current_work', 'none')}\n"
            f"Difficulty: {problem_info.get('difficulty_level', 'unknown')}\n"
            f"Key concepts: {', '.join(problem_info.get('key_concepts', []))}\n"
            f"Suggested approach: {problem_info.get('suggested_approach', '')}"
        )
        if problem_info.get('has_errors'):
            ctx += f"\nError to address: {problem_info.get('error_description')}"

        history_text = "\n".join(
            f"{'Student' if t['role'] == 'user' else 'Tutor'}: {t['content'][:100]}"
            for t in conversation_history[-6:]
        )

        if step == 0:
            step_note = (
                "This is the first explanation. Greet the student warmly, "
                "identify what they need to solve, and walk through step 1 in detail."
            )
        else:
            step_note = f"Continue the explanation with step {step + 1}. Build on what was covered before."

        prompt = (
            f"Problem to explain:\n{ctx}\n\n"
            f"Recent conversation:\n{history_text or 'None yet'}\n\n"
            f"{step_note}\n\n"
            "Write a thorough, encouraging explanation (aim for 4-6 sentences minimum). "
            "Break down each step clearly. Use simple language a student can follow.\n"
            "Plain text only — no markdown, no bullet points (this is read aloud).\n"
            "If a visual diagram would genuinely help understanding, add on a new line: "
            "[DIAGRAM]: brief description of what to draw\n"
            "Only add [DIAGRAM] if it truly helps — not for every problem."
        )

        response = await self.vision_model.generate_content_async(
            prompt,
            generation_config=self._make_config(0.7, 1024),
            stream=True
        )
        async for chunk in response:
            if chunk.text:
                yield chunk.text

    async def handle_interruption(
        self,
        user_question: str,
        current_explanation: str,
        problem_info: dict
    ) -> AsyncIterator[str]:
        prompt = (
            f'Student interrupted while you said: "{current_explanation[:200]}"\n\n'
            f'Student asked: "{user_question}"\n'
            f"Problem: {problem_info.get('problem', '')}\n\n"
            "Respond naturally in 2-3 sentences, plain text, no markdown."
        )
        response = await self.dialogue_model.generate_content_async(
            prompt,
            generation_config=self._make_config(0.8, 256),
            stream=True
        )
        async for chunk in response:
            if chunk.text:
                yield chunk.text

    async def generate_diagram_code(self, description: str, problem_info: dict = None) -> str:
        """
        Two-step approach:
        1. Ask Gemini to SOLVE the problem and return solution steps as JSON
        2. Build the SVG from those solution steps — never shows the question
        """
        logger.info(f"generate_diagram_code called for: {description[:80]}")

        problem  = problem_info.get('problem', description) if problem_info else description
        subject  = problem_info.get('subject', 'math') if problem_info else 'math'
        approach = problem_info.get('suggested_approach', '') if problem_info else ''

        # ── Step 1: Solve using dedicated solver model ────────────────────────
        solve_prompt = (
            f'Problem: {problem}\n\n'
            f"Solve completely. Output this exact JSON structure — no other text:\n"
            f'{{"title":"Solving {problem[:20]}","steps":['
            f'{{"label":"2x+6-6=14-6","value":"2x=8","color":"#3B82F6"}},'
            f'{{"label":"2x/2=8/2","value":"x=4","color":"#10B981"}}'
            f'],"answer":"x = 4"}}\n\n'
            f"Now do the same for the actual problem above.\n"
            f"label = the equation with the operation applied (e.g. '2x+6-6=14-6')\n"
            f"value = the simplified result (e.g. '2x=8')\n"
            f"ONLY equations in label/value — no words whatsoever.\n"
            f"2-4 steps max. answer = final answer only."
        )

        try:
            import asyncio as _asyncio
            solve_response = await _asyncio.wait_for(
                self.solver_model.generate_content_async(
                    solve_prompt,
                    generation_config=self._make_config(0.0, 512)
                ),
                timeout=12.0
            )
            raw = solve_response.text.strip()
            logger.info(f"solution raw response: {raw[:400]}")

            # Robust JSON extraction
            import re as _re
            cleaned = self._strip_fences(raw)
            if not cleaned.startswith('{'):
                match = _re.search(r'\{.*\}', cleaned, _re.DOTALL)
                if match:
                    cleaned = match.group(0)
            solved = json.loads(cleaned)
            logger.info(f"solution parsed OK: {solved.get('title')} — {len(solved.get('steps', []))} steps")

            colors = ["#3B82F6", "#10B981", "#8B5CF6", "#F59E0B", "#EF4444"]
            items = []
            for i, s in enumerate(solved.get("steps", [])[:4]):
                label = str(s.get("label", ""))
                value = str(s.get("value", ""))
                # Reject prose — if label contains spaces and no = or operators, skip
                has_equation = any(c in label for c in "=+-*/")
                if not has_equation and len(label.split()) > 3:
                    logger.warning(f"Skipping prose label: {label}")
                    continue
                items.append({
                    "label": label,
                    "value": value,
                    "color": s.get("color", colors[i % len(colors)]),
                })

            if not items:
                raise ValueError("No valid equation steps returned")

            data = {
                "title": solved.get("title", f"Solving {problem[:20]}")[:30],
                "type": "steps",
                "items": items,
                "explanation": solved.get("answer", "")[:80],
            }

        except Exception as e:
            logger.error(f"solve step failed: {type(e).__name__}: {e} — building fallback from approach")
            # Better fallback: use suggested_approach split into lines if available
            if approach:
                fallback_items = []
                colors = ["#3B82F6", "#10B981", "#8B5CF6", "#F59E0B", "#EF4444"]
                for i, line in enumerate(approach.split('.')[:4]):
                    line = line.strip()
                    if line:
                        fallback_items.append({"label": f"Step {i+1}", "value": line[:18], "color": colors[i % len(colors)]})
                data = {
                    "title": "Solution",
                    "type": "steps",
                    "items": fallback_items or [{"label": "See explanation", "value": "in main panel", "color": "#3B82F6"}],
                    "explanation": approach[:80],
                }
            else:
                data = {
                    "title": "Solution",
                    "type": "steps",
                    "items": [{"label": "See explanation", "value": "in main panel", "color": "#3B82F6"}],
                    "explanation": "Check the explanation panel for the full solution.",
                }

        svg = self._build_svg(data, description)
        logger.info(f"generate_diagram_code final SVG length: {len(svg)}")
        return svg

    async def generate_practice_question(self, problem_info: dict) -> dict:
        """
        Generate a practice question based on the same concept as the problem.
        Returns dict with question, hint, answer.
        Fast: single Gemini call with JSON response, 5s timeout.
        """
        logger.info(f"generate_practice_question for: {problem_info.get('problem', '')[:60]}")

        prompt = (
            f"A student just solved this problem:\n"
            f"Subject: {problem_info.get('subject', 'unknown')}\n"
            f"Problem: {problem_info.get('problem', 'unknown')}\n"
            f"Difficulty: {problem_info.get('difficulty_level', 'unknown')}\n"
            f"Key concepts: {', '.join(problem_info.get('key_concepts', []))}\n\n"
            "Create ONE similar practice problem to test understanding.\n"
            "Respond ONLY with raw JSON, no markdown, no code fences.\n"
            "Use exactly this structure:\n"
            '{"question":"the practice question",'
            '"hint":"a helpful hint without giving away the answer",'
            '"answer":"the complete worked answer"}'
            "\n\nKeep the question similar in difficulty. Make it slightly different so they can't just copy the original answer."
        )

        import asyncio
        try:
            response = await asyncio.wait_for(
                self.vision_model.generate_content_async(
                    prompt,
                    generation_config=self._make_config(0.7, 512)
                ),
                timeout=15.0
            )

            raw = response.text.strip()
            logger.info(f"practice question raw: {raw[:200]}")
            data = json.loads(self._strip_fences(raw))

            # Validate required fields
            return {
                "question": data.get("question", "Try solving a similar problem."),
                "hint":     data.get("hint", "Think about the key concepts."),
                "answer":   data.get("answer", "Work through it step by step."),
            }

        except asyncio.TimeoutError:
            logger.error("generate_practice_question timed out")
            return self._fallback_practice(problem_info)

        except json.JSONDecodeError as e:
            logger.error(f"practice question JSON error: {e}")
            return self._fallback_practice(problem_info)

        except Exception as e:
            logger.error(f"generate_practice_question error: {e}", exc_info=True)
            return self._fallback_practice(problem_info)

    def _fallback_practice(self, problem_info: dict) -> dict:
        subject = problem_info.get("subject", "this topic")
        return {
            "question": f"Try another problem similar to: {problem_info.get('problem', 'the one you just solved')[:80]}",
            "hint": f"Use the same approach you learned for {subject}.",
            "answer": "Work through it step by step using the same method.",
        }

    def _build_svg(self, data: dict, description: str) -> str:
        """Clean minimal layout — white background, no colored boxes."""
        title       = data.get("title", "Solution")[:40]
        items       = data.get("items", [])[:5]
        explanation = data.get("explanation", "")
        n           = len(items)

        ITEM_H = 70
        GAP    = 8
        TOP    = 56
        BOTTOM = 60 if explanation else 16
        W      = 560
        H      = TOP + n * (ITEM_H + GAP) + BOTTOM
        H      = max(H, 200)

        lines = [
            f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg">',
            f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
            # Title
            f'<text x="{W//2}" y="36" text-anchor="middle" '
            f'font-family="Arial,sans-serif" font-size="18" font-weight="bold" fill="#1e293b">'
            f'{_esc(title)}</text>',
            # Divider under title
            f'<line x1="24" y1="46" x2="{W-24}" y2="46" stroke="#e2e8f0" stroke-width="1.5"/>',
        ]

        for i, item in enumerate(items):
            label = str(item.get("label", ""))  # e.g. "2x+6-6=14-6"
            value = str(item.get("value", ""))  # e.g. "2x=8"
            iy    = TOP + i * (ITEM_H + GAP)

            # Step label header — "Step 1" in grey
            lines.append(
                f'<text x="24" y="{iy + 18}" '
                f'font-family="Arial,sans-serif" font-size="12" font-weight="bold" fill="#94a3b8">'
                f'Step {i+1}</text>'
            )

            # Working equation (label) in medium grey
            label_text = _esc(label[:52])
            lines.append(
                f'<text x="24" y="{iy + 36}" '
                f'font-family="Arial,sans-serif" font-size="14" fill="#64748b">'
                f'{label_text}</text>'
            )
            if len(label) > 52:
                lines.append(
                    f'<text x="24" y="{iy + 52}" '
                    f'font-family="Arial,sans-serif" font-size="14" fill="#64748b">'
                    f'{_esc(label[52:104])}</text>'
                )

            # Result (value) in bold black — the simplified equation
            val_y = iy + 56 if len(label) <= 52 else iy + 68
            lines.append(
                f'<text x="24" y="{val_y}" '
                f'font-family="Arial,sans-serif" font-size="20" font-weight="bold" fill="#1e293b">'
                f'{_esc(value[:48])}</text>'
            )

            # Divider between steps
            if i < n - 1:
                div_y = iy + ITEM_H + GAP - 2
                lines.append(
                    f'<line x1="24" y1="{div_y}" x2="{W-24}" y2="{div_y}" '
                    f'stroke="#f1f5f9" stroke-width="1"/>'
                )

        # Final answer
        if explanation:
            ans_y = TOP + n * (ITEM_H + GAP) + 10
            lines += [
                f'<line x1="24" y1="{ans_y - 6}" x2="{W-24}" y2="{ans_y - 6}" stroke="#e2e8f0" stroke-width="1.5"/>',
                # Green tick circle
                f'<circle cx="36" cy="{ans_y + 14}" r="12" fill="#10b981"/>',
                f'<text x="36" y="{ans_y + 19}" text-anchor="middle" '
                f'font-family="Arial,sans-serif" font-size="14" font-weight="bold" fill="#ffffff">✓</text>',
                # Answer text
                f'<text x="56" y="{ans_y + 10}" '
                f'font-family="Arial,sans-serif" font-size="13" font-weight="bold" fill="#10b981">Answer:</text>',
                f'<text x="110" y="{ans_y + 10}" '
                f'font-family="Arial,sans-serif" font-size="13" font-weight="bold" fill="#1e293b">{_esc(explanation[:60])}</text>',
            ]
            if len(explanation) > 60:
                lines.append(
                    f'<text x="56" y="{ans_y + 26}" '
                    f'font-family="Arial,sans-serif" font-size="13" fill="#1e293b">{_esc(explanation[60:120])}</text>'
                )

        lines.append('</svg>')
        return "\n".join(lines)


def _esc(text: str) -> str:
    """Escape special XML characters for safe SVG text."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )