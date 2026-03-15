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

    async def generate_diagram_code(self, description: str) -> str:
        """
        Two-step approach:
        1. Ask Gemini for diagram DATA as JSON (what to draw, not SVG code)
        2. Build the SVG ourselves in Python — guaranteed complete and valid
        """
        logger.info(f"generate_diagram_code called for: {description[:80]}")

        # Step 1: Get diagram data as JSON
        prompt = (
            f'For this educational concept: "{description}"\n\n'
            "Describe a simple diagram using JSON. "
            "Respond ONLY with raw JSON, no markdown, no code fences.\n\n"
            "Use this exact structure:\n"
            '{"title":"short title under 30 chars",'
            '"type":"equation or steps or comparison or definition",'
            '"items":['
            '{"label":"text to show","value":"optional number or formula","color":"#3B82F6"},'
            '{"label":"text to show","value":"optional number or formula","color":"#10B981"}'
            '],'
            '"explanation":"one sentence summary"}'
            "\n\nMaximum 5 items. Keep all labels under 20 characters."
        )

        try:
            response = await self.vision_model.generate_content_async(
                prompt,
                generation_config=self._make_config(0.2, 512)
            )

            raw = response.text.strip()
            logger.info(f"diagram JSON raw: {raw[:200]}")
            data = json.loads(self._strip_fences(raw))

        except Exception as e:
            logger.warning(f"diagram JSON generation failed: {e} — using description directly")
            # Fallback: build a simple label diagram from the description
            data = {
                "title": description[:30],
                "type": "definition",
                "items": [{"label": description[:40], "value": "", "color": "#3B82F6"}],
                "explanation": ""
            }

        # Step 2: Build SVG from the JSON data — always complete and valid
        svg = self._build_svg(data, description)
        logger.info(f"generate_diagram_code final SVG length: {len(svg)}")
        return svg

    def _build_svg(self, data: dict, description: str) -> str:
        """Build a complete, valid SVG from diagram data dict."""
        title = data.get("title", description[:30])
        items = data.get("items", [])[:5]
        explanation = data.get("explanation", "")
        diagram_type = data.get("type", "steps")

        W, H = 600, 400
        lines = [
            f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
            f'xmlns="http://www.w3.org/2000/svg">',
            f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
            # Title
            f'<text x="{W//2}" y="40" text-anchor="middle" '
            f'font-family="Arial" font-size="20" font-weight="bold" fill="#1e293b">'
            f'{_esc(title)}</text>',
        ]

        if not items:
            # No items — just show description as text
            lines.append(
                f'<text x="{W//2}" y="{H//2}" text-anchor="middle" '
                f'font-family="Arial" font-size="16" fill="#475569">'
                f'{_esc(description[:60])}</text>'
            )
        elif diagram_type in ("equation", "comparison") and len(items) >= 2:
            # Side-by-side comparison layout
            box_w, box_h = 220, 100
            gap = 40
            total_w = 2 * box_w + gap
            start_x = (W - total_w) // 2
            y = 80

            for i, item in enumerate(items[:2]):
                x = start_x + i * (box_w + gap)
                color = item.get("color", "#3B82F6")
                label = item.get("label", "")[:20]
                value = item.get("value", "")[:20]
                lines += [
                    f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" '
                    f'rx="12" fill="{color}" fill-opacity="0.15" '
                    f'stroke="{color}" stroke-width="2"/>',
                    f'<text x="{x + box_w//2}" y="{y + 35}" text-anchor="middle" '
                    f'font-family="Arial" font-size="14" fill="#475569">{_esc(label)}</text>',
                    f'<text x="{x + box_w//2}" y="{y + 65}" text-anchor="middle" '
                    f'font-family="Arial" font-size="22" font-weight="bold" fill="{color}">'
                    f'{_esc(value)}</text>',
                ]

            # Arrow between boxes
            ax1 = start_x + box_w + 4
            ax2 = start_x + box_w + gap - 4
            ay = y + box_h // 2
            lines.append(
                f'<line x1="{ax1}" y1="{ay}" x2="{ax2}" y2="{ay}" '
                f'stroke="#94a3b8" stroke-width="2" '
                f'marker-end="url(#arrowhead)"/>'
            )
            # Arrow marker def
            lines.insert(1,
                '<defs><marker id="arrowhead" markerWidth="8" markerHeight="6" '
                'refX="8" refY="3" orient="auto">'
                '<polygon points="0 0, 8 3, 0 6" fill="#94a3b8"/>'
                '</marker></defs>'
            )

            # Remaining items as steps below
            if len(items) > 2:
                step_y = y + box_h + 40
                for j, item in enumerate(items[2:]):
                    color = item.get("color", "#8B5CF6")
                    label = item.get("label", "")[:30]
                    value = item.get("value", "")[:20]
                    ix = W // 2
                    lines += [
                        f'<rect x="{ix - 180}" y="{step_y + j*60}" width="360" height="44" '
                        f'rx="8" fill="{color}" fill-opacity="0.1" stroke="{color}" stroke-width="1.5"/>',
                        f'<text x="{ix}" y="{step_y + j*60 + 26}" text-anchor="middle" '
                        f'font-family="Arial" font-size="14" fill="#334155">'
                        f'{_esc(label)} {_esc(value)}</text>',
                    ]

        else:
            # Vertical steps layout
            item_h = 52
            total_h = len(items) * item_h + (len(items) - 1) * 12
            start_y = max(70, (H - total_h - 60) // 2)
            colors = ["#3B82F6", "#10B981", "#8B5CF6", "#F59E0B", "#EF4444"]

            for i, item in enumerate(items):
                color = item.get("color", colors[i % len(colors)])
                label = item.get("label", "")[:35]
                value = item.get("value", "")[:20]
                iy = start_y + i * (item_h + 12)
                cx = 60
                # Step circle
                lines += [
                    f'<circle cx="{cx}" cy="{iy + item_h//2}" r="20" '
                    f'fill="{color}" fill-opacity="0.2" stroke="{color}" stroke-width="2"/>',
                    f'<text x="{cx}" y="{iy + item_h//2 + 6}" text-anchor="middle" '
                    f'font-family="Arial" font-size="16" font-weight="bold" fill="{color}">'
                    f'{i+1}</text>',
                    # Box
                    f'<rect x="92" y="{iy}" width="470" height="{item_h}" '
                    f'rx="8" fill="{color}" fill-opacity="0.08" '
                    f'stroke="{color}" stroke-width="1.5"/>',
                    f'<text x="112" y="{iy + 22}" '
                    f'font-family="Arial" font-size="14" fill="#334155">{_esc(label)}</text>',
                ]
                if value:
                    lines.append(
                        f'<text x="112" y="{iy + 40}" '
                        f'font-family="Arial" font-size="13" fill="{color}">'
                        f'{_esc(value)}</text>'
                    )
                # Connector line between steps
                if i < len(items) - 1:
                    line_y1 = iy + item_h
                    line_y2 = iy + item_h + 12
                    lines.append(
                        f'<line x1="{cx}" y1="{line_y1}" x2="{cx}" y2="{line_y2}" '
                        f'stroke="#cbd5e1" stroke-width="2" stroke-dasharray="3,3"/>'
                    )

        # Explanation text at bottom
        if explanation:
            lines.append(
                f'<text x="{W//2}" y="{H - 15}" text-anchor="middle" '
                f'font-family="Arial" font-size="12" fill="#94a3b8">'
                f'{_esc(explanation[:80])}</text>'
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