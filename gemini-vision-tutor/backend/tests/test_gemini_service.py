"""
Tests for GeminiService error handling.

Run with:  cd backend && python -m pytest tests/test_gemini_service.py -v
"""
import asyncio
import base64
import json
import os
import sys
import types
import pytest

# ---------------------------------------------------------------------------
# Helpers: build a minimal fake google.generativeai module so we can test
# without a real API key or network connection.
# ---------------------------------------------------------------------------

def _make_fake_genai(raise_exc=None, return_text=None):
    """
    Return a fake `genai` module whose GenerativeModel.generate_content_async
    either raises `raise_exc` or returns a fake response with `return_text`.
    """
    genai = types.ModuleType("google.generativeai")

    class FakeResponse:
        def __init__(self, text):
            self.text = text

    class FakeModel:
        async def generate_content_async(self, *args, **kwargs):
            if raise_exc is not None:
                raise raise_exc
            return FakeResponse(return_text)

    class FakeTypes:
        class GenerationConfig:
            def __init__(self, **kwargs):
                pass

    genai.GenerativeModel = lambda **kwargs: FakeModel()
    genai.configure = lambda **kwargs: None
    genai.types = FakeTypes()
    return genai


# ---------------------------------------------------------------------------
# Fixture: patch env + import so each test gets a fresh GeminiService
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def set_fake_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-testing")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)


def _make_service(fake_genai):
    """Construct a GeminiService using an injected fake genai module."""
    # Late import so env var is already set
    if "services.gemini_service" in sys.modules:
        del sys.modules["services.gemini_service"]

    # Patch the loader so it returns our fake
    import importlib
    import unittest.mock as mock

    # Make sure the package path is available
    backend_dir = os.path.join(os.path.dirname(__file__), "..")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    from services.gemini_service import GeminiService

    svc = GeminiService.__new__(GeminiService)
    svc.genai = fake_genai
    svc.vision_model = fake_genai.GenerativeModel()
    svc.dialogue_model = fake_genai.GenerativeModel()
    return svc


# Minimal 1-pixel JPEG in base64 (valid input so base64.b64decode doesn't fail)
_TINY_JPEG_B64 = base64.b64encode(
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
    b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
    b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e;"
    b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
    b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
    b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xf5\x0f\xff\xd9"
).decode()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAnalyzeFrameSuccess:
    def test_returns_parsed_json_on_success(self):
        payload = {
            "subject": "math",
            "problem": "2 + 2 = ?",
            "current_work": "",
            "difficulty_level": "elementary",
            "suggested_approach": "add",
            "key_concepts": [],
            "has_errors": False,
            "error_description": "",
        }
        fg = _make_fake_genai(return_text=json.dumps(payload))
        svc = _make_service(fg)

        result = asyncio.get_event_loop().run_until_complete(
            svc.analyze_frame(_TINY_JPEG_B64)
        )
        assert result["subject"] == "math"
        assert result["problem"] == "2 + 2 = ?"
        assert result["has_errors"] is False


class TestAnalyzeFrameJsonError:
    def test_returns_fallback_on_bad_json(self):
        fg = _make_fake_genai(return_text="This is definitely not JSON")
        svc = _make_service(fg)

        result = asyncio.get_event_loop().run_until_complete(
            svc.analyze_frame(_TINY_JPEG_B64)
        )
        # Fallback dict should be returned without raising
        assert result["subject"] == "unknown"
        # No API_ERROR marker — just a JSON parse failure
        assert not result.get("error_description", "").startswith("API_ERROR")


class TestAnalyzeFrameApiError:
    def test_surfaces_api_error_in_result(self):
        class FakeApiError(Exception):
            pass

        fg = _make_fake_genai(raise_exc=FakeApiError("404 model not found"))
        svc = _make_service(fg)

        result = asyncio.get_event_loop().run_until_complete(
            svc.analyze_frame(_TINY_JPEG_B64)
        )
        # Must NOT re-raise — returns a dict instead
        assert isinstance(result, dict)
        # Must carry the API_ERROR marker so websocket can detect it
        assert result["error_description"].startswith("API_ERROR:")
        assert "model not found" in result["error_description"]
        # Problem text must be human-readable
        assert "Analysis failed" in result["problem"]

    def test_auth_error_shows_error_type(self):
        class PermissionDenied(Exception):
            pass

        fg = _make_fake_genai(raise_exc=PermissionDenied("API key not valid"))
        svc = _make_service(fg)

        result = asyncio.get_event_loop().run_until_complete(
            svc.analyze_frame(_TINY_JPEG_B64)
        )
        assert "API key not valid" in result["error_description"]


class TestFallbackInfo:
    def test_no_error_msg(self):
        from services.gemini_service import GeminiService
        # Build a bare instance just to call _fallback_info
        svc = object.__new__(GeminiService)
        fb = svc._fallback_info()
        assert fb["subject"] == "unknown"
        assert not fb["has_errors"]
        assert not fb["error_description"]

    def test_with_error_msg(self):
        from services.gemini_service import GeminiService
        svc = object.__new__(GeminiService)
        fb = svc._fallback_info(error_msg="InvalidArgument: model not found")
        assert fb["has_errors"] is True
        assert fb["error_description"].startswith("API_ERROR:")
        assert "Analysis failed" in fb["problem"]


class TestStripFences:
    def test_strips_json_fence(self):
        from services.gemini_service import GeminiService
        svc = object.__new__(GeminiService)
        raw = "```json\n{\"a\": 1}\n```"
        assert svc._strip_fences(raw) == '{"a": 1}'

    def test_no_fence_unchanged(self):
        from services.gemini_service import GeminiService
        svc = object.__new__(GeminiService)
        assert svc._strip_fences('{"a": 1}') == '{"a": 1}'
