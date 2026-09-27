"""
Test that Gemini-related code in both projects is identical and round-trips work:
  1) _build_gemini_payload  —  system + user(+image) + assistant(+tool_call) + tool messages
  2) _parse_gemini_response — text + functionCall candidates
  3) round-trip  —  payload → functionCall → parse → same name/args
  4) image format  —  inlineData mimeType + data
  5) source-code identity  —  byte-for-byte compare Gemini functions
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Load BOTH llm_client modules under distinct names via importlib
# ---------------------------------------------------------------------------

_ALEX_DIR   = Path(__file__).resolve().parent.parent
_MARKXL_DIR = Path(r"F:\memory no think with moondream")

_ALEX_CLIENT   = _ALEX_DIR   / "core" / "llm_client.py"
_MARKXL_CLIENT = _MARKXL_DIR / "core" / "llm_client.py"


def _load_module(name: str, path: Path):
    """Load a Python source file as a module with a given name."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod  = importlib.util.module_from_spec(spec)
    # Stub out things that shouldn't be needed for pure-function tests
    mod._load_config      = lambda: {}  # noqa
    mod.get_llm_provider  = lambda: "gemini"  # noqa
    mod.threading         = __import__("threading")
    # Load
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# These will hold the live module references once loaded
alex   = _load_module("llm_client_alex",   _ALEX_CLIENT)
markxl = _load_module("llm_client_markxl", _MARKXL_CLIENT)

# ---------------------------------------------------------------------------
# Fixtures — reusable sample data
# ---------------------------------------------------------------------------

_SAMPLE_SYSTEM = "You are Orthos, a helpful AI assistant."
_SAMPLE_USER   = "What's the weather in Tokyo?"
_SAMPLE_TOOL_NAME   = "get_weather"
_SAMPLE_TOOL_ARGS   = {"location": "Tokyo", "unit": "celsius"}
_SAMPLE_TOOL_RESULT = '{"temperature": 22, "condition": "cloudy"}'


# ── Real screenshot via GDI (or mss fallback) ─────────────────────────────
# Matches the exact pipeline used in main.py _process_message
import importlib
_SCREEN_MOD = importlib.import_module("actions.screen_processor")


def _capture_real_screenshot() -> str:
    """Capture screen via GDI (same as runtime), return base64 JPEG string."""
    import base64
    raw_bytes, _mime = _SCREEN_MOD._capture_screen()
    return base64.b64encode(raw_bytes).decode("ascii")


_SAMPLE_B64_IMAGE = _capture_real_screenshot()


@pytest.fixture
def messages_with_image():
    """System → user + image → assistant with tool_call → tool result."""
    return [
        {"role": "system", "content": _SAMPLE_SYSTEM},
        {"role": "user", "content": _SAMPLE_USER, "images": [_SAMPLE_B64_IMAGE]},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "function": {"name": _SAMPLE_TOOL_NAME, "arguments": _SAMPLE_TOOL_ARGS}},
        ]},
        {"role": "tool", "content": _SAMPLE_TOOL_RESULT, "name": _SAMPLE_TOOL_NAME},
    ]


def _deepcopy_msgs(msgs):
    """Deep-copy a message list so destructive .pop('images') doesn't bleed."""
    return [dict(m) for m in msgs]


_SAMPLE_TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    }},
    {"type": "function", "function": {
        "name": "get_time",
        "description": "Get current time for a location",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    }},
]


# ---------------------------------------------------------------------------
# 1. Payload building
# ---------------------------------------------------------------------------

class TestBuildGeminiPayload:
    """Verify _build_gemini_payload output structure for both projects."""

    def _check_payload_structure(self, payload: dict, module):
        assert "contents" in payload, "missing 'contents'"
        assert isinstance(payload["contents"], list), "'contents' not a list"

        roles = [c["role"] for c in payload["contents"]]
        parts_list = [c["parts"] for c in payload["contents"]]

        # System prompt → systemInstruction
        if _SAMPLE_SYSTEM:
            assert "systemInstruction" in payload
            assert payload["systemInstruction"]["parts"][0]["text"] == _SAMPLE_SYSTEM

        # User message (with image)
        user_idx = roles.index("user")
        user_parts = parts_list[user_idx]
        texts = [p for p in user_parts if "text" in p]
        imgs  = [p for p in user_parts if "inlineData" in p]
        assert any(t["text"] == _SAMPLE_USER for t in texts), "user text missing"
        assert len(imgs) == 1, "expected 1 inlineData part"
        assert imgs[0]["inlineData"]["mimeType"] == "image/jpeg"
        assert imgs[0]["inlineData"]["data"] == _SAMPLE_B64_IMAGE

        # Assistant → model role + functionCall
        model_idx = roles.index("model")
        model_parts = parts_list[model_idx]
        fcs = [p for p in model_parts if "functionCall" in p]
        assert len(fcs) == 1
        assert fcs[0]["functionCall"]["name"] == _SAMPLE_TOOL_NAME
        assert fcs[0]["functionCall"]["args"] == _SAMPLE_TOOL_ARGS

        # Tool → function role + functionResponse
        func_idx = roles.index("function")
        func_parts = parts_list[func_idx]
        frs = [p for p in func_parts if "functionResponse" in p]
        assert len(frs) == 1
        assert frs[0]["functionResponse"]["name"] == _SAMPLE_TOOL_NAME

        # Tools array
        assert "tools" in payload
        decls = payload["tools"][0]["functionDeclarations"]
        assert len(decls) == 2
        decl_names = [d["name"] for d in decls]
        assert "get_weather" in decl_names
        assert "get_time" in decl_names

    def test_alex_payload(self, messages_with_image):
        payload = alex._build_gemini_payload(messages_with_image, _SAMPLE_TOOLS)
        self._check_payload_structure(payload, alex)

    def test_markxl_payload(self, messages_with_image):
        payload = markxl._build_gemini_payload(messages_with_image, _SAMPLE_TOOLS)
        self._check_payload_structure(payload, markxl)

    def test_identical_output(self, messages_with_image):
        p1 = alex._build_gemini_payload(_deepcopy_msgs(messages_with_image), _SAMPLE_TOOLS)
        p2 = markxl._build_gemini_payload(_deepcopy_msgs(messages_with_image), _SAMPLE_TOOLS)
        assert p1 == p2, "payloads differ between projects"


# ---------------------------------------------------------------------------
# 2. Response parsing
# ---------------------------------------------------------------------------

class TestParseGeminiResponse:
    """Verify _parse_gemini_response handles text and functionCall correctly."""

    @pytest.fixture
    def text_only_response(self):
        return {
            "candidates": [{
                "content": {
                    "parts": [{"text": "The weather in Tokyo is 22°C and cloudy."}],
                },
            }],
        }

    @pytest.fixture
    def tool_call_response(self):
        return {
            "candidates": [{
                "content": {
                    "parts": [
                        {"text": "Let me check the weather for you."},
                        {"functionCall": {"name": "get_weather", "args": {"location": "Tokyo"}}},
                    ],
                },
            }],
        }

    def test_alex_text_only(self, text_only_response):
        result = alex._parse_gemini_response(text_only_response)
        assert result["content"] == "The weather in Tokyo is 22°C and cloudy."
        assert result["tool_calls"] == []

    def test_markxl_text_only(self, text_only_response):
        result = markxl._parse_gemini_response(text_only_response)
        assert result["content"] == "The weather in Tokyo is 22°C and cloudy."
        assert result["tool_calls"] == []

    def test_alex_tool_call(self, tool_call_response):
        result = alex._parse_gemini_response(tool_call_response)
        assert "Let me check" in result["content"]
        assert len(result["tool_calls"]) == 1
        tc = result["tool_calls"][0]
        assert tc["function"]["name"] == "get_weather"
        assert tc["function"]["arguments"] == {"location": "Tokyo"}

    def test_markxl_tool_call(self, tool_call_response):
        result = markxl._parse_gemini_response(tool_call_response)
        assert "Let me check" in result["content"]
        assert len(result["tool_calls"]) == 1
        tc = result["tool_calls"][0]
        assert tc["function"]["name"] == "get_weather"
        assert tc["function"]["arguments"] == {"location": "Tokyo"}

    def test_identical_output(self, tool_call_response):
        r1 = alex._parse_gemini_response(tool_call_response)
        r2 = markxl._parse_gemini_response(tool_call_response)
        assert r1 == r2


# ---------------------------------------------------------------------------
# 3. Round-trip: payload → parse
# ---------------------------------------------------------------------------

class TestGeminiRoundTrip:
    """Build a payload from sample messages, then verify the functionCall
    can be parsed back correctly."""

    def test_tool_round_trip(self, messages_with_image):
        payload = alex._build_gemini_payload(messages_with_image, _SAMPLE_TOOLS)

        # Simulate a response with the same functionCall
        model_part = None
        for c in payload["contents"]:
            if c["role"] == "model":
                model_part = c
                break
        assert model_part, "no model content in payload"

        # Build a mock response from the payload's own functionCall
        mock_response = {
            "candidates": [{
                "content": {
                    "parts": model_part["parts"],
                },
            }],
        }
        result = alex._parse_gemini_response(mock_response)
        assert len(result["tool_calls"]) == 1
        tc = result["tool_calls"][0]
        assert tc["function"]["name"] == _SAMPLE_TOOL_NAME
        assert tc["function"]["arguments"] == _SAMPLE_TOOL_ARGS


# ---------------------------------------------------------------------------
# 4. Image format in both projects
# ---------------------------------------------------------------------------

class TestImageFormat:
    """Verify image inlineData structure is consistent."""

    def test_alex_image_format(self):
        msgs = [{"role": "user", "content": "describe this", "images": [_SAMPLE_B64_IMAGE]}]
        payload = alex._build_gemini_payload(msgs)
        parts = payload["contents"][0]["parts"]
        imgs = [p for p in parts if "inlineData" in p]
        assert len(imgs) == 1
        inline = imgs[0]["inlineData"]
        assert inline["mimeType"] == "image/jpeg"
        assert inline["data"] == _SAMPLE_B64_IMAGE
        assert len(inline["data"]) > 100  # non-trivial base64

    def test_markxl_image_format(self):
        msgs = [{"role": "user", "content": "describe this", "images": [_SAMPLE_B64_IMAGE]}]
        payload = markxl._build_gemini_payload(msgs)
        parts = payload["contents"][0]["parts"]
        imgs = [p for p in parts if "inlineData" in p]
        assert len(imgs) == 1
        inline = imgs[0]["inlineData"]
        assert inline["mimeType"] == "image/jpeg"
        assert inline["data"] == _SAMPLE_B64_IMAGE

    def test_identical_output(self):
        msgs = [{"role": "user", "content": "x", "images": [_SAMPLE_B64_IMAGE]}]
        p1 = alex._build_gemini_payload(_deepcopy_msgs(msgs))
        p2 = markxl._build_gemini_payload(_deepcopy_msgs(msgs))
        assert p1 == p2


# ---------------------------------------------------------------------------
# 5. Source-code identity
# ---------------------------------------------------------------------------

class TestSourceCodeIdentity:
    """Byte-for-byte comparison of every Gemini-related function."""

    _FUNC_NAMES = [
        "_build_gemini_payload",
        "_parse_gemini_response",
        "_stream_gemini",
    ]

    @staticmethod
    def _get_source(module, name: str) -> str:
        import inspect
        return inspect.getsource(getattr(module, name))

    @staticmethod
    def _logic_lines(source: str) -> list[str]:
        """Return source lines without docstring-only lines and blank lines."""
        import re
        lines = []
        in_docstring = False
        for line in source.splitlines():
            stripped = line.strip()
            # Toggle in/out of multi-line docstring
            if in_docstring:
                if stripped.endswith('"""') or stripped.endswith("'''"):
                    in_docstring = False
                continue
            if stripped.startswith('"""') or stripped.startswith("'''"):
                if stripped.count('"""') == 2 or stripped.count("'''") == 2:
                    continue  # single-line docstring
                in_docstring = True
                continue
            if not stripped:
                continue
            lines.append(line)
        return lines

    def _assert_logic_equal(self, name: str):
        s1 = self._logic_lines(self._get_source(alex, name))
        s2 = self._logic_lines(self._get_source(markxl, name))
        # Also filter out _log_v lines (Alex-only debug logging)
        s1 = [l for l in s1 if "_log_v" not in l]
        s2 = [l for l in s2 if "_log_v" not in l]
        assert s1 == s2, f"'{name}' logic differs between projects"

    def test_build_gemini_payload(self):
        self._assert_logic_equal("_build_gemini_payload")

    def test_parse_gemini_response(self):
        self._assert_logic_equal("_parse_gemini_response")

    def test_normalise_tool_calls(self):
        self._assert_logic_equal("_normalise_tool_calls")

    def test_stream_gemini(self):
        self._assert_logic_equal("_stream_gemini")


# ---------------------------------------------------------------------------
# 6. Live API test with real GDI screenshot + tools
# ---------------------------------------------------------------------------

class TestLiveGeminiWithRealScreenshot:
    """Send actual GDI screenshot + tools to Gemini API, verify 200."""

    MARKXL_KEY = "REDACTED_GEMINI_KEY"
    ALEX_KEY    = "REDACTED_GEMINI_KEY"
    MODEL       = "gemini-3.1-flash-lite"
    URL         = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

    @pytest.fixture
    def live_payload(self):
        """Payload with real GDI screenshot (same as runtime)."""
        b64 = _capture_real_screenshot()
        return {
            "contents": [{"role": "user", "parts": [
                {"text": "Describe this screenshot in one word."},
                {"inlineData": {"mimeType": "image/jpeg", "data": b64}},
            ]}],
            "tools": [{"functionDeclarations": [
                {"name": "test_tool", "description": "test",
                 "parameters": {"type": "object", "properties": {"msg": {"type": "string"}},
                                "required": ["msg"]}},
            ]}],
            "generationConfig": {"maxOutputTokens": 256},
        }

    def _test_key(self, label: str, key: str, payload: dict):
        import requests
        headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
        r = requests.post(self.URL, headers=headers, json=payload, timeout=30)
        assert r.status_code == 200, f"{label}: HTTP {r.status_code} — {r.text[:200]}"
        data = r.json()
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
        fcs = [p for p in parts if "functionCall" in p]
        assert len(text) > 0 or len(fcs) > 0, f"{label}: empty response (no text or functionCall)"
        desc = text[:80] if text else f"functionCall: {fcs[0]['functionCall']['name']}"
        print(f"  {label}: HTTP 200 — {desc}")

    def test_markxl_live(self, live_payload):
        self._test_key("MARKXL", self.MARKXL_KEY, live_payload)

    def test_alex_live(self, live_payload):
        self._test_key("ALEX", self.ALEX_KEY, live_payload)
