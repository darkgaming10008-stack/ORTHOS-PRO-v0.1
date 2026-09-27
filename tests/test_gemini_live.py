import json
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import requests

from core import llm_client
from core.gemini_live_provider import GeminiLiveProvider, QuotaExceededError, _is_quota_error
from google.genai import types
from core.gemini_live_provider import GeminiLiveProvider, QuotaExceededError, _is_quota_error

_ALEX_DIR = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _ALEX_DIR / "config" / "api_keys.json"

_MCP_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"},
            },
            "required": ["location"],
            "additionalProperties": False,
        },
    },
}


@pytest.fixture(autouse=True)
def _restore_http():
    llm_client._HTTP = requests.Session()


@pytest.fixture(autouse=True)
def _patch_llm_config():
    """Set provider to gemini_live for all tests via llm_client._load_config."""
    cfg = _load_real_config()
    patched = {**cfg, "llm_provider": "gemini_live", "gemini_model": "gemini-3.1-flash-live-preview"}
    with patch.object(llm_client, "_load_config", return_value=patched):
        yield


def _load_real_config() -> dict:
    try:
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _has_api_key() -> bool:
    return bool(_load_real_config().get("gemini_api_key"))


# ── 1. warmup_model() routing ─────────────────────────────────---------------


class TestWarmupModel:
    def test_uses_provider_warmup_instead_of_post(self):
        result = llm_client.warmup_model("You are a helpful assistant.")
        assert isinstance(result, bool)

    def test_does_not_call_http_post(self):
        with patch.object(llm_client._HTTP, "post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"message": {"content": "ok"}}
            llm_client.warmup_model()
            mock_post.assert_not_called()


# ── 2. ensure_ollama_running() with empty health_url ──────────────────────────


class TestEnsureOllamaRunning:
    def test_gemini_live_returns_true(self):
        result = llm_client.ensure_ollama_running()
        assert result is True

    def test_direct_logic_catches_error(self):
        cfg = llm_client._get_provider_config()
        assert cfg["provider"] in ("gemini", "gemini_live", "cloudflare")

        from requests.exceptions import MissingSchema
        try:
            resp = llm_client._HTTP.get(cfg["health_url"], timeout=3)
            ok = resp.status_code == 200
        except MissingSchema:
            ok = True
        assert ok is True


# ── 3. GeminiLiveProvider ─────────────────────────────────────────────────────


class TestGeminiLiveProviderWarmup:
    def test_warmup_returns_bool(self):
        provider = GeminiLiveProvider()
        result = provider.warmup()
        assert isinstance(result, bool)

    def test_warmup_connection_error(self):
        provider = GeminiLiveProvider()
        with patch.object(provider, "_api_key", "fake-key"):
            with patch("google.genai.Client") as MockClient:
                MockClient.side_effect = Exception("API unreachable")
                assert provider.warmup() is False


# ── 4. Tool schema ────────────────────────────────────────────────────────────


class TestToolSchema:
    def test_function_declaration_with_extra_fields(self):
        from google.genai import types
        func = _MCP_TOOL["function"]
        try:
            decl = types.FunctionDeclaration(
                name=func["name"],
                description=func.get("description", ""),
                parameters_json_schema=func.get("parameters", {"type": "object", "properties": {}}),
            )
        except Exception as e:
            pytest.fail(f"FunctionDeclaration rejected extra fields: {e}")
        assert decl.name == "get_weather"


# ── 5. End-to-end chat via GeminiLiveProvider ─────────────────────────────────


class TestGeminiLiveChat:
    @pytest.fixture
    def live_provider(self):
        if not _has_api_key():
            pytest.skip("No gemini_api_key configured")
        return GeminiLiveProvider()

    def test_warmup_then_chat(self, live_provider):
        if not live_provider.warmup():
            pytest.skip("Model not available via API")
        messages = [
            {"role": "system", "content": "Reply in 1 word."},
            {"role": "user", "content": "Say hello"},
        ]
        result = live_provider.chat(messages, timeout=30)
        assert isinstance(result, dict)
        assert "content" in result
        assert "tool_calls" in result
        print(f"  Response: {result['content'][:200]}")

    def test_chat_stream_yields_sentences(self, live_provider):
        if not live_provider.warmup():
            pytest.skip("Model not available via API")
        messages = [
            {"role": "system", "content": "Reply in 1 short sentence."},
            {"role": "user", "content": "What color is the sky?"},
        ]
        sentences = []
        final = None
        for event in live_provider.chat_stream(messages, timeout=30):
            if event["type"] == "sentence":
                sentences.append(event["text"])
            elif event["type"] in ("done", "cancelled"):
                final = event
        assert len(sentences) > 0
        print(f"  Sentences: {sentences}")
        print(f"  Final content: {final['content'][:200] if final else 'N/A'}")

    def test_chat_with_tools(self, live_provider):
        if not live_provider.warmup():
            pytest.skip("Model not available via API")
        messages = [
            {"role": "system", "content": "Use tools when appropriate."},
            {"role": "user", "content": "What's the weather in Tokyo?"},
        ]
        result = live_provider.chat(messages, tools=[_MCP_TOOL], timeout=30)
        print(f"  Content: {result.get('content', '')[:200]}")
        print(f"  Tool calls: {len(result.get('tool_calls', []))}")
        for tc in result.get("tool_calls", []):
            assert "function" in tc
            assert "name" in tc["function"]

    def test_chat_cancel(self, live_provider):
        if not live_provider.warmup():
            pytest.skip("Model not available via API")
        cancel = threading.Event()
        messages = [{"role": "user", "content": "Tell me a very long story"}]
        def _do_cancel():
            time.sleep(5)
            cancel.set()
        threading.Thread(target=_do_cancel, daemon=True).start()
        for event in live_provider.chat_stream(messages, timeout=60, cancel_event=cancel):
            if event["type"] == "cancelled":
                print(f"  Cancelled, had content: {bool(event['content'])}")
                return
        pytest.fail("Expected cancelled event")


# ── 6. Full routing via call_llm_stream ───────────────────────────────────────


class TestFullRouting:
    def test_call_llm_stream_routes_to_gemini_live(self):
        messages = [{"role": "user", "content": "say hi"}]
        gen = llm_client.call_llm_stream(messages, timeout=10)
        try:
            for event in gen:
                pass
        except RuntimeError as e:
            assert "Invalid URL" not in str(e)

    def test_call_llm_raises_for_gemini_live(self):
        messages = [{"role": "user", "content": "hi"}]
        try:
            result = llm_client.call_llm(messages, timeout=10)
            assert "content" in result
        except RuntimeError as e:
            assert "Invalid URL" not in str(e)


# ── 7. Config isolation ──────────────────────────────────────────────────────


class TestConfigIsolation:
    def test_gemini_live_config_shape(self):
        cfg = llm_client._get_provider_config()
        assert cfg["provider"] == "gemini_live"
        assert cfg["chat_endpoint"] == ""
        assert cfg["health_url"] == ""

    def test_gemini_config_untouched(self, monkeypatch):
        monkeypatch.setattr(llm_client, "_load_config", lambda: {
            "llm_provider": "gemini", "gemini_api_key": "test-key", "gemini_model": "gemini-2.5-flash",
        })
        cfg = llm_client._get_provider_config()
        assert cfg["provider"] == "gemini"
        assert "generateContent" in cfg["chat_endpoint"]
        assert cfg["health_url"] != ""

    def test_openai_config_untouched(self, monkeypatch):
        monkeypatch.setattr(llm_client, "_load_config", lambda: {
            "llm_provider": "openai", "llm_url": "http://localhost:1234", "llm_model": "my-model",
        })
        cfg = llm_client._get_provider_config()
        assert cfg["provider"] == "openai"
        assert "/v1/chat/completions" in cfg["chat_endpoint"]
        assert "/v1/models" in cfg["health_url"]

    def test_ollama_config_untouched(self, monkeypatch):
        monkeypatch.setattr(llm_client, "_load_config", lambda: {
            "llm_provider": "ollama", "llm_url": "http://localhost:11434", "llm_model": "gemma4:cloud",
        })
        cfg = llm_client._get_provider_config()
        assert cfg["provider"] == "ollama"
        assert "/api/chat" in cfg["chat_endpoint"]
        assert "/api/tags" in cfg["health_url"]


class TestGeminiLiveErrors:
    def _make_provider(self):
        p = GeminiLiveProvider()
        p._api_key = "fake-key"
        return p

    def test_quota_markers_detection(self):
        assert _is_quota_error("quotaValue: '10' quota exceeded")
        assert _is_quota_error("1011 RESOURCE_EXHAUSTED exceeded your current quota")
        assert not _is_quota_error("connection reset by peer")

    def test_chat_stream_raises_quota_exceeded_error(self):
        p = self._make_provider()
        quota_msg = (
            "received 1011 (internal error) You exceeded your current quota, "
            "please check your plan and billing details."
        )
        class _RaisingSession:
            async def __aenter__(self):
                raise Exception(quota_msg)
            async def __aexit__(self, *a):
                return False
        with patch("google.genai.Client") as MockClient:
            MockClient.return_value.aio.live.connect = MagicMock(return_value=_RaisingSession())
            with pytest.raises(QuotaExceededError) as ei:
                list(p.chat_stream([{"role": "user", "content": "hi"}], timeout=10))
        assert "quota" in str(ei.value).lower()

    def test_chat_stream_raises_generic_on_non_quota_error(self):
        p = self._make_provider()
        with patch("google.genai.Client") as MockClient:
            class _RaisingSession:
                async def __aenter__(self):
                    raise Exception("Connection refused by server")
                async def __aexit__(self, *a):
                    return False
            MockClient.return_value.aio.live.connect = MagicMock(return_value=_RaisingSession())
            with pytest.raises(RuntimeError) as ei:
                list(p.chat_stream([{"role": "user", "content": "hi"}], timeout=10))
        assert not isinstance(ei.value, QuotaExceededError)

    def test_config_includes_context_window_compression(self):
        p = self._make_provider()
        p._model = "gemini-3.1-flash-live-preview"
        cfg = p._build_config()
        assert cfg.context_window_compression is not None
        assert cfg.context_window_compression.trigger_tokens == 25000
        assert cfg.context_window_compression.sliding_window.target_tokens == 8000
        assert getattr(cfg, "session_resumption", None) is None

    def test_config_2x_omits_compression(self):
        p = self._make_provider()
        p._model = "gemini-2.5-flash-native-audio-preview-12-2025"
        cfg = p._build_config()
        assert getattr(cfg, "context_window_compression", None) is None

    def test_config_resumption_only_with_handle(self):
        p = self._make_provider()
        p._model = "gemini-3.1-flash-live-preview"
        cfg = p._build_config()
        assert getattr(cfg, "session_resumption", None) is None
        p._resumption_handle = "abc123"
        cfg = p._build_config()
        assert cfg.session_resumption is not None
        assert cfg.session_resumption.handle == "abc123"

