"""Thinking events (ollama `thinking` + OpenAI `reasoning_content`) and the
Kaggle provider: config resolution, payload shape, response parsing.

All network access is mocked.
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import llm_client as lc  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_http():
    lc._HTTP = MagicMock()
    yield


def _ollama_stream_response(chunks: list[dict]):
    resp = MagicMock()
    resp.status_code = 200
    resp.iter_lines.return_value = [json.dumps(c).encode("utf-8") for c in chunks]
    ctx = MagicMock()
    ctx.__enter__.return_value = resp
    ctx.__exit__.return_value = False
    lc._HTTP.post.return_value = ctx
    return resp


# ── Ollama thinking deltas ───────────────────────────────────────────────────

class TestOllamaThinking:
    def _run(self, chunks):
        _ollama_stream_response(chunks)
        with patch.object(lc, "_get_provider_config", return_value={
            "chat_endpoint": "http://localhost:11434/api/chat",
            "headers": {}, "model": "qwen3", "provider": "ollama",
            "base_url": "http://localhost:11434", "health_url": "http://localhost:11434/api/tags",
        }):
            return list(lc.call_llm_stream([{"role": "user", "content": "hi"}]))

    def test_thinking_events_emitted(self):
        events = self._run([
            {"message": {"thinking": "let me"}, "done": False},
            {"message": {"thinking": " think..."}, "done": False},
            {"message": {"content": "Hello!"}, "done": False},
            {"message": {"content": ""}, "done": True},
        ])
        kinds = [e["type"] for e in events]
        assert kinds.count("thinking") == 2
        assert kinds[0] == "thinking"  # arrives before the sentence
        assert events[0]["text"] == "let me"
        assert "sentence" in kinds

    def test_done_carries_full_thinking(self):
        events = self._run([
            {"message": {"thinking": "hmm "}, "done": False},
            {"message": {"content": "Answer"}, "done": False},
            {"message": {}, "done": True},
        ])
        done = [e for e in events if e["type"] == "done"][0]
        assert done["thinking"] == "hmm"
        assert done["content"] == "Answer"

    def test_no_thinking_key_still_works(self):
        events = self._run([
            {"message": {"content": "Plain reply"}, "done": False},
            {"message": {}, "done": True},
        ])
        assert not any(e["type"] == "thinking" for e in events)
        done = [e for e in events if e["type"] == "done"][0]
        assert done["content"] == "Plain reply"

    def test_thinking_not_in_sentences(self):
        """Thinking text must never leak into the spoken/visible reply."""
        events = self._run([
            {"message": {"thinking": "SECRET-REASONING"}, "done": False},
            {"message": {"content": "Visible"}, "done": False},
            {"message": {}, "done": True},
        ])
        joined = " ".join(e["text"] for e in events if e["type"] == "sentence")
        assert "SECRET-REASONING" not in joined


# ── OpenAI-compatible reasoning_content deltas ──────────────────────────────

class TestOpenAIThinking:
    def _run(self, chunks, provider="openai"):
        resp = MagicMock()
        resp.status_code = 200
        resp.iter_lines.return_value = [
            ("data: " + json.dumps(c)).encode("utf-8") for c in chunks
        ] + [b"data: [DONE]"]
        ctx = MagicMock()
        ctx.__enter__.return_value = resp
        ctx.__exit__.return_value = False
        lc._HTTP.post.return_value = ctx
        with patch.object(lc, "_get_provider_config", return_value={
            "chat_endpoint": "http://x/v1/chat/completions",
            "headers": {}, "model": "m", "provider": provider,
            "base_url": "http://x", "health_url": "http://x/v1/models",
        }):
            return list(lc.call_llm_stream([{"role": "user", "content": "hi"}]))

    def test_reasoning_content_emitted(self):
        events = self._run([
            {"choices": [{"delta": {"reasoning_content": "step 1 "}}]},
            {"choices": [{"delta": {"content": "Answer"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ])
        kinds = [e["type"] for e in events]
        assert "thinking" in kinds
        assert events[0]["type"] == "thinking"
        assert events[0]["text"] == "step 1 "

    def test_reasoning_alias_field(self):
        events = self._run([
            {"choices": [{"delta": {"reasoning": "why"}}]},
            {"choices": [{"delta": {"content": "A"}}, ]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ])
        assert any(e["type"] == "thinking" and e["text"] == "why" for e in events)

    def test_plain_provider_no_reasoning_events(self):
        events = self._run([
            {"choices": [{"delta": {"reasoning_content": "ignored"}}]},
            {"choices": [{"delta": {"content": "A"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ], provider="cloudflare")
        assert not any(e["type"] == "thinking" for e in events)

    def test_done_includes_thinking(self):
        events = self._run([
            {"choices": [{"delta": {"reasoning_content": "deep thought"}}]},
            {"choices": [{"delta": {"content": "A"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ])
        done = [e for e in events if e["type"] == "done"][0]
        assert done["thinking"] == "deep thought"


# ── Kaggle provider ──────────────────────────────────────────────────────────

_KAGGLE_CFG = {
    "chat_endpoint": "http://100.64.0.5:11434/api/chat",
    "headers": {}, "model": "qwen3.8-27b-uncensored-mtp", "provider": "kaggle",
    "base_url": "http://100.64.0.5:11434", "health_url": "http://100.64.0.5:11434/api/tags",
}


class TestKaggleProvider:
    def test_config_resolution(self):
        with patch.object(lc, "_load_config", return_value={
            "llm_provider": "kaggle",
            "kaggle_url": "http://100.64.0.5:11434/",
            "kaggle_model": "qwen3.8-27b-uncensored-mtp",
        }):
            cfg = lc._get_provider_config()
        assert cfg["provider"] == "kaggle"
        assert cfg["chat_endpoint"] == "http://100.64.0.5:11434/api/chat"
        assert cfg["base_url"] == "http://100.64.0.5:11434"  # trailing / stripped

    def test_config_default_model(self):
        with patch.object(lc, "_load_config", return_value={"llm_provider": "kaggle"}):
            cfg = lc._get_provider_config()
        assert cfg["model"] == "qwen3.8-27b-uncensored-mtp"
        assert cfg["base_url"] == "http://localhost:11434"

    def test_payload_uses_ollama_shape(self):
        from unittest.mock import patch as _p
        with _p.object(lc, "get_model_context_window", create=True, return_value=262144):
            payload = lc._build_payload(
                [{"role": "user", "content": "hi"}], tools=None,
                stream=False, model="m", provider="kaggle",
            )
        assert payload["keep_alive"] == -1
        assert "options" in payload and "num_gpu" in payload["options"]
        assert "tools" not in payload

    def test_parse_response_with_thinking(self):
        data = {"message": {"content": "Reply", "thinking": "thought"}}
        out = lc._parse_chat_response(data, "kaggle")
        assert out["content"] == "Reply"
        assert out["thinking"] == "thought"
        assert out["tool_calls"] == []

    def test_stream_works_over_kaggle(self):
        _ollama_stream_response([
            {"message": {"thinking": "hmm"}, "done": False},
            {"message": {"content": "Hi"}, "done": False},
            {"message": {}, "done": True},
        ])
        with patch.object(lc, "_get_provider_config", return_value=dict(_KAGGLE_CFG)):
            events = list(lc.call_llm_stream([{"role": "user", "content": "hi"}]))
        assert any(e["type"] == "thinking" for e in events)
        done = [e for e in events if e["type"] == "done"][0]
        assert done["content"] == "Hi"
