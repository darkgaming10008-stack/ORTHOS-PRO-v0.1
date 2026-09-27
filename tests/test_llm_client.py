import json
import threading
from unittest.mock import patch, MagicMock, PropertyMock

import pytest
from core import llm_client


# ── Fixture: fresh _HTTP mock per test ────────────────────────────────────────


@pytest.fixture(autouse=True)
def _fresh_http_mock():
    """Reset the _HTTP mock between tests so side_effects don't bleed."""
    llm_client._HTTP = MagicMock()
    yield


# ── get_llm_settings ─────────────────────────────────────────────────────────


class TestGetLLMSettings:
    def test_returns_defaults(self):
        with patch.object(llm_client, "_load_config", return_value={}):
            url, model = llm_client.get_llm_settings()
        assert url == "http://localhost:11434"
        assert model == "gemma4:31b-cloud"

    def test_uses_config_values(self):
        with patch.object(
            llm_client, "_load_config",
            return_value={"llm_url": "http://example.com:8080", "llm_model": "my-model"},
        ):
            url, model = llm_client.get_llm_settings()
        assert url == "http://example.com:8080"
        assert model == "my-model"


# ── get_llm_provider ─────────────────────────────────────────────────────────


class TestGetLLMProvider:
    def test_default_is_ollama(self):
        with patch.object(llm_client, "_load_config", return_value={}):
            assert llm_client.get_llm_provider() == "ollama"

    def test_openai_raw(self):
        with patch.object(llm_client, "_load_config", return_value={"llm_provider": "openai"}):
            assert llm_client.get_llm_provider() == "openai"

    def test_lmstudio_normalised(self):
        with patch.object(llm_client, "_load_config", return_value={"llm_provider": "lmstudio"}):
            assert llm_client.get_llm_provider() == "openai"

    def test_llamacpp_normalised(self):
        with patch.object(llm_client, "_load_config", return_value={"llm_provider": "llamacpp"}):
            assert llm_client.get_llm_provider() == "openai"


# ── call_llm (Ollama backend) ────────────────────────────────────────────────


class TestCallLLMOllama:
    def test_returns_content_and_tool_calls(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "message": {
                "content": "Hello world",
                "tool_calls": [
                    {"id": "tc1", "function": {"name": "test_tool", "arguments": {"arg1": "val1"}}},
                ],
            },
        }
        llm_client._HTTP.post.return_value = resp
        result = llm_client.call_llm([{"role": "user", "content": "hi"}])
        assert result["content"] == "Hello world"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "test_tool"

    def test_connection_error_triggers_restart(self):
        from requests.exceptions import ConnectionError
        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = {"message": {"content": "Recovered", "tool_calls": []}}
        llm_client._HTTP.post.side_effect = [ConnectionError(), ok_resp]
        with patch.object(llm_client, "ensure_ollama_running", return_value=True):
            result = llm_client.call_llm([{"role": "user", "content": "hi"}])
        assert result["content"] == "Recovered"

    def test_connection_error_fails_gracefully(self):
        from requests.exceptions import ConnectionError
        llm_client._HTTP.post.side_effect = ConnectionError()
        with patch.object(llm_client, "ensure_ollama_running", return_value=False):
            with pytest.raises(RuntimeError, match="Cannot connect to Ollama"):
                llm_client.call_llm([{"role": "user", "content": "hi"}])

    def test_timeout_raises(self):
        from requests.exceptions import Timeout
        llm_client._HTTP.post.side_effect = Timeout()
        with pytest.raises(RuntimeError, match="timed out"):
            llm_client.call_llm([{"role": "user", "content": "hi"}])

    def test_http_error_raises(self):
        from requests.exceptions import HTTPError
        err = HTTPError()
        err.response = MagicMock()
        err.response.status_code = 500
        err.response.text = "Internal Server Error"
        llm_client._HTTP.post.side_effect = err
        with pytest.raises(RuntimeError, match="HTTP error"):
            llm_client.call_llm([{"role": "user", "content": "hi"}])


# ── call_llm (OpenAI backend) ────────────────────────────────────────────────


class TestCallLLMOpenAI:
    def test_openai_returns_content(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "choices": [{
                "message": {
                    "content": "Hello from OpenAI",
                    "tool_calls": [],
                },
            }],
        }
        llm_client._HTTP.post.return_value = resp
        with patch.object(llm_client, "get_llm_provider", return_value="openai"):
            result = llm_client.call_llm([{"role": "user", "content": "hi"}])
        assert result["content"] == "Hello from OpenAI"

    def test_openai_normalises_tool_calls(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "choices": [{
                "message": {
                    "content": "",
                    "tool_calls": [{
                        "id": "call_123",
                        "function": {
                            "name": "my_tool",
                            "arguments": json.dumps({"x": 1}),
                        },
                    }],
                },
            }],
        }
        llm_client._HTTP.post.return_value = resp
        with patch.object(llm_client, "get_llm_provider", return_value="openai"):
            result = llm_client.call_llm([{"role": "user", "content": "hi"}])
        assert result["tool_calls"][0]["function"]["arguments"] == {"x": 1}


# ── call_llm_stream ──────────────────────────────────────────────────────────


class TestCallLLMStream:
    def test_yields_sentences_and_done(self):
        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.iter_lines.return_value = iter([
            b'{"message":{"content":"Hello"},"done":false}',
            b'{"message":{"content":" world"},"done":false}',
            b'{"message":{"content":""},"done":true}',
        ])
        mock_resp.status_code = 200
        llm_client._HTTP.post.return_value = mock_resp
        events = list(llm_client.call_llm_stream([{"role": "user", "content": "hi"}]))
        done_events = [e for e in events if e["type"] == "done"]
        assert len(done_events) == 1
        done = done_events[0]
        assert done["content"] == "Hello world"

    def test_cancel_event_stops_stream(self):
        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.iter_lines.return_value = iter([
            b'{"message":{"content":"first"},"done":false}',
        ])
        mock_resp.status_code = 200
        llm_client._HTTP.post.return_value = mock_resp
        cancel = threading.Event()
        cancel.set()
        events = list(llm_client.call_llm_stream([{"role": "user", "content": "hi"}], cancel_event=cancel))
        assert any(e["type"] == "cancelled" for e in events)

    def test_openai_stream_yields_sentences(self):
        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.iter_lines.return_value = iter([
            b"data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}",
            b"data: {\"choices\":[{\"delta\":{\"content\":\" world\"},\"finish_reason\":\"stop\"}]}",
            b"data: [DONE]",
        ])
        mock_resp.status_code = 200
        llm_client._HTTP.post.return_value = mock_resp
        with patch.object(llm_client, "get_llm_provider", return_value="openai"):
            events = list(llm_client.call_llm_stream([{"role": "user", "content": "hi"}]))
        done_events = [e for e in events if e["type"] == "done"]
        assert len(done_events) == 1
        assert "Hello" in done_events[0]["content"]


# ── ensure_ollama_running ────────────────────────────────────────────────────


class TestEnsureOllamaRunning:
    def test_ollama_already_running(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        llm_client._HTTP.get.return_value = mock_resp
        assert llm_client.ensure_ollama_running() is True
        call_url = llm_client._HTTP.get.call_args[0][0]
        assert "/api/tags" in call_url

    def test_ollama_not_running_starts_it(self):
        from requests.exceptions import ConnectionError
        ok_resp = MagicMock()
        ok_resp.status_code = 200
        llm_client._HTTP.get.side_effect = [ConnectionError(), ok_resp]
        with patch("subprocess.Popen"):
            result = llm_client.ensure_ollama_running(timeout=5)
        assert result is True

    def test_ollama_not_found(self):
        from requests.exceptions import ConnectionError
        llm_client._HTTP.get.side_effect = ConnectionError()
        with patch("subprocess.Popen", side_effect=FileNotFoundError):
            result = llm_client.ensure_ollama_running(timeout=2)
        assert result is False

    def test_openai_provider_pings_v1_models(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        llm_client._HTTP.get.return_value = mock_resp
        with patch.object(llm_client, "get_llm_provider", return_value="openai"):
            assert llm_client.ensure_ollama_running() is True
        call_url = llm_client._HTTP.get.call_args[0][0]
        assert "/v1/models" in call_url


# ── call_llm_text ────────────────────────────────────────────────────────────


class TestCallLLMText:
    def test_returns_content(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"message": {"content": "Simple reply"}}
        llm_client._HTTP.post.return_value = resp
        result = llm_client.call_llm_text("Hello")
        assert result == "Simple reply"

    def test_includes_system_prompt(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"message": {"content": "With system"}}
        llm_client._HTTP.post.return_value = resp
        llm_client.call_llm_text("Hello", system="Be helpful")
        call_kwargs = llm_client._HTTP.post.call_args[1]
        messages = call_kwargs["json"]["messages"]
        assert messages[0]["role"] == "system"


# ── warmup_model ─────────────────────────────────────────────────────────────


class TestWarmupModel:
    def test_warmup_sends_keep_alive(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"message": {"content": "ok"}}
        llm_client._HTTP.post.return_value = resp
        result = llm_client.warmup_model("You are helpful.")
        assert result is True
        call_kwargs = llm_client._HTTP.post.call_args[1]
        assert call_kwargs["json"]["keep_alive"] == -1

    def test_warmup_openai(self):
        resp = MagicMock()
        resp.status_code = 200
        llm_client._HTTP.post.return_value = resp
        with patch.object(llm_client, "get_llm_provider", return_value="openai"):
            result = llm_client.warmup_model("You are helpful.")
        assert result is True


# ── _select_tools_for_provider (dynamic tool selection) ──────────────────────

def _make_legacy_tools():
    """Return all 42 legacy tool declarations (name + minimal function stub)."""
    names = [
        "open_app", "web_search", "webfetch", "weather_report",
        "send_message", "reminder", "youtube_video", "screen_process",
        "screen_locate", "computer_settings", "file_controller",
        "desktop_control", "code_helper", "dev_agent", "agent_task",
        "computer_control", "game_updater", "flight_finder", "run_terminal",
        "list_mcp_servers", "shutdown_orthos", "file_processor",
        "save_memory", "forget_memory", "search_memory", "list_memories",
        "recall_conversation",
        "save_procedure", "list_procedures", "core_memory_append",
        "core_memory_replace", "archival_memory_search", "context_status",
        "search_timeline", "save_timeline_event",
        "list_projects", "create_project", "set_active_project",
        "get_project_memories", "save_tool_state", "get_tool_state",
        "search_knowledge_graph",
    ]
    return [{"type": "function", "function": {"name": n, "description": "", "parameters": {}}} for n in names]


def _make_mcp_tools(server: str, count: int) -> list:
    """Generate *count* synthetic MCP tool declarations for *server*."""
    tools = []
    for i in range(count):
        tools.append({
            "type": "function",
            "function": {
                "name": f"mcp_{server}_tool_{i}",
                "description": "",
                "parameters": {},
            },
        })
    return tools


def _all_tools() -> list:
    t = _make_legacy_tools()
    t.extend(_make_mcp_tools("playwright", 51))
    t.extend(_make_mcp_tools("winapp", 54))
    return t


class TestDynamicToolSelection:
    """Tests for _select_tools_for_provider — the dynamic tool selection logic."""

    def test_unlimited_provider_gets_all_tools(self):
        """openrouter / cloudflare / nvidia are not in _TOOL_LIMITS → all 147 tools returned."""
        tools = _all_tools()
        result = llm_client._select_tools_for_provider(tools, [{"role": "user", "content": "hi"}], "openrouter")
        assert len(result) == len(tools), "openrouter should receive all tools"

    def test_under_limit_returns_all(self):
        """Only 42 legacy tools (under 128) → no filtering needed."""
        tools = _make_legacy_tools()
        result = llm_client._select_tools_for_provider(tools, [{"role": "user", "content": "hi"}], "groq")
        assert len(result) == len(tools)

    def test_essential_tools_always_present(self):
        """Essential tool names must appear in the result for groq."""
        tools = _all_tools()
        result = llm_client._select_tools_for_provider(tools, [{"role": "user", "content": "hello"}], "groq")
        names = {t["function"]["name"] for t in result}
        for essential in llm_client._TOOL_ESSENTIAL_NAMES:
            assert essential in names, f"Essential tool '{essential}' missing from selection"

    def test_respects_tool_limit(self):
        """Result size must not exceed 128 for groq."""
        tools = _all_tools()
        result = llm_client._select_tools_for_provider(tools, [{"role": "user", "content": "hello"}], "groq")
        assert len(result) <= 128, f"Expected ≤128 tools, got {len(result)}"

    def test_search_keyword_prioritises_web_search(self):
        """User says 'search' → web_search should be in result."""
        tools = _all_tools()
        result = llm_client._select_tools_for_provider(tools, [{"role": "user", "content": "search the web for AI news"}], "groq")
        names = [t["function"]["name"] for t in result]
        assert "web_search" in names, "web_search should be selected when user says 'search'"

    def test_browser_keyword_includes_playwright_tools(self):
        """User says 'browser' → playwright tools should appear."""
        tools = _all_tools()
        result = llm_client._select_tools_for_provider(tools, [{"role": "user", "content": "open browser and go to google"}], "groq")
        mcp_names = [t["function"]["name"] for t in result if t["function"]["name"].startswith("mcp_playwright_")]
        assert len(mcp_names) >= 1, "Expected at least one playwright tool when user says 'browser'"

    def test_desktop_keyword_includes_winapp_tools(self):
        """User says 'app' → winapp tools should appear."""
        tools = _all_tools()
        result = llm_client._select_tools_for_provider(tools, [{"role": "user", "content": "open the calculator app"}], "groq")
        mcp_names = [t["function"]["name"] for t in result if t["function"]["name"].startswith("mcp_winapp_")]
        assert len(mcp_names) >= 1, "Expected at least one winapp tool when user says 'app'"

    def test_non_mcp_tools_preferred_over_mcp_when_equal_score(self):
        """With no keywords matched, legacy tools should be preferred over MCP tools when filling slots."""
        tools = _all_tools()
        result = llm_client._select_tools_for_provider(tools, [{"role": "user", "content": "zzzz"}], "groq")
        names = [t["function"]["name"] for t in result]
        # Find positions of legacy vs MCP tools
        last_non_mcp_idx = -1
        first_mcp_idx = -1
        for i, name in enumerate(names):
            if name.startswith("mcp_"):
                if first_mcp_idx < 0:
                    first_mcp_idx = i
            else:
                last_non_mcp_idx = i
        # After essential tools, non-MCP should come before MCP (or no MCP at all)
        assert first_mcp_idx < 0 or last_non_mcp_idx < first_mcp_idx, \
            "Non-MCP tools should appear before MCP tools when scores are equal"

    def test_image_message_without_text_extracts_empty_keywords(self):
        """Messages with only image content (list, no text) should not crash."""
        tools = _all_tools()
        msg = {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/..."}}]}
        result = llm_client._select_tools_for_provider(tools, [msg], "groq")
        assert len(result) == 128  # no keywords → fill with non-MCP first

    def test_image_message_with_text_extracts_text(self):
        """Messages with both image and text should extract the text part."""
        tools = _all_tools()
        msg = {"role": "user", "content": [
            {"type": "text", "text": "search the web"},
            {"type": "image_url", "image_url": {"url": "data:image/..."}},
        ]}
        result = llm_client._select_tools_for_provider(tools, [msg], "groq")
        names = [t["function"]["name"] for t in result]
        assert "web_search" in names

    def test_only_system_messages_no_user(self):
        """No user message in the conversation → should not crash, fills with non-MCP."""
        tools = _all_tools()
        result = llm_client._select_tools_for_provider(tools, [{"role": "system", "content": "be helpful"}], "groq")
        assert len(result) == 128

    def test_player_metadata_untouched_for_non_limited_providers(self):
        """openai is in _TOOL_LIMITS → should still be filtered. Verify it works."""
        tools = _all_tools()
        result = llm_client._select_tools_for_provider(tools, [{"role": "user", "content": "hello"}], "openai")
        assert len(result) == 128
