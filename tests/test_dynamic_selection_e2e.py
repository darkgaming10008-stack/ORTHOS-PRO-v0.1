"""Dynamic tool selection tests for Groq/OpenAI (limited) vs other providers.

Verifies that _build_payload from llm_client.py:
  - limited providers (groq, openai) get <= 128 tools with essentials always present
  - unlimited providers receive every tool unchanged
"""

import pytest

from core.llm_client import _build_payload, _TOOL_LIMITS, _TOOL_ESSENTIAL_NAMES
from core.tool_declarations import TOOL_DECLARATIONS as LEGACY_TOOLS


def mock_mcp(server: str, count: int) -> list:
    return [
        {"name": f"mcp_{server}_tool_{i}", "description": "", "parameters": {}}
        for i in range(count)
    ]


MCP_PLAYWRIGHT = mock_mcp("playwright", 51)
MCP_WINAPP = mock_mcp("winapp", 54)
ALL_TOOLS = LEGACY_TOOLS + MCP_PLAYWRIGHT + MCP_WINAPP

MESSAGES_SEARCH = [{"role": "user", "content": "search the web for AI news"}]
MESSAGES_DESKTOP = [{"role": "user", "content": "open calculator app and click button"}]
MESSAGES_GENERIC = [{"role": "user", "content": "hello"}]

LIMITED_PROVIDERS = sorted(_TOOL_LIMITS)
UNLIMITED_PROVIDERS = ["openrouter", "cloudflare", "nvidia", "ollama"]

MSG_CASES = [
    ("search", MESSAGES_SEARCH),
    ("desktop", MESSAGES_DESKTOP),
    ("generic", MESSAGES_GENERIC),
]


def _tool_name(t: dict) -> str:
    return t.get("name", t.get("function", {}).get("name", ""))


def _selected_names(provider: str, messages: list) -> list:
    payload = _build_payload(
        messages, ALL_TOOLS, stream=False, model="test-model", provider=provider
    )
    return [_tool_name(t) for t in payload.get("tools", [])]


class TestLimitedProviders:
    @pytest.mark.parametrize("provider", LIMITED_PROVIDERS)
    @pytest.mark.parametrize("label,msgs", MSG_CASES, ids=[c[0] for c in MSG_CASES])
    def test_essentials_always_present(self, provider, label, msgs):
        names = _selected_names(provider, msgs)
        missing = [n for n in _TOOL_ESSENTIAL_NAMES if n not in names]
        assert not missing, f"{provider}/{label} missing essential tools: {missing}"

    @pytest.mark.parametrize("provider", LIMITED_PROVIDERS)
    @pytest.mark.parametrize("label,msgs", MSG_CASES, ids=[c[0] for c in MSG_CASES])
    def test_within_tool_limit(self, provider, label, msgs):
        names = _selected_names(provider, msgs)
        assert len(names) <= _TOOL_LIMITS[provider]

    def test_no_duplicate_tool_names(self):
        for provider in LIMITED_PROVIDERS:
            names = _selected_names(provider, MESSAGES_GENERIC)
            assert len(names) == len(set(names))


class TestUnlimitedProviders:
    @pytest.mark.parametrize("provider", UNLIMITED_PROVIDERS)
    def test_gets_all_tools(self, provider):
        names = _selected_names(provider, MESSAGES_GENERIC)
        assert len(names) == len(ALL_TOOLS)

    @pytest.mark.parametrize("provider", UNLIMITED_PROVIDERS)
    def test_ollama_keeps_raw_format(self, provider):
        payload = _build_payload(
            MESSAGES_GENERIC, ALL_TOOLS, stream=False, model="test-model", provider=provider
        )
        tools = payload.get("tools", [])
        if provider == "ollama":
            assert all("name" in t for t in tools)
        else:
            assert all(t.get("type") == "function" and "function" in t for t in tools)


class TestKeywordRelevance:
    def test_search_query_prioritizes_web_tools(self):
        names = _selected_names("groq", MESSAGES_SEARCH)
        ranked = [n for n in names if n not in _TOOL_ESSENTIAL_NAMES]
        web_tools = [n for n in ranked if "web" in n or "search" in n or "fetch" in n]
        assert web_tools, "expected web/search tools among selected non-essential tools"
        assert ranked.index(web_tools[0]) < len(ranked) // 2

    def test_desktop_query_prioritizes_app_tools(self):
        names = _selected_names("groq", MESSAGES_DESKTOP)
        ranked = [n for n in names if n not in _TOOL_ESSENTIAL_NAMES]
        app_tools = [n for n in ranked if "app" in n or "window" in n or "desktop" in n]
        assert app_tools, "expected app/window tools among selected non-essential tools"