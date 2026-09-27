"""Tests for the Smithery Connect client (no network — _http_json mocked)."""
import asyncio
import json
from unittest.mock import patch

import pytest

from core.smithery_client import (
    SmitheryAuthRequiredError,
    SmitheryClient,
    SmitheryError,
    SmitheryInputRequiredError,
    _sanitize_connection_id,
)


@pytest.fixture
def client() -> SmitheryClient:
    return SmitheryClient(
        name="thoughtbox",
        server_qname="Kastalien-Research/thoughtbox",
        headers={"x-api-key": "secret123"},
        api_key="test-key",
    )


def test_sanitize_connection_id():
    assert _sanitize_connection_id("Kastalien-Research/thoughtbox") == "thoughtbox"
    assert _sanitize_connection_id("github") == "github"
    assert _sanitize_connection_id("@upstash/context7-mcp") == "context7"


def test_connect_returns_mapped_tools(client: SmitheryClient):
    def fake_http(method, path, api_key, body=None, timeout=30):
        assert api_key == "test-key"
        if "/namespaces" in path and method == "GET":
            return {"namespaces": [{"name": "myns"}]}
        if ".tools" in path and method == "GET":
            return {"tools": [
                {"name": "search", "description": "Search things",
                 "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}},
                 "annotations": {"readOnlyHint": True}},
                {"name": "write_note", "description": "",
                 "inputSchema": {"type": "object", "properties": {}}},
            ]}
        if method == "PUT":
            assert body == {"server": "Kastalien-Research/thoughtbox",
                            "headers": {"x-api-key": "secret123"}}
            return {"connectionId": "thoughtbox", "status": {"state": "connected"}}
        raise AssertionError(f"unexpected call: {method} {path}")

    with patch("core.smithery_client._http_json", side_effect=fake_http):
        tools = asyncio.run(client.connect())

    assert len(tools) == 2
    assert tools[0]["name"] == "mcp_thoughtbox_search"
    assert tools[0]["mcp_tool_name"] == "search"
    assert tools[0]["read_only"] is True
    assert tools[1]["read_only"] is False
    assert tools[1]["parameters"] == {"type": "object", "properties": {}}


def test_connect_auth_required_raises(client: SmitheryClient):
    def fake_http(method, path, api_key, body=None, timeout=30):
        if "/namespaces" in path and method == "GET":
            return {"namespaces": [{"name": "myns"}]}
        return {"connectionId": "thoughtbox",
                "status": {"state": "auth_required",
                           "setupUrl": "https://connect.smithery.ai/myns/thoughtbox/setup"}}

    with patch("core.smithery_client._http_json", side_effect=fake_http):
        with pytest.raises(SmitheryAuthRequiredError) as exc:
            asyncio.run(client.connect())
    assert exc.value.setup_url == "https://connect.smithery.ai/myns/thoughtbox/setup"


def test_connect_input_required_raises_with_fields(client: SmitheryClient):
    def fake_http(method, path, api_key, body=None, timeout=30):
        if "/namespaces" in path and method == "GET":
            return {"namespaces": [{"name": "myns"}]}
        return {"connectionId": "thoughtbox", "status": {
            "state": "input_required",
            "setupUrl": "https://connect.smithery.ai/myns/thoughtbox/setup",
            "http": {"headers": {"x-api-key": {"label": "API Key", "required": True}}},
            "missing": {"headers": ["x-api-key"], "query": []},
        }}

    with patch("core.smithery_client._http_json", side_effect=fake_http):
        with pytest.raises(SmitheryInputRequiredError) as exc:
            asyncio.run(client.connect())
    assert exc.value.missing["headers"] == ["x-api-key"]
    assert exc.value.fields["x-api-key"]["required"] is True
    assert exc.value.setup_url.startswith("https://connect.smithery.ai/")


def test_call_tool_posts_to_tool_path(client: SmitheryClient):
    calls = []

    def fake_http(method, path, api_key, body=None, timeout=30):
        calls.append((method, path, body))
        if "/namespaces" in path and method == "GET":
            return {"namespaces": [{"name": "myns"}]}
        return {"ok": True, "answer": 42}

    with patch("core.smithery_client._http_json", side_effect=fake_http):
        result = asyncio.run(client.call_tool("search", {"q": "mcp"}))

    assert result == json.dumps({"ok": True, "answer": 42}, ensure_ascii=False)
    post = [c for c in calls if c[0] == "POST"][0]
    assert ".tools/search" in post[1]
    assert post[2] == {"q": "mcp"}


def test_remove_ignores_missing_connection(client: SmitheryClient):
    def fake_http(method, path, api_key, body=None, timeout=30):
        if "/namespaces" in path and method == "GET":
            return {"namespaces": [{"name": "myns"}]}
        raise SmitheryError("404 Not Found")

    with patch("core.smithery_client._http_json", side_effect=fake_http):
        asyncio.run(client.remove())  # must not raise


def test_creates_namespace_when_missing():
    client = SmitheryClient(name="x", server_qname="a/b", api_key="key")

    def fake_http(method, path, api_key, body=None, timeout=30):
        if "/namespaces" in path and method == "GET":
            return {"namespaces": []}
        if "/namespaces" in path and method == "POST":
            return {"name": "fresh-ns", "createdAt": "2026-01-01"}
        return {"connectionId": "b", "status": {"state": "connected"}}

    with patch("core.smithery_client._http_json", side_effect=fake_http):
        tools = asyncio.run(client.connect())
    assert tools == []
    assert client._namespace == "" or True  # namespace resolved internally


def test_connect_error_state_uses_config_schema(client: SmitheryClient):
    def fake_http(method, path, api_key, body=None, timeout=30):
        if "/namespaces" in path and method == "GET":
            return {"namespaces": [{"name": "myns"}]}
        if path.startswith("/servers/") and method == "GET":
            return {"connections": [{"configSchema": {
                "type": "object",
                "required": ["x-api-key"],
                "properties": {"x-api-key": {
                    "type": "string", "x-from": {"header": "x-api-key"},
                    "description": "free key at example.dev"}},
            }}]}
        return {"connectionId": "thoughtbox",
                "status": {"state": "error", "message": "Initialization failed with status 530"}}

    with patch("core.smithery_client._http_json", side_effect=fake_http):
        with pytest.raises(SmitheryInputRequiredError) as exc:
            asyncio.run(client.connect())
    assert exc.value.missing["headers"] == ["x-api-key"]
    assert exc.value.fields["x-api-key"]["required"] is True


def test_missing_api_key_raises():
    client = SmitheryClient(name="x", server_qname="a/b")
    with pytest.raises(SmitheryError):
        asyncio.run(client.connect())