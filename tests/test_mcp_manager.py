"""Tests for the MCP manager backend (no network, no real servers)."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from core.mcp_manager import (
    MCPManager, _sanitize_server_name, get_global_manager,
)


@pytest.fixture
def manager(tmp_path: Path) -> MCPManager:
    cfg = tmp_path / "mcp_servers.json"
    cfg.write_text(json.dumps({}), encoding="utf-8")
    mgr = MCPManager(config_path=cfg)
    return mgr


def test_sanitize_server_name():
    assert _sanitize_server_name("@modelcontextprotocol/server-github") == "github"
    assert _sanitize_server_name("mcp-server-sqlite") == "sqlite"
    assert _sanitize_server_name("firecrawl-mcp") == "firecrawl"
    assert _sanitize_server_name("exa-mcp-server") == "exa"


def test_global_manager_singleton():
    m1 = get_global_manager()
    m2 = get_global_manager()
    assert m1 is m2


def test_load_missing_config_returns_empty(tmp_path: Path):
    mgr = MCPManager(config_path=tmp_path / "nope.json")
    assert mgr.load_config() == {}


def test_install_disabled_server_no_connect(manager: MCPManager):
    result = manager.install_server_detailed(
        "my-server",
        {"enabled": False, "category": "Dev", "command": "npx",
         "args": ["-y", "some-pkg"], "description": "test"},
    )
    assert result["ok"] is True
    assert "disabled" in result["message"]
    assert "my-server" in manager.load_config()


def test_install_duplicate_rejected(manager: MCPManager):
    cfg = {"enabled": False, "category": "Dev", "command": "npx", "args": ["-y", "x"]}
    manager.install_server_detailed("dup", dict(cfg))
    result = manager.install_server_detailed("dup", dict(cfg))
    assert result["ok"] is False
    assert result["error"] == "exists"


def test_save_creates_backup(manager: MCPManager, tmp_path: Path):
    manager.install_server_detailed("a", {"enabled": False, "category": "Dev",
                                          "command": "npx", "args": ["-y", "a"]})
    manager.install_server_detailed("b", {"enabled": False, "category": "Dev",
                                          "command": "npx", "args": ["-y", "b"]})
    backups = manager.list_backups()
    assert len(backups) >= 1


def test_rollback_restores_config(manager: MCPManager):
    manager.install_server_detailed("only-a", {"enabled": False, "category": "Dev",
                                               "command": "npx", "args": ["-y", "a"]})
    backup = manager.list_backups()[-1]
    manager.install_server_detailed("only-b", {"enabled": False, "category": "Dev",
                                               "command": "npx", "args": ["-y", "b"]})
    assert "only-b" in manager.load_config()
    msg = manager.rollback_backup(backup)
    assert "Rolled back" in msg
    assert "only-b" not in manager.load_config()
    assert "only-a" in manager.load_config()


def test_rollback_rejects_path_traversal(manager: MCPManager):
    msg = manager.rollback_backup("../evil.json")
    assert "Invalid" in msg


def test_tool_meta_write_detection(manager: MCPManager):
    manager._tool_meta["srv"] = {
        "read_tool": {"name": "mcp_srv_read_tool", "read_only": True,
                      "mcp_tool_name": "read_tool", "description": "r", "parameters": {}},
        "write_tool": {"name": "mcp_srv_write_tool", "read_only": False,
                       "mcp_tool_name": "write_tool", "description": "w", "parameters": {}},
    }
    assert manager.is_write_tool("mcp_srv_read_tool") is False
    assert manager.is_write_tool("mcp_srv_write_tool") is True


def test_per_tool_toggle_persists(manager: MCPManager):
    manager.install_server_detailed(
        "srv",
        {"enabled": False, "category": "Dev", "command": "npx", "args": ["-y", "x"],
         "_tools": {"t1": {"enabled": True}}},
    )
    msg = manager.set_tool_enabled("srv", "mcp_srv_t1", False)
    assert "disabled" in msg
    assert manager.is_tool_enabled("srv", "mcp_srv_t1") is False


def test_search_falls_back_to_popular(manager: MCPManager):
    with patch.object(manager, "_search_npm", return_value=[]), \
         patch.object(manager, "_search_official", return_value=[]), \
         patch.object(manager, "_search_pypi", return_value=[]), \
         patch.object(manager, "_search_smithery", return_value=[]):
        results = manager.search_registry("anything")
    assert len(results) > 0
    assert all(r["source"] == "popular" for r in results)


def test_search_cache_hit(manager: MCPManager):
    with patch.object(manager, "_search_npm", return_value=[]) as npm, \
         patch.object(manager, "_search_official", return_value=[]) as off, \
         patch.object(manager, "_search_pypi", return_value=[]), \
         patch.object(manager, "_search_smithery", return_value=[]):
        manager.search_registry("cacheme")
        assert npm.call_count == 1
        manager.search_registry("cacheme")
        assert npm.call_count == 1


def test_uninstall_removes_config(manager: MCPManager):
    manager.install_server_detailed("gone", {"enabled": False, "category": "Dev",
                                             "command": "npx", "args": ["-y", "g"]})
    msg = manager.uninstall_server("gone")
    assert "Uninstalled" in msg
    assert "gone" not in manager.load_config()


def test_settings_roundtrip(manager: MCPManager):
    with patch("core.mcp_manager.DEFAULT_SETTINGS_PATH",
               manager._config_path.parent / "mcp_settings_test.json"):
        manager.save_settings(confirm_write=False)
        assert manager.confirm_write() is False
        manager.save_settings(confirm_write=True)
        assert manager.confirm_write() is True


class _FakeResp:
    """Context-manager response stub for urlopen mocks."""

    def __init__(self, payload: dict):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps(self._payload).encode()


_SMITHERY_FIXTURE = {
    "servers": [
        {"qualifiedName": "github", "displayName": "GitHub",
         "description": "Connect your AI agents to GitHub",
         "useCount": 6059, "verified": True, "remote": True, "isDeployed": True},
        {"qualifiedName": "smithery-ai/github", "displayName": "Github",
         "description": "Access the GitHub API",
         "useCount": 2984, "verified": False, "remote": True, "isDeployed": True},
        {"qualifiedName": "kiennd/reference-servers", "displayName": "Sequential Thinking",
         "description": "stdio only", "remote": False, "isDeployed": True},
    ],
    "pagination": {"currentPage": 1, "pageSize": 25, "totalPages": 75, "totalCount": 150},
}


def test_search_smithery_no_key_returns_empty(manager: MCPManager):
    manager._settings.pop("smithery_api_key", None)
    assert manager._search_smithery("github") == []


def test_search_smithery_parses_and_filters(manager: MCPManager):
    manager._settings["smithery_api_key"] = "test-key"
    captured: dict = {}

    def fake_urlopen(req, timeout=4):
        captured["url"] = req.get_full_url()
        captured["auth"] = req.get_header("Authorization")
        return _FakeResp(_SMITHERY_FIXTURE)

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        results = manager._search_smithery("github")

    assert "api.smithery.ai/servers" in captured["url"]
    assert captured["auth"] == "Bearer test-key"
    # stdio-only (remote: false) entries are filtered out
    names = [r["name"] for r in results]
    assert names == ["github", "smithery-ai/github"]
    assert "kiennd/reference-servers" not in names

    github = next(r for r in results if r["name"] == "github")
    assert github["source"] == "smithery"
    assert github["transport"] == "smithery"
    assert github["server_qname"] == "github"
    assert github["verified"] is True
    assert github["downloads"] == 6059
    assert manager._smithery_pages["github"] == 75


def test_search_smithery_passes_page_params(manager: MCPManager):
    manager._settings["smithery_api_key"] = "test-key"
    captured: dict = {}

    def fake_urlopen(req, timeout=4):
        captured["url"] = req.get_full_url()
        return _FakeResp({"servers": [], "pagination": {"totalPages": 5}})

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        manager._search_smithery("q", page=3, limit=25)
    assert "page=3" in captured["url"]
    assert "pageSize=25" in captured["url"]
    assert "q=q" in captured["url"]


def test_dedup_priority_official_over_smithery(manager: MCPManager):
    with patch.object(manager, "_search_official",
                      return_value=[{"name": "github", "source": "official"}]), \
         patch.object(manager, "_search_smithery",
                      return_value=[{"name": "github", "source": "smithery"}]), \
         patch.object(manager, "_search_npm", return_value=[]), \
         patch.object(manager, "_search_pypi", return_value=[]):
        results = manager.search_registry("github")
    assert results[0]["name"] == "github"
    assert results[0]["source"] == "official"


def test_pagination_has_more_and_page_merge(manager: MCPManager):
    manager._smithery_pages["q"] = 3
    with patch.object(manager, "_search_official", return_value=[]), \
         patch.object(manager, "_search_smithery",
                      return_value=[{"name": "srv", "source": "smithery"}]), \
         patch.object(manager, "_search_npm", return_value=[]), \
         patch.object(manager, "_search_pypi", return_value=[]):
        page1 = manager.search_registry_page("q", 1)
        page3 = manager.search_registry_page("q", 3)
        page3b = manager.search_registry_page("q", 3)
    assert page1["page"] == 1
    assert page1["has_more"] is True
    assert page3["has_more"] is False
    assert page3b["results"] == page3["results"]  # cached same shape


def test_load_more_appends_new_results(manager: MCPManager):
    def smithery_page(q, page, limit):
        return [{"name": f"srv-{page}", "source": "smithery"}]

    with patch.object(manager, "_search_official", return_value=[]), \
         patch.object(manager, "_search_smithery", side_effect=smithery_page), \
         patch.object(manager, "_search_npm", return_value=[]), \
         patch.object(manager, "_search_pypi", return_value=[]):
        page1 = manager.search_registry_page("q", 1)
        page2 = manager.search_registry_page("q", 2)

    assert [r["name"] for r in page1["results"]] == ["srv-1"]
    assert [r["name"] for r in page2["results"]] == ["srv-2"]


def test_search_falls_back_to_popular_with_smithery_patched(manager: MCPManager):
    with patch.object(manager, "_search_npm", return_value=[]), \
         patch.object(manager, "_search_official", return_value=[]), \
         patch.object(manager, "_search_pypi", return_value=[]), \
         patch.object(manager, "_search_smithery", return_value=[]):
        results = manager.search_registry("anything")
    assert len(results) > 0
    assert all(r["source"] == "popular" for r in results)
