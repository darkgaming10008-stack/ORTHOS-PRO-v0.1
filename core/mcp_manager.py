from __future__ import annotations

import concurrent.futures
import json
import os
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from core.mcp_client import MCPClient
from core.smithery_client import (
    SmitheryAuthRequiredError,
    SmitheryClient,
    SmitheryInputRequiredError,
)
from core.tool_registry import ToolRegistry
from gateway.async_worker import get_async_worker


def _is_mcp_package(name: str, description: str) -> bool:
    """Heuristic: is this package likely an MCP server?"""
    desc_lower = description.lower()
    name_lower = name.lower()
    keywords = ("mcp", "model-context-protocol", "modelcontextprotocol")
    return any(k in name_lower for k in keywords) or any(k in desc_lower for k in keywords)


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "mcp_servers.json"
DEFAULT_SETTINGS_PATH = BASE_DIR / "config" / "mcp_settings.json"
BACKUP_DIR = BASE_DIR / "config" / "mcp_backups"
MAX_BACKUPS = 20
SEARCH_CACHE_TTL = 600
SEARCH_TIMEOUT = 6
SEARCH_PAGE_SIZE = 25

STATUS_STOPPED = "stopped"
STATUS_CONNECTING = "connecting"
STATUS_RUNNING = "running"
STATUS_ERROR = "error"

# ---------------------------------------------------------------------------
# Shared runtime — lets the UI and main.py operate on the same registry/index
# ---------------------------------------------------------------------------

_runtime_lock = threading.Lock()
_shared_registry: ToolRegistry | None = None
_shared_index: Any | None = None
_shared_loaded: set[str] | None = None


def attach_runtime(registry: ToolRegistry | None = None,
                   tool_index: Any | None = None,
                   loaded_set: set[str] | None = None) -> None:
    """Attach the app's live runtime so MCP tools register/unregister instantly."""
    global _shared_registry, _shared_index, _shared_loaded
    with _runtime_lock:
        _shared_registry = registry
        _shared_index = tool_index
        _shared_loaded = loaded_set


def detach_runtime() -> None:
    global _shared_registry, _shared_index, _shared_loaded
    with _runtime_lock:
        _shared_registry = None
        _shared_index = None
        _shared_loaded = None


_manager_singleton: "MCPManager | None" = None
_manager_lock = threading.Lock()


def get_global_manager() -> "MCPManager":
    """Return the process-wide MCPManager singleton."""
    global _manager_singleton
    if _manager_singleton is None:
        with _manager_lock:
            if _manager_singleton is None:
                _manager_singleton = MCPManager()
    return _manager_singleton


def _sanitize_server_name(package: str) -> str:
    """Turn a package name into a stable, safe server key."""
    name = package.split("/")[-1] if "/" in package else package
    for marker in ("server-", "-mcp", "mcp-", "-server", "mcp_"):
        name = name.replace(marker, "")
    name = name.replace("@", "").replace("_", "-")
    parts = [p for p in name.split("-") if p]
    return "-".join(parts) if parts else package


class MCPManager:
    """Manages multiple MCP server connections and tool registration."""

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH):
        self._config_path = Path(config_path)
        self._clients: dict[str, MCPClient] = {}
        self._worker = get_async_worker()
        self._enabled = True

        self._states: dict[str, str] = {}
        self._last_error: dict[str, str] = {}
        self._tool_meta: dict[str, dict] = {}   # server -> short_name -> meta
        self._search_cache: dict[str, tuple[float, list[dict], bool]] = {}
        self._smithery_pages: dict[str, int] = {}
        self._npm_totals: dict[str, int] = {}
        self._probe_cache: dict[str, tuple[float, dict]] = {}
        self._settings: dict[str, Any] = {}

        self._runtime_lock = threading.Lock()
        self._load_settings()

    # ------------------------------------------------------------------
    # Settings (app-level, separate file so config stays clean)
    # ------------------------------------------------------------------

    def _load_settings(self) -> None:
        try:
            if DEFAULT_SETTINGS_PATH.exists():
                self._settings = json.loads(DEFAULT_SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            self._settings = {}

    def get_settings(self) -> dict:
        return dict(self._settings)

    def save_settings(self, **kwargs) -> bool:
        self._settings.update(kwargs)
        try:
            DEFAULT_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            DEFAULT_SETTINGS_PATH.write_text(json.dumps(self._settings, indent=4), encoding="utf-8")
            return True
        except Exception as e:
            print(f"[MCP] Settings save error: {e}")
            return False

    def confirm_write(self) -> bool:
        return bool(self._settings.get("confirm_write", True))

    def get_smithery_key(self) -> str:
        return str(self._settings.get("smithery_api_key", "") or "").strip()

    # ------------------------------------------------------------------
    # Config + backups
    # ------------------------------------------------------------------

    def load_config(self) -> dict[str, dict]:
        try:
            if not self._config_path.exists():
                print(f"[MCP] Config not found: {self._config_path}")
                self._enabled = False
                return {}
            data = json.loads(self._config_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                servers = {}
                for svc in data:
                    name = svc.get("name", "unknown")
                    servers[name] = {
                        "enabled": True,
                        "category": "Uncategorized",
                        "command": svc.get("command", "npx"),
                        "args": svc.get("args", []),
                        "description": svc.get("description", f"MCP server: {name}"),
                    }
                    if svc.get("env"):
                        servers[name]["env"] = svc["env"]
                return servers
            if isinstance(data, dict):
                return {k: v for k, v in data.items() if not k.startswith("_")}
            return {}
        except Exception as e:
            print(f"[MCP] Failed to load config: {e}")
            self._enabled = False
            return {}

    def save_config(self, config: dict[str, dict] | None = None) -> bool:
        if config is None:
            config = self.load_config()
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            self._backup_config()
            self._config_path.write_text(json.dumps(config, indent=4), encoding="utf-8")
            return True
        except Exception as e:
            print(f"[MCP] Config save error: {e}")
            return False

    def _backup_config(self) -> None:
        if not self._config_path.exists():
            return
        try:
            BACKUP_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            target = BACKUP_DIR / f"mcp_servers_{stamp}.json"
            target.write_text(self._config_path.read_text(encoding="utf-8"), encoding="utf-8")
            backups = sorted(BACKUP_DIR.glob("mcp_servers_*.json"), key=lambda p: p.name, reverse=True)
            for old in backups[MAX_BACKUPS:]:
                try:
                    old.unlink()
                except OSError:
                    pass
        except Exception as e:
            print(f"[MCP] Backup failed: {e}")

    def list_backups(self) -> list[str]:
        try:
            if not BACKUP_DIR.exists():
                return []
            return sorted(p.name for p in BACKUP_DIR.glob("mcp_servers_*.json"))
        except Exception:
            return []

    def rollback_backup(self, backup_name: str) -> str:
        if not backup_name or "/" in backup_name or "\\" in backup_name:
            return "[MCP] Invalid backup name"
        path = BACKUP_DIR / backup_name
        if not path.exists():
            return f"[MCP] Backup '{backup_name}' not found"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            return f"[MCP] Backup corrupt: {e}"
        self.stop_all()
        if self.save_config(data):
            return f"[MCP] Rolled back to '{backup_name}'"
        return "[MCP] Rollback failed"

    def get_all_servers_config(self) -> dict[str, dict]:
        return self.load_config()

    # ------------------------------------------------------------------
    # Tool config helpers (per-tool toggles, write detection)
    # ------------------------------------------------------------------

    def get_tool_config(self, server: str) -> dict:
        cfg = self.load_config().get(server, {})
        return cfg.get("_tools", {}) or {}

    def is_tool_enabled(self, server: str, tool_full_name: str) -> bool:
        short = tool_full_name.replace(f"mcp_{server}_", "", 1)
        entry = self.get_tool_config(server).get(short)
        return entry.get("enabled", True) if isinstance(entry, dict) else True

    def is_write_tool(self, tool_full_name: str) -> bool:
        for server, metas in self._tool_meta.items():
            short = tool_full_name.replace(f"mcp_{server}_", "", 1)
            if short != tool_full_name and short in metas:
                return not bool(metas[short].get("read_only"))
        return True

    def set_tool_enabled(self, server: str, tool_full_name: str, enabled: bool) -> str:
        config = self.load_config()
        srv = config.get(server)
        if not srv:
            return f"[MCP] Server '{server}' not found"
        short = tool_full_name.replace(f"mcp_{server}_", "", 1)
        tools_cfg = srv.setdefault("_tools", {})
        entry = tools_cfg.setdefault(short, {})
        entry["enabled"] = bool(enabled)
        if not self.save_config(config):
            return "[MCP] Failed to save config"
        self._sync_server_tools(server)
        return f"[MCP] '{tool_full_name}' {'enabled' if enabled else 'disabled'}"

    def _register_one(self, server: str, meta: dict) -> None:
        """Register a single tool into the shared runtime (registry + index)."""
        full_name = meta["name"]
        with self._runtime_lock:
            registry = _shared_registry
            index = _shared_index
        if registry is not None:
            registry.register(
                func=self._make_mcp_handler(server, meta["mcp_tool_name"]),
                name=full_name,
                description=meta["description"],
                parameters=meta.get("parameters", {}),
            )
        if index is not None:
            if full_name not in index._mcp_tool_names:
                index._mcp_tool_names.append(full_name)
            index._mcp_summaries[full_name] = meta["description"][:120]
            index._mcp_server_tools.setdefault(server, [])
            if full_name not in index._mcp_server_tools[server]:
                index._mcp_server_tools[server].append(full_name)
            index._mcp_embeddings = None

    def _unregister_all(self, server: str) -> None:
        """Remove all of a server's tools from the shared runtime."""
        with self._runtime_lock:
            registry = _shared_registry
            index = _shared_index
            loaded = _shared_loaded
        names = list(index._mcp_server_tools.get(server, [])) if index is not None else []
        if registry is not None:
            for n in names:
                registry.unregister(n)
        if index is not None:
            for n in names:
                if n in index._mcp_tool_names:
                    index._mcp_tool_names.remove(n)
                index._mcp_summaries.pop(n, None)
                if loaded is not None:
                    loaded.discard(n)
            index._mcp_server_tools.pop(server, None)
            index._mcp_embeddings = None

    def _sync_server_tools(self, server: str) -> None:
        """Re-register a server's enabled tools (drop disabled ones)."""
        self._unregister_all(server)
        metas = self._tool_meta.get(server, {})
        for short, meta in metas.items():
            if self.is_tool_enabled(server, meta["name"]):
                self._register_one(server, meta)

    # ------------------------------------------------------------------
    # Discovery & registration
    # ------------------------------------------------------------------

    def discover_and_register(self, registry: ToolRegistry) -> list[dict]:
        """Connect to all enabled servers, discover tools, and register them.

        Returns tool declarations to append to the LLM's tool list.
        """
        servers = self.load_config()
        if not servers or not self._enabled:
            return []

        declarations: list[dict] = []

        for name, cfg in servers.items():
            if not cfg.get("enabled", True):
                continue
            decls = self._connect_server(name, cfg, registry=registry)
            declarations.extend(decls)

        return declarations

    def _connect_server(self, name: str, cfg: dict,
                        registry: ToolRegistry | None = None) -> list[dict]:
        """Connect to one server, register its enabled tools, track state."""
        command = cfg.get("command")
        args = cfg.get("args", [])
        env = cfg.get("env")
        url = cfg.get("url")
        headers = cfg.get("headers")

        self._states[name] = STATUS_CONNECTING
        self._last_error.pop(name, None)

        if cfg.get("transport") == "smithery":
            client: Any = SmitheryClient(
                name=name,
                server_qname=cfg.get("server_qname") or name,
                headers=headers,
                api_key=self.get_smithery_key() or None,
                connection_id=cfg.get("connection_id"),
            )
            try:
                tools = self._worker.run_sync(client.connect(), timeout=90)
            except SmitheryAuthRequiredError as e:
                self._states[name] = STATUS_ERROR
                self._last_error[name] = (
                    f"Smithery OAuth required — open in your browser: {e.setup_url}")
                print(f"[MCP] Server '{name}': {self._last_error[name]}")
                return []
            except SmitheryInputRequiredError as e:
                missing = list(e.missing.get("headers", [])) + list(e.missing.get("query", []))
                self._states[name] = STATUS_ERROR
                msg = f"Smithery server needs configuration: {', '.join(missing) or 'see setup page'}"
                if e.setup_url:
                    msg += f" — open in your browser: {e.setup_url}"
                self._last_error[name] = msg
                print(f"[MCP] Server '{name}': {msg}")
                return []
            except Exception as e:
                self._states[name] = STATUS_ERROR
                self._last_error[name] = str(e)
                print(f"[MCP] Failed to connect server '{name}': {e}")
                return []
        else:
            client = MCPClient(name=name, command=command, args=args, env=env, url=url, headers=headers)
            try:
                tools = self._worker.run_sync(client.connect(), timeout=180)
            except Exception as e:
                self._states[name] = STATUS_ERROR
                self._last_error[name] = str(e)
                print(f"[MCP] Failed to connect server '{name}': {e}")
                return []

        self._clients[name] = client
        self._states[name] = STATUS_RUNNING
        self._tool_meta[name] = {}

        declarations: list[dict] = []
        for info in tools:
            mcp_full_name = info["name"]
            short = mcp_full_name.replace(f"mcp_{name}_", "", 1)
            self._tool_meta[name][short] = {
                "name": mcp_full_name,
                "mcp_tool_name": info["mcp_tool_name"],
                "description": info["description"],
                "parameters": info.get("parameters", {}),
                "read_only": bool(info.get("read_only", False)),
            }
            if not self.is_tool_enabled(name, mcp_full_name):
                continue
            self._register_one(name, self._tool_meta[name][short])
            declarations.append({
                "type": "function",
                "function": {
                    "name": mcp_full_name,
                    "description": info["description"],
                    "parameters": info.get("parameters", {"type": "object", "properties": {}}),
                },
                "mcp_server": info.get("mcp_server", name),
            })

        print(f"[MCP] Server '{name}': {len(tools)} tool(s) discovered, {len(declarations)} registered")
        return declarations

    # ------------------------------------------------------------------
    # Per-server lifecycle
    # ------------------------------------------------------------------

    def get_server_status(self, name: str) -> str:
        return self._states.get(name, STATUS_STOPPED)

    def get_last_error(self, name: str) -> str:
        return self._last_error.get(name, "")

    def get_server_tools_meta(self, name: str) -> list[dict]:
        metas = self._tool_meta.get(name, {})
        out = []
        for short, meta in metas.items():
            out.append({
                "name": meta["name"],
                "short": short,
                "description": meta["description"],
                "read_only": bool(meta.get("read_only", False)),
                "enabled": self.is_tool_enabled(name, meta["name"]),
            })
        return sorted(out, key=lambda t: t["name"])

    def start_server(self, name: str) -> str:
        if name in self._clients and self._states.get(name) == STATUS_RUNNING:
            return f"[MCP] Server '{name}' is already running"
        cfg = self.load_config().get(name)
        if not cfg:
            return f"[MCP] Server '{name}' not found in config"
        decls = self._connect_server(name, cfg)
        if decls or self._states.get(name) == STATUS_RUNNING:
            n = len([d for d in decls])
            return f"[MCP] Started '{name}' ({n} tool(s) registered)"
        err = self._last_error.get(name, "unknown error")
        return f"[MCP] Failed to start '{name}': {err}"

    def stop_server(self, name: str) -> str:
        client = self._clients.pop(name, None)
        if client:
            try:
                self._worker.run_sync(client.disconnect(), timeout=10)
            except Exception:
                pass
        self._unregister_all(name)
        self._tool_meta.pop(name, None)
        self._states[name] = STATUS_STOPPED
        return f"[MCP] Stopped '{name}'"

    def stop_all(self) -> None:
        for name in list(self._clients.keys()):
            self.stop_server(name)

    def restart_server(self, name: str) -> str:
        self.stop_server(name)
        return self.start_server(name)

    def toggle_server(self, name: str) -> str:
        config = self.load_config()
        cfg = config.get(name)
        if not cfg:
            return f"[MCP] Server '{name}' not found"
        if cfg.get("enabled", False):
            self.stop_server(name)
            cfg["enabled"] = False
            self.save_config(config)
            return f"[MCP] '{name}' disabled and stopped"
        else:
            cfg["enabled"] = True
            self.save_config(config)
            msg = self.start_server(name)
            return f"[MCP] '{name}' enabled. {msg}"

    # ------------------------------------------------------------------
    # Health probe
    # ------------------------------------------------------------------

    def probe_server(self, name: str, force: bool = False) -> dict:
        """Real health check: re-list tools on the live session."""
        cached = self._probe_cache.get(name)
        if cached and not force and time.time() - cached[0] < 10:
            return dict(cached[1])

        client = self._clients.get(name)
        if client is None:
            result = {"ok": False, "tools": 0, "error": "not running"}
        else:
            try:
                tools = self._worker.run_sync(client.list_tools_live(), timeout=10)
                result = {"ok": True, "tools": len(tools), "error": None}
                self._states[name] = STATUS_RUNNING
            except Exception as e:
                self._states[name] = STATUS_ERROR
                self._last_error[name] = str(e)
                result = {"ok": False, "tools": 0, "error": str(e)}

        self._probe_cache[name] = (time.time(), result)
        return dict(result)

    # ------------------------------------------------------------------
    # Install pipeline (ChatGPT-style: preview -> scan tools -> connect)
    # ------------------------------------------------------------------

    def preview_server(self, config: dict) -> tuple[list[dict], str | None, dict | None]:
        """Connect to a not-yet-installed server and return its tool list.

        Returns (tools, error, info). For Smithery servers, `info` may carry
        {"kind": "auth_required", "setup_url": ...} (OAuth needed) or
        {"kind": "input_required", "fields": ..., "missing": ...} (config keys
        needed). Does NOT persist anything.
        """
        name = "preview"
        try:
            if config.get("transport") == "smithery":
                client = SmitheryClient(
                    name=name,
                    server_qname=config.get("server_qname") or "server",
                    headers=config.get("headers"),
                    api_key=self.get_smithery_key() or None,
                    connection_id=config.get("connection_id"),
                )
                tools = self._worker.run_sync(client.connect(), timeout=90)
            else:
                client = MCPClient(
                    name=name,
                    command=config.get("command"),
                    args=config.get("args", []),
                    env=config.get("env"),
                    url=config.get("url"),
                    headers=config.get("headers"),
                )
                tools = self._worker.run_sync(client.connect(), timeout=180)
            return tools, None, None
        except SmitheryAuthRequiredError as e:
            return [], str(e), {"kind": "auth_required", "setup_url": e.setup_url}
        except SmitheryInputRequiredError as e:
            return [], str(e), {
                "kind": "input_required",
                "fields": e.fields,
                "missing": e.missing,
                "setup_url": e.setup_url,
            }
        except Exception as e:
            return [], str(e), None
        finally:
            try:
                self._worker.run_sync(client.disconnect(), timeout=10)
            except Exception:
                pass

    def install_server(self, name: str, config_dict: dict) -> str:
        """Backwards-compatible install that returns a status string."""
        result = self.install_server_detailed(name, config_dict)
        return result["message"]

    def install_server_detailed(self, name: str, config_dict: dict,
                                enabled_tools: list[str] | None = None) -> dict:
        """Install, connect, probe, and (if runtime attached) register live.

        enabled_tools: optional list of short tool names to enable (rest disabled).
        Returns {"ok": bool, "message": str, "tools": list, "error": str|None}.
        """
        config = self.load_config()
        if name in config:
            return {"ok": False, "message": f"[MCP] Server '{name}' already exists",
                    "tools": [], "error": "exists"}

        if enabled_tools is not None:
            config_dict = dict(config_dict)
            config_dict["_tools"] = {t: {"enabled": True} for t in enabled_tools}

        config[name] = config_dict
        if not self.save_config(config):
            return {"ok": False, "message": f"[MCP] Failed to save config for '{name}'",
                    "tools": [], "error": "save_failed"}

        if not config_dict.get("enabled", True):
            return {"ok": True, "message": f"[MCP] Added '{name}' (disabled)",
                    "tools": [], "error": None}

        decls = self._connect_server(name, config_dict)
        ok = self._states.get(name) == STATUS_RUNNING
        probe = self.probe_server(name, force=True) if ok else {"ok": False, "tools": 0, "error": self._last_error.get(name)}
        tools = self.get_server_tools_meta(name)
        message = (
            f"[MCP] Installed '{name}' — {len(tools)} tool(s) online ✓"
            if ok else
            f"[MCP] Installed '{name}' but connection failed: {self._last_error.get(name, '?')}"
        )
        return {"ok": ok, "message": message, "tools": tools, "error": None if ok else self._last_error.get(name)}

    def uninstall_server(self, name: str) -> str:
        self.stop_server(name)
        config = self.load_config()
        if name not in config:
            return f"[MCP] Server '{name}' not found in config"
        cfg = config[name]
        del config[name]
        self.save_config(config)
        if cfg.get("transport") == "smithery":
            try:
                client = SmitheryClient(
                    name=name,
                    server_qname=cfg.get("server_qname") or name,
                    api_key=self.get_smithery_key() or None,
                    connection_id=cfg.get("connection_id"),
                )
                self._worker.run_sync(client.remove(), timeout=20)
            except Exception as e:
                print(f"[MCP] Smithery connection cleanup failed for '{name}': {e}")
        return f"[MCP] Uninstalled '{name}'"

    # ------------------------------------------------------------------
    # Update / refresh (ChatGPT-style "Refresh" for tool metadata)
    # ------------------------------------------------------------------

    def update_server(self, name: str) -> dict:
        """Reconnect, re-scan tools, update per-tool config, return diff summary."""
        cfg = self.load_config().get(name)
        if not cfg:
            return {"ok": False, "message": f"[MCP] Server '{name}' not found", "diff": []}

        before = {meta["name"] for meta in self.get_server_tools_meta(name)}
        decls = self._connect_server(name, cfg)
        after = {meta["name"] for meta in self.get_server_tools_meta(name)}

        added = sorted(after - before)
        removed = sorted(before - after)
        return {
            "ok": self._states.get(name) == STATUS_RUNNING,
            "message": f"[MCP] '{name}' refreshed — {len(added)} added, {len(removed)} removed",
            "diff": {"added": added, "removed": removed},
        }

    # ------------------------------------------------------------------
    # Registry search — parallel federated search with cache + pagination
    # ------------------------------------------------------------------

    def search_registry(self, query: str = "", force: bool = False) -> list[dict]:
        """Federated search across Official + Smithery + npm + PyPI."""
        return self.search_registry_page(query, 1, force)["results"]

    def search_registry_page(self, query: str = "", page: int = 1,
                             force: bool = False) -> dict:
        """Search with pagination support.

        Page 1 queries all four sources; later pages paginate the sources that
        support it (Smithery via page/pageSize, npm via from). Returns:
        {"results": [...], "page": page, "has_more": bool}
        """
        key = (query or "").strip().lower()
        page = max(1, int(page))
        cache_key = (key, page)
        cached = self._search_cache.get(cache_key)
        if cached and not force and time.time() - cached[0] < SEARCH_CACHE_TTL:
            return {"results": list(cached[1]), "page": page, "has_more": cached[2]}

        seen: dict[str, dict] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            future_official = ex.submit(self._search_official, query, SEARCH_PAGE_SIZE)
            future_smithery = ex.submit(self._search_smithery, query, page, SEARCH_PAGE_SIZE)
            future_npm = ex.submit(self._search_npm, query, SEARCH_PAGE_SIZE,
                                   (page - 1) * SEARCH_PAGE_SIZE)
            future_pypi = ex.submit(self._search_pypi, query, SEARCH_PAGE_SIZE)

            # Priority: official overwrites, everything else setdefaults.
            for r in self._safe_future(future_official):
                seen[r["name"]] = r
            for r in self._safe_future(future_smithery):
                seen.setdefault(r["name"], r)
            for r in self._safe_future(future_npm):
                seen.setdefault(r["name"], r)
            for r in self._safe_future(future_pypi):
                seen.setdefault(r["name"], r)

        if not seen:
            results = self._popular_fallback()
            has_more = False
        else:
            results = list(seen.values())
            has_more = (
                self._smithery_pages.get(key, 0) > page
                or self._npm_totals.get(key, 0) > page * SEARCH_PAGE_SIZE
            )

        self._search_cache[cache_key] = (time.time(), results, has_more)
        return {"results": results, "page": page, "has_more": has_more}

    @staticmethod
    def _safe_future(future) -> list[dict]:
        try:
            return future.result(timeout=SEARCH_TIMEOUT + 1) or []
        except Exception:
            return []

    def _search_npm(self, query: str, limit: int = SEARCH_PAGE_SIZE,
                    from_index: int = 0) -> list[dict]:
        """Search npm registry for MCP-related packages."""
        results = []
        try:
            text = f"mcp {query}" if query else "mcp"
            url = f"https://registry.npmjs.org/-/v1/search?text={urllib.parse.quote(text)}&size={limit}"
            if from_index:
                url += f"&from={from_index}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=SEARCH_TIMEOUT) as resp:
                data = json.loads(resp.read().decode())
                self._npm_totals[(query or "").strip().lower()] = int(data.get("total", 0) or 0)
                for pkg in data.get("objects", []):
                    p = pkg.get("package", {})
                    name: str = p.get("name", "")
                    if not name or not _is_mcp_package(name, p.get("description", "")):
                        continue
                    downloads = pkg.get("downloads")
                    if isinstance(downloads, dict):
                        downloads = downloads.get("monthly")
                    try:
                        downloads = int(downloads) if downloads is not None else None
                    except (TypeError, ValueError):
                        downloads = None
                    results.append({
                        "name": name,
                        "description": (p.get("description", "") or "")[:200],
                        "version": p.get("version", ""),
                        "source": "npm",
                        "command": "npx",
                        "args_base": ["-y", name],
                        "downloads": downloads,
                        "verified": False,
                    })
        except Exception:
            pass
        return results

    def _search_official(self, query: str, limit: int = 15) -> list[dict]:
        """Search the official MCP Registry (registry.modelcontextprotocol.io).

        2026 v0.1 format: {"servers": [{ "_meta": {...}, "server": { name, description,
        version, packages: [{registryType, identifier, runtimeHint, runtimeArguments,
        transport, environmentVariables}], remotes: [{type, url, headers}] } }]}
        """
        results = []
        try:
            params = ""
            if query:
                params = f"?search={urllib.parse.quote(query)}&limit={limit}"
            else:
                params = f"?limit={limit}"
            url = f"https://registry.modelcontextprotocol.io/v0/servers{params}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=SEARCH_TIMEOUT) as resp:
                data = json.loads(resp.read().decode())
                for entry in data.get("data", data.get("servers", [])):
                    server = entry.get("server", entry) if isinstance(entry, dict) else entry
                    if not isinstance(server, dict):
                        continue
                    name: str = server.get("name", server.get("id", ""))
                    if not name:
                        continue

                    # ── Local install (packages) ──
                    packages = server.get("packages") or []
                    if packages:
                        for pk in packages:
                            if pk.get("registryType") not in ("npm", "pypi", "python"):
                                continue
                            identifier = pk.get("identifier", "")
                            if not identifier:
                                continue
                            runtime = pk.get("runtimeHint") or (
                                "npx" if pk.get("registryType") == "npm" else "uvx")
                            args_base = []
                            for ra in pk.get("runtimeArguments") or []:
                                if ra.get("type") == "positional":
                                    args_base.append(ra.get("value"))
                            if identifier not in args_base:
                                args_base.append(identifier)
                            env_meta = []
                            for ev in pk.get("environmentVariables") or []:
                                env_meta.append({
                                    "name": ev.get("name", ""),
                                    "required": bool(ev.get("isRequired", False)),
                                    "secret": bool(ev.get("isSecret", False)),
                                    "description": ev.get("description", ""),
                                })
                            results.append({
                                "name": name,
                                "description": (server.get("description", "") or "")[:200],
                                "version": pk.get("version") or server.get("version", "latest"),
                                "source": "official",
                                "command": runtime,
                                "args_base": args_base,
                                "env": env_meta,
                                "downloads": None,
                                "verified": True,
                            })
                            break
                        continue

                    # ── Remote install (remotes / streamable-http) ──
                    remotes = server.get("remotes") or []
                    if remotes:
                        r = remotes[0]
                        headers_meta = []
                        for h in r.get("headers") or []:
                            value = h.get("value", "")
                            var = None
                            if "{" in value and "}" in value:
                                var = value[value.find("{") + 1:value.find("}")]
                            headers_meta.append({
                                "name": h.get("name", "Authorization"),
                                "var": var or "API_KEY",
                                "value_template": value,
                                "required": bool(h.get("isRequired", False)),
                                "secret": bool(h.get("isSecret", False)),
                                "description": h.get("description", ""),
                            })
                        results.append({
                            "name": name,
                            "description": (server.get("description", "") or "")[:200],
                            "version": server.get("version", "latest"),
                            "source": "official",
                            "command": None,
                            "args_base": [],
                            "url": r.get("url", ""),
                            "headers": headers_meta,
                            "downloads": None,
                            "verified": True,
                        })
        except Exception:
            pass
        return results

    def _search_smithery(self, query: str, page: int = 1,
                         limit: int = SEARCH_PAGE_SIZE) -> list[dict]:
        """Search Smithery's public registry (needs smithery_api_key).

        Remotely hosted servers install through the Connect API; stdio-only
        listings (`remote: false`) are skipped because they need the CLI.
        """
        api_key = self.get_smithery_key()
        if not api_key:
            return []
        results = []
        try:
            params: dict[str, Any] = {"page": int(page), "pageSize": int(limit)}
            if query:
                params["q"] = query
            url = ("https://api.smithery.ai/servers?"
                   + urllib.parse.urlencode(params))
            req = urllib.request.Request(url, headers={
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "Mozilla/5.0",
            })
            with urllib.request.urlopen(req, timeout=SEARCH_TIMEOUT) as resp:
                data = json.loads(resp.read().decode())
            key = (query or "").strip().lower()
            pagination = data.get("pagination") or {}
            self._smithery_pages[key] = int(pagination.get("totalPages", 0) or 0)
            for s in data.get("servers", []):
                if not isinstance(s, dict) or not s.get("qualifiedName"):
                    continue
                if not s.get("remote") or not s.get("isDeployed"):
                    continue
                name: str = s["qualifiedName"]
                results.append({
                    "name": name,
                    "description": (s.get("description") or "")[:200],
                    "version": s.get("version") or "latest",
                    "source": "smithery",
                    "command": None,
                    "args_base": [],
                    "url": None,
                    "transport": "smithery",
                    "server_qname": name,
                    "headers": [],
                    "downloads": s.get("useCount"),
                    "verified": bool(s.get("verified")),
                    "display_name": s.get("displayName") or name,
                })
        except Exception:
            pass
        return results

    def _search_pypi(self, query: str, limit: int = SEARCH_PAGE_SIZE) -> list[dict]:
        """Search PyPI via XML-RPC for Python MCP packages."""
        results = []
        try:
            import xmlrpc.client
            client = xmlrpc.client.ServerProxy("https://pypi.org/pypi")
            search_terms = f"mcp {query}" if query else "mcp"
            hits = client.search(
                {"name": search_terms, "summary": search_terms, "description": search_terms},
                "or",
            )
            for hit in hits[:limit]:
                name: str = hit.get("name", "")
                if not name or not _is_mcp_package(name, hit.get("summary", "")):
                    continue
                results.append({
                    "name": name,
                    "description": (hit.get("summary", "") or "")[:200],
                    "version": hit.get("version", "latest"),
                    "source": "pypi",
                    "command": "uvx",
                    "args_base": [name],
                    "score": None,
                    "verified": False,
                })
        except Exception:
            pass
        return results

    def _popular_fallback(self) -> list[dict]:
        """Return curated popular servers when all searches fail."""
        return [
            {"name": "@modelcontextprotocol/server-github", "description": "GitHub API — repos, issues, PRs", "source": "popular", "command": "npx", "args_base": ["-y", "@modelcontextprotocol/server-github"], "downloads": 250000, "verified": True},
            {"name": "@modelcontextprotocol/server-playwright", "description": "Browser automation", "source": "popular", "command": "npx", "args_base": ["-y", "@modelcontextprotocol/server-playwright"], "downloads": 220000, "verified": True},
            {"name": "@modelcontextprotocol/server-brave-search", "description": "Web search via Brave", "source": "popular", "command": "npx", "args_base": ["-y", "@modelcontextprotocol/server-brave-search"], "downloads": 90000, "verified": True},
            {"name": "@modelcontextprotocol/server-postgres", "description": "PostgreSQL database", "source": "popular", "command": "npx", "args_base": ["-y", "@modelcontextprotocol/server-postgres"], "downloads": 70000, "verified": True},
            {"name": "@modelcontextprotocol/server-filesystem", "description": "File system access", "source": "popular", "command": "npx", "args_base": ["-y", "@modelcontextprotocol/server-filesystem"], "downloads": 120000, "verified": True},
            {"name": "@modelcontextprotocol/server-sequential-thinking", "description": "Step-by-step reasoning", "source": "popular", "command": "npx", "args_base": ["-y", "@modelcontextprotocol/server-sequential-thinking"], "downloads": 80000, "verified": True},
            {"name": "@upstash/context7-mcp", "description": "Up-to-date library docs", "source": "popular", "command": "npx", "args_base": ["-y", "@upstash/context7-mcp"], "downloads": 65000, "verified": True},
            {"name": "firecrawl-mcp", "description": "Web scraping + search", "source": "popular", "command": "npx", "args_base": ["-y", "firecrawl-mcp"], "downloads": 55000, "verified": False},
            {"name": "exa-mcp-server", "description": "AI-powered search engine", "source": "popular", "command": "npx", "args_base": ["-y", "exa-mcp-server"], "downloads": 40000, "verified": False},
            {"name": "minimax-mcp", "description": "TTS, image & video generation", "source": "popular", "command": "npx", "args_base": ["-y", "minimax-mcp"], "downloads": 20000, "verified": False},
        ]

    # ------------------------------------------------------------------
    # Tool handler factory
    # ------------------------------------------------------------------

    def _make_mcp_handler(self, server_name: str, tool_name: str) -> Callable:
        """Return a synchronous handler that calls the MCP tool via the async worker."""

        def handler(parameters=None, response=None, player=None, **kwargs):
            args = parameters or kwargs or {}
            client = self._clients.get(server_name)
            if client is None:
                return f"Error: MCP server '{server_name}' not connected"
            try:
                return self._worker.run_sync(client.call_tool(tool_name, args), timeout=60)
            except Exception as e:
                return f"MCP error on '{server_name}/{tool_name}': {e}"

        return handler

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self):
        """Disconnect all MCP servers."""
        self.stop_all()
