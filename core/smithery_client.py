"""Smithery Connect client — 2026 REST API integration.

Smithery hosts MCP servers and brokers connections through its Connect API:
  - GET  /namespaces                         -> user's namespace
  - POST /namespaces                         -> create namespace (generated name)
  - PUT  /connect/{namespace}/{connectionId} -> create/update an MCP connection
  - GET  /connect/{namespace}/{connectionId}/.tools
  - POST /connect/{namespace}/{connectionId}/.tools/{toolPath}
  - DELETE /connect/{namespace}/{connectionId}

All endpoints authenticate with the Smithery API key as `Authorization: Bearer`.
"""
from __future__ import annotations

import asyncio
import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

SMITHERY_API_BASE = "https://api.smithery.ai"
_API_TIMEOUT = 30

_headers_lock = threading.Lock()
_namespace_cache: dict[str, str] = {}


class SmitheryError(Exception):
    """Base error for the Smithery REST API."""


class SmitheryAuthRequiredError(SmitheryError):
    """The upstream server needs OAuth authorization (hosted setupUrl)."""

    def __init__(self, setup_url: str, message: str = "OAuth authorization required"):
        super().__init__(message)
        self.setup_url = setup_url
        self.kind = "auth_required"


class SmitheryInputRequiredError(SmitheryError):
    """The server needs configuration values (API keys etc.) before use."""

    def __init__(self, fields: dict, missing: dict, setup_url: str,
                 message: str = "Server requires configuration"):
        super().__init__(message)
        self.fields = fields or {}
        self.missing = missing or {}
        self.setup_url = setup_url
        self.kind = "input_required"


def _http_json(method: str, path: str, api_key: str, body: Any | None = None,
               timeout: int = _API_TIMEOUT) -> Any:
    """JSON request against the Smithery API; raises SmitheryError on failure."""
    url = f"{SMITHERY_API_BASE}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (alex-mcp-manager)",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            body = json.loads(e.read().decode("utf-8", errors="replace"))
            detail = f": {body.get('message', body.get('error', ''))}"
        except Exception:
            pass
        raise SmitheryError(f"Smithery API {e.code} on {method} {path}{detail}") from None
    except Exception as e:
        raise SmitheryError(f"Smithery API request failed ({method} {path}): {e}") from None


def _sanitize_connection_id(name: str) -> str:
    """Turn a qualified name into a stable Smithery connection ID."""
    base = name.split("/")[-1] if "/" in name else name
    for marker in ("server-", "-mcp", "mcp-", "-server", "mcp_"):
        base = base.replace(marker, "")
    base = base.replace("@", "").replace("_", "-")
    parts = [p for p in base.split("-") if p]
    return "-".join(parts) if parts else "server"


class SmitheryClient:
    """Connects to a Smithery-hosted MCP server via the Connect REST API.

    Presents the same async surface as MCPClient (connect / list_tools_live /
    call_tool / disconnect) so the manager can treat it identically.
    """

    def __init__(
        self,
        name: str,
        server_qname: str | None = None,
        headers: dict[str, str] | None = None,
        api_key: str | None = None,
        namespace: str | None = None,
        connection_id: str | None = None,
    ):
        self.name = name
        self.server_qname = server_qname or name
        self._headers = headers or {}
        self._api_key = (api_key or "").strip()
        self._namespace = (namespace or "").strip()
        self._connection_id = connection_id or _sanitize_connection_id(self.server_qname)

    # ── API plumbing ──────────────────────────────────────────────────

    def _require_key(self) -> str:
        if not self._api_key:
            raise SmitheryError("Smithery API key is not set — add it in MCP Manager (Browse tab)")
        return self._api_key

    def _ensure_namespace(self) -> str:
        """Reuse the user's namespace, creating one if needed. Cached per key."""
        key = self._require_key()
        global _namespace_cache
        with _headers_lock:
            cached = _namespace_cache.get(key)
        if cached:
            return cached
        data = _http_json("GET", "/namespaces", key)
        namespaces = (data or {}).get("namespaces") or []
        if namespaces:
            name = str(namespaces[0].get("name", ""))
        else:
            created = _http_json("POST", "/namespaces", key)
            name = str((created or {}).get("name", ""))
        if not name:
            raise SmitheryError("Could not resolve a Smithery namespace for this API key")
        with _headers_lock:
            _namespace_cache[key] = name
        return name

    def _put_connection(self, namespace: str) -> dict:
        body: dict[str, Any] = {"server": self.server_qname}
        if self._headers:
            body["headers"] = self._headers
        return _http_json(
            "PUT", f"/connect/{urllib.parse.quote(namespace)}/{urllib.parse.quote(self._connection_id)}",
            self._require_key(), body,
        )

    @staticmethod
    def _config_schema_fields(config_schema: dict | None) -> tuple[dict, dict]:
        """Normalize a server's configSchema into (fields, missing) like the
        connection input_required payload."""
        fields: dict[str, dict] = {}
        missing: dict[str, list[str]] = {"headers": [], "query": []}
        if not isinstance(config_schema, dict):
            return fields, missing
        required = set(config_schema.get("required") or [])
        for name, prop in (config_schema.get("properties") or {}).items():
            if not isinstance(prop, dict):
                continue
            xfrom = prop.get("x-from") or {}
            dest = "headers" if xfrom.get("header") else "query"
            fields[name] = {
                "label": str(prop.get("title") or name),
                "description": str(prop.get("description") or ""),
                "required": name in required,
            }
            if name in required:
                missing[dest].append(name)
        return fields, missing

    def _detail_config_schema(self) -> dict | None:
        """Fetch the server's declared config schema (headers/query it needs)."""
        try:
            detail = _http_json(
                "GET",
                f"/servers/{urllib.parse.quote(self.server_qname, safe='')}",
                self._require_key(),
            )
            conns = (detail or {}).get("connections") or []
            for c in conns:
                schema = c.get("configSchema")
                if isinstance(schema, dict):
                    return schema
        except Exception:
            pass
        return None

    @staticmethod
    def _map_tools(server_name: str, tools: list[dict]) -> list[dict]:
        out = []
        for t in tools or []:
            tname = t.get("name", "")
            if not tname:
                continue
            params = t.get("inputSchema") or {"type": "object", "properties": {}}
            read_only = bool(((t.get("annotations") or {}).get("readOnlyHint")))
            out.append({
                "name": f"mcp_{server_name}_{tname}",
                "description": t.get("description") or f"MCP tool from {server_name}: {tname}",
                "parameters": params,
                "mcp_server": server_name,
                "mcp_tool_name": tname,
                "read_only": read_only,
            })
        return out

    # ── async interface (called via the async worker) ────────────────

    async def connect(self) -> list[dict]:
        """Create/refresh the connection and return tool declarations.

        Raises SmitheryAuthRequiredError / SmitheryInputRequiredError so the
        caller can route the user to the hosted setup flow.
        """
        namespace = await asyncio.to_thread(self._ensure_namespace)
        conn = await asyncio.to_thread(self._put_connection, namespace)
        status = conn.get("status") or {}
        state = status.get("state")
        if state == "connected":
            tools = await asyncio.to_thread(
                _http_json, "GET",
                f"/connect/{urllib.parse.quote(namespace)}/{urllib.parse.quote(self._connection_id)}/.tools",
                self._require_key(),
            )
            return self._map_tools(self.name, (tools or {}).get("tools") or [])
        if state == "auth_required":
            raise SmitheryAuthRequiredError(status.get("setupUrl") or "")
        if state == "input_required":
            raise SmitheryInputRequiredError(
                (status.get("http") or {}).get("headers") or {},
                status.get("missing") or {},
                status.get("setupUrl") or "",
            )
        if state == "error":
            fields, missing = self._config_schema_fields(self._detail_config_schema())
            if missing.get("headers"):
                raise SmitheryInputRequiredError(fields, missing, "")
        raise SmitheryError(status.get("message") or f"Smithery connection in unexpected state: {state}")

    async def list_tools_live(self) -> list[dict]:
        """Re-list tools (health probe / refresh)."""
        namespace = await asyncio.to_thread(self._ensure_namespace)
        tools = await asyncio.to_thread(
            _http_json, "GET",
            f"/connect/{urllib.parse.quote(namespace)}/{urllib.parse.quote(self._connection_id)}/.tools",
            self._require_key(),
        )
        return self._map_tools(self.name, (tools or {}).get("tools") or [])

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Invoke a tool via the Connect API and return the result as a string."""
        try:
            namespace = await asyncio.to_thread(self._ensure_namespace)
            result = await asyncio.to_thread(
                _http_json, "POST",
                f"/connect/{urllib.parse.quote(namespace)}/{urllib.parse.quote(self._connection_id)}"
                f"/.tools/{urllib.parse.quote(tool_name, safe='')}",
                self._require_key(), arguments or {},
            )
            if result is None:
                return "OK (no content)"
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return f"MCP tool '{tool_name}' error: {e}"

    async def disconnect(self) -> None:
        """Stateless REST client — nothing to tear down."""

    async def remove(self) -> None:
        """Delete the connection (used on uninstall). Ignores already-gone."""
        try:
            namespace = await asyncio.to_thread(self._ensure_namespace)
            await asyncio.to_thread(
                _http_json, "DELETE",
                f"/connect/{urllib.parse.quote(namespace)}/{urllib.parse.quote(self._connection_id)}",
                self._require_key(),
            )
        except SmitheryError:
            pass