from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.auth import OAuthClientProvider
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
from mcp.client.auth.oauth2 import TokenStorage

from core.tool_index import resolve_refs


class FileTokenStorage(TokenStorage):
    """Persists OAuth tokens + client info to a JSON file on disk."""

    def __init__(self, path: Path):
        self._path = Path(path)
        self._tokens: OAuthToken | None = None
        self._client_info: OAuthClientInformationFull | None = None
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                if data.get("tokens"):
                    self._tokens = OAuthToken.model_validate(data["tokens"])
                if data.get("client_info"):
                    self._client_info = OAuthClientInformationFull.model_validate(data["client_info"])
            except Exception as e:
                print(f"[MCP] Could not load OAuth tokens for {self._path.name}: {e}")

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "tokens": self._tokens.model_dump(mode="json") if self._tokens else None,
            "client_info": self._client_info.model_dump(mode="json") if self._client_info else None,
        }
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    async def get_tokens(self) -> OAuthToken | None:
        return self._tokens

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self._tokens = tokens
        self._save()

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        return self._client_info

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self._client_info = client_info
        self._save()


class OAuthCallbackServer:
    """Local HTTP server that captures the OAuth redirect callback."""

    DEFAULT_PORT = 8765

    def __init__(self, port: int = DEFAULT_PORT):
        self._port = port
        self._server: asyncio.AbstractServer | None = None
        self._result: tuple[str, str | None] | None = None
        self._event = asyncio.Event()

    @property
    def redirect_uri(self) -> str:
        return f"http://127.0.0.1:{self._port}/callback"

    async def start(self):
        self._event = asyncio.Event()
        self._result = None
        try:
            self._server = await asyncio.start_server(self._handle, "127.0.0.1", self._port)
        except OSError:
            self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self._port = self._server.sockets[0].getsockname()[1]
        return self

    async def _handle(self, reader, writer):
        try:
            request_line = (await reader.readline()).decode("latin-1").strip()
            parts = request_line.split(" ")
            path = parts[1] if len(parts) > 1 else "/"
            while True:
                line = (await reader.readline()).decode("latin-1").strip()
                if not line:
                    break
            from urllib.parse import parse_qs, urlparse

            qs = parse_qs(urlparse(path).query)
            code = qs.get("code", [None])[0]
            state = qs.get("state", [None])[0]
            body = b"<html><body><h1>Orthos connected to Spotify</h1><p>You can close this tab now.</p></body></html>"
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n"
                b"Content-Length: " + str(len(body)).encode() + b"\r\n"
                b"Connection: close\r\n\r\n" + body
            )
            await writer.drain()
            self._result = (code, state)
            self._event.set()
        except Exception:
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def wait_for_callback(self) -> tuple[str, str | None]:
        await asyncio.wait_for(self._event.wait(), timeout=300)
        return self._result or ("", None)

    async def close(self):
        if self._server:
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception:
                pass


def _open_browser(url: str) -> None:
    """Open the auth URL in the debug Chrome (port 9222) if available, else default browser."""
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp("http://127.0.0.1:9222", timeout=4000)
            context = browser.contexts[0] if browser.contexts else browser.new_context()
            page = context.pages[0] if context.pages else context.new_page()
            page.goto(url)
            return
    except Exception:
        pass
    try:
        import webbrowser

        webbrowser.open(url)
    except Exception:
        print(f"[MCP] Open this URL to authorize: {url}")


class MCPClient:
    """Connects to a single MCP server via stdio or streamable-HTTP transport."""

    def __init__(
        self,
        name: str,
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        url: str | None = None,
        headers: dict[str, str] | None = None,
    ):
        self.name = name
        self.url = url
        self._headers = headers or {}
        self._params = StdioServerParameters(command=command or "npx", args=args or [], env=env)
        self._session: ClientSession | None = None
        self._ctx: Any = None
        self._callback_server: OAuthCallbackServer | None = None

    @property
    def is_remote(self) -> bool:
        return bool(self.url)

    def _token_path(self) -> Path:
        base = Path(__file__).resolve().parent.parent / "config" / "oauth_tokens"
        safe = "".join(c if c.isalnum() else "_" for c in self.name)
        return base / f"{safe}.json"

    async def connect(self) -> list[dict]:
        """Connect, initialise, and return tool declarations."""
        if self.is_remote:
            callback = OAuthCallbackServer()
            await callback.start()
            self._callback_server = callback

            storage = FileTokenStorage(self._token_path())
            client_info = await storage.get_client_info()
            if client_info is not None and callback.redirect_uri not in [str(u) for u in client_info.redirect_uris]:
                print(f"[MCP] '{self.name}': registered callback URI changed, re-registering OAuth client")
                await storage.set_client_info(None)

            provider = OAuthClientProvider(
                server_url=self.url,
                client_metadata=OAuthClientMetadata(
                    client_name="alex",
                    redirect_uris=[callback.redirect_uri],
                ),
                storage=storage,
                redirect_handler=lambda url: asyncio.to_thread(_open_browser, url),
                callback_handler=callback.wait_for_callback,
            )
            self._ctx = streamablehttp_client(self.url, headers=self._headers or None, auth=provider)
        else:
            self._ctx = stdio_client(self._params)

        self._streams = await self._ctx.__aenter__()
        read, write, *_ = self._streams

        self._session = ClientSession(read, write)
        await self._session.__aenter__()
        await self._session.initialize()

        tools_result = await self._session.list_tools()
        tools = []
        for tool in tools_result.tools:
            params = tool.inputSchema if hasattr(tool, "inputSchema") else {}
            if params:
                params = resolve_refs(dict(params))
            read_only = None
            annotations = getattr(tool, "annotations", None)
            if annotations is not None:
                read_only = getattr(annotations, "readOnlyHint", None)
            tools.append({
                "name": f"mcp_{self.name}_{tool.name}",
                "description": tool.description or f"MCP tool from {self.name}: {tool.name}",
                "parameters": params,
                "mcp_server": self.name,
                "mcp_tool_name": tool.name,
                "read_only": bool(read_only),
            })
        return tools

    async def list_tools_live(self) -> list[dict]:
        """Re-list tools on the live session (used for health probes / refresh)."""
        if self._session is None:
            raise RuntimeError(f"MCP server '{self.name}' not connected")
        tools_result = await self._session.list_tools()
        names = []
        for tool in tools_result.tools:
            read_only = None
            annotations = getattr(tool, "annotations", None)
            if annotations is not None:
                read_only = getattr(annotations, "readOnlyHint", None)
            names.append({
                "name": f"mcp_{self.name}_{tool.name}",
                "description": tool.description or f"MCP tool from {self.name}: {tool.name}",
                "read_only": bool(read_only),
            })
        return names

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Call a tool and return the result as a string."""
        if self._session is None:
            return f"Error: MCP server '{self.name}' not connected"
        try:
            result = await self._session.call_tool(tool_name, arguments)
            if hasattr(result, "content") and result.content:
                parts = []
                for item in result.content:
                    if hasattr(item, "text"):
                        parts.append(item.text)
                    else:
                        parts.append(str(item))
                return "\n".join(parts)
            return str(result)
        except Exception as e:
            return f"MCP tool '{tool_name}' error: {e}"

    async def disconnect(self) -> None:
        """Disconnect from the server."""
        if self._session is not None:
            await self._session.__aexit__(None, None, None)
            self._session = None
        if self._ctx is not None:
            await self._ctx.__aexit__(None, None, None)
            self._ctx = None
        if self._callback_server is not None:
            await self._callback_server.close()
            self._callback_server = None
