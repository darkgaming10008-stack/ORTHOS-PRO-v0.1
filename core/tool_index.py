"""
Tool index with semantic search for dynamic tool discovery.

Leverages existing sentence-transformers (all-MiniLM-L6-v2) — same model
used by ChromaDB — to embed tool descriptions for semantic retrieval.
Falls back to BM25-style keyword matching if the embedding model is
unavailable.

Usage:
    from core.tool_index import ToolIndex

    idx = ToolIndex(tool_registry)
    results = idx.search_tools("send a message", k=3)
    full = idx.get_tool_definition("web_search")
"""

import json
import re
import threading
from typing import Any

_embedder = None
_embedder_lock = threading.Lock()


def _get_embedder():
    global _embedder
    if _embedder is not None:
        return _embedder
    with _embedder_lock:
        if _embedder is not None:
            return _embedder
        try:
            from memory.vector_index import _get_model as _vi_get_model
            _embedder = _vi_get_model()
            return _embedder
        except Exception:
            pass
        try:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        except Exception:
            _embedder = False
    return _embedder


def _embed(texts: list[str]) -> list[list[float]] | None:
    try:
        from memory.memory_client import _is_healthy, memory_embed
        if _is_healthy():
            arr = memory_embed(texts)
            return arr.tolist()
    except Exception:
        pass
    model = _get_embedder()
    if not model:
        return None
    try:
        emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return emb.tolist()
    except Exception:
        return None


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def resolve_refs(schema: dict) -> dict:
    """Inline $ref pointers in JSON Schema for LLM compatibility.
    
    MCP servers often emit $ref/$defs from Pydantic model_json_schema().
    LLM clients cannot resolve these, so we inline definitions.
    """
    schema = dict(schema)
    defs = schema.pop("$defs", None) or schema.pop("definitions", None)
    if not defs:
        return schema

    def _resolve(obj):
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref = obj["$ref"]
                if ref.startswith("#/$defs/") or ref.startswith("#/definitions/"):
                    name = ref.rsplit("/", 1)[-1]
                    if name in defs:
                        resolved = dict(defs[name])
                        for k, v in obj.items():
                            if k != "$ref":
                                resolved[k] = v
                        return _resolve(resolved)
            return {k: _resolve(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_resolve(item) for item in obj]
        return obj

    return _resolve(schema)


TOOL_SUMMARIES: dict[str, str] = {
    "open_app": "Launch any application, website, or program on the computer",
    "web_search": "Search the web for information, research, or current events",
    "webfetch": "Fetch and read content from a specific URL",
    "weather_report": "Get weather report for a city",
    "send_message": "Send a message via WhatsApp, Telegram, or similar",
    "reminder": "Set a timed reminder using Task Scheduler",
    "youtube_video": "Control YouTube: play, summarize, get info, trending",
    "screen_process": "Capture screen/camera and analyze with vision",
    "screen_locate": "Find any visual element on screen by description",
    "computer_settings": "Control system settings: volume, brightness, WiFi, shutdown",
    "file_controller": "Manage files and folders: create, delete, move, copy, read, write",
    "desktop_control": "Control desktop: wallpaper, organize, clean, stats",
    "code_helper": "Write, edit, explain, run, or build code files",
    "dev_agent": "Build complete multi-file projects from scratch",
    "agent_task": "Execute complex multi-step tasks requiring multiple tools",
    "computer_control": "Direct mouse/keyboard: type, click, scroll, hotkeys, screenshot",
    "game_updater": "Manage Steam/Epic games: install, update, list installed",
    "flight_finder": "Search Google Flights for flight options and prices",
    "run_terminal": "Execute shell/terminal commands and return output",
    "list_mcp_servers": "List configured MCP servers with status and tool counts",
    "shutdown_orthos": "Shut down the assistant completely",
    "file_processor": "Process uploaded files: images, PDFs, docs, CSV, archives",
    "save_memory": "Save any information to permanent long-term memory",
    "forget_memory": "Delete specific information from long-term memory",
    "search_memory": "Search all memory systems for context and past facts",
    "recall_conversation": "Search raw past chat messages verbatim (exact quotes with dates)",
    "list_memories": "Show complete summary of everything known about the user",
    "save_procedure": "Store a learned workflow, rule, or how-to knowledge",
    "list_procedures": "Show all learned procedures and behavioral rules",
    "core_memory_append": "Append a new permanent fact to core identity memory",
    "core_memory_replace": "Replace an existing fact in core identity memory",
    "archival_memory_search": "Search archived evicted memories",
    "context_status": "Show memory usage, token estimates, and index health",
    "search_timeline": "Search project timeline for historical events and milestones",
    "save_timeline_event": "Log an important event to the project timeline",
    "list_projects": "List all known projects with their IDs",
    "create_project": "Create a new project for organizing sessions and memories",
    "set_active_project": "Switch current session to a different project",
    "get_project_memories": "Get all memories and context for a specific project",
    "save_tool_state": "Save tool state for continuity across calls",
    "get_tool_state": "Get remembered state for a tool",
    "search_knowledge_graph": "Search entity relationships in the knowledge graph",
    "read": "Read file contents with optional line range support for large files",
    "edit": "Modify files using precise string replacement - oldString to newString",
    "grep": "Search file contents using regex patterns with file pattern filtering",
    "glob": "Find files by glob patterns like **/*.py with recursive search support",
}


META_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_tools",
            "description": (
                "Search for available tools by capability. "
                "Returns matching tool names, descriptions, and full parameter schemas. "
                "Use this when you need to find a tool for a specific task."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What capability you need, e.g. 'search web', 'send message', 'file operations'"
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of results to return (default 5, max 10)"
                    }
                },
                "required": ["query"]
            }
        }
    },
        {
            "type": "function",
            "function": {
                "name": "get_tool_definition",
                "description": (
                    "Load the full parameter schema for a tool or ALL tools from an MCP server. "
                    "Call this with an individual tool name (from search_tools) to load its full schema, "
                    "or pass an MCP server name (from search_tools results) to load all tools from that server at once. "
                    "After loading, the tool(s) will be available with full schemas."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "description": "Tool name (e.g. 'web_search', 'mcp_playwright_navigate') OR MCP server name (e.g. 'playwright', 'filesystem') to load all its tools"
                        }
                    },
                    "required": ["tool_name"]
                }
            }
        },
]


class ToolIndex:
    def __init__(self, tool_registry=None):
        self._registry = tool_registry
        self._tool_names: list[str] = list(TOOL_SUMMARIES.keys())
        self._embeddings: list[list[float]] | None = None
        self._embed_lock = threading.Lock()
        self._mcp_tool_names: list[str] = []
        self._mcp_summaries: dict[str, str] = {}
        self._mcp_embeddings: list[list[float]] | None = None
        self._mcp_server_tools: dict[str, list[str]] = {}

    def _build_embeddings(self):
        built_internal = self._embeddings is not None
        built_mcp = not self._mcp_tool_names or self._mcp_embeddings is not None
        if built_internal and built_mcp:
            return True
        with self._embed_lock:
            if self._embeddings is None:
                texts = [f"{name}: {TOOL_SUMMARIES[name]}" for name in self._tool_names]
                result = _embed(texts)
                if result is not None:
                    self._embeddings = result
            if self._mcp_tool_names and self._mcp_embeddings is None:
                texts = [f"{name}: {self._mcp_summaries.get(name, name)}" for name in self._mcp_tool_names]
                result = _embed(texts)
                if result is not None:
                    self._mcp_embeddings = result
        return self._embeddings is not None

    def search_tools(self, query: str, k: int = 5) -> list[dict]:
        k = max(1, min(k, 10))
        all_scored: list[tuple[str, float]] = []
        used_semantic = False

        self._build_embeddings()
        q_emb = _embed([query]) if self._embeddings else None

        if q_emb:
            used_semantic = True
            if self._embeddings:
                scores = [
                    sum(a * b for a, b in zip(q_emb[0], t_emb))
                    for t_emb in self._embeddings
                ]
                for name, score in zip(self._tool_names, scores):
                    all_scored.append((name, score))
            if self._mcp_embeddings:
                scores = [
                    sum(a * b for a, b in zip(q_emb[0], t_emb))
                    for t_emb in self._mcp_embeddings
                ]
                for name, score in zip(self._mcp_tool_names, scores):
                    all_scored.append((name, score))

        if not used_semantic:
            q_tokens = _tokenize(query)
            for name in self._tool_names:
                text = f"{name} {TOOL_SUMMARIES.get(name, '')}".lower()
                mc = sum(1 for t in q_tokens if t in text)
                if mc:
                    all_scored.append((name, mc))
            for name in self._mcp_tool_names:
                text = f"{name} {self._mcp_summaries.get(name, '')}".lower()
                mc = sum(1 for t in q_tokens if t in text)
                if mc:
                    all_scored.append((name, mc))

        if not all_scored:
            return []

        all_scored.sort(key=lambda x: -x[1])
        results = []
        for name, _ in all_scored[:k]:
            summary = TOOL_SUMMARIES.get(name) or self._mcp_summaries.get(name) or ""
            results.append({
                "name": name,
                "summary": summary,
                "definition": self.get_tool_definition(name),
            })
        return results

    def get_tool_definition(self, name: str) -> dict | None:
        if not self._registry:
            return None
        tools = self._registry.list_tools()
        for t in tools:
            if t["name"] == name:
                from core.tool_registry import _to_ollama_declaration
                params = t.get("parameters", {})
                if params:
                    params = resolve_refs(dict(params))
                return _to_ollama_declaration(
                    name=t["name"],
                    description=t["description"],
                    parameters=params,
                )
        return None

    def get_summary_declarations(self) -> list[dict]:
        from core.tool_registry import _to_ollama_declaration
        decls = []
        for name in self._tool_names:
            decls.append(
                _to_ollama_declaration(
                    name=name,
                    description=TOOL_SUMMARIES.get(name, name),
                    parameters={},
                )
            )
        return decls

    def get_summary_token_estimate(self) -> int:
        total = 0
        for name, summary in TOOL_SUMMARIES.items():
            total += len(name) + len(summary) + 50
        for meta in META_TOOL_DEFINITIONS:
            total += len(json.dumps(meta, ensure_ascii=False))
        return total // 4


def make_search_tools_handler(tool_index: ToolIndex):
    def handler(parameters=None, response=None, player=None, **kwargs):
        args = parameters or {}
        query = args.get("query", "")
        k = args.get("k", 5)
        if isinstance(k, str):
            try:
                k = int(k)
            except (ValueError, TypeError):
                k = 5
        if not query:
            return "Error: query is required"
        results = tool_index.search_tools(query, k)
        if not results:
            hint = ""
            servers = sorted(tool_index._mcp_server_tools.keys())
            if servers:
                hint = (
                    f"\n  Tip: {len(tool_index._mcp_tool_names)} MCP tools are available across "
                    f"{len(servers)} server(s): {', '.join(servers)}. "
                    f"Pass a server name to get_tool_definition to load all its tools at once. "
                    f"Try broader terms like 'browser', 'file', 'database'."
                )
            return f"No tools found for: {query}{hint}"
        lines = [f"[Tool Search Results for: {query}]"]
        for r in results:
            name = r["name"]
            summary = r["summary"]
            lines.append(f"\n  {name}: {summary}")
            if r.get("definition"):
                fn = r["definition"].get("function", {})
                params = fn.get("parameters", {})
                props = params.get("properties", {})
                required = params.get("required", [])
                if props:
                    param_lines = []
                    for pname, pinfo in props.items():
                        ptype = pinfo.get("type", "string")
                        req_mark = " (required)" if required and pname in required else ""
                        param_lines.append(f"{pname}: {ptype}{req_mark}")
                    lines.append(f"    Parameters: {', '.join(param_lines)}")
        lines.append(
            "\n\nTo load a tool's full schema, call get_tool_definition(tool_name='<tool_name>'). "
            "To load ALL tools from an MCP server, pass the server name: get_tool_definition(tool_name='<server_name>')."
        )
        return "\n".join(lines)
    handler.__name__ = "search_tools"
    return handler


def make_get_tool_definition_handler(tool_index: ToolIndex, loaded_set: set):
    def handler(parameters=None, response=None, player=None, **kwargs):
        args = parameters or {}
        name = args.get("tool_name", "")
        if not name:
            return "Error: tool_name is required"

        # Case 1: Individual tool name
        definition = tool_index.get_tool_definition(name)
        if definition:
            loaded_set.add(name)
            fn = definition.get("function", {})
            tool_type = "MCP" if name.startswith("mcp_") else "Internal"
            return (
                f"Tool '{name}' loaded. Full schema available from next turn.\n"
                f"Type: {tool_type}\n"
                f"Description: {fn.get('description', '')}"
            )

        # Case 2: MCP server name — load all tools from that server
        server_tools = tool_index._mcp_server_tools.get(name)
        if server_tools:
            loaded = []
            for tool_name in server_tools:
                loaded_set.add(tool_name)
                loaded.append(tool_name)
            return (
                f"Server '{name}' loaded — {len(loaded)} tool(s) added.\n"
                f"Tools: {', '.join(sorted(loaded))}\n"
                f"Full schemas available from next turn."
            )

        # Not found — show available tools AND servers
        av = sorted(tool_index._tool_names + tool_index._mcp_tool_names)
        servers = sorted(tool_index._mcp_server_tools.keys())
        hint = ""
        if servers:
            hint = (
                f"\nAvailable MCP servers: {', '.join(servers)}. "
                f"Pass a server name to load all its tools at once."
            )
        return f"Tool or server '{name}' not found. Available tools: {', '.join(av)}.{hint}"
    handler.__name__ = "get_tool_definition"
    return handler
