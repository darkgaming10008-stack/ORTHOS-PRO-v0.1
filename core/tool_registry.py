"""
Dynamic tool registry for Orthos.

Provides:
  - @tool(name, description, parameters) decorator
  - ToolRegistry class with register, get, list_tools, execute, get_declarations
  - discover() auto-scans actions/ directory for registered tools
"""

import importlib
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Callable


_TYPE_MAP = {
    "OBJECT": "object",
    "STRING": "string",
    "ARRAY": "array",
    "INTEGER": "integer",
    "BOOLEAN": "boolean",
    "NUMBER": "number",
}


def _convert_type(t: str) -> str:
    return _TYPE_MAP.get(t, t.lower()) if isinstance(t, str) else t


def _convert_props(props: dict) -> dict:
    out = {}
    for k, v in props.items():
        nv = dict(v)
        if "type" in nv:
            nv["type"] = _convert_type(nv["type"])
        if "items" in nv and isinstance(nv["items"], dict):
            nv["items"] = {"type": _convert_type(nv["items"].get("type", "string"))}
        out[k] = nv
    return out


def _to_ollama_declaration(name: str, description: str, parameters: dict) -> dict:
    """Convert a tool declaration to the OLLAMA_TOOLS format used by main.py."""
    params = parameters or {}
    new_params: dict = {
        "type": "object",
        "properties": _convert_props(params.get("properties", {})),
    }
    req = params.get("required")
    if req:
        new_params["required"] = req
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": new_params,
        },
    }


# ── Global registry singleton ────────────────────────────────────────────────

_TOOLS: dict[str, dict] = {}       # name → {"fn": Callable, "name": str, "description": str, "parameters": dict}


def tool(name: str, description: str = "", parameters: dict | None = None):
    """Decorator that registers a tool function.

    Usage:
        @tool("my_tool", description="Does something", parameters={...})
        def my_tool(parameters=None, response=None, player=None):
            ...
    """
    parameters = parameters or {}

    def decorator(func: Callable) -> Callable:
        _TOOLS[name] = {
            "fn": func,
            "name": name,
            "description": description,
            "parameters": parameters,
        }
        return func

    return decorator


class ToolRegistry:
    """Dynamic tool registry that mirrors the OLLAMA_TOOLS pattern from main.py."""

    def __init__(self):
        self._tools: dict[str, dict] = {}

    def register(
        self,
        func: Callable,
        name: str,
        description: str = "",
        parameters: dict | None = None,
    ) -> None:
        """Register a tool function manually."""
        parameters = parameters or {}
        self._tools[name] = {
            "fn": func,
            "name": name,
            "description": description,
            "parameters": parameters,
        }

    def unregister(self, name: str) -> bool:
        """Remove a dynamically registered tool (no-op for built-in tools)."""
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get(self, name: str) -> Callable | None:
        """Return the tool function by name, or None."""
        entry = self._tools.get(name) or _TOOLS.get(name)
        if entry:
            return entry["fn"]
        return None

    def list_tools(self) -> list[dict]:
        """Return all registered tool declarations (raw format)."""
        tools = list(self._tools.values())
        for entry in _TOOLS.values():
            if entry["name"] not in {t["name"] for t in tools}:
                tools.append(entry)
        return tools

    def execute(self, name: str, args: dict) -> Any:
        """Execute a tool by name with the given args dict.

        Tools are called with the same signature used in main.py:
            func(parameters=args, ...)
        """
        entry = self._tools.get(name) or _TOOLS.get(name)
        if not entry:
            raise KeyError(f"Tool '{name}' not found")
        return entry["fn"](parameters=args, response=None, player=None)

    def register_legacy(self, declarations: list[dict]) -> None:
        """Register tools from a legacy TOOL_DECLARATIONS list.

        Creates generic wrappers that dispatch to action modules at call time.
        Falls back to no-op wrappers for tools handled inline in main.py.
        """
        from core.tool_declarations import _make_legacy_handler

        for decl in declarations:
            name = decl["name"]
            handler = _make_legacy_handler(name)
            self.register(
                func=handler,
                name=name,
                description=decl["description"],
                parameters=decl.get("parameters", {}),
            )

    def get_declarations(self) -> list[dict]:
        """Return all tool declarations in OLLAMA_TOOLS format.

        This exactly matches the output of _to_ollama_tools() in main.py,
        which is the format expected by call_llm_stream() in core/llm_client.py:
            {
                "type": "function",
                "function": {
                    "name": ...,
                    "description": ...,
                    "parameters": {"type": "object", "properties": ..., "required": ...}
                }
            }
        """
        decls = self.list_tools()
        result = []
        for d in decls:
            result.append(
                _to_ollama_declaration(
                    name=d["name"],
                    description=d["description"],
                    parameters=d["parameters"],
                )
            )
        return result

    def register_tool(self, name: str, fn: Callable, description: str = "", parameters: dict | None = None) -> None:
        """Register a tool function directly (alias for register)."""
        self.register(fn, name, description, parameters)


def discover(actions_path: str | Path | None = None) -> ToolRegistry:
    """Scan the actions/ directory and discover all registered tools.

    Imports every .py file in the actions directory. Any function decorated
    with @tool or registered via ToolRegistry is collected into a new
    ToolRegistry instance.

    Args:
        actions_path: Path to the actions directory. Defaults to
                      ``<project_root>/actions/``.

    Returns:
        A ToolRegistry populated with all discovered tools.
    """
    if actions_path is None:
        base = Path(__file__).resolve().parent.parent
        actions_path = base / "actions"
    else:
        actions_path = Path(actions_path).resolve()

    if not actions_path.is_dir():
        raise NotADirectoryError(f"actions path not found: {actions_path}")

    # Ensure the actions directory is on sys.path so we can import modules
    actions_str = str(actions_path)
    if actions_str not in sys.path:
        sys.path.insert(0, actions_str)

    # Import every .py file (except __pycache__ and underscores)
    for entry in sorted(os.listdir(actions_str)):
        if entry.startswith("_") or entry.startswith(".") or not entry.endswith(".py"):
            continue
        mod_name = entry[:-3]
        try:
            importlib.import_module(mod_name)
        except Exception as e:
            print(f"[ToolRegistry] Failed to import {mod_name}: {e}")

    # Build a registry from everything collected in the global _TOOLS dict
    registry = ToolRegistry()
    for name, entry in _TOOLS.items():
        registry.register(
            func=entry["fn"],
            name=entry["name"],
            description=entry["description"],
            parameters=entry["parameters"],
        )
    return registry
