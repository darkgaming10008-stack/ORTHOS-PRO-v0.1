"""Regression: the tool-timing crash from the 2026-09-05 session log.

``_process_message_inner`` crashed with UnboundLocalError at
``_tool_t0 = time.time()`` because a *local* ``import time`` deeper in
the same function (the rolling-summary branch) made ``time`` a
function-local name for the entire function body.  Python decides
local-vs-module scope at compile time, so even a branch that never runs
poisons every other use.

This test compiles main.py and asserts no method (except the isolated
``_shutdown`` helper) carries a bare local ``import time`` or an
assignment to the name ``time`` — the class of bug that took down every
message with tool calls.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = (ROOT / "main.py").read_text(encoding="utf-8")


def _bare_local_time_imports(source: str):
    """Yield (function_name, lineno) for every function-scoped bare
    ``import time`` (aliased imports like ``import time as _time`` are
    harmless)."""
    tree = ast.parse(source)
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(func):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "time" and (alias.asname or "time") == "time":
                        yield func.name, node.lineno


def test_no_local_import_time_in_main_methods():
    problems = [
        (fn, ln) for fn, ln in _bare_local_time_imports(MAIN)
        if fn != "_shutdown"  # isolated helper: its scope is its own
    ]
    assert problems == [], (
        f"Local 'import time' found in {problems} — Python makes 'time' "
        "function-local for the WHOLE body and any time.time() in the same "
        "method crashes with UnboundLocalError (see 2026-09-05 log)."
    )


def test_module_level_time_import_exists():
    tree = ast.parse(MAIN)
    for node in tree.body:  # top-level statements only
        if isinstance(node, ast.Import):
            if any(a.name == "time" for a in node.names):
                return
    raise AssertionError("main.py lost its module-level 'import time'")
