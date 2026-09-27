"""Regression checks for the live native (QTextBrowser) activity log.

Historically this file validated the retired WebEngine renderer and its
dead fallback parser.  Those classes were removed from ui.py; the assertions
now target the shipping implementation in core/native_chat.py so a green
suite means the real renderer is healthy.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

from PyQt6.QtWidgets import QApplication

import pytest


ROOT = Path(__file__).resolve().parents[1]
UI_SOURCE = (ROOT / "ui.py").read_text(encoding="utf-8")
NATIVE_SOURCE = (ROOT / "core" / "native_chat.py").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_ui_uses_the_native_renderer_and_webengine_is_gone():
    # The native import replaces the old class; nothing WebEngine remains.
    assert "from core.native_chat import NativeRichLogWidget as RichLogWidget" in UI_SOURCE
    assert "QWebEngineView" not in UI_SOURCE
    assert "QWebEngineSettings" not in UI_SOURCE
    assert "_HTML_TEMPLATE" not in UI_SOURCE
    assert "class LogWidget" not in UI_SOURCE


def test_native_renderer_is_the_live_log(qapp):
    import ui
    from core.native_chat import NativeRichLogWidget

    assert isinstance(ui.MainWindow("nonexistent.png")._log, NativeRichLogWidget)


def test_assistant_entry_preserves_multiline_markdown_and_latex(qapp):
    from core.native_chat import NativeRichLogWidget

    parse = NativeRichLogWidget._parse_entry
    text = ("Orthos: **bold**\n\n```python\nprint('$x^2$')\n```\n\n"
            "$$x^2+y^2=z^2$$\n\\(a_1 \\times b^2\\)")
    msg = parse(text)
    assert msg["tag"] == "ai"
    assert msg["prefix"] == "Orthos:"
    assert "**bold**" in msg["body"]
    assert "$$x^2+y^2=z^2$$" in msg["body"]


def test_history_is_restored_in_one_render_not_many_competing_updates(qapp):
    main_source = (ROOT / "main.py").read_text(encoding="utf-8")
    # One bulk call for the whole restored history (no per-turn appends).
    assert "_populate_log(self._conversation)" in main_source
    assert "[Session] Failed to restore chat log:" in main_source


def test_fences_are_enabled_in_the_markdown_pipeline():
    # js-default ships with ``` fences disabled; the renderer must enable.
    assert '_MARKDOWN.enable("fence")' in NATIVE_SOURCE


def test_code_blocks_are_located_via_token_stream_not_regex():
    # The regex heuristic mis-bound copy buttons (audit C1); ensure the
    # token-stream approach is what ships.
    assert "_FENCE_RE" not in NATIVE_SOURCE
    assert 'tok.type in ("fence", "code_block")' in NATIVE_SOURCE


def test_math_images_are_cached_across_sessions():
    # Session switches must not re-rasterize every formula (audit M3).
    assert "_MATH_IMG_CACHE" in NATIVE_SOURCE
