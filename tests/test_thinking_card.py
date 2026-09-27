"""UI tests for the collapsible thinking card in NativeRichLogWidget."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PyQt6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def widget(qapp):
    from core.native_chat import NativeRichLogWidget
    w = NativeRichLogWidget()
    yield w
    w.deleteLater()


class TestThinkingCard:
    def test_start_creates_card(self, widget):
        widget.thinking_start()
        assert widget._thinking_active is True
        assert widget._thinking_text == ""
        widget.thinking_abort()

    def test_append_accumulates(self, widget):
        widget.thinking_start()
        widget.thinking_append("step one ")
        widget.thinking_append("step two")
        assert widget._thinking_text == "step one step two"
        widget.thinking_abort()

    def test_append_auto_starts(self, widget):
        widget.thinking_append("auto")
        assert widget._thinking_active is True
        widget.thinking_abort()

    def test_end_collapses_and_keeps_toggle_text(self, widget):
        widget.thinking_start()
        widget.thinking_append("reasoning here")
        widget.thinking_end(None)
        assert widget._thinking_active is False
        # toggle text survives so the chip can expand it later
        assert widget._thinking_text == "reasoning here"
        widget.thinking_abort()

    def test_abort_clears_everything(self, widget):
        widget.thinking_start()
        widget.thinking_append("x")
        widget.thinking_abort()
        assert widget._thinking_active is False
        assert widget._thinking_text == ""

    def test_toggle_flips_state(self, widget):
        widget.thinking_start()
        assert widget._thinking_expanded is False
        widget.thinking_toggle()
        assert widget._thinking_expanded is True
        widget.thinking_toggle()
        assert widget._thinking_expanded is False
        widget.thinking_abort()

    def test_threadsafe_signals_exist(self, widget):
        for sig in ("thinking_start_threadsafe", "thinking_append_threadsafe",
                    "thinking_end_threadsafe", "thinking_abort_threadsafe"):
            assert callable(getattr(widget, sig))

    def test_card_html_contains_marker_and_toggle(self, widget):
        from core.native_chat import _THINK_MARK
        widget.thinking_start()
        widget.thinking_append("some reasoning")
        html = widget._thinking_card_html()
        assert "thinktoggle://" in html
        assert _THINK_MARK in html
        assert "Thinking" in html
        widget.thinking_abort()

    def test_expanded_html_shows_body(self, widget):
        widget.thinking_start()
        widget.thinking_append("visible reasoning body")
        widget.thinking_toggle()
        html = widget._thinking_card_html()
        assert "visible reasoning body" in html
        widget.thinking_abort()

    def test_end_keeps_final_chip_in_document(self, widget):
        widget.thinking_start()
        widget.thinking_append("kept reasoning")
        widget.thinking_end(None)
        doc_text = widget.toPlainText()
        assert "Thought" in doc_text or "Thinking" in doc_text
        widget.thinking_abort()
