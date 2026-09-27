"""End-to-end checks for the browser-free chat renderer."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QTextDocument
from PyQt6.QtWidgets import QApplication

from core.native_chat import NativeRichLogWidget


def test_native_renderer_keeps_markdown_code_and_math_visible():
    app = QApplication.instance() or QApplication([])
    chat = NativeRichLogWidget()
    chat.set_logs([
        "You: Explain $x^2$",
        "Orthos: # Result\n\n**bold** and *italic*\n\n```python\nprint('$x^2$')\n```\n\n$$x^2+y^2=z^2$$\n\n$a_1 \\times b^2$",
    ])
    plain = chat.toPlainText()
    html = chat.document().toHtml()
    assert "Result" in plain
    assert "bold" in plain
    assert "print('$x^2$')" in plain  # Math inside code stays code.
    assert "<img" in html             # Actual equations are native images.


def test_native_math_images_are_transparent_not_white_rectangles():
    app = QApplication.instance() or QApplication([])
    chat = NativeRichLogWidget()
    chat._math_image(r"2^4 \times 3", False)
    # Resource URLs are sequential: the first formula is /f/1.
    image = chat.document().resource(
        QTextDocument.ResourceType.ImageResource, QUrl("orthos-math:/f/1")
    )
    assert image is not None, "math image was not registered as a resource"
    assert image.pixelColor(0, 0).alpha() == 0
    assert any(
        image.pixelColor(x, y).alpha() > 0
        for y in range(image.height()) for x in range(image.width())
    )


def test_role_cards_keep_user_ai_system_and_tool_visually_distinct():
    app = QApplication.instance() or QApplication([])
    chat = NativeRichLogWidget()
    chat.set_logs([
        "You: Hello", "Orthos: Hi", "SYS: Ready", "SYS: ▶ search_tools",
    ])
    html = chat.document().toHtml()
    assert NativeRichLogWidget._parse_entry("SYS: ▶ search_tools")["tag"] == "tool"
    # 2026 flat design: roles stay distinct via bubble tint, label color,
    # and row accent — verified through the shipped card builders.
    cards = [chat._card_html(m) for m in chat._messages]
    assert "background-color:#0d2f42" in cards[0]          # YOU bubble tint
    assert "#4dd8ff" in cards[1]                           # AI label accent
    assert "#d6a019" in cards[2]                           # SYS accent
    assert "#00a877" in cards[3]                           # TOOL accent


def test_streaming_display_math_flag_reaches_renderer():
    """Bug #2 regression: display-math detection is USE, not dead code."""
    import inspect
    from core import native_chat as nc
    src = inspect.getsource(nc.NativeRichLogWidget._render_math)
    assert "is_display = False" not in src, "display-math flag still hardcoded"
    assert "is_display = display" in src or "is_display = expression.startswith" in src


def test_session_switch_kills_inflight_stream_state():
    """Bug #3 regression: old session stream must not leak into new session."""
    app = QApplication.instance() or QApplication([])
    chat = NativeRichLogWidget()
    chat.streaming_start()
    chat.streaming_append("old session draft text ")
    assert chat._streaming_active
    chat.set_logs_with_timestamps(["You: new session msg"], None)
    assert not chat._streaming_active, "streaming state survived session switch"
    assert not chat._thinking_active
    assert chat._streaming_text == ""
    # Orphan tokens from the old session's queued signals are dropped:
    chat.streaming_append("leaked tail")
    assert all("old session" not in m["body"] for m in chat._messages)
    chat.streaming_end(None)
    assert all("leaked tail" not in m["body"] for m in chat._messages),         "suppressed orphan must not commit a card"


def test_streaming_throttle_long_body():
    """Bug #6 regression: long bodies re-render at most once per flush window."""
    import time as _t
    from unittest.mock import patch
    app = QApplication.instance() or QApplication([])
    chat = NativeRichLogWidget()
    chat.streaming_start()
    calls = {"n": 0}
    orig = NativeRichLogWidget._streaming_card_html
    def counting(self, text):
        calls["n"] += 1
        return orig(self, text)
    with patch.object(NativeRichLogWidget, "_streaming_card_html", counting):
        for i in range(40):
            chat.streaming_append("word%d " % i)
            chat._streaming_text += "pad" * 250  # cross inline limit fast
        qapp = app
        qapp.processEvents()
        _t.sleep(0.15)
        qapp.processEvents()
    assert calls["n"] <= 5, f"expected throttling, got {calls['n']} full re-renders"
    chat.streaming_end(None)


def test_load_older_accepts_timestamp_tuples():
    """Bug #5 regression: older pages keep their DB timestamps."""
    app = QApplication.instance() or QApplication([])
    chat = NativeRichLogWidget()
    chat.set_history_window(
        ["You: recent"], ["2026-09-27T10:00:00"], 3,
        load_older_cb=lambda shown, want: (
            ["You: older one", "Orthos: older two"],
            ["2026-09-25T08:15:00", "2026-09-25T08:16:00"],
        ))
    chat._load_older_page()
    msgs = chat._messages
    assert msgs[0]["ts"] == "2026-09-25T08:15:00", f"older ts lost: {msgs[0]}"
    assert msgs[1]["ts"] == "2026-09-25T08:16:00"


def test_code_store_fifo_eviction():
    """Bug #7 regression: code store stays bounded."""
    app = QApplication.instance() or QApplication([])
    chat = NativeRichLogWidget()
    for i in range(300):
        chat._register_code("code block %d" % i)
    assert len(chat._code_store) <= 250, "code store grew unbounded"


def test_activity_panel_uses_a_persisted_resizable_splitter():
    source = (os.path.dirname(os.path.dirname(__file__)) + "/ui.py")
    text = open(source, encoding="utf-8").read()
    assert "self._chat_splitter = QSplitter(Qt.Orientation.Horizontal)" in text
    assert "self._chat_splitter.splitterMoved.connect(self._save_activity_panel_width)" in text
    assert "layout/activity_panel_width" in text
