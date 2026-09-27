"""Native, no-browser chat rendering for the Orthos desktop UI.

This deliberately avoids QWebEngine: Chromium rendering can fail independently
of the rest of a PyQt application.  Markdown is parsed locally and equations
are rendered to in-memory images by Matplotlib's MathText engine.

2026 architecture notes:
- Code blocks are located via markdown-it's own token stream (parse once),
  never by regex — the old heuristic mis-bound copy buttons and leaked math
  into indented/unbalanced fences.
- Fences are explicitly enabled; markdown-it's js-default disables ``` fences
  (only ~~~ worked), which silently degraded every code reply.
- Math images are cached by expression hash and their alpha is extracted with
  PIL/numpy instead of a per-pixel Python loop — session switches no longer
  re-rasterize the whole history on the GUI thread.
- Appends at the bottom use cursor insertHtml instead of a full setHtml of
  every cached card (O(1) per message instead of O(document)).
- Per-message HTML cache is content-addressed and evicted below the render
  window; the DB remains the authoritative history.
"""

from __future__ import annotations

import hashlib
import html as _html
import io
import re
import time

from markdown_it import MarkdownIt
from matplotlib.mathtext import math_to_image
from PyQt6.QtCore import Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QTextCursor, QTextDocument
from PyQt6.QtWidgets import QApplication, QTextBrowser

try:
    from pygments import highlight as _pyg_highlight
    from pygments.formatters.html import HtmlFormatter
    from pygments.lexers import get_lexer_by_name, TextLexer
    from pygments.util import ClassNotFound
    _HAS_PYGMENTS = True
except ImportError:  # graceful: plain <pre><code> blocks still render
    _HAS_PYGMENTS = False


_MARKDOWN = MarkdownIt("js-default", {"breaks": True, "html": False})
_MARKDOWN.enable("fence")  # js-default ships with ``` fences disabled.
_MATH_RE = re.compile(
    r"\$\$(.+?)\$\$|\\\[(.+?)\\\]|\\\((.+?)\\\)|(?<!\\)\$(?!\$)(.+?)(?<!\\)\$",
    re.DOTALL,
)
# Rendered history cap: the DB is authoritative; the widget only needs a
# readable tail.  Keeps setHtml() payloads small on very long chats.
_MAX_RENDERED_MESSAGES = 400
# Lazy-load page size (F6): messages fetched per "Load older" click.
_OLDER_PAGE = 100
_COPY_PREFIX = "copy://"
# Streaming render throttle: bodies above this many chars are re-rendered
# at most every _STREAM_FLUSH_MS (short replies stay instant).  Without the
# floor, a long code stream re-parsed + re-highlighted the WHOLE body on
# every token (O(n^2) on the GUI thread — visible as typing lag).
_STREAM_INLINE_LIMIT = 1500
_STREAM_FLUSH_MS = 80
# End-card scan depth: live cards (streaming draft, typing, thinking, action
# bar) are end-anchored, but a long streaming reply can span far more than a
# dozen blocks (each code fence adds label+pre blocks).  The old fixed 12
# window silently failed on those, leaking duplicate cards.
_END_SCAN_BLOCKS = 160
# Copy-button code store cap (FIFO) — bounds memory on very long sessions.
_CODE_STORE_MAX = 200

# ── Module-level caches (shared across widget rebuilds) ────────────────
# expression-hash -> rendered QImage; survives session switches so a math
# formula is rasterized once per process instead of once per switch.
_MATH_IMG_CACHE: dict[str, QImage] = {}

# ── Syntax highlighting (Pygments, dark-cyber palette) ─────────────────
if _HAS_PYGMENTS:
    # Qt rich text ignores external CSS classes — inline styles only.
    # "one-dark"-flavoured token colors tuned for the #050f16 pre background.
    _PYG_FORMATTER = HtmlFormatter(
        style="one-dark", noclasses=True,
        prestyles="background:#050f16;margin:0;padding:8px;"
                  "font-family:'Cascadia Code','Cascadia Mono',Consolas,monospace;"
                  "font-size:9pt;",
    )

    _LEXER_ALIASES = {
        "js": "javascript", "ts": "typescript", "py": "python",
        "rb": "ruby", "sh": "bash", "shell": "bash", "yml": "yaml",
        "cs": "csharp", "cpp": "cpp", "c++": "cpp", "md": "markdown",
    }

    def _highlight_code(code: str, lang: str) -> str:
        """Return a Qt-renderable highlighted <pre> block, or plain fallback."""
        try:
            name = _LEXER_ALIASES.get(lang.lower(), lang.lower())
            lexer = get_lexer_by_name(name, stripall=False)
        except (ClassNotFound, ValueError):
            try:
                lexer = TextLexer(stripall=False)
            except Exception:
                return None
        try:
            out = _pyg_highlight(code, lexer, _PYG_FORMATTER)
            return out
        except Exception:
            return None
else:
    def _highlight_code(code: str, lang: str) -> str:
        return None

# Typing-indicator frame sequence (three pulsing dots).
_TYPING_FRAMES = (
    '<span style="color:#00b0dd;">●</span>'
    '<span style="color:#07202c;">●</span>'
    '<span style="color:#07202c;">●</span>',
    '<span style="color:#00b0dd;">●</span>'
    '<span style="color:#00b0dd;">●</span>'
    '<span style="color:#07202c;">●</span>',
    '<span style="color:#00b0dd;">●</span>'
    '<span style="color:#00b0dd;">●</span>'
    '<span style="color:#00b0dd;">●</span>',
)
_TYPING_CARD = (
    '<p style="margin:10px 0 4px 0;">'
    '<span style="color:#00b0dd;font-weight:600;font-family:Cascadia Code,'
    ' Consolas, monospace;font-size:9pt;letter-spacing:0.8px;">Orthos</span>'
    ' <span style="font-size:9pt;color:#3a8a9a;">is thinking</span> {dots}'
    '<span style="font-size:1pt;color:#071018;">\ue903</span>'
    '</p>'
)

# Invisible end-anchors.  End-anchored live cards (typing indicator,
# streaming draft, action bar) are removed by SCANNING the document tail
# for these markers — position tracking broke across setHtml rebuilds and
# could delete arbitrary message text (corruption).  The marks are
# private-use codepoints that never occur in message bodies.
_STREAM_MARK = "\ue901"   # inside the streaming caret line
_ACTION_MARK = "\ue902"   # trailing after the action-bar chips
_TYPING_MARK = "\ue903"   # inside the typing indicator line
_THINK_MARK = "\ue904"    # inside the collapsible thinking card header


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()


class NativeRichLogWidget(QTextBrowser):
    """Copyable/scrollable GFM-like Markdown and math without a web view."""

    _sig = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._messages: list[dict] = []
        # content-hash -> cached HTML card.  Content-addressed keys make the
        # cache immune to index shifts and direct _messages edits (an edited
        # body simply misses the cache).
        self._html_cache: dict[str, str] = {}
        # id -> raw code text, filled while rendering fenced code blocks
        self._code_store: dict[str, str] = {}
        self._code_seq = 0
        self.setReadOnly(True)
        # Anchor clicks are routed manually so copy:// works alongside links,
        # and no scheme (file://, qrc:) can navigate/destroy the document.
        self.setOpenExternalLinks(False)
        self.setOpenLinks(False)
        self.anchorClicked.connect(self._on_anchor)
        self.setFont(QFont("Segoe UI Variable Text", 10))
        self.setPlaceholderText("Start a conversation with Orthos…")
        self.setStyleSheet("""
            QTextBrowser { background: #071018; color: #c9eef7; border: 1px solid #123c50;
                border-radius: 4px; padding: 8px; selection-background-color: #0a4d66; }
            QScrollBar:vertical { background: #040b10; width: 8px; border: none; }
            QScrollBar::handle:vertical { background: #1d5a75; border-radius: 4px; min-height: 24px; }
            QScrollBar::handle:vertical:hover { background: #2a7c9a; }
        """)
        self.document().setDefaultStyleSheet("""
            body { color: #c9eef7; font-family: 'Segoe UI Variable Text', 'Segoe UI', system-ui, sans-serif; font-size: 10.5pt; }
            p { margin: 5px 0; } h1,h2,h3 { color: #5fd9f5; margin: 12px 0 5px; }
            h1 { font-size: 15pt; } h2 { font-size: 13pt; } h3 { font-size: 11.5pt; }
            strong { color: #f2fcff; } em { color: #ffd27a; }
            pre { background: #050f16; border: 1px solid #11455e; padding: 8px; }
            code { color: #7dffc4; font-family: 'Cascadia Code', 'Cascadia Mono', Consolas, monospace; }
            blockquote { border-left: 3px solid #00a9cf; color: #93d9e8; padding-left: 10px; }
            table { border-collapse: collapse; } th { background: #0a2230; color: #7ee4ff; }
            th,td { border: 1px solid #17516a; padding: 4px 7px; }
            a { color: #56dfff; }
        """)
        self._sig.connect(self._enqueue)
        self._connect_streaming_signals()

        # ── Typing indicator state ──────────────────────────────────
        self._typing = False
        self._typing_frame = 0
        self._typing_pos = -1  # document anchor for the animated card
        self._typing_timer = QTimer(self)
        self._typing_timer.setInterval(400)
        self._typing_timer.timeout.connect(self._tick_typing)

        # ── Streaming reply card state (F1) ─────────────────────────
        self._streaming_active = False
        self._streaming_text = ""
        self._stream_paint_pending = False   # throttle flag (_stream_flush)
        self._suppress_stream = False        # session-switch orphan guard

        # ── Reasoning/thinking card (collapsible, ChatGPT-style) ────
        self._thinking_active = False
        self._thinking_text = ""
        self._thinking_expanded = False
        self._thinking_started_at = 0.0

        # ── Lazy history paging (F6) ─────────────────────────────────
        # ``_total_messages`` counts messages that exist but are not yet
        # rendered; a top sentinel offers to fetch them page by page.
        self._total_messages = 0            # authoritative count (DB)
        self._load_older_cb = None          # (before_idx, count) -> list[str]
        self._loading_older = False

        # ── Inline follow-up actions (F7) ────────────────────────────
        self._on_action_cb = None           # "regenerate" | "continue"

        # ── Unread tracking for the scroll FAB ──────────────────────
        self._unread = 0
        self.verticalScrollBar().valueChanged.connect(self._on_scroll_moved)

    # ── Public API ────────────────────────────────────────────────────

    def append_log(self, text: str):
        self._sig.emit(text)

    def set_logs(self, entries: list[str]):
        self.set_logs_with_timestamps(entries, None)

    def set_logs_with_timestamps(self, entries: list[str], timestamps: list[str] | None = None):
        """Replace history, attaching DB turn timestamps when provided."""
        # Session switch: kill any in-flight live state so a streaming draft
        # or thinking card from the OLD session cannot leak into the new one
        # via still-queued cross-thread signals.  _suppress_stream drops the
        # orphan tokens those queued signals still deliver; the next explicit
        # streaming_start() (a real new reply) clears it.
        was_streaming = self._streaming_active
        self._stream_paint_pending = False
        self.streaming_abort()
        self.thinking_abort()
        self._suppress_stream = was_streaming
        self._messages = [self._parse_entry(entry) for entry in entries]
        if timestamps:
            for msg, ts in zip(self._messages, timestamps):
                if ts:
                    msg["ts"] = ts
        self._code_store.clear()
        self._code_seq = 0
        self.set_typing(False)
        self._unread = 0
        self._full_render()

    def clear(self):  # Qt API used by MainWindow
        self._messages.clear()
        self._code_store.clear()
        self._code_seq = 0
        self.set_typing(False)
        self._unread = 0
        super().clear()

    # ── Typing indicator ───────────────────────────────────────────

    def set_typing(self, active: bool):
        """Show/hide the animated ORTHOS thinking card (F4)."""
        if active == self._typing:
            return
        self._typing = active
        if active:
            self._show_typing_card()
            self._typing_timer.start()
        else:
            self._typing_timer.stop()
            self._remove_typing_card()

    # ── Streaming render (live-typing reply card) ───────────────────

    # Thread-safe bridges: the LLM loop runs off the GUI thread, so all
    # streaming mutations are marshalled through queued signal connections.
    _stream_start_sig = pyqtSignal()
    _stream_append_sig = pyqtSignal(str)
    _stream_end_sig = pyqtSignal(object)
    _stream_abort_sig = pyqtSignal()
    _think_start_sig = pyqtSignal()
    _think_append_sig = pyqtSignal(str)
    _think_end_sig = pyqtSignal(object)
    _think_abort_sig = pyqtSignal()

    def _connect_streaming_signals(self):
        self._stream_start_sig.connect(
            self.streaming_start, Qt.ConnectionType.QueuedConnection)
        self._stream_append_sig.connect(
            self.streaming_append, Qt.ConnectionType.QueuedConnection)
        self._stream_end_sig.connect(
            self._streaming_end_slot, Qt.ConnectionType.QueuedConnection)
        self._stream_abort_sig.connect(
            self.streaming_abort, Qt.ConnectionType.QueuedConnection)
        self._think_start_sig.connect(
            self.thinking_start, Qt.ConnectionType.QueuedConnection)
        self._think_append_sig.connect(
            self.thinking_append, Qt.ConnectionType.QueuedConnection)
        self._think_end_sig.connect(
            self._thinking_end_slot, Qt.ConnectionType.QueuedConnection)
        self._think_abort_sig.connect(
            self.thinking_abort, Qt.ConnectionType.QueuedConnection)

    def streaming_start_threadsafe(self):
        self._stream_start_sig.emit()

    def streaming_append_threadsafe(self, text: str):
        self._stream_append_sig.emit(text)

    def streaming_end_threadsafe(self, final_text: str | None = None):
        self._stream_end_sig.emit(final_text)

    def streaming_abort_threadsafe(self):
        self._stream_abort_sig.emit()

    # ── Reasoning/thinking card (thread-safe bridges) ────────────────
    def thinking_start_threadsafe(self):
        self._think_start_sig.emit()

    def thinking_append_threadsafe(self, text: str):
        self._think_append_sig.emit(text)

    def thinking_end_threadsafe(self, final_text: str | None = None):
        self._think_end_sig.emit(final_text)

    def thinking_abort_threadsafe(self):
        self._think_abort_sig.emit()

    def _streaming_end_slot(self, final_text):
        self.streaming_end(final_text)

    def streaming_start(self):
        """Begin a live-streamed assistant reply card (F1).

        Tokens appended via streaming_append render progressively in a
        flat ORTHOS block; streaming_end finalizes it as a normal message.
        """
        if self._streaming_active:
            self._remove_streaming_card()
        self.set_typing(False)
        self._streaming_active = True
        self._streaming_text = ""
        self._stream_paint_pending = False
        self._suppress_stream = False        # explicit start = legit new reply
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._insert_html_block(cursor, self._streaming_card_html(""))
        # History readers keep their position; the live card waits at the
        # document tail until they choose to come down (unread badge shows).
        bar = self.verticalScrollBar()
        if bar.maximum() - bar.value() <= 60:
            self.scroll_to_bottom_hard()

    def streaming_append(self, text: str):
        """Append streamed tokens to the live card (cheap partial render).

        Bodies beyond _STREAM_INLINE_LIMIT chars are throttled to one
        re-render per _STREAM_FLUSH_MS — markdown parse + pygments over a
        growing body on EVERY token was O(n^2) on the GUI thread.  The
        final authoritative render happens in streaming_end().
        """
        if not self._streaming_active:
            if self._suppress_stream:
                return  # orphan token from a switched-away session — drop it
            self.streaming_start()
        self._streaming_text += text
        try:
            if len(self._streaming_text) > _STREAM_INLINE_LIMIT:
                if self._stream_paint_pending:
                    return  # a repaint is already scheduled — skip this one
                self._stream_paint_pending = True
                QTimer.singleShot(_STREAM_FLUSH_MS, self._stream_flush)
                return
            self._remove_streaming_card()
            cursor = self.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self._insert_html_block(cursor, self._streaming_card_html(self._streaming_text))
            bar = self.verticalScrollBar()
            if bar.maximum() - bar.value() <= 60:
                self._snap_to_bottom()
        except RuntimeError:
            pass

    def _stream_flush(self):
        """Throttled repaint for long streaming bodies."""
        self._stream_paint_pending = False
        if not self._streaming_active:
            return
        try:
            self._remove_streaming_card()
            cursor = self.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self._insert_html_block(cursor, self._streaming_card_html(self._streaming_text))
            bar = self.verticalScrollBar()
            if bar.maximum() - bar.value() <= 60:
                self._snap_to_bottom()
        except RuntimeError:
            pass

    def streaming_end(self, final_text: str | None = None) -> str:
        """Finalize the stream; returns the assembled text (or given final)."""
        text = final_text if final_text is not None else self._streaming_text
        self._stream_paint_pending = False   # cancel any throttled repaint
        self._remove_streaming_card()
        self._streaming_active = False
        self._streaming_text = ""
        if text:
            self._enqueue(f"Orthos: {text}")
        return text

    def streaming_abort(self):
        """Cancel the stream without emitting a message card."""
        self._stream_paint_pending = False   # cancel any throttled repaint
        self._remove_streaming_card()
        self._streaming_active = False
        self._streaming_text = ""

    # ── Reasoning/thinking card (ChatGPT/OpenCode-style) ────────────

    def _thinking_end_slot(self, final_text):
        self.thinking_end(final_text)

    def thinking_start(self):
        """Begin a live collapsible thinking card above the reply."""
        if self._thinking_active:
            self._remove_thinking_card()
        self._thinking_active = True
        self._thinking_text = ""
        self._thinking_expanded = False
        self._thinking_started_at = time.time()
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._insert_html_block(cursor, self._thinking_card_html())
        bar = self.verticalScrollBar()
        if bar.maximum() - bar.value() <= 60:
            self.scroll_to_bottom_hard()

    def thinking_append(self, text: str):
        """Append reasoning tokens; re-render the (usually collapsed) card."""
        if not self._thinking_active:
            if self._suppress_stream:
                return  # orphan reasoning from a switched-away session
            self.thinking_start()
        self._thinking_text += text
        try:
            self._remove_thinking_card()
            cursor = self.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self._insert_html_block(cursor, self._thinking_card_html())
            bar = self.verticalScrollBar()
            if bar.maximum() - bar.value() <= 60:
                self._snap_to_bottom()
        except RuntimeError:
            pass

    def thinking_end(self, final_text: str | None = None):
        """Collapse the card; keeps it browsable via the toggle chip.

        The reasoning text is retained in ``_thinking_text`` so the chip can
        expand it later; ``thinking_start``/``thinking_abort`` reset it.
        """
        if final_text is not None:
            self._thinking_text = final_text
        self._thinking_active = False
        try:
            self._remove_thinking_card()
            if self._thinking_text.strip():
                cursor = self.textCursor()
                cursor.movePosition(QTextCursor.MoveOperation.End)
                self._insert_html_block(cursor, self._thinking_card_html())
        except RuntimeError:
            pass
        self._thinking_expanded = False

    def thinking_abort(self):
        """Cancel a live thinking card without leaving any trace."""
        self._remove_thinking_card()
        self._thinking_active = False
        self._thinking_text = ""
        self._thinking_expanded = False

    def thinking_toggle(self):
        """Expand/collapse the last thinking card via the toggle chip."""
        self._thinking_expanded = not self._thinking_expanded
        if self._thinking_active:
            self._remove_thinking_card()
            cursor = self.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self._insert_html_block(cursor, self._thinking_card_html())
        elif self._thinking_text.strip():
            cursor = self.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self._insert_html_block(cursor, self._thinking_card_html())

    def _thinking_card_html(self) -> str:
        """Collapsible reasoning chip (header always, body when expanded)."""
        elapsed = max(0.0, time.time() - self._thinking_started_at)
        dur = f"{elapsed:.0f}s" if self._thinking_active else f"{elapsed:.0f}s"
        state = "Thinking" if self._thinking_active else "Thought"
        if self._thinking_active and not self._thinking_text.strip():
            state = "Thinking…"
        words = len(self._thinking_text.split())
        arrow = "▾" if self._thinking_expanded else "▸"
        header = (
            f'<a href="thinktoggle://" style="text-decoration:none;color:#7fb8d0;'
            f'font-size:8.5pt;">💭 {state} ({dur} · {words} tok) {arrow}</a>'
            f'<span style="color:transparent;font-size:1px;">{_THINK_MARK}</span>'
        )
        if self._thinking_expanded and self._thinking_text.strip():
            body = (
                '<div style="color:#6e93a6;font-size:9pt;background:#0a1520;'
                'border-left:2px solid #17475e;padding:6px 8px;margin:2px 0 4px 0;">'
                + _html.escape(self._thinking_text).replace("\n", "<br>")
                + "</div>"
            )
        else:
            body = ""
        return f'<p style="margin:8px 0 2px 0;">{header}</p>{body}'

    def _remove_thinking_card(self):
        self._remove_end_card(_THINK_MARK)

    def _streaming_card_html(self, text: str) -> str:
        """Flat ORTHOS label + partial body + live caret (2026 style)."""
        partial = self._markdown_with_code(text) if text.strip() else ""
        # The caret (with its invisible removal mark) always shows while
        # streaming — even before the first token arrives.
        caret = (
            f'<span style="color:#00b0dd;font-weight:700;">\u258c</span>'
        )
        label = self._streaming_label_html()
        return (
            f'<p style="margin:10px 0 2px 0;">{label}</p>{partial}{caret}'
        )

    def _remove_streaming_card(self):
        self._remove_end_card(_STREAM_MARK)


    def _remove_end_card(self, mark: str) -> bool:
        """Delete the end-anchored live card tagged with ``mark``.

        Scans document blocks from the END backwards; the card is found by
        its invisible marker, and everything from that block's start to the
        document end is removed.  Survives setHtml() rebuilds (no stale
        positions) and never touches message text above the marker.
        Scan depth is _END_SCAN_BLOCKS (a long streaming draft outgrows the
        old fixed 12-block window and then never got removed).

        Block removal leaves its separator behind as an empty husk; each
        typing tick / streaming token used to leak one of those — the
        blank gap users saw between their message and the reply.  After
        removing the card, every CONSECUTIVE empty tail block (keeping only
        the final implicit one) is merged away with deletePreviousChar().
        """
        try:
            doc = self.document()
            block = doc.lastBlock() if hasattr(doc, "lastBlock") else doc.end()
            # Walk up to _END_SCAN_BLOCKS back — live cards sit at the end,
            # but a long streaming reply spans many blocks (fences expand).
            for _ in range(_END_SCAN_BLOCKS):
                if not block.isValid():
                    return False
                text = block.text()
                if mark in text:
                    cursor = QTextCursor(doc)
                    cursor.setPosition(block.position())
                    end = doc.characterCount() - 1
                    if end > block.position():
                        cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
                        cursor.removeSelectedText()
                        # ── Husk cleanup: merge consecutive empty blocks ──
                        # After the removal the cursor sits in the leftover
                        # empty block.  While that block AND the one before
                        # it are both empty, deletePreviousChar() eats the
                        # separator and merges them — leaving exactly one
                        # implicit empty tail block, never a growing gap.
                        for _ in range(_END_SCAN_BLOCKS):
                            cur = cursor.block()
                            if not cur.isValid() or cur.position() == 0:
                                break
                            if cur.text().strip():
                                break  # content reached — stop
                            cursor.deletePreviousChar()
                    return True
                block = block.previous()
        except RuntimeError:
            pass
        return False

    def _insert_html_block(self, cursor: QTextCursor, html: str):
        """Insert HTML in its OWN block — never merge into the tail block.

        Qt rich text merges consecutive inline HTML into the current block;
        without this boundary, end-anchored cards fuse with the previous
        message and scan-removal would eat message text.

        When the cursor already sits at the end of an EMPTY tail block, the
        HTML is written INTO it instead of appending yet another block —
        this is what stops live-card churn (typing ticks, streaming tokens)
        from accumulating blank lines between messages.
        """
        block = cursor.block()
        if (cursor.atEnd() and block.isValid() and block.position() > 0
                and not block.text().strip()):
            cursor.insertHtml(html)   # reuse the empty tail block
            return
        cursor.insertBlock()
        cursor.insertHtml(html)

    def _show_typing_card(self):
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._insert_html_block(
            cursor, _TYPING_CARD.format(dots=_TYPING_FRAMES[0]))
        # Follow the view ONLY when the reader is parked at the bottom —
        # a history reader must never be yanked down by the indicator.
        bar = self.verticalScrollBar()
        if bar.maximum() - bar.value() <= 60:
            self.scroll_to_bottom_hard()

    def _remove_typing_card(self):
        self._remove_end_card(_TYPING_MARK)
        self._typing_pos = -1

    def _tick_typing(self):
        """Advance the three-dot pulse without touching message cards."""
        if not self._typing:
            return
        self._typing_frame = (self._typing_frame + 1) % len(_TYPING_FRAMES)
        try:
            # The remove+insert bounce shifts the viewport; a reader parked
            # at the bottom must stay there across every animation tick.
            bar = self.verticalScrollBar()
            was_at_bottom = bar.maximum() - bar.value() <= 60
            self._remove_typing_card()
            cursor = self.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self._insert_html_block(
                cursor, _TYPING_CARD.format(dots=_TYPING_FRAMES[self._typing_frame]))
            if was_at_bottom:
                self._snap_to_bottom()
                self._deferred_snaps(1, 0)
        except RuntimeError:
            pass

    # ── Scroll FAB + unread badge ──────────────────────────────────

    def _on_scroll_moved(self, _value: int):
        bar = self.verticalScrollBar()
        if bar.maximum() - bar.value() <= 60:
            self._unread = 0

    def resizeEvent(self, event):
        super().resizeEvent(event)
        bar = self.verticalScrollBar()
        if bar.maximum() - bar.value() <= 60:
            self._unread = 0

    def _unread_changed(self):
        """Call after appends when the reader is not at the bottom."""
        bar = self.verticalScrollBar()
        if bar.maximum() - bar.value() > 60:
            self._unread += 1

    # ── Timestamps ─────────────────────────────────────────────────────

    @staticmethod
    def _short_time(value: str | None) -> str:
        """Human-friendly timestamp from an ISO-ish string; '' when unknown.

        Today's messages show HH:MM; older ones include the date
        ("05 Sep 14:32") so history reads without ambiguity.
        """
        if not value:
            return ""
        text = str(value).strip()
        iso = re.search(
            r"(\d{4})-(\d{2})-(\d{2})[T ](\d{1,2}):(\d{2})", text)
        if iso:
            try:
                from datetime import date, datetime
                then = datetime(
                    int(iso.group(1)), int(iso.group(2)), int(iso.group(3)),
                    int(iso.group(4)), int(iso.group(5)),
                )
                hh = f"{int(iso.group(4)):02d}:{iso.group(5)}"
                if then.date() == date.today():
                    return hh
                return f"{then.strftime('%d %b')} {hh}"
            except ValueError:
                return ""  # invalid date: show nothing, never garbage
        match = re.search(r"(\d{1,2}):(\d{2})", text)
        if not match:
            return ""
        return f"{int(match.group(1)):02d}:{match.group(2)}"

    # ── Copy anchors ───────────────────────────────────────────────────

    def _on_anchor(self, url: QUrl):
        target = url.toString()
        if target.startswith("action://"):
            action = target[len("action://"):]
            handler = getattr(self, "_on_action_cb", None)
            if handler:
                try:
                    handler(action)
                except Exception:
                    pass
            return
        if target.startswith("older://"):
            self._load_older_page()
            return
        if target.startswith("thinktoggle://"):
            self.thinking_toggle()
            return
        if target.startswith(_COPY_PREFIX):
            code_id = target[len(_COPY_PREFIX):]
            code = self._code_store.get(code_id, "")
            if code:
                # Strip the parser's trailing newline; a dead empty block
                # never registers an anchor at all.
                QApplication.clipboard().setText(code.rstrip("\n"))
                self._flash_copy_feedback(code_id)
            return
        if target.startswith("msg://"):
            # Hover-copy button on message cards: copies the raw body text.
            idx_part = target[len("msg://"):]
            try:
                idx = int(idx_part.rsplit("copy-", 1)[-1])
            except ValueError:
                return
            msg = self._messages[idx] if 0 <= idx < len(self._messages) else None
            if msg:
                QApplication.clipboard().setText(msg.get("body", ""))
            return
        if target.startswith(("http://", "https://", "mailto:", "ftp://")):
            from PyQt6.QtGui import QDesktopServices
            QDesktopServices.openUrl(QUrl(target))

    def _flash_copy_feedback(self, code_id: str):
        """Brief 'Copied ✓' on the matching copy anchor.

        The anchor href lives in a char format, not in the document text,
        so the label is found by walking block fragments and checking
        ``anchorHref`` — a plain text search never sees it.
        """
        doc = self.document()
        block = doc.firstBlock()
        while block.isValid():
            it = block.begin()
            while not it.atEnd():
                frag = it.fragment()
                it += 1
                if not frag.isValid():
                    continue
                href = frag.charFormat().anchorHref()
                if not href or not href.startswith(_COPY_PREFIX):
                    continue
                if href[len(_COPY_PREFIX):] != code_id:
                    continue
                cursor = QTextCursor(doc)
                cursor.setPosition(frag.position())
                cursor.setPosition(frag.position() + frag.length(),
                                   QTextCursor.MoveMode.KeepAnchor)
                cursor.insertText("Copied ✓", frag.charFormat())
                restore_pos = frag.position()
                restore_fmt = frag.charFormat()

                def restore(_pos=restore_pos, _fmt=restore_fmt):
                    try:
                        doc2 = self.document()
                        c2 = QTextCursor(doc2)
                        c2.setPosition(min(_pos, doc2.characterCount() - 1))
                        c2.setPosition(min(_pos + len("Copied ✓"),
                                           doc2.characterCount() - 1),
                                       QTextCursor.MoveMode.KeepAnchor)
                        if c2.selectedText() == "Copied ✓":
                            c2.insertText("⧉ Copy", _fmt)
                    except RuntimeError:
                        pass  # widget deleted during the flash — nothing to do

                QTimer.singleShot(1200, restore)
                return
            block = block.next()

    # ── Message intake ─────────────────────────────────────────────────

    def _enqueue(self, text: str):
        msg = self._parse_entry(text)
        if not msg.get("ts"):
            msg["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._messages.append(msg)
        self._append_message(msg)

    # Unprefixed operational lines that are file/attachment events, not chat.
    _FILE_PREFIXES = ("attached:", "ready:", "removed:", "file:")

    @staticmethod
    def _parse_entry(text: str) -> dict:
        prefixes = (
            ("you:", "you", "You:"), ("orthos:", "ai", "Orthos:"),
            ("orthos:", "ai", "Orthos:"), ("file:", "file", "File:"),
            ("sys:", "sys", "SYS:"),
        )
        stripped = text.lstrip()
        lowered = stripped.lower()
        for marker, tag, prefix in prefixes:
            if lowered.startswith(marker):
                body = stripped[len(marker):].lstrip()
                if tag == "sys" and (body.startswith("▶") or body.lower().startswith("tool result")):
                    return {"tag": "tool", "prefix": "TOOL", "body": body}
                return {"tag": tag, "prefix": prefix, "body": body}
        # Bracketed tool/action tags: "[YouTube] …" -> operational row.
        bracket = re.match(r"^\[(\w+)\]\s*", lowered)
        if bracket:
            return {"tag": "tool", "prefix": bracket.group(1).upper(), "body": stripped}
        # Attachment-lifecycle lines emitted without a role prefix.
        for marker in NativeRichLogWidget._FILE_PREFIXES:
            if lowered.startswith(marker):
                return {"tag": "file", "prefix": "FILE", "body": stripped}
        # Only line-initial error markers recolor the row; mid-sentence
        # mentions of "error" stay system text.
        if re.match(r"^\s*(err|error|failed|✗)\b", lowered):
            return {"tag": "err", "prefix": "ERROR", "body": stripped}
        return {"tag": "sys", "prefix": "", "body": text}

    # ── Math (cached, vectorized) ──────────────────────────────────────

    def _math_image(self, expression: str, display: bool) -> str:
        key = _sha(expression)
        image = _MATH_IMG_CACHE.get(key)
        if image is None:
            try:
                image = self._rasterize_math(expression)
                _MATH_IMG_CACHE[key] = image
            except Exception:
                # Unsupported MathText stays readable rather than vanishing.
                return f"<code>{_html.escape(expression)}</code>"
        # Resources are per-document; re-registering the same cached image
        # under a fresh id keeps rendering consistent after setHtml resets.
        self._math_seq = getattr(self, "_math_seq", 0) + 1
        resource = QUrl(f"orthos-math:/f/{self._math_seq}")
        self.document().addResource(QTextDocument.ResourceType.ImageResource, resource, image)
        align = "center" if display else "middle"
        return f'<img src="{resource.toString()}" style="vertical-align:{align};">'

    def _rasterize_math(self, expression: str) -> QImage:
        buf = io.BytesIO()
        math_to_image(f"${expression}$", buf, dpi=150, format="png",
                      color="#eaf7fc")
        image = QImage.fromData(buf.getvalue(), "PNG")
        if image.isNull():
            raise ValueError("empty math image")
        return self._remove_white_background(image)

    @staticmethod
    def _remove_white_background(image: QImage) -> QImage:
        """Recover transparency from MathText's white-composited pixels.

        Vectorized with PIL/numpy: the old per-pixel loop was ~18k Qt calls
        per glyph, which froze the UI for seconds on math-heavy sessions.
        """
        from PIL import Image
        import numpy as np

        src = Image.frombytes(
            "RGBA", (image.width(), image.height()),
            image.constBits().asstring(image.sizeInBytes()) if hasattr(image, "constBits")
            else bytes(image.bits().asstring(image.sizeInBytes())),
        ).convert("RGB")
        arr = np.asarray(src, dtype=np.float32)
        # Foreground #eaf7fc over white: alpha = (255 - red) / (255 - 234).
        red = arr[:, :, 0]
        alpha = np.clip((255.0 - red) / (255 - 234.0), 0.0, 1.0)
        out = np.zeros((image.height(), image.width(), 4), dtype=np.uint8)
        out[:, :, 0] = 234; out[:, :, 1] = 247; out[:, :, 2] = 252
        out[:, :, 3] = (alpha * 255).astype(np.uint8)
        pil_out = Image.fromarray(out, "RGBA")
        buf = io.BytesIO()
        pil_out.save(buf, format="PNG")
        result = QImage.fromData(buf.getvalue(), "PNG")
        return result if not result.isNull() else image

    # ── Markdown + code blocks (token-stream based) ────────────────────

    def _markdown_with_code(self, body: str) -> str:
        """Render markdown with fenced code copy-buttons and math images."""
        tokens = _MARKDOWN.parse(body)
        code_spans: list[tuple[str, str]] = []  # (raw code, language)
        math_parts: list[tuple[str, list[str]]] = []  # (protected, formulas)

        # Pass 1: split inline math out of each token, keep code verbatim.
        lines_out: list[str] = []
        for tok in tokens:
            if tok.type in ("fence", "code_block"):
                code_spans.append((tok.content, (tok.info or "").strip()))
                # Placeholder line that renders as a distinct block.
                lines_out.append(("CODE", len(code_spans) - 1))
            elif tok.type == "inline":
                protected, formulas = self._protect_math(tok.content)
                math_parts.append((protected, formulas))
                lines_out.append(("TEXT", protected))
            else:
                lines_out.append(("TEXT", ""))

        # Rebuild a markdown source where code is fenced placeholders and
        # math is protected text; then render and substitute.
        src_lines: list[str] = []
        for kind, payload in lines_out:
            if kind == "CODE":
                # Use a unique fence so the placeholder survives rendering.
                idx = payload
                src_lines.append(f"\n```\nORTHOSCODEBLOCK{idx}\n```\n")
            elif payload:
                src_lines.append(payload)
        html_out = _MARKDOWN.render("\n".join(src_lines))

        # Pass 2: replace math markers with <img> (cached rasterization).
        for protected, formulas in math_parts:
            for idx, expression in enumerate(formulas):
                marker = f"ORTHOSMATH{idx}TOKEN"
                if marker in html_out:
                    img = self._math_image(expression, False)
                    html_out = html_out.replace(marker, img)

        # Pass 3: code blocks -> highlighted <pre> with copy anchor.
        for idx, (code, lang) in enumerate(code_spans):
            if not code.strip():
                # Empty fence: render the block, skip the dead copy button.
                html_out = html_out.replace(
                    f"<pre><code>ORTHOSCODEBLOCK{idx}\n</code></pre>",
                    "<pre><code></code></pre>")
                html_out = html_out.replace(f"ORTHOSCODEBLOCK{idx}", "")
                continue
            code_id = self._register_code(code)
            # Pygments highlight (language label + colored tokens); fall back
            # to a plain escaped block when the lexer or module is missing.
            highlighted = _highlight_code(code, lang)
            lang_label = (
                f'<span style="color:#3a8a9a;font-size:7pt;'
                f'font-family:Cascadia Code, Consolas, monospace;">{lang}</span>'
                if lang else ""
            )
            if highlighted:
                # Pygments emits its own <pre>; inject the label line above.
                body_html = (
                    f'<p style="margin:0 0 2px 0;">{lang_label}</p>'
                    + highlighted
                )
            else:
                body_html = (
                    f'<p style="margin:0 0 2px 0;">{lang_label}</p>'
                    f'<pre><code>{_html.escape(code)}</code></pre>'
                )
            block = (
                f'<p style="margin:0;"><a href="{_COPY_PREFIX}{code_id}">'
                f'<span style="font-family:Cascadia Code, Consolas, monospace;font-size:8pt;'
                f'text-decoration:none;">⧉ Copy</span></a></p>'
                + body_html
            )
            # The placeholder renders inside <pre><code>; swap the whole block.
            html_out = html_out.replace(
                f"<pre><code>ORTHOSCODEBLOCK{idx}\n</code></pre>", block)
            html_out = html_out.replace(
                f"ORTHOSCODEBLOCK{idx}", block)
        return html_out

    def _protect_math(self, source: str) -> str:
        """Replace math with rasterize-on-demand markers (rendered later)."""
        formulas: list[str] = []

        def replace_math(match: re.Match) -> str:
            expression = next(part for part in match.groups() if part is not None).strip()
            formulas.append(expression)
            return f"ORTHOSMATH{len(formulas) - 1}TOKEN"

        protected = _MATH_RE.sub(replace_math, source)
        return protected, formulas

    def _render_math(self, protected: str, formulas: list[str]) -> str:
        """Replace math markers with <img> tags (cached by expression)."""
        out = protected
        for idx, expression in enumerate(formulas):
            # $$...$$ and \\...\\ render display-style (larger, centered);
            # the old code computed this and then passed a hardcoded False.
            is_display = expression.startswith("$$") or "\\[" in expression
            marker = f"ORTHOSMATH{idx}TOKEN"
            if marker not in out:
                continue
            img = self._math_image(expression, is_display)
            out = out.replace(marker, img)
        return out

    def _register_code(self, raw_code: str) -> str:
        self._code_seq += 1
        code_id = f"c{self._code_seq}"
        # FIFO evict: bounds memory on marathon sessions (a copy chip for a
        # block evicted long ago is cheaper than unbounded growth).
        if len(self._code_store) >= _CODE_STORE_MAX:
            for old_id in list(self._code_store.keys())[:64]:
                self._code_store.pop(old_id, None)
        self._code_store[code_id] = raw_code
        return code_id

    # ── Cards ──────────────────────────────────────────────────────────

    _ATTACH_RE = re.compile(r"\[ATTACHED FILES:\s*(.+?)\]", re.DOTALL)

    def _body_html(self, message: dict) -> str:
        """Single-pass body render; every attachment block styled, tail kept."""
        body_src = message["body"]
        matches = list(self._ATTACH_RE.finditer(body_src))
        if not matches:
            return self._markdown_with_code(body_src)
        parts: list[str] = []
        cursor = 0
        for m in matches:
            if m.start() > cursor:
                parts.append(self._markdown_with_code(body_src[cursor:m.start()]))
            inner = self._markdown_with_code(m.group(1))
            parts.append(
                f'<div style="margin:6px 0;padding:5px 9px;background-color:#061a24;'
                f'border:1px solid #1d5a75;border-left:3px solid #3d9db8;'
                f'border-radius:3px;color:#8ed6e5;font-size:9pt;">'
                f'📎 {inner}</div>'
            )
            cursor = m.end()
        if cursor < len(body_src):
            parts.append(self._markdown_with_code(body_src[cursor:]))
        return "".join(parts)

    def _ai_label_html(self) -> str:
        """Flat 2026 assistant label — name only, no box (Claude-style)."""
        return (
            '<span style="color:#4dd8ff;font-weight:600;'
            'font-family:Cascadia Code, Consolas, monospace;font-size:9pt;'
            'letter-spacing:0.8px;">Orthos</span>'
        )

    def _streaming_label_html(self) -> str:
        """Streaming variant of the label: carries the removal mark so the
        FIRST block of the live card can be scan-found even when the body
        spans several blocks (code fences, lists)."""
        return (
            '<span style="color:#4dd8ff;font-weight:600;'
            'font-family:Cascadia Code, Consolas, monospace;font-size:9pt;'
            f'letter-spacing:0.8px;">Orthos{_STREAM_MARK}</span>'
        )

    def _card_html(self, message: dict, idx: int | None = None) -> str:
        """2026 flat layout: whitespace separation, no heavy boxes."""
        roles = {
            "you":  ("YOU",     "#3d9db8", "#eaf7fc"),
            "ai":   ("ORTHOS",  "#00b0dd", "#d4f4fd"),
            "sys":  ("SYSTEM",  "#d6a019", "#ffe9a8"),
            "tool": ("TOOL",    "#00a877", "#c6ffdf"),
            "file": ("FILE",    "#00a877", "#c6ffdf"),
            "err":  ("ERROR",   "#e8496a", "#ffd6de"),
        }
        label, accent, text = roles[message["tag"]]
        ts = self._short_time(message.get("ts", ""))
        body = self._body_html(message)
        # Click-to-copy chip in the header: works for every chat card.
        copy_chip = ""
        if idx is not None:
            copy_chip = (
                f' <a href="msg://copy-{idx}">'
                f'<span style="font-size:8pt;color:#3a8a9a;text-decoration:none;">⧉</span></a>'
            )

        if message["tag"] == "you":
            # Modern user bubble: borderless tint, generous rounding; the
            # "YOU" header is dropped — bubbles are self-evident (2026 norm).
            return (
                f'<table align="right" width="78%" cellspacing="0" cellpadding="0"><tr><td>'
                f'<div style="margin:6px 0 14px;padding:10px 14px;'
                f'background-color:#0d2f42;color:{text};'
                f'border:none;border-radius:14px;">'
                f'{body}'
                + (f'<span style="color:#3a8a9a;font-size:8pt;">{copy_chip} · {ts}</span>'
                   if ts else copy_chip)
                + '</div>'
                f'</td></tr></table>'
            )

        if message["tag"] == "ai":
            # Flat assistant block: bold accent label, content directly on
            # the background — no card chrome (ChatGPT/Claude style).
            ts_chip = (f'<span style="color:#3a8a9a;font-size:8pt;"> · {ts}</span>'
                       if ts else "")
            return (
                f'<p style="margin:12px 0 2px 0;">'
                f'{self._ai_label_html()}{ts_chip}{copy_chip}</p>'
                f'<div style="margin:0 0 4px 2px;color:#d4f4fd;">{body}</div>'
            )

        if message["tag"] in ("sys", "tool", "file"):
            # Minimal operational row: dim dot + mono label, no chrome.
            prefix_label = message.get("prefix") or label
            header = (
                f'<span style="color:{accent};font-weight:600;'
                f'font-family:Cascadia Code, Consolas, monospace;font-size:8pt;'
                f'letter-spacing:0.6px;">{prefix_label}</span>'
                + (f'<span style="color:#3a8a9a;font-size:8pt;"> · {ts}</span>' if ts else "")
            )
            return (
                f'<p style="margin:5px 0 2px 0;color:#7fa3b0;font-size:9pt;">'
                f'{header} {body}</p>'
            )

        # ERROR: slim red-tint row — visible, never shouty.
        return (
            f'<div style="margin:6px 0 10px;padding:6px 10px;'
            f'background-color:#2a1118;border-left:3px solid {accent};'
            f'border-radius:3px;color:{text};font-size:9pt;">'
            f'{body}</div>'
        )

    def _cache_key(self, message: dict) -> str:
        return _sha(f"{message.get('tag')}|{message.get('ts','')}|{message.get('body','')}")

    # ── Rendering ──────────────────────────────────────────────────────

    def _append_message(self, msg: dict):
        """Fast path: append one card without re-laying-out the document.

        End-anchored live cards (typing indicator, streaming draft, action
        bar) are scan-removed by their invisible markers and re-appended
        after the new card — O(tail blocks), never a full-document rebuild.
        """
        scroll = self.verticalScrollBar()
        at_bottom = scroll.maximum() - scroll.value() <= 60
        if scroll.maximum() <= 0:
            at_bottom = True
        # Telegram-style: sending your own message means you want to follow
        # it — always jump to the bottom, even from deep history browsing.
        _is_own_send = msg.get("tag") == "you"
        if not at_bottom and not _is_own_send:
            # Reader is browsing history: rebuild (typing card excluded —
            # _full_render only renders _messages) and count the miss.
            self._full_render()
            self._unread += 1
            self._unread_changed_fab()
            return
        card = self._card_html(msg, len(self._messages) - 1)
        self._html_cache[self._cache_key(msg)] = card
        # Lift any live cards off the tail, insert the message, then drop
        # them back so the newest content is always visible above them.
        had_typing = self._typing
        had_streaming = self._streaming_active
        # Always lift the action bar if it's in the document — the enabled
        # check below sees the message ALREADY appended, so presence is the
        # correct source of truth for lifting.
        had_actions = self._remove_end_card(_ACTION_MARK)
        if had_typing:
            self._remove_typing_card()
        if had_streaming:
            self._remove_streaming_card()
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._insert_html_block(cursor, card)
        if had_typing:
            self._insert_html_block(
                cursor, _TYPING_CARD.format(dots=_TYPING_FRAMES[self._typing_frame]))
        if had_streaming:
            self._insert_html_block(cursor, self._streaming_card_html(self._streaming_text))
        # Re-add the bar only when the new tail message is an AI reply —
        # including the FIRST AI reply (bar was never present before).
        if self._action_bar_enabled:
            self._append_action_bar()
        self._snap_to_bottom()
        self._deferred_snaps(1, 0)

    def _unread_changed_fab(self):
        """Notify the host window that unread state changed (FAB badge)."""
        handler = getattr(self, "_on_unread_changed_cb", None)
        if handler:
            try:
                handler(self._unread)
            except Exception:
                pass

    def _full_render(self):
        scroll = self.verticalScrollBar()
        at_bottom = scroll.maximum() - scroll.value() <= 60
        if scroll.maximum() <= 0:
            at_bottom = True
        saved_ratio = (
            scroll.value() / scroll.maximum() if scroll.maximum() > 0 else 1.0
        )

        # Only the visible tail is rendered; the DB keeps the full history.
        start = max(0, len(self._messages) - _MAX_RENDERED_MESSAGES)
        chunks: list[str] = []

        # F6: when the DB holds older turns than this window shows, a top
        # sentinel offers to load them instead of silently truncating.
        hidden_older = self._total_messages - len(self._messages)
        if hidden_older > 0 and self._load_older_cb is not None:
            chunks.append(self._older_sentinel_html(hidden_older))

        for offset, message in enumerate(self._messages[start:]):
            index = start + offset
            key = self._cache_key(message)
            html = self._html_cache.get(key)
            if html is None:
                html = self._card_html(message, index)
                self._html_cache[key] = html
            chunks.append(html)

        # F7: follow-up actions under the newest assistant card — only when
        # the conversation ends with an AI reply and a handler is wired.
        if (getattr(self, "_on_action_cb", None) is not None
                and self._messages
                and self._messages[-1].get("tag") == "ai"):
            chunks.append(self._action_bar_html())

        # Evict entries that fell out of the render window (M1).
        if len(self._messages) > _MAX_RENDERED_MESSAGES and len(self._html_cache) > _MAX_RENDERED_MESSAGES * 2:
            self._html_cache.clear()

        self.setHtml("<html><body>" + "".join(chunks) + "</body></html>")

        if at_bottom:
            cursor = self.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.setTextCursor(cursor)
            self.ensureCursorVisible()
            self._deferred_snaps(1, 0)
        else:
            def _restore():
                try:
                    bar = self.verticalScrollBar()
                    if bar.maximum() > 0:
                        bar.setValue(int(saved_ratio * bar.maximum()))
                except RuntimeError:
                    pass  # widget destroyed during the deferred restore
            QTimer.singleShot(0, _restore)

    # ── Lazy history paging (F6) ─────────────────────────────────────

    def set_history_window(self, entries: list[str], timestamps: list[str] | None,
                           total_messages: int, load_older_cb=None):
        """Show the newest page of a longer history.

        ``total_messages`` is the authoritative count held by the DB;
        ``load_older_cb(before_idx, count)`` returns older raw entries.
        """
        self._total_messages = max(total_messages, len(entries))
        self._load_older_cb = load_older_cb
        self.set_logs_with_timestamps(entries, timestamps)

    def _older_sentinel_html(self, hidden: int) -> str:
        return (
            f'<div style="margin:0 0 8px;padding:6px;text-align:center;">'
            f'<a href="older://{min(hidden, _OLDER_PAGE)}">'
            f'<span style="font-size:9pt;color:#3a8a9a;text-decoration:none;">'
            f'↑ Load older messages ({hidden} earlier in history)</span></a></div>'
        )

    # ── Inline follow-up actions (F7) ───────────────────────────────

    @property
    def _action_bar_enabled(self) -> bool:
        return (getattr(self, "_on_action_cb", None) is not None
                and bool(self._messages)
                and self._messages[-1].get("tag") == "ai")

    def _action_bar_html(self) -> str:
        """Regenerate / Continue chips under the newest assistant card."""
        def _chip(href: str, label: str) -> str:
            return (
                f'<a href="{href}">'
                f'<span style="font-size:8pt;font-family:Cascadia Code,Consolas,'
                f'monospace;color:#3a8a9a;text-decoration:none;'
                f'border:1px solid #1d5a75;border-radius:3px;padding:2px 8px;'
                f'background-color:#061a24;">{label}</span></a>'
            )
        # Invisible end-anchor so the fast path can lift/move this bar.
        return (
            f'<p style="margin:0 0 10px;">'
            + _chip("action://regenerate", "↻ Regenerate")
            + "&nbsp;&nbsp;"
            + _chip("action://continue", "▸ Continue")
            + f'<span style="font-size:1pt;color:#071018;">{_ACTION_MARK}</span>'
            + "</p>"
        )

    def _append_action_bar(self):
        """Drop a fresh action bar at the document end (fast path helper)."""
        if not self._action_bar_enabled:
            return
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._insert_html_block(cursor, self._action_bar_html())

    def _load_older_page(self):
        if self._loading_older or self._load_older_cb is None:
            return
        hidden = self._total_messages - len(self._messages)
        if hidden <= 0:
            return
        count = min(hidden, _OLDER_PAGE)
        self._loading_older = True
        try:
            result = self._load_older_cb(len(self._messages), count)
        except Exception:
            result = None
        self._loading_older = False
        # Callback may return entries only (legacy) or (entries, timestamps).
        if isinstance(result, tuple) and len(result) == 2:
            older, older_ts = (result[0] or []), (result[1] or [])
        else:
            older, older_ts = (result or []), None
        if not older:
            # Nothing more to fetch: drop the sentinel permanently.
            self._total_messages = len(self._messages)
            self._full_render()
            return
        parsed = [self._parse_entry(e) for e in older]
        if older_ts:
            for msg, ts in zip(parsed, older_ts):
                if ts:
                    msg["ts"] = ts
        self._messages = parsed + self._messages
        self._html_cache.clear()  # index-addressed chips must re-render
        self._full_render()

    def _snap_to_bottom(self):
        """Synchronous bottom set. Callers that may race pending layout
        add their own deferred (guarded) passes — this stays pure."""
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _deferred_snaps(self, count: int = 1, delay_ms: int = 0):
        """Queue guarded bottom-follow passes that survive widget death."""
        def _safe_snap():
            try:
                bar = self.verticalScrollBar()
                bar.setValue(bar.maximum())
            except RuntimeError:
                pass  # widget destroyed — nothing to scroll

        for _ in range(count):
            QTimer.singleShot(delay_ms, _safe_snap)

    def scroll_to_bottom_hard(self):
        """Bullet-proof scroll to the true bottom (FAB click path).

        ``_snap_to_bottom`` reads ``maximum()`` synchronously, which is
        stale while the document still has pending layout (typing ticker,
        streaming inserts).  This three-pass variant is layout-safe:
          1. cursor-based scroll (independent of scrollbar math),
          2. deferred pass once the event loop flushes pending layout,
          3. a late pass for slow images / late settling.
        """
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()
        self._snap_to_bottom()
        self._deferred_snaps(2, 0)
        self._deferred_snaps(1, 120)
    # Back-compat: tests and callers may call _render directly.
    def _render(self):
        self._full_render()
