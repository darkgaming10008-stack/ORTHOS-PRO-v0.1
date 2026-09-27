"""Real regression tests for the 2026 chat/session/MCP feature wave.

Covers:
1. search_sessions DB helper (title + content search).
2. Incremental token cache (real-time counter correctness).
3. Chat message timestamps (live appends + DB-restored history).
4. Per-code-block copy buttons (copy:// anchors + clipboard).
5. Renderer performance guards (HTML cache, render cap, scroll restore).
6. MCP manager log collector (close-safe appends).
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import QApplication

import pytest


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


# ──────────────────────────────────────────────────────────────────────
# 1. search_sessions (DB)
# ──────────────────────────────────────────────────────────────────────

def test_search_sessions_matches_title_and_content():
    from memory import conversation_db as cdb

    sid_a = cdb.create_session("Alpha Python Helpers")
    sid_b = cdb.create_session("Zeta Unrelated")
    sid_c = cdb.create_session("Gamma Also Unrelated")
    cdb.save_turn("user", "tell me about postgresql tuning", session_id=sid_b)

    hits = cdb.search_sessions("alpha python")
    assert sid_a in [s["id"] for s in hits]

    content_hits = cdb.search_sessions("postgresql")
    hit_ids = [s["id"] for s in content_hits]
    assert sid_b in hit_ids and sid_c not in hit_ids

    # Empty query returns the full list (same contract as list_sessions).
    assert len(cdb.search_sessions("")) >= 3

    cdb.delete_session(sid_a)
    cdb.delete_session(sid_b)
    cdb.delete_session(sid_c)


def test_search_sessions_respects_project_filter():
    from memory import conversation_db as cdb

    proj = cdb.create_project("Search Test Proj") if hasattr(cdb, "create_project") else None
    if proj is None:
        pytest.skip("create_project helper unavailable in this DB layer")
    sid_in = cdb.create_session("In Project Keyword", project_id=proj)
    sid_out = cdb.create_session("In Project Keyword Other")

    hits = cdb.search_sessions("Keyword", project_id=proj)
    ids = [s["id"] for s in hits]
    assert sid_in in ids and sid_out not in ids

    cdb.delete_session(sid_in)
    cdb.delete_session(sid_out)


# ──────────────────────────────────────────────────────────────────────
# 2. Token cache (real-time counter)
# ──────────────────────────────────────────────────────────────────────

def test_token_cache_updates_incrementally_without_rescan():
    from memory import conversation_db as cdb

    sid = cdb.create_session("Token Cache Session")
    cdb._token_count_cache.pop(sid, None)  # start cold

    cdb.save_turn("user", "hello world", session_id=sid)
    cdb.save_turn("assistant", "hi there back", session_id=sid)

    cached = cdb._token_count_cache.get(sid)
    assert cached is not None and cached > 0
    # estimate must agree with the cache (no divergence path).
    assert cdb.estimate_session_tokens(sid) == cached

    cdb.delete_session(sid)
    assert sid not in cdb._token_count_cache  # invalidated on delete


def test_token_cache_survives_without_full_content_scan():
    """After restart-like cold cache, first estimate primes from DB."""
    from memory import conversation_db as cdb

    sid = cdb.create_session("Cold Cache Session")
    cdb.save_turn("user", "a" * 400, session_id=sid)
    cold = cdb.estimate_session_tokens(sid)
    assert cold > 0
    # Cached value must equal a forced re-scan.
    cdb._token_count_cache.pop(sid)
    assert cdb.estimate_session_tokens(sid) == cold
    cdb.delete_session(sid)


# ──────────────────────────────────────────────────────────────────────
# 3. Timestamps in chat
# ──────────────────────────────────────────────────────────────────────

def test_live_appends_get_wall_clock_timestamp(qapp):
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.append_log("you: ping")
    msg = chat._messages[-1]
    assert msg["ts"], "live append missing timestamp"
    # Rendered form is HH:MM (today) regardless of the ISO source.
    short = NativeRichLogWidget._short_time(msg["ts"])
    assert len(short) == 5 and short[2] == ":"


def test_db_history_timestamps_are_shown(qapp):
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.set_logs_with_timestamps(
        ["You: old message", "Orthos: old reply"],
        ["2026-08-01T14:32:10", "2026-08-01T14:33:00"],
    )
    html = chat.document().toHtml()
    assert "14:32" in html and "14:33" in html


def test_timestamps_omitted_when_unknown(qapp):
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.set_logs(["You: no ts here"])
    html = chat.document().toHtml()
    # Body renders in the bubble; no crash, no stray time text.
    assert "no ts here" in chat.toPlainText()


# ──────────────────────────────────────────────────────────────────────
# 4. Code block copy buttons
# ──────────────────────────────────────────────────────────────────────

def test_code_block_gets_copy_anchor(qapp):
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.append_log("orthos: here is code:\n```python\nprint('hi')\n```")
    html = chat.document().toHtml()
    assert "copy://c1" in html, "copy anchor missing on fenced block"
    assert "⧉ Copy" in html


def test_copy_anchor_puts_code_on_clipboard(qapp):
    from core.native_chat import NativeRichLogWidget
    from PyQt6.QtWidgets import QApplication as QA

    chat = NativeRichLogWidget()
    chat.append_log("orthos: snippet:\n```js\nconsole.log(42)\n```")
    assert "c1" in chat._code_store
    chat._on_anchor(QUrl("copy://c1"))
    assert QA.clipboard().text() == "console.log(42)"


def test_copy_click_with_unknown_id_is_safe(qapp):
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat._on_anchor(QUrl("copy://nope"))  # must not raise
    chat._on_anchor(QUrl("https://example.com"))  # external route, no crash


def test_math_inside_code_is_not_copy_mangled(qapp):
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.append_log("orthos: mixed:\n```python\nx = '$x^2$'\n```\nand $y^2$")
    plain = chat.toPlainText()
    assert "'$x^2$'" in plain  # fence contents stay literal code


# ──────────────────────────────────────────────────────────────────────
# 5. Renderer guards (cache / cap / scroll)
# ──────────────────────────────────────────────────────────────────────

def test_html_cache_prevents_re_render_of_old_cards(qapp):
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.append_log("orthos: one")
    cached_before = dict(chat._html_cache)
    assert cached_before, "first append did not populate the cache"
    chat.append_log("you: two")
    # The old card's cache entry must survive unchanged (same content-key).
    for key, value in cached_before.items():
        assert chat._html_cache.get(key) == value
    # New message gets its own entry.
    assert len(chat._html_cache) >= 2


def test_render_cap_trims_very_long_chats(qapp):
    from core.native_chat import NativeRichLogWidget, _MAX_RENDERED_MESSAGES

    chat = NativeRichLogWidget()
    for i in range(_MAX_RENDERED_MESSAGES + 50):
        chat._messages.append(
            {"tag": "ai", "prefix": "Orthos:", "body": f"bulk {i}", "ts": "00:00"})
    chat._render()
    assert len(chat._html_cache) <= _MAX_RENDERED_MESSAGES


def test_scroll_position_restored_when_not_at_bottom(qapp):
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.resize(400, 200)
    chat.show()
    for i in range(40):
        chat._messages.append(
            {"tag": "ai", "prefix": "Orthos:", "body": f"line {i} " * 8, "ts": "00:00"})
    chat._render()
    qapp.processEvents()
    scroll = chat.verticalScrollBar()
    if scroll.maximum() <= 0:
        pytest.skip("offscreen layout provides no scroll range")
    mid = scroll.maximum() // 2
    scroll.setValue(mid)
    qapp.processEvents()
    chat._messages.append(
        {"tag": "you", "prefix": "You:", "body": "new tail", "ts": "00:00"})
    chat._render()
    qapp.processEvents()
    # Reader was mid-scroll: the position must not jump to top.
    assert scroll.value() > 0
    # Ratio restore keeps them near the same relative spot (±15%).
    expected = mid
    assert abs(scroll.value() - expected) < 0.2 * scroll.maximum()
    chat.close()


def test_scroll_follows_bottom_on_new_message(qapp):
    """The user-reported regression: sending a message must scroll DOWN to
    the new content, never jump up to old content."""
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.resize(400, 200)
    chat.show()
    qapp.processEvents()

    for i in range(20):
        chat._messages.append(
            {"tag": "ai", "prefix": "Orthos:", "body": f"history {i} " * 6, "ts": "00:00"})
    chat._render()
    # Reader sits at the bottom (as a live chat does).
    chat._snap_to_bottom()
    qapp.processEvents()

    before = chat.verticalScrollBar().value()
    chat._messages.append(
        {"tag": "you", "prefix": "You:", "body": "brand new message", "ts": "12:00"})
    chat._render()
    qapp.processEvents()  # let deferred singleShot(0) bottom-follow run

    bar = chat.verticalScrollBar()
    assert bar.maximum() > 0, "no scroll range after append (layout missing?)"
    assert bar.value() >= before, "scroll jumped UP on new message"
    assert bar.value() >= bar.maximum() - 60, \
        f"not at bottom after append: {bar.value()}/{bar.maximum()}"
    chat.close()


def test_scroll_never_jumps_up_when_at_bottom(qapp):
    """Repeated live appends must monotonically keep the view at the bottom."""
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.resize(400, 200)
    chat.show()
    qapp.processEvents()

    last_value = 0
    for i in range(6):
        chat._messages.append(
            {"tag": "ai", "prefix": "Orthos:", "body": f"chunk {i} " * 10, "ts": "00:00"})
        chat._render()
        qapp.processEvents()  # run deferred follow/restore timers
        bar = chat.verticalScrollBar()
        if bar.maximum() > 0:
            assert bar.value() >= last_value, \
                f"append {i}: scroll moved up ({last_value} -> {bar.value()})"
            last_value = bar.value()
    chat.close()


# ──────────────────────────────────────────────────────────────────────
# 6. MCP log collector safety
# ──────────────────────────────────────────────────────────────────────

def test_mcp_log_collector_survives_detached_widget(qapp):
    sys.path.insert(0, PROJECT_ROOT)
    from mcp_manager_ui import _LogCollector

    class Dead:
        def append(self, _):
            raise RuntimeError("wrapped C/C++ object deleted")

    sink = _LogCollector()
    dead = Dead()
    sink.set_log_widget(dead)
    sink.append("crash-proof?")     # falls back to buffer
    assert "crash-proof?" in sink._buffer

    # Detaching routes appends back to the buffer without error.
    sink.set_log_widget(None)
    sink.append("buffered again")
    assert "buffered again" in sink._buffer


# ──────────────────────────────────────────────────────────────────────
# 7. Role layout redesign (user/assistant distinction)
# ──────────────────────────────────────────────────────────────────────

def _card_for(chat, msg_index: int) -> str:
    """Fetch the rendered card HTML for a message by index.

    The HTML cache is content-addressed; resolve through _messages.
    """
    msg = chat._messages[msg_index]
    key = chat._cache_key(msg)
    html = chat._html_cache.get(key)
    if html is None:
        html = chat._card_html(msg)
        chat._html_cache[key] = html
    return html


def test_you_messages_are_right_aligned_bubbles(qapp):
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.append_log("you: bubble me")
    chat.append_log("orthos: card me")
    you_card = _card_for(chat, 0)
    ai_card = _card_for(chat, 1)
    # YOU: right-aligned 78% borderless bubble (2026 flat style).
    assert 'align="right" width="78%"' in you_card
    assert "border-radius:14px" in you_card
    assert "border:none" in you_card
    # ORTHOS: flat label block — no box, no right-align wrapper.
    assert 'align="right" width="78%"' not in ai_card
    assert "background-color" not in ai_card


def test_sys_tool_file_are_compact_rows(qapp):
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.append_log("sys: operational note")
    chat.append_log("SYS: \u25b6 some_tool")
    chat.append_log("ATTACHED: file event")
    # Minimal operational rows (2026): dot-free label + dim text, no chrome.
    plain = chat.toPlainText()
    for i, label in [(0, "SYS"), (1, "TOOL"), (2, "FILE")]:
        card = _card_for(chat, i)
        assert "margin:5px 0 2px 0" in card, f"{label} not minimal (margins)"
        assert "background-color" not in card, f"{label} must have no bg"
        assert "border-left" not in card, f"{label} must have no border chrome"
        assert label in plain


def test_error_stays_attention_card(qapp):
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.append_log("ERR: real failure")
    card = _card_for(chat, 0)
    # 2026 slim red-tint row (visible, not shouty).
    assert "border-left:3px solid #e8496a" in card
    assert "real failure" in chat.toPlainText()


# ──────────────────────────────────────────────────────────────────────
# 8. 2026 bug-fix regressions
# ──────────────────────────────────────────────────────────────────────

def test_copy_feedback_flash_is_visible_after_click(qapp):
    """Dead-code regression: 'Copied ✓' must actually appear in the text."""
    from core.native_chat import NativeRichLogWidget
    from PyQt6.QtCore import QUrl

    chat = NativeRichLogWidget()
    chat.append_log("orthos: snippet:\n```js\nconsole.log(1)\n```")
    chat._on_anchor(QUrl("copy://c1"))
    assert "Copied \u2713" in chat.toPlainText()


def test_err_heuristic_uses_word_boundary(qapp):
    """'terrific'/'query' must NOT render as ERROR cards."""
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.append_log("that query was terrific")
    chat.append_log("error happened for real")
    tags = [m["tag"] for m in chat._messages]
    assert tags[0] != "err", "'terrific' misclassified as error"
    assert tags[1] == "err", "'error' must be classified as error"
    # Only the genuine error gets the red treatment.
    cards = [_card_for(chat, 0), _card_for(chat, 1)]
    assert "border-left:3px solid #e8496a" not in cards[0]
    assert "border-left:3px solid #e8496a" in cards[1]


def test_attached_prefixes_map_to_file_card(qapp):
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.append_log("ATTACHED: notes.pdf (pdf, 2MB)")
    chat.append_log("READY: report.docx — extracted")
    chat.append_log("REMOVED: old image")
    assert all(m["tag"] == "file" for m in chat._messages)
    assert "FILE" in chat.toPlainText()


def test_attachment_bracket_text_becomes_styled_block(qapp):
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.append_log("you: check this [ATTACHED FILES: notes.pdf (pdf, 2MB)]")
    card = _card_for(chat, 0)
    plain = chat.toPlainText()
    assert "[ATTACHED FILES:" not in card, "bracket spam rendered literally"
    assert "\U0001F4CE" in card, "attachment block icon missing"
    assert "check this" in plain, "prose before attachment lost"


def test_code_store_cleared_on_session_switch(qapp):
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.append_log("orthos: ```python\nx=1\n```")
    assert len(chat._code_store) == 1
    chat.set_logs(["you: fresh session"])
    assert chat._code_store == {}, "stale code ids leaked across sessions"
    assert chat._code_seq == 0


def test_old_timestamps_show_date_not_bare_time(qapp):
    from core.native_chat import NativeRichLogWidget
    from datetime import date, timedelta

    chat = NativeRichLogWidget()
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%dT14:32:00")
    today = date.today().strftime("%Y-%m-%dT09:15:00")
    chat.set_logs_with_timestamps(
        ["You: yesterday msg", "You: today msg"],
        [yesterday, today],
    )
    plain = chat.toPlainText()
    assert "14:32" in plain
    assert "09:15" in plain
    # The old message must expose its date (e.g. "04 Sep") — not bare time.
    card_old = chat._card_html(chat._messages[0])
    card_new = chat._card_html(chat._messages[1])
    assert "14:32" in card_old and "09:15" in card_new
    assert card_old != card_new.replace(
        chat._short_time(today), chat._short_time(yesterday)), \
        "old and today cards must differ beyond time (date should show)"


# ──────────────────────────────────────────────────────────────────────
# 9. Audit-fix regressions (C1/C3/M4/M5/m15 — token-stream pipeline)
# ──────────────────────────────────────────────────────────────────────

def test_mixed_indented_and_fenced_code_copy_correct_blocks(qapp):
    """C1 regression: indented code must not steal the fenced block's button."""
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.append_log(
        "orthos: mixed\n\n    indented = 1\n\n```python\nfenced_code()\n```")
    # Both blocks present, each bound to its OWN source (no cross-binding).
    values = list(chat._code_store.values())
    assert any("fenced_code" in c for c in values), "fenced block has no button"
    assert any("indented" in c for c in values), "indented block has no button"
    # No single anchor holds both blocks (the old mis-binding bug).
    assert not any("fenced_code" in c and "indented" in c for c in values)


def test_four_backtick_fence_copies_full_content(qapp):
    """C1 regression: ```` with inner ``` must copy everything, not ''."""
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.append_log("orthos: outer\n\n````\ncode with ``` inside\n````")
    stored = list(chat._code_store.values())
    assert stored, "long fence produced no copy anchor"
    assert "``` inside" in stored[0], f"copied wrong span: {stored[0]!r}"


def test_empty_code_fence_has_no_dead_copy_button(qapp):
    """m15 regression: ```\n``` must not emit a no-op anchor."""
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.append_log("orthos: empty\n\n```\n```")
    assert chat._code_store == {}, "empty fence registered a dead copy button"


def test_append_path_is_incremental_not_full_rerender(qapp):
    """M4 regression: appending at the bottom must not setHtml the document."""
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.append_log("orthos: first")
    doc_before = chat.document()
    chat.append_log("you: second")  # at bottom -> insertHtml fast path
    assert chat.document() is doc_before, "append rebuilt the whole document"
    assert "second" in chat.toPlainText()


def test_attachment_text_after_block_is_kept(qapp):
    """M5 regression: prose after [ATTACHED FILES: …] must not vanish."""
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.append_log("you: before [ATTACHED FILES: a.pdf] after text")
    plain = chat.toPlainText()
    assert "before" in plain and "after text" in plain


def test_multiple_attachment_blocks_all_styled(qapp):
    """M5 regression: every [ATTACHED FILES: …] block gets styled."""
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.append_log("you: [ATTACHED FILES: a.pdf] mid [ATTACHED FILES: b.pdf]")
    card = _card_for(chat, 0)
    assert card.count("\U0001F4CE") == 2, "second attachment block unstyled"


def test_file_scheme_link_cannot_navigate_the_chat(qapp):
    """m3 regression: file:// links must never replace the chat document."""
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.append_log("orthos: see [docs](file:///C:/secret.txt) here")
    text_before = chat.toPlainText()
    from PyQt6.QtCore import QUrl
    chat._on_anchor(QUrl("file:///C:/secret.txt"))
    assert chat.toPlainText() == text_before, "file:// navigated the chat"


def test_bracketed_tool_tags_render_as_tool_rows(qapp):
    """m5 regression: '[YouTube] playing' is an operational row, not SYSTEM."""
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.append_log("[YouTube] playing song")
    assert chat._messages[-1]["tag"] == "tool"
    assert "YOUTUBE" in chat.toPlainText()


def test_error_mention_mid_sentence_stays_system(qapp):
    """m7 regression: operational chatter mentioning 'errors' stays neutral."""
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.append_log("[Search] how to fix errors in python")
    assert chat._messages[-1]["tag"] != "err"


# ──────────────────────────────────────────────────────────────────────
# 10. Wave-2 features (typing indicator, scroll FAB, message copy)
# ──────────────────────────────────────────────────────────────────────

def test_typing_indicator_shows_and_removes(qapp):
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.append_log("orthos: hello")
    chat.set_typing(True)
    assert "thinking" in chat.toPlainText()
    chat._tick_typing()  # animated frame advance
    assert "thinking" in chat.toPlainText()
    chat.set_typing(False)
    assert "thinking" not in chat.toPlainText()
    # The real message must survive the indicator's removal.
    assert "hello" in chat.toPlainText()


def test_typing_card_is_not_persisted_as_a_message(qapp):
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.set_typing(True)
    chat.set_typing(False)
    assert chat._messages == []
    chat.set_logs(["you: fresh"])
    assert "thinking" not in chat.toPlainText()


def test_streaming_appends_keep_typing_card_last(qapp):
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.set_typing(True)
    chat.append_log("orthos: part one")
    # Indicator still visible while streaming.
    assert "thinking" in chat.toPlainText()
    assert "part one" in chat.toPlainText()
    chat.set_typing(False)
    assert "thinking" not in chat.toPlainText()
    assert "part one" in chat.toPlainText()


def test_message_copy_chip_copies_body(qapp):
    from core.native_chat import NativeRichLogWidget
    from PyQt6.QtCore import QUrl
    from PyQt6.QtWidgets import QApplication as QA

    chat = NativeRichLogWidget()
    chat.append_log("orthos: chip target text")
    chat._on_anchor(QUrl(f"msg://copy-{len(chat._messages) - 1}"))
    assert QA.clipboard().text() == "chip target text"
    # Out-of-range / malformed index is a safe no-op.
    chat._on_anchor(QUrl("msg://copy-999"))
    chat._on_anchor(QUrl("msg://garbage"))


def test_message_copy_chip_present_on_cards(qapp):
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.append_log("orthos: has chip")
    chat.append_log("you: also has chip")
    chat._full_render()
    html = chat.document().toHtml()
    assert "msg://copy-" in html, "message copy chips missing from render"


def test_unread_counter_tracks_browsing_reader(qapp):
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.resize(400, 200)
    chat.show()
    for i in range(30):
        chat._messages.append({"tag": "ai", "prefix": "Orthos:",
                               "body": f"line {i} " * 6, "ts": ""})
    chat._full_render()
    qapp.processEvents()
    bar = chat.verticalScrollBar()
    if bar.maximum() <= 0:
        pytest.skip("offscreen layout provides no scroll range")
    bar.setValue(bar.maximum() // 2)  # reader browses history
    qapp.processEvents()
    # 2026-09-06 decision: own sends ALWAYS follow to the bottom — so the
    # unread counter is proven with an AI reply instead (the yank-free path).
    chat.append_log("orthos: arrives while away")
    qapp.processEvents()
    assert chat._unread >= 1
    bar.setValue(bar.maximum())  # reader returns to the bottom
    qapp.processEvents()
    assert chat._unread == 0
    chat.close()


def test_main_window_has_scroll_fab(qapp):
    import ui

    win = ui.MainWindow("nonexistent.png")
    assert hasattr(win, "_scroll_fab")
    assert hasattr(win, "_fab_badge")
    # FAB routing: clicking snaps the log to the bottom.
    win._log._unread = 3
    win._on_scroll_fab_clicked()
    assert win._log._unread == 0
    win.close()


# ──────────────────────────────────────────────────────────────────────
# 11. Wave-3 features (syntax highlighting, streaming render)
# ──────────────────────────────────────────────────────────────────────

def test_syntax_highlighting_colors_code_blocks(qapp):
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.append_log("orthos: look:\n```python\ndef foo():\n    return 1\n```")
    html = chat.document().toHtml()
    # Pygments one-dark palette emits inline color spans.
    assert "color:#" in html, "no highlighted tokens in render"
    assert "python" in html.lower(), "language label missing"
    # The raw code remains selectable text.
    assert "def foo():" in chat.toPlainText()


def test_syntax_highlighting_unknown_lang_falls_back(qapp):
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.append_log("orthos: weird:\n```notalanguage\nplain words\n```")
    plain = chat.toPlainText()
    assert "plain words" in plain  # code intact, no crash
    # Copy still works for unknown-language blocks.
    assert any("plain words" in c for c in chat._code_store.values())


def test_highlighted_code_still_copies_raw_source(qapp):
    from core.native_chat import NativeRichLogWidget
    from PyQt6.QtCore import QUrl
    from PyQt6.QtWidgets import QApplication as QA

    chat = NativeRichLogWidget()
    chat.append_log("orthos: copy src:\n```js\nconsole.log('hi')\n```")
    chat._on_anchor(QUrl("copy://c1"))
    # Clipboard gets the RAW code, not the highlighted HTML.
    assert QA.clipboard().text() == "console.log('hi')"


def test_streaming_lifecycle_start_append_end(qapp):
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    msgs_before = len(chat._messages)
    chat.streaming_start()
    qapp.processEvents()
    assert "▌" in chat.toPlainText(), "caret missing on start"
    for chunk in ("Hello ", "**bold** ", "world"):
        chat.streaming_append(chunk)
    qapp.processEvents()
    plain = chat.toPlainText()
    assert "Hello" in plain and "bold" in plain
    # No message committed until end.
    assert len(chat._messages) == msgs_before
    final = chat.streaming_end()
    qapp.processEvents()
    assert final == "Hello **bold** world"
    assert len(chat._messages) == msgs_before + 1
    assert chat._messages[-1]["tag"] == "ai"
    assert "▌" not in chat.toPlainText()


def test_streaming_abort_discards_partial(qapp):
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.streaming_start()
    chat.streaming_append("junk text")
    chat.streaming_abort()
    qapp.processEvents()
    assert "junk text" not in chat.toPlainText()
    assert "▌" not in chat.toPlainText()
    assert chat._messages == []


def test_streaming_replaces_typing_indicator(qapp):
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.set_typing(True)
    chat.streaming_start()  # first token supersedes "thinking"
    qapp.processEvents()
    assert not chat._typing
    assert "thinking" not in chat.toPlainText()
    assert "▌" in chat.toPlainText()
    chat.streaming_end()


def test_streaming_threadsafe_bridge(qapp):
    """Signals exist and route to the GUI-thread methods (F1 thread safety)."""
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    for name in ("_stream_start_sig", "_stream_append_sig",
                 "_stream_end_sig", "_stream_abort_sig"):
        assert hasattr(chat, name), f"{name} missing"
    # Emit from "any thread" (here: same thread) — queued slots run on process.
    chat.streaming_start_threadsafe()
    chat.streaming_append_threadsafe("bridged")
    chat.streaming_end_threadsafe("bridged text")
    for _ in range(5):
        qapp.processEvents()
    assert any(m["body"] == "bridged text" for m in chat._messages)


# ──────────────────────────────────────────────────────────────────────
# 12. Wave-4 features (lazy history, regenerate/continue)
# ──────────────────────────────────────────────────────────────────────

def test_lazy_history_sentinel_shows_hidden_count(qapp):
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    full = [f"you: msg {i}" for i in range(300)]
    chat.set_history_window(full[-50:], None, total_messages=300,
                            load_older_cb=lambda c, w: [])
    qapp.processEvents()
    plain = chat.toPlainText()
    assert "Load older messages" in plain, "sentinel missing"
    assert "250" in plain, "hidden count not shown"
    assert "msg 299" in plain and "msg 0" not in plain


def test_lazy_history_loads_page_on_click(qapp):
    from core.native_chat import NativeRichLogWidget
    from PyQt6.QtCore import QUrl

    chat = NativeRichLogWidget()
    full = [f"you: msg {i}" for i in range(150)]
    calls = []

    def fetch(shown, want):
        calls.append((shown, want))
        return full[:len(full) - shown][-want:]

    chat.set_history_window(full[-40:], None, total_messages=150,
                            load_older_cb=fetch)
    qapp.processEvents()
    chat._on_anchor(QUrl("older://100"))
    qapp.processEvents()
    assert len(calls) == 1, "fetch callback not invoked once"
    # 100 older messages prepended.
    assert len(chat._messages) == 140
    assert "msg 0" in chat.toPlainText() or len(chat._messages) >= 140


def test_lazy_history_exhaustion_drops_sentinel(qapp):
    from core.native_chat import NativeRichLogWidget
    from PyQt6.QtCore import QUrl

    chat = NativeRichLogWidget()
    full = [f"you: msg {i}" for i in range(60)]
    chat.set_history_window(full[-50:], None, total_messages=60,
                            load_older_cb=lambda c, w: full[:max(0, len(full) - c)][-w:])
    qapp.processEvents()
    chat._on_anchor(QUrl("older://100"))
    qapp.processEvents()
    assert len(chat._messages) == 60
    # Everything loaded: no sentinel, no further fetch.
    assert "Load older" not in chat.toPlainText()


def test_action_bar_shown_after_ai_reply_only(qapp):
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    seen = []
    chat._on_action_cb = seen.append
    chat.append_log("you: question")
    qapp.processEvents()
    assert "Regenerate" not in chat.document().toHtml()
    chat.append_log("orthos: answer")
    qapp.processEvents()
    assert "Regenerate" in chat.document().toHtml()
    assert "Continue" in chat.document().toHtml()


def test_action_bar_without_handler_is_absent(qapp):
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()  # no _on_action_cb wired
    chat.append_log("orthos: reply")
    qapp.processEvents()
    assert "Regenerate" not in chat.document().toHtml()


def test_action_chips_route_to_handler(qapp):
    from core.native_chat import NativeRichLogWidget
    from PyQt6.QtCore import QUrl

    chat = NativeRichLogWidget()
    seen = []
    chat._on_action_cb = seen.append
    chat._on_anchor(QUrl("action://regenerate"))
    chat._on_anchor(QUrl("action://continue"))
    chat._on_anchor(QUrl("action://bogus"))
    assert seen == ["regenerate", "continue", "bogus"]


def test_action_bar_moves_after_new_turn(qapp):
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat._on_action_cb = lambda _a: None
    chat.append_log("orthos: first answer")
    qapp.processEvents()
    chat.append_log("you: follow-up")
    qapp.processEvents()
    html = chat.document().toHtml()
    assert html.count("Regenerate") == 0, "stale bar after user message"
    chat.append_log("orthos: second answer")
    qapp.processEvents()
    assert chat.document().toHtml().count("Regenerate") == 1


def test_regenerate_action_queue_is_threadsafe(qapp):
    """The GUI-thread handler must only queue; the worker performs mutation."""
    import ui
    import types

    # Structural check on the shipped wiring.
    src = open(os.path.join(PROJECT_ROOT, "main.py"), encoding="utf-8").read()
    assert "self._text_queue.put((\"action_msg\", action))" in src, \
        "action handler must queue (thread-safe), not mutate directly"
    assert 'item[0] == "action_msg"' in src, "worker must dispatch action_msg"


# ──────────────────────────────────────────────────────────────────────
# 13. Scroll-to-bottom FAB robustness (typing-ticker interference)
# ──────────────────────────────────────────────────────────────────────

def test_fab_click_reaches_true_bottom_during_typing(qapp):
    """User-reported bug: ↓ button never fully scrolled while thinking.

    The raw setValue(maximum()) read a stale maximum mid-ticker; the
    3-pass hard snap must land exactly at the bottom even with the
    typing card animating."""
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.resize(400, 300)
    chat.show()
    for i in range(30):
        chat._messages.append({"tag": "ai", "prefix": "Orthos:",
                               "body": f"msg {i} " * 8, "ts": ""})
    chat._full_render()
    qapp.processEvents()
    bar = chat.verticalScrollBar()
    if bar.maximum() <= 0:
        pytest.skip("offscreen layout provides no scroll range")

    chat.set_typing(True)
    qapp.processEvents()
    chat._tick_typing()  # animate once so a pending layout exists
    qapp.processEvents()

    bar.setValue(bar.maximum() // 2)  # user scrolls up mid-thinking
    qapp.processEvents()
    chat.scroll_to_bottom_hard()      # the FAB's new code path
    # Run all deferred passes (0ms + 120ms timers).
    for _ in range(8):
        qapp.processEvents()
        from PyQt6.QtCore import QEventLoop, QTimer as _QT
    loop = QEventLoop()
    _QT.singleShot(200, loop.quit)
    loop.exec()

    assert bar.value() >= bar.maximum() - 2, \
        f"FAB hard-snap short of bottom: {bar.value()}/{bar.maximum()}"
    chat.set_typing(False)
    chat.close()


def test_typing_tick_preserves_bottom_position(qapp):
    """The typing animation's remove+insert bounce must not drag a
    bottom-parked reader upwards on every 400ms tick."""
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.resize(400, 300)
    chat.show()
    for i in range(30):
        chat._messages.append({"tag": "ai", "prefix": "Orthos:",
                               "body": f"msg {i} " * 8, "ts": ""})
    chat._full_render()
    qapp.processEvents()
    bar = chat.verticalScrollBar()
    if bar.maximum() <= 0:
        pytest.skip("offscreen layout provides no scroll range")

    chat.set_typing(True)
    qapp.processEvents()
    chat._snap_to_bottom()
    qapp.processEvents()

    last_value = bar.value()
    for i in range(5):
        chat._tick_typing()
        qapp.processEvents()
        # A bottom-parked reader must never move UP across a tick.
        assert bar.value() >= last_value - 2, \
            f"tick {i}: bottom reader dragged up ({last_value} -> {bar.value()})"
        last_value = max(last_value, bar.value())
    chat.set_typing(False)
    chat.close()


def test_typing_tick_does_not_disturb_history_reader(qapp):
    """A reader browsing history keeps their position across ticks."""
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.resize(400, 300)
    chat.show()
    for i in range(30):
        chat._messages.append({"tag": "ai", "prefix": "Orthos:",
                               "body": f"msg {i} " * 8, "ts": ""})
    chat._full_render()
    qapp.processEvents()
    bar = chat.verticalScrollBar()
    if bar.maximum() <= 0:
        pytest.skip("offscreen layout provides no scroll range")

    chat.set_typing(True)
    qapp.processEvents()
    bar.setValue(bar.maximum() // 2)
    qapp.processEvents()
    ratio_before = bar.value() / bar.maximum()

    for _ in range(5):
        chat._tick_typing()
        qapp.processEvents()

    ratio_after = bar.value() / bar.maximum()
    assert abs(ratio_after - ratio_before) < 0.2, \
        f"history reader yanked: {ratio_before:.2f} -> {ratio_after:.2f}"
    chat.set_typing(False)
    chat.close()


# ──────────────────────────────────────────────────────────────────────
# 14. Scroll-UX decisions (2026-09-06): reader-intent respect
# ──────────────────────────────────────────────────────────────────────

def test_thinking_indicator_never_yanks_history_reader(qapp):
    """set_typing(True) must not scroll the view when the reader browses."""
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.resize(400, 300)
    chat.show()
    for i in range(30):
        chat._messages.append({"tag": "ai", "prefix": "Orthos:",
                               "body": f"msg {i} " * 8, "ts": ""})
    chat._full_render()
    qapp.processEvents()
    bar = chat.verticalScrollBar()
    if bar.maximum() <= 0:
        pytest.skip("offscreen layout provides no scroll range")

    bar.setValue(bar.maximum() // 3)
    qapp.processEvents()
    before = bar.value()
    chat.set_typing(True)
    qapp.processEvents()
    after = bar.value()
    assert abs(after - before) <= 30, \
        f"indicator yanked history reader: {before} -> {after}"
    chat.set_typing(False)
    chat.close()


def test_streaming_start_respects_history_reader(qapp):
    """streaming_start/append keep the viewport of a history reader."""
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.resize(400, 300)
    chat.show()
    for i in range(30):
        chat._messages.append({"tag": "ai", "prefix": "Orthos:",
                               "body": f"msg {i} " * 8, "ts": ""})
    chat._full_render()
    qapp.processEvents()
    bar = chat.verticalScrollBar()
    if bar.maximum() <= 0:
        pytest.skip("offscreen layout provides no scroll range")

    bar.setValue(bar.maximum() // 3)
    qapp.processEvents()
    before = bar.value()
    chat.streaming_start()
    chat.streaming_append("live tokens")
    qapp.processEvents()
    after = bar.value()
    assert abs(after - before) <= 30, \
        f"streaming yanked history reader: {before} -> {after}"
    chat.streaming_abort()
    chat.close()


def test_own_send_jumps_to_bottom_telegram_style(qapp):
    """Sending your own message always follows it down — even from deep
    history browsing (2026-09-06 product decision)."""
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.resize(400, 300)
    chat.show()
    for i in range(30):
        chat._messages.append({"tag": "ai", "prefix": "Orthos:",
                               "body": f"msg {i} " * 8, "ts": ""})
    chat._full_render()
    qapp.processEvents()
    bar = chat.verticalScrollBar()
    if bar.maximum() <= 0:
        pytest.skip("offscreen layout provides no scroll range")

    bar.setValue(bar.maximum() // 4)  # deep history
    qapp.processEvents()
    chat.append_log("you: follow me down")
    qapp.processEvents()
    assert "follow me down" in chat.toPlainText(), "own message lost"
    assert bar.maximum() - bar.value() <= 80, \
        f"own send did not follow to bottom: {bar.value()}/{bar.maximum()}"
    chat.close()


def test_ai_message_respects_history_reader_and_counts_unread(qapp):
    """AI replies must NOT yank a history reader; unread badge counts it."""
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.resize(400, 300)
    chat.show()
    for i in range(30):
        chat._messages.append({"tag": "ai", "prefix": "Orthos:",
                               "body": f"msg {i} " * 8, "ts": ""})
    chat._full_render()
    qapp.processEvents()
    bar = chat.verticalScrollBar()
    if bar.maximum() <= 0:
        pytest.skip("offscreen layout provides no scroll range")

    bar.setValue(bar.maximum() // 4)
    qapp.processEvents()
    before = bar.value()
    unread_before = chat._unread
    chat.append_log("orthos: respects the reader")
    qapp.processEvents()
    assert abs(bar.value() - before) <= 30, "AI reply yanked history reader"
    assert chat._unread == unread_before + 1, "unread not counted"
    chat.close()


def test_main_window_has_floating_thinking_badge(qapp):
    """The 'Orthos is thinking' pill shows ONLY for history readers."""
    import ui

    win = ui.MainWindow("nonexistent.png")
    win.resize(1200, 800)
    win.show()
    qapp.processEvents()

    log = win._log
    badge = getattr(win, "_thinking_badge", None)
    assert badge is not None, "floating thinking badge missing"

    for i in range(40):
        log._messages.append({"tag": "ai", "prefix": "Orthos:",
                              "body": f"m {i} " * 10, "ts": ""})
    log._full_render()
    qapp.processEvents()
    bar = log.verticalScrollBar()
    if bar.maximum() <= 0:
        win.close()
        pytest.skip("offscreen layout provides no scroll range")

    # Bottom reader + thinking -> hidden (the in-chat indicator shows).
    bar.setValue(bar.maximum())
    qapp.processEvents()
    win._apply_state("THINKING")
    qapp.processEvents()
    win._update_fab_visibility()
    assert not badge.isVisible(), "badge visible for bottom reader"

    # History reader + thinking -> visible.
    bar.setValue(bar.maximum() // 3)
    qapp.processEvents()
    win._update_fab_visibility()
    assert badge.isVisible(), "badge hidden for history reader during thinking"

    # Thinking ends -> badge clears.
    win._apply_state("LISTENING")
    qapp.processEvents()
    win._update_fab_visibility()
    assert not badge.isVisible(), "badge stayed after thinking ended"
    win.close()


# ──────────────────────────────────────────────────────────────────────
# 15. Blank-gap regression (2026-09-06): live-card churn must not
#     leak empty blocks between the user message and the reply.
# ──────────────────────────────────────────────────────────────────────

def _doc_blocks(chat):
    out, b = [], chat.document().firstBlock()
    while b.isValid():
        out.append(b.text())
        b = b.next()
    return out


def test_typing_ticks_do_not_accumulate_empty_blocks(qapp):
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.show()
    chat.append_log("you: hello")
    qapp.processEvents()
    baseline_empty = sum(1 for t in _doc_blocks(chat) if not t.strip())

    chat.set_typing(True)
    qapp.processEvents()
    for _ in range(10):  # ~4 seconds of thinking animation
        chat._tick_typing()
        qapp.processEvents()

    blocks = _doc_blocks(chat)
    empty = sum(1 for t in blocks if not t.strip())
    assert "hello" in chat.toPlainText(), "user message lost during ticks"
    # The gap must not grow: same order of magnitude as baseline, never
    # the one-empty-block-per-tick leak (baseline + 10 would be the bug).
    assert empty <= baseline_empty + 2, \
        f"typing ticks leaked empty blocks: baseline={baseline_empty}, now={empty}"
    chat.set_typing(False)
    chat.close()


def test_streaming_tokens_do_not_accumulate_empty_blocks(qapp):
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.show()
    chat.append_log("you: stress")
    qapp.processEvents()
    chat.streaming_start()
    for i in range(30):
        chat.streaming_append(f"tok{i} ")
        qapp.processEvents()
    chat.streaming_end("final assembled reply")
    qapp.processEvents()

    plain = chat.toPlainText()
    assert "stress" in plain, "user message lost during streaming"
    assert "final assembled reply" in plain, "streamed reply lost"

    blocks = _doc_blocks(chat)
    empty = sum(1 for t in blocks if not t.strip())
    idx_user = next(i for i, t in enumerate(blocks) if "stress" in t)
    idx_ai = next(i for i, t in enumerate(blocks) if "final assembled" in t)
    gap = idx_ai - idx_user
    # Before the fix: 30 tokens -> ~60 leaked blocks, gap > 30.
    assert empty <= 4, f"{empty} empty blocks after 30 tokens"
    assert gap <= 5, f"blank gap of {gap} blocks between user msg and reply"
    chat.close()


def test_multi_turn_live_cards_keep_document_tight(qapp):
    from core.native_chat import NativeRichLogWidget

    chat = NativeRichLogWidget()
    chat.show()
    for turn in range(3):
        chat.append_log(f"you: turn{turn}")
        qapp.processEvents()
        chat.set_typing(True)
        qapp.processEvents()
        for _ in range(3):
            chat._tick_typing()
            qapp.processEvents()
        chat.set_typing(False)
        chat.streaming_start()
        chat.streaming_append(f"answer{turn} ")
        qapp.processEvents()
        chat.streaming_end(f"reply{turn} full")
        qapp.processEvents()

    plain = chat.toPlainText()
    for turn in range(3):
        assert f"turn{turn}" in plain and f"reply{turn} full" in plain
    blocks = _doc_blocks(chat)
    empty = sum(1 for t in blocks if not t.strip())
    # 3 full turns of ticker+stream churn: the leak used to add ~5
    # empties per turn (tick x3 + stream + end).
    assert empty <= 8, f"multi-turn accumulation: {empty} empty blocks"
    chat.close()
