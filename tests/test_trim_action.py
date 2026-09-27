"""Tests for the partial-trim conversation policy (2026-09-06 user decision).

Raw budget hits 50k -> summarize ONLY the oldest ~15k tokens; recent turns
stay raw in the prompt. Tool-call chains must never be split mid-pair.
"""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest


# ──────────────────────────────────────────────────────────────────────
# _oldest_trim_split: pure logic tests
# ──────────────────────────────────────────────────────────────────────

def _mk(role, content, **kw):
    m = {"role": role, "content": content}
    m.update(kw)
    return m


def test_split_respects_token_budget():
    from core.conversation_trim import oldest_trim_indices

    # 50 turns, ~250 tokens each ≈ 12,500 total. chunk target 15000 covers all.
    msgs = [_mk("user", "word " * 250) for _ in range(50)]
    idxs = oldest_trim_indices(msgs, 15000)
    assert len(idxs) == 50, "chunk must include everything under budget"


def test_trim_never_splits_tool_chain():
    """Trim boundary must never fall between assistant(tool_calls) and tool result."""
    from core.conversation_trim import oldest_trim_indices

    # A tool chain: user -> assistant(tool_calls) -> tool results
    msgs = [
        _mk("user", "question?"),
        _mk("assistant", "", tool_calls=[{"id": "tc9", "name": "run_terminal"}]),
        _mk("tool", "result 1", tool_call_id="tc9"),
        _mk("assistant", "answer"),
        _mk("user", "follow up"),
        _mk("assistant", "more answer"),
    ]
    idxs = oldest_trim_indices(msgs, chunk_tokens=300)
    # If a tool-call assistant gets trimmed, its tool result must be too.
    trimmed = {i for i in idxs}
    for i in trimmed:
        m = msgs[i]
        for tc in (m.get("tool_calls") or []):
            cid = tc.get("id") if isinstance(tc, dict) else None
            if cid:
                # find that tool result's index
                for j, other in enumerate(msgs):
                    if other.get("tool_call_id") == cid:
                        assert j in trimmed or j < i, \
                            f"tool_calls at {i} trimmed but its result at {j} left raw"


def test_trim_includes_tool_pairs_when_exceeding_budget():
    """When a tool chain is the only way past budget, include the full pair."""
    from core.conversation_trim import oldest_trim_indices

    big_text = "word " * 800  # ~800 tokens each
    msgs = [
        _mk("user", big_text),
        _mk("assistant", "", tool_calls=[{"id": "tc1", "name": "run_terminal"}]),
        _mk("tool", big_text, tool_call_id="tc1"),
        _mk("assistant", big_text),
        _mk("user", "q"),
        _mk("assistant", "a"),
    ]
    idxs = oldest_trim_indices(msgs, chunk_tokens=900)
    trimmed = set(idxs)
    # No orphaned tool result left in `rest`: every trimmed tool_calls keeps its result.
    for i in trimmed:
        for tc in (msgs[i].get("tool_calls") or []):
            cid = tc.get("id") if isinstance(tc, dict) else None
            if cid:
                for j, other in enumerate(msgs):
                    if other.get("tool_call_id") == cid:
                        assert j in trimmed, \
                            f"tool result {j} orphaned after trim of assistant {i}"


# ──────────────────────────────────────────────────────────────────────
# main.py trim-block wiring (structure assertions — heavy flow mocked)
# ──────────────────────────────────────────────────────────────────────

def test_main_uses_auto_compact_not_full_summary():
    src = os.path.join(PROJECT_ROOT, "main.py")
    text = open(src, encoding="utf-8").read()
    # Auto-compact (2026-09-25): oldest slice summarized, recent stays raw.
    assert "[Compact] budget hit" in text
    assert "_oldest_trim_split(" in text
    # Force-trim (summarize-everything) was removed by spec.
    assert "Force-trimmed to conv_budget" not in text
    assert "[Trim] partial:" not in text


def test_trigger_and_chunk_constants_exist():
    from core import conversation_trim
    assert conversation_trim.RAW_CONV_TRIGGER_TOKENS == 50000
    assert conversation_trim.TRIM_CHUNK_TOKENS == 15000


def test_auto_compact_saves_unsaved_before_dropping():
    """A4 regression: compact must persist unsaved turns before dropping."""
    src = os.path.join(PROJECT_ROOT, "main.py")
    text = open(src, encoding="utf-8").read()
    start = text.find("def _auto_compact_if_needed")
    assert start != -1, "_auto_compact_if_needed not found"
    pos_assign = text.find("self._conversation = rest + conv[-2:]", start)
    assert pos_assign != -1, "compact rebuild assignment not found"
    region = text[start:pos_assign]
    assert "save_conversation(unsaved" in region, \
        "auto-compact drops unsaved turns without saving them"


def test_user_turns_recent_stay_raw_after_trim():
    """End-to-end-ish: trim keeps recent turns raw while oldest get summarized."""
    from core.conversation_trim import oldest_trim_indices, TRIM_CHUNK_TOKENS

    # Conversation: 20 user/assistant pairs (~1k each) = 40 turns
    conv = []
    for i in range(20):
        conv.append({"role": "user", "content": "word " * 500})
        conv.append({"role": "assistant", "content": "answer"})

    obs_msgs = conv.copy()
    current_msg = obs_msgs.pop()       # latest assistant (in-flight answer)
    keep_msg = obs_msgs.pop()          # its user question
    trim_idx = oldest_trim_indices(obs_msgs, TRIM_CHUNK_TOKENS)
    oldest = [obs_msgs[i] for i in trim_idx]
    rest = [m for i, m in enumerate(obs_msgs) if i not in set(trim_idx)]

    rebuilt = rest + [keep_msg, current_msg]
    assert len(rebuilt) < len(conv), "trim must reduce raw history"
    assert rebuilt[-1] is current_msg, "current message stays last"
    assert rebuilt[-1]["content"] == "answer", "latest answer stays raw"
    # The most recent USER message is the one right before answer — kept raw.
    assert rebuilt[-2]["content"] == "word " * 500, "recent user message must stay raw"


def test_current_message_never_summarized():
    """The in-flight message must never join the trimmed chunk."""
    from core.conversation_trim import oldest_trim_indices, TRIM_CHUNK_TOKENS

    big = "word " * 12000  # ~12k tokens
    msgs = [
        {"role": "user", "content": big},
        {"role": "assistant", "content": big},
        {"role": "user", "content": "current user question"},
    ]
    # The trim helper only sees non-current messages; the caller pops the last
    # message(s) first, so by contract the last element is never in the chunk.
    idxs = oldest_trim_indices(msgs, TRIM_CHUNK_TOKENS)
    assert all(m.get("content") != "current user question" for i, m in enumerate(msgs)
               if i in idxs), "in-flight message must never be summarized"
