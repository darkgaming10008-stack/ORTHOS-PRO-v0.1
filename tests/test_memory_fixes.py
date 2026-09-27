"""Regression tests for the P0/P2 memory + session fixes (2026-09-06).

Covers: compression death-loop (T1), version-snapshot-on-change (M1),
source_quality preserve (M2), fork-from-last-turn (C4), archived-turn
filtering (C2), decay skips internal keys (A5), sidebar fork nonlocal (A1),
Chroma upsert (H1), archive search (M4).
"""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MEM = os.path.join(PROJECT_ROOT, "memory")

import pytest


# ── A1: sidebar fork crash (source has nonlocal child_sessions) ──

def test_sidebar_fork_render_declares_nested_nonlocal():
    src = open(os.path.join(PROJECT_ROOT, "ui.py"), encoding="utf-8").read()
    # The nested _render_session reassigns child_sessions — it MUST declare
    # nonlocal or every forked session crashes the sidebar with UnboundLocalError.
    idx = src.find("def _render_session(s, depth=0):")
    assert idx != -1, "session render function not found"
    region = src[idx: idx + 400]
    assert "nonlocal child_sessions" in region, \
        "child_sessions reassignment lacks nonlocal (fork crash)"


# ── A5: decay skips internal (_-prefixed) keys ──

def test_decay_skips_internal_keys():
    src = open(os.path.join(PROJECT_ROOT, "main.py"), encoding="utf-8").read()
    region = src[src.find("def _decay_old_memories"):]
    assert "key.startswith(\"_\")" in region, \
        "decay still deletes _rolling_summary / _summary_* internal keys"


# ── T1 + C2: compression operates on active turns only ──

def test_compress_oldest_turns_filters_archived():
    src = open(os.path.join(MEM, "conversation_db.py"), encoding="utf-8").read()
    region = src[src.find("def compress_oldest_turns"):src.find("def ", src.find("def compress_oldest_turns") + 10)]
    assert "include_archived=True" in region, "compress must read all to classify active"
    assert "not t.get(\"archived\")" in region, "active filter missing (T1 loop)"


def test_get_turns_by_session_excludes_archived_by_default():
    src = open(os.path.join(MEM, "conversation_db.py"), encoding="utf-8").read()
    start = src.find("def get_turns_by_session")
    assert start != -1, "get_turns_by_session must exist"
    end = src.find("\ndef ", start + 10)   # next top-level def AFTER it
    region = src[start: end if end != -1 else len(src)]
    assert "archived = 0" in region, "get_turns_by_session must exclude archived by default"


def test_token_cache_decremented_on_compress():
    src = open(os.path.join(MEM, "conversation_db.py"), encoding="utf-8").read()
    region = src[src.find("def compress_oldest_turns"):]
    assert "_token_count_cache[session_id] = max(" in region, \
        "compression must decrement the token cache (death-loop fix)"


# ── M1 + M2: version only on content change + source_quality preserve ──

def test_update_by_key_only_versions_on_change():
    src = open(os.path.join(MEM, "conversation_db.py"), encoding="utf-8").read()
    region = src[src.find("def update_memory_item_by_key"): src.find("def delete_memory_item_by_key")]
    assert "content_changed = (content != old_content)" in region, \
        "version gating missing (M1 explosion)"
    assert "if content_changed:" in region


def test_source_quality_preserve_default():
    src = open(os.path.join(MEM, "conversation_db.py"), encoding="utf-8").read()
    region = src[src.find("def update_memory_item_by_key"): src.find("def delete_memory_item_by_key")]
    assert "source_quality: str | None = None" in region, \
        "source_quality default must be None to preserve (M2)"
    assert 'if source_quality is None:' in region


# ── C4: fork uses last active turn ──

def test_fork_from_zero_copies_latest_active():
    src = open(os.path.join(MEM, "conversation_db.py"), encoding="utf-8").read()
    region = src[src.find("def fork_session"): src.find("def get_turns_up_to")]
    assert "from_turn_id <= 0" in region, "fork must treat <=0 as 'latest active turn'"
    assert "archived = 0" in region, "fork must exclude archived turns"


# ── H1: Chroma upsert ──

def test_chroma_add_uses_upsert():
    src = open(os.path.join(MEM, "chroma_memory.py"), encoding="utf-8").read()
    region = src[src.find("def add_memory"): src.find("def _remote_search")]
    assert "self._collection.upsert(" in region, \
        "Chroma add_memory must upsert so re-saves update the vector (H1)"


# ── H2: delete_memory correct arity ──

def test_core_memory_replace_uses_two_arg_delete():
    src = open(os.path.join(MEM, "memory_manager.py"), encoding="utf-8").read()
    assert 'cm.delete_memory("identity", key)' in src or 'cm.delete_memory("identity", key)' in src, \
        "core memory replace must call delete_memory(category, key) — not the one-arg misuse"


# ── M4: archive search passes include_archived ──

def test_archival_search_includes_archived():
    src = open(os.path.join(MEM, "memory_manager.py"), encoding="utf-8").read()
    region = src[src.find("def archival_memory_search"): src.find("def ", src.find("def archival_memory_search") + 10)]
    # not strictly bounded; check the whole function body has include_archived
    assert "include_archived=True" in src[src.find("def archival_memory_search"):
                                          src.find("def ", src.find("def archival_memory_search") + 10)
                                          if src.find("def ", src.find("def archival_memory_search") + 10) > src.find("def archival_memory_search") else len(src)], \
        "archive search must request archived items (M4)"


# ── Dreaming lock ──

def test_dream_trigger_is_locked():
    src = open(os.path.join(MEM, "dreaming.py"), encoding="utf-8").read()
    region = src[src.find("def trigger_dream"): src.find("def start_dream_loop")]
    assert "with _dream_lock:" in region, "trigger_dream must guard with _dream_lock"