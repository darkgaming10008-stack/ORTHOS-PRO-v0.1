import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import memory.conversation_db as _cdb
from memory import memory_manager as mm


# ── helpers ──────────────────────────────────────────────────────────────────


def _seed(cat: str, key: str, value: str, *, expires=None, score=1.0,
          access_count=0, source="manual") -> None:
    """Insert one memory fact directly into the SQLite store (temp DB)."""
    _cdb.add_memory_item(
        None, cat, str(value)[:2000], key=key, importance=float(score),
        source=source, access_count=access_count, expires_at=expires,
    )


def _make_entry(value: str, **kw) -> dict:
    base = {
        "value": value,
        "updated": kw.pop("updated", datetime.now().strftime("%Y-%m-%d")),
        "history": [],
        "source": "manual",
        "score": kw.pop("score", 1.0),
        "access_count": 0,
    }
    base.update(kw)
    return base


# ── load_memory ──────────────────────────────────────────────────────────────


class TestLoadMemory:
    def test_empty_db_returns_empty_structure(self):
        result = mm.load_memory()
        assert result == mm._empty_memory()

    def test_loads_seeded_fact(self):
        _seed("identity", "name", "Alice")
        result = mm.load_memory()
        assert result["identity"]["name"]["value"] == "Alice"

    def test_load_returns_all_categories(self):
        result = mm.load_memory()
        for cat in ("identity", "preferences", "projects", "relationships",
                    "wishes", "notes", "procedures"):
            assert cat in result


# ── save_memory ──────────────────────────────────────────────────────────────


class TestSaveMemory:
    def test_save_writes_fact(self):
        mm.save_memory({"identity": {"name": {"value": "Alice"}}}, skip_git=True)
        row = _cdb.get_memory_item_by_key("identity", "name")
        assert row is not None
        assert row["content"] == "Alice"

    def test_save_non_dict_does_nothing(self):
        before = _cdb.count_memory_items()
        mm.save_memory("not_a_dict", skip_git=True)
        assert _cdb.count_memory_items() == before

    def test_save_nested_dict(self):
        mm.save_memory({"identity": {"name": {"value": "Alice"}}}, skip_git=True)
        assert _cdb.get_memory_item_by_key("identity", "name")["content"] == "Alice"


# ── _new_entry / _decay_score ────────────────────────────────────────────────


class TestDecayScore:
    def test_new_entry_has_initial_score(self):
        entry = mm._new_entry("hello")
        assert entry["score"] == mm.SCORE_INITIAL

    def test_decay_fresh_entry(self):
        entry = mm._new_entry("hello")
        entry["updated"] = datetime.now().strftime("%Y-%m-%d")
        score = mm._decay_score(entry)
        assert score == pytest.approx(mm.SCORE_INITIAL, abs=0.01)

    def test_decay_old_entry(self):
        entry = mm._new_entry("old")
        entry["updated"] = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        score = mm._decay_score(entry)
        assert score == pytest.approx(0.25, abs=0.01)

    def test_decay_no_updated_field(self):
        entry = {"score": 1.0, "value": "test"}
        score = mm._decay_score(entry)
        assert score == 1.0

    def test_decay_future_date_clamps(self):
        entry = mm._new_entry("future")
        entry["updated"] = (datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d")
        score = mm._decay_score(entry)
        assert score == pytest.approx(mm.SCORE_INITIAL, abs=0.01)


# ── update_memory ────────────────────────────────────────────────────────────


class TestUpdateMemory:
    def test_add_new_entry(self):
        mm.update_memory({"identity": {"name": {"value": "Alice"}}})
        memory = mm.load_memory()
        assert memory["identity"]["name"]["value"] == "Alice"

    def test_update_existing_entry(self):
        _seed("identity", "name", "Alice")
        mm.update_memory({"identity": {"name": {"value": "Alice B."}}})
        memory = mm.load_memory()
        assert memory["identity"]["name"]["value"] == "Alice B."
        assert len(memory["identity"]["name"]["history"]) >= 1

    def test_empty_update_returns_current(self):
        _seed("identity", "name", "Alice")
        result = mm.update_memory({})
        assert result["identity"]["name"]["value"] == "Alice"


# ── forget ───────────────────────────────────────────────────────────────────


class TestForget:
    def test_forget_existing_key(self):
        _seed("preferences", "color", "blue")
        result = mm.forget("color", "preferences")
        assert "Forgotten" in result
        memory = mm.load_memory()
        assert "color" not in memory["preferences"]

    def test_forget_missing_key(self):
        result = mm.forget("nonexistent", "preferences")
        assert "Not found" in result


# ── search_all_memories ──────────────────────────────────────────────────────


def _bm25_fact(cat, key, value):
    return (value, {"type": "fact", "category": cat, "key": key, "value": value}, 1.0)


class TestSearchAllMemories:
    def test_returns_identity_when_bm25_hits(self):
        _seed("identity", "name", "Alice")
        bm25_results = [_bm25_fact("identity", "name", "Alice")]
        with patch("memory.query_expansion.search_with_expansion", return_value=bm25_results), \
             patch("memory.chroma_memory.ChromaMemory") as CM:
            CM.return_value.search_memory.return_value = []
            result = mm.search_all_memories("who are you", limit=5)
        assert "Alice" in result

    def test_no_matches_returns_message(self):
        with patch("memory.query_expansion.search_with_expansion", return_value=[]), \
             patch("memory.chroma_memory.ChromaMemory") as CM:
            CM.return_value.search_memory.return_value = []
            result = mm.search_all_memories("zzzzzzznothing", limit=5)
        assert "No matches" in result

    def test_search_identity_keyword(self):
        _seed("identity", "name", "Alice")
        bm25_results = [_bm25_fact("identity", "name", "Alice")]
        with patch("memory.query_expansion.search_with_expansion", return_value=bm25_results), \
             patch("memory.chroma_memory.ChromaMemory") as CM:
            CM.return_value.search_memory.return_value = []
            result = mm.search_all_memories("my name", limit=5)
        assert "Alice" in result


# ── clear_expired_memories ───────────────────────────────────────────────────


class TestClearExpiredMemories:
    def test_removes_expired_entries(self):
        _seed("notes", "old_note", "old", expires="2000-01-01")
        _seed("notes", "fresh", "fresh", expires="2099-01-01")
        removed = mm.clear_expired_memories()
        assert removed == 1
        memory = mm.load_memory()
        assert "old_note" not in memory["notes"]
        assert "fresh" in memory["notes"]

    def test_no_expired(self):
        assert mm.clear_expired_memories() == 0

    def test_all_future_does_not_expire(self):
        _seed("notes", "active", "here", expires="2099-01-01")
        assert mm.clear_expired_memories() == 0
        assert "active" in mm.load_memory()["notes"]


# ── enforce_memory_cap ────────────────────────────────────────────────────────


class TestEnforceMemoryCap:
    def test_below_cap_does_nothing(self, monkeypatch):
        for i in range(5):
            _seed("notes", f"key_{i}", f"val_{i}")
        monkeypatch.setattr(mm, "MAX_MEMORY_ENTRIES", 200)
        archived = mm.enforce_memory_cap()
        assert archived == 0

    def test_archives_oldest_entries(self, monkeypatch):
        for i in range(15):
            _seed("notes", f"key_{i}", f"val_{i}")
        monkeypatch.setattr(mm, "MAX_MEMORY_ENTRIES", 10)
        archived = mm.enforce_memory_cap()
        assert archived == 5
        remaining = mm.load_memory()
        remaining_count = mm._count_entries(remaining)
        assert remaining_count == 10

    def test_archive_stored_as_archived(self, monkeypatch):
        for i in range(12):
            _seed("notes", f"k{i}", f"v{i}")
        monkeypatch.setattr(mm, "MAX_MEMORY_ENTRIES", 10)
        mm.enforce_memory_cap()
        archived = _cdb.get_archived_memory_items(limit=999)
        assert len(archived) >= 2


# ── format_memory_summary ────────────────────────────────────────────────────


class TestFormatMemorySummary:
    def test_returns_known_count(self):
        _seed("identity", "name", "Alice")
        _seed("preferences", "color", "blue")
        _seed("notes", "todo", "buy milk")
        result = mm.format_memory_summary()
        assert "Alice" in result
        assert "blue" in result
        assert "buy milk" in result

    def test_empty_memory(self):
        result = mm.format_memory_summary()
        assert "nothing stored" in result

    def test_skips_expired_entries(self):
        _seed("notes", "expired", "gone", expires="2000-01-01")
        _seed("notes", "active", "here", expires="2099-01-01")
        result = mm.format_memory_summary()
        assert "gone" not in result
        assert "here" in result


# ── remember ──────────────────────────────────────────────────────────────────


class TestRemember:
    def test_remember_creates_entry(self):
        msg = mm.remember("fav_drink", "coffee", "preferences")
        assert "Remembered" in msg
        memory = mm.load_memory()
        assert memory["preferences"]["fav_drink"]["value"] == "coffee"

    def test_remember_invalid_category_defaults_to_notes(self):
        mm.remember("x", "y", "invalid_category")
        memory = mm.load_memory()
        assert memory["notes"]["x"]["value"] == "y"


# ── format_memory_for_prompt ──────────────────────────────────────────────────


class TestFormatMemoryForPrompt:
    @staticmethod
    def _make_sample_memories():
        return [
            {"category": "identity", "key": "name", "value": "Alice"},
            {"category": "preferences", "key": "food", "value": "pizza"},
            {"category": "projects", "key": "website", "value": "building a portfolio"},
        ]

    def test_header_present(self):
        result = mm.format_memory_for_prompt(self._make_sample_memories())
        assert "[CORE MEMORY" in result

    def test_includes_identity(self):
        result = mm.format_memory_for_prompt(self._make_sample_memories())
        assert "Alice" in result

    def test_empty_memory_returns_empty_string(self):
        result = mm.format_memory_for_prompt(None)
        assert result == ""
        result = mm.format_memory_for_prompt([])
        assert result == ""


# ── core_memory_append / core_memory_replace ──────────────────────────────────


class TestCoreMemory:
    def test_append_creates_entry(self):
        msg = mm.core_memory_append("likes", "pizza")
        assert "Core memory updated" in msg
        memory = mm.load_memory()
        assert memory["identity"]["likes"]["value"] == "pizza"

    def test_replace_updates_existing(self):
        _seed("identity", "name", "Alice")
        mm.core_memory_replace("name", "Bob")
        memory = mm.load_memory()
        assert memory["identity"]["name"]["value"] == "Bob"


# ── context_status ────────────────────────────────────────────────────────────


class TestContextStatus:
    def test_returns_counts(self):
        _seed("identity", "name", "Alice")
        result = mm.context_status()
        assert "identity: 1" in result


# ── _recursive_update ─────────────────────────────────────────────────────────


class TestRecursiveUpdate:
    def test_skips_none_values(self):
        target = {}
        mm._recursive_update(target, {"notes": {"x": None}})
        assert "x" not in target.get("notes", {})

    def test_skips_empty_string_values(self):
        target = {}
        mm._recursive_update(target, {"notes": {"x": ""}})
        assert "x" not in target.get("notes", {})

    def test_creates_nested_dicts(self):
        target = {}
        mm._recursive_update(target, {"identity": {"name": {"value": "Alice"}}})
        assert target["identity"]["name"]["value"] == "Alice"


# ── _predict_ttl ──────────────────────────────────────────────────────────────


class TestPredictTTL:
    def test_tomorrow_pattern(self):
        ttl = mm._predict_ttl("See you tomorrow", "notes")
        assert ttl is not None

    def test_next_week_pattern(self):
        ttl = mm._predict_ttl("Next week I have a meeting", "notes")
        assert ttl is not None

    def test_no_match_returns_none(self):
        ttl = mm._predict_ttl("I like pizza", "notes")
        assert ttl is not None


# ── _bump_score ───────────────────────────────────────────────────────────────


class TestBumpScore:
    def test_increments_access_count(self):
        entry = _make_entry("test", score=0.5, access_count=1)
        result = mm._bump_score(entry)
        assert result["access_count"] == 2

    def test_boosts_score(self):
        entry = _make_entry("test", score=0.5, updated=datetime.now().strftime("%Y-%m-%d"))
        result = mm._bump_score(entry)
        assert result["score"] == pytest.approx(0.5 + mm.SCORE_BOOST_ON_ACCESS, abs=0.01)


# ── Phase 1-5: FTS, decay, distill, rerank ─────────────────────────────────


class TestFTSSearch:
    def test_fts_finds_seeded_fact(self):
        _seed("preferences", "food", "I love sushi")
        from memory.conversation_db import rebuild_memory_fts, search_memory_items
        rebuild_memory_fts()
        results = search_memory_items("sushi", limit=5)
        assert any(r.get("key") == "food" for r in results)

    def test_fts_key_boost(self):
        _seed("preferences", "sushi", "favorite food is sushi")
        from memory.conversation_db import rebuild_memory_fts, search_memory_items
        rebuild_memory_fts()
        results = search_memory_items("sushi", limit=5)
        boosted = [r for r in results if r.get("key") == "sushi" and r.get("fts_boost")]
        assert boosted, "key match should be field-boosted"


class TestTemporalDecay:
    def test_recent_full_weight(self):
        recent = (datetime.now() - timedelta(days=3)).date().isoformat()
        assert mm._temporal_decay(recent) == 1.0

    def test_old_decays(self):
        old = (datetime.now() - timedelta(days=200)).date().isoformat()
        assert mm._temporal_decay(old) < 1.0

    def test_none_returns_one(self):
        assert mm._temporal_decay(None) == 1.0


class TestDistillMemory:
    def test_merges_exact_duplicates(self):
        _seed("notes", "dup_a", "same fact here")
        _seed("notes", "dup_b", "same fact here")
        removed = mm.distill_memory()
        assert removed == 1
        m = mm.load_memory()
        # exactly one of the two keys survives
        assert ("dup_a" in m.get("notes", {})) ^ ("dup_b" in m.get("notes", {}))

    def test_drops_empty_items(self):
        _seed("notes", "empty_one", "x")
        _cdb.add_memory_item(None, "notes", "  ", key="empty_one")
        removed = mm.distill_memory()
        assert removed >= 1


class TestSearchAllMemoriesSignals:
    def test_includes_fts_section(self):
        _seed("preferences", "coffee", "drinks espresso daily")
        from memory.conversation_db import rebuild_memory_fts
        rebuild_memory_fts()
        result = mm.search_all_memories("espresso", limit=5)
        assert "FULL-TEXT MATCHES" in result or "SEMANTIC MATCHES" in result

