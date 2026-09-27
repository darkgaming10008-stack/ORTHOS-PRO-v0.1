"""Tests for the Sept 2026 budget/prompt system.

Covers: pinned context windows (A1), pinned input budget (A2), past-summary
condense caching (B), summaries JOIN + cache (C), auto-compact slice math and
wiring (F), background prefetch presence (D/E).
"""

import json
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from unittest.mock import patch

import pytest


def _read(relpath: str) -> str:
    return open(os.path.join(PROJECT_ROOT, relpath), encoding="utf-8").read()


# ──────────────────────────────────────────────────────────────────────
# A1 — pinned context windows
# ──────────────────────────────────────────────────────────────────────

class TestContextPins:
    def test_kaggle_pinned_to_60k(self):
        from memory.conversation_db import get_model_context_window
        # Real kaggle server window — must NOT resolve via cache/patterns
        # (the old chain returned 128k from ctx_cache.json / "qwen" match).
        assert get_model_context_window("qwen3.8-27b-uncensored-mtp", "kaggle") == 60_000

    def test_other_providers_pinned_to_100k(self):
        from memory.conversation_db import get_model_context_window
        assert get_model_context_window("gemini-3.1-flash-lite", "gemini") == 100_000
        assert get_model_context_window("llama-3.3-70b-versatile", "groq") == 100_000
        assert get_model_context_window("gpt-5", "openai") == 100_000

    def test_no_provider_defaults_to_100k(self):
        from memory.conversation_db import get_model_context_window
        assert get_model_context_window("mystery-model", "") == 100_000

    def test_pattern_no_longer_overrides(self):
        from memory.conversation_db import get_model_context_window
        # "qwen" pattern used to win with 128k regardless of provider truth
        assert get_model_context_window("qwen-anything", "kaggle") == 60_000
        assert get_model_context_window("qwen-anything", "ollama") == 100_000


# ──────────────────────────────────────────────────────────────────────
# A2 — pinned input budget
# ──────────────────────────────────────────────────────────────────────

class TestInputBudget:
    def test_budget_pins_in_source(self):
        text = _read("main.py")
        assert '25_000 if cfg.get("provider", "").lower().strip() == "kaggle" else 50_000' in text
        assert "int(ctx_window * 0.50)" not in text, "old ctx-derived budget still present"


# ──────────────────────────────────────────────────────────────────────
# B — past-summary condense cache
# ──────────────────────────────────────────────────────────────────────

class TestCondenseCache:
    def setup_method(self, _):
        import memory.conversation_db as cdb
        cdb._past_condense_cache.clear()
        # Isolate disk persistence from the developer's real cache file.
        cdb._CONDENSE_CACHE_PATH = (
            cdb.Path(tempfile.mkdtemp(prefix="condense_test_")) / ".condense_cache.json"
        )

    def test_same_input_cached_single_llm_call(self):
        import memory.conversation_db as cdb
        calls = {"n": 0}

        def fake_summ(text, timeout_s=15, target_note=None):
            calls["n"] += 1
            return "CONDENSED"

        with patch.object(cdb, "_summarize_with_llm", side_effect=fake_summ):
            r1 = cdb._condense_past_summaries("big raw summaries", 5000, block_ok=True)
            r2 = cdb._condense_past_summaries("big raw summaries", 5000, block_ok=True)
        assert r1 == r2 == "CONDENSED"
        assert calls["n"] == 1, "identical input must not re-call the LLM"

    def test_different_target_invalidates(self):
        import memory.conversation_db as _unused_cdb  # noqa: F841
        import memory.conversation_db as cdb
        calls = {"n": 0}

        def fake_summ(text, timeout_s=15, target_note=None):
            calls["n"] += 1
            return "C"

        with patch.object(cdb, "_summarize_with_llm", side_effect=fake_summ):
            cdb._condense_past_summaries("raw", 5000, block_ok=True)
            cdb._condense_past_summaries("raw", 4000, block_ok=True)
        assert calls["n"] == 2

    def test_failure_negative_cached(self):
        import memory.conversation_db as cdb
        calls = {"n": 0}

        def fake_summ(text, timeout_s=15, target_note=None):
            calls["n"] += 1
            return None   # e.g. Gemini 503

        with patch.object(cdb, "_summarize_with_llm", side_effect=fake_summ):
            r1 = cdb._condense_past_summaries("raw-fail", 5000, block_ok=True)
            r2 = cdb._condense_past_summaries("raw-fail", 5000, block_ok=True)
        assert r1 is None and r2 is None
        assert calls["n"] == 1, "recent failure must be negative-cached (no 503 spam)"

    def test_hot_path_never_blocks(self):
        import memory.conversation_db as cdb
        import threading as _threading
        calls = {"n": 0}
        started = _threading.Event()

        def fake_summ(text, timeout_s=15, target_note=None):
            calls["n"] += 1
            started.set()
            return "CONDENSED"

        with patch.object(cdb, "_summarize_with_llm", side_effect=fake_summ):
            r = cdb._condense_past_summaries("hot path raw", 5000, block_ok=False)
            assert r is None, "hot path must NOT wait on the LLM"
            assert started.wait(timeout=5), "background condense should have run"
        assert calls["n"] == 1, "background condense warms the cache for the next message"

    def test_condense_cache_persisted_to_disk(self):
        import memory.conversation_db as cdb
        calls = {"n": 0}

        def fake_summ(text, timeout_s=15, target_note=None):
            calls["n"] += 1
            return "PERSISTED"

        with patch.object(cdb, "_summarize_with_llm", side_effect=fake_summ):
            cdb._condense_past_summaries("disk raw", 5000, block_ok=True)
        raw = json.loads(cdb._CONDENSE_CACHE_PATH.read_text(encoding="utf-8"))
        assert any(v.get("text") == "PERSISTED" for v in raw.values()), \
            "condense result must be written to the disk cache"


# ──────────────────────────────────────────────────────────────────────
# C — summaries JOIN + exclude-session cache
# ──────────────────────────────────────────────────────────────────────

class TestSummariesQuery:
    def setup_method(self, _):
        import memory.conversation_db as cdb
        cdb._summaries_cache.clear()

    def test_no_n_plus_one_get_session(self):
        import memory.conversation_db as cdb
        with patch.object(cdb, "get_session",
                          side_effect=AssertionError("N+1 regression: per-row get_session")):
            txt = cdb._get_all_summaries_text(exclude_session="not-a-real-session-id")
        assert isinstance(txt, str)

    def test_exclude_session_key_cached(self):
        import memory.conversation_db as cdb
        cdb._get_all_summaries_text(exclude_session="sess-abc")
        assert "sess-abc" in cdb._summaries_cache, \
            "hot path (exclude=current sid) must be cached under its own key"

    def test_save_summary_invalidates_cache(self):
        import memory.conversation_db as cdb
        from unittest.mock import MagicMock
        cdb._summaries_cache["x"] = {"text": "stale", "built_at": 0.0}
        fake_conn = MagicMock()
        with patch("sqlite3.connect", return_value=fake_conn), \
             patch.object(cdb, "invalidate_search_cache"), \
             patch("memory.chroma_memory.embed_session_summary", create=True):
            cdb.save_summary("sess-z", "S", 1)
        assert "x" not in cdb._summaries_cache, \
            "save_summary must invalidate the summaries cache"


# ──────────────────────────────────────────────────────────────────────
# F — auto-compact slice math
# ──────────────────────────────────────────────────────────────────────

class TestCompactSliceMath:
    def test_non_kaggle_exact_spec(self):
        from core.conversation_trim import compact_slice_tokens
        raw = 34200   # conv space at 50k trigger (50k − tools − base)
        s = compact_slice_tokens(raw)
        assert s == 15000, "cap must be 15k for a ~34k conversation"
        assert 50_000 - s + s // 3 == 40_000, "50k−15k+5k=40k spec violated"

    def test_kaggle_scaled_slice(self):
        from core.conversation_trim import compact_slice_tokens
        raw = 9200    # conv space at 25k trigger
        s = compact_slice_tokens(raw)
        assert s == 4140
        result = 25_000 - s + s // 3
        assert 21_000 <= result <= 23_000, "kaggle must land ~20% under budget"
        assert raw - s >= raw * 0.5, "majority of recent conversation stays raw"

    def test_floor(self):
        from core.conversation_trim import compact_slice_tokens
        assert compact_slice_tokens(1500) == 1000
        assert compact_slice_tokens(500) == 1000


class TestCompactWiring:
    def setup_method(self, _):
        self.text = _read("main.py")

    def test_compact_runs_before_prompt_build(self):
        # Order fix: refreshed rolling summary must land in THIS turn's prompt.
        pos_call = self.text.find("self._auto_compact_if_needed(_tool_cost_est)")
        pos_build = self.text.find('{"role": "system", "content": self._build_system_prompt()}')
        assert pos_call != -1 and pos_build != -1
        assert pos_call < pos_build, "compact must run BEFORE prompt build"

    def test_force_trim_removed(self):
        assert "Force-trimmed to conv_budget" not in self.text
        assert "Force aggressive trim" not in self.text

    def test_circuit_breaker_present(self):
        assert "_COMPACT_FAIL_LIMIT = 3" in self.text
        assert "_compact_blocked_until" in self.text

    def test_soft_precompact_at_90_percent(self):
        assert "int(budget * 0.90)" in self.text

    def test_recent_stays_raw_rebuild(self):
        # Only the oldest slice is dropped; last 2 (current turn) appended back.
        assert "self._conversation = rest + conv[-2:]" in self.text


class TestNoWordDrop:
    def test_summarize_turns_returns_none_on_llm_failure(self):
        from memory.conversation_db import summarize_turns
        turns = [{"role": "user", "content": "word " * 3000}]
        with patch("memory.conversation_db._summarize_with_llm", return_value=None):
            assert summarize_turns(turns, target_tokens=100) is None, \
                "LLM failure must return None, not a word-soup fallback"

    def test_summarize_turns_condense_retry(self):
        from memory.conversation_db import summarize_turns
        turns = [{"role": "user", "content": "word " * 3000}]
        with patch("memory.conversation_db._summarize_with_llm",
                   side_effect=["over " * 2000, "OK"]) as m:
            r = summarize_turns(turns, target_tokens=50)
        assert r == "OK"
        assert m.call_count == 2, "over-cap summary must get one condense retry"

    def test_word_drop_code_gone(self):
        db_text = _read(os.path.join("memory", "conversation_db.py"))
        assert "words[:int(target_tokens * 0.9)]" not in db_text
        main_text = _read("main.py")
        assert "words[-int(self._summary_rolling_budget" not in main_text


# ──────────────────────────────────────────────────────────────────────
# D/E — prefetch wiring
# ──────────────────────────────────────────────────────────────────────

class TestPrefetchWiring:
    def test_response_hook_spawns_prefetch(self):
        text = _read("main.py")
        assert "target=self._prefetch_next_payload" in text

    def test_static_and_query_segments_split(self):
        text = _read("main.py")
        assert "def _static_prompt_segments" in text
        assert "def _build_query_prompt_segments" in text
        assert "_PROMPT_ORDER" in text

    def test_typing_prefetch_wired(self):
        text = _read("main.py")
        assert "self.ui.on_query_typing = self._prefetch_query" in text
        ui_text = _read("ui.py")
        assert "textChanged.connect(self._on_query_typing)" in ui_text
        assert "setInterval(500)" in ui_text
        assert "def _prefetch_query" in text

    def test_query_prefetch_consumed_on_match(self):
        text = _read("main.py")
        assert 'cached.get("query") == q' in text
