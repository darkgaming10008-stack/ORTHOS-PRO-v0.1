"""Tests: raw conversation (turns) searchability.

Covers:
  - _sanitize_fts_query (FTS5 operators / punctuation / Hinglish safety)
  - save_turn live turns_fts indexing + search_turns_fts
  - backfill_turns_fts idempotency
  - delete_session FTS cleanup
  - hybrid_search turn signal
  - memory_manager fusion [RAW CONVERSATION MATCHES] block
  - Orthos._recall_conversation handler (recall_conversation tool)
"""
import sys
from unittest.mock import MagicMock

from memory import conversation_db as cdb


# ── _sanitize_fts_query ─────────────────────────────────────────────────

def test_sanitize_strips_fts_operators():
    q = 'what did I say about "servicing" AND (garage) OR bike*^'
    out = cdb._sanitize_fts_query(q)
    assert "AND" not in out
    assert "(" not in out and ")" not in out
    assert "*" not in out and "^" not in out
    # terms survive as quoted phrases joined by OR
    assert '"servicing"' in out
    assert '"garage"' in out
    assert '"bike"' in out


def test_sanitize_empty_and_junk():
    assert cdb._sanitize_fts_query("") == ""
    assert cdb._sanitize_fts_query('   ') == ""
    assert cdb._sanitize_fts_query('"*:^()-"') == ""
    assert cdb._sanitize_fts_query("the a an of") == ""


def test_sanitize_dedupes_terms():
    out = cdb._sanitize_fts_query("bike BIKE Bike servicing")
    assert out.count('"bike"') == 1
    assert out.count('"servicing"') == 1


# ── save_turn → live FTS index ──────────────────────────────────────────

def test_save_turn_live_fts_and_search():
    sid = cdb.create_session("fts live test")
    cdb.save_turn("user", "meri bike ki servicing Sharma garage se karwani hai", sid)
    cdb.save_turn("assistant", "theek hai bhai, Sharma garage ka number note kar liya", sid)

    hits = cdb.search_turns_fts("sharma garage", limit=5)
    assert hits, "freshly saved turn must be immediately searchable"
    assert any("Sharma garage" in h["content"] for h in hits)
    assert hits[0]["type"] == "turn"
    assert hits[0]["session_title"] == "fts live test"
    assert hits[0]["role"] in ("user", "assistant")
    # snippet must be present and highlighted or non-empty
    assert hits[0]["snippet"]


def test_search_turns_hinglish_punctuation_query():
    sid = cdb.create_session("hinglish fts")
    cdb.save_turn("user", "yaad rakhna mere phone ka PIN 4321 hai", sid)
    # punctuation-heavy query must not crash MATCH and must still find it
    hits = cdb.search_turns_fts("phone!!! PIN???", limit=5)
    assert any("4321" in h["content"] for h in hits)


def test_search_turns_session_scoped():
    s1 = cdb.create_session("scoped a")
    s2 = cdb.create_session("scoped b")
    cdb.save_turn("user", "quantum flux capacitor discussion", s1)
    cdb.save_turn("user", "quantum flux capacitor discussion again", s2)
    hits = cdb.search_turns_fts("quantum capacitor", limit=10, session_id=s2)
    assert hits
    assert all(h["session_id"] == s2 for h in hits)


# ── backfill idempotency ────────────────────────────────────────────────

def test_backfill_idempotent():
    sid = cdb.create_session("backfill test")
    cdb.save_turn("user", "backfill probe message alpha nine", sid)
    n1 = cdb.backfill_turns_fts()
    assert n1 >= 0
    n2 = cdb.backfill_turns_fts()
    assert n2 == 0, "second backfill must index nothing new"


# ── delete cleanup ──────────────────────────────────────────────────────

def test_delete_session_cleans_fts():
    sid = cdb.create_session("delete me fts")
    cdb.save_turn("user", "zebra unicornmagic phrase", sid)
    assert cdb.search_turns_fts("zebra unicornmagic")
    cdb.delete_session(sid)
    cdb.invalidate_search_cache()
    assert not cdb.search_turns_fts("zebra unicornmagic")


# ── hybrid_search includes turns ────────────────────────────────────────

def test_hybrid_search_includes_turn_signal():
    sid = cdb.create_session("hybrid turns")
    cdb.save_turn("user", "mango lassi recipe with saffron", sid)
    cdb.invalidate_search_cache()
    results = cdb.hybrid_search("mango lassi saffron", limit=5)
    assert any(r.get("type") == "turn" for r in results), \
        "hybrid_search must surface raw conversation turns"


# ── fusion block in search_all_memories ─────────────────────────────────

def test_fusion_raw_conversation_block(monkeypatch):
    monkeypatch.setenv("ALEX_RERANKER", "0")   # skip cross-encoder HTTP
    monkeypatch.setenv("ALEX_HYDE", "0")
    sid = cdb.create_session("fusion turns")
    cdb.save_turn("user", "pani puri challenge at sarita bazar", sid)
    cdb.invalidate_search_cache()

    from memory.memory_manager import search_all_memories
    out = search_all_memories("pani puri sarita bazar", limit=5)
    assert "[RAW CONVERSATION MATCHES" in out
    # snippet() wraps matches in >>...<< highlight markers — strip before
    # checking the phrase survives.
    flat = out.replace(">>", "").replace("<<", "").lower()
    assert "pani puri" in flat
    assert "fusion turns" in out


# ── recall_conversation tool handler ────────────────────────────────────

# ── Import main.py with heavy/stub-incompatible modules faked ──────────
_STUBS = [
    "ui",
    "memory.memory_manager",
    "memory.chroma_memory",
    "actions.file_processor", "actions.flight_finder", "actions.open_app",
    "actions.weather_report", "actions.send_message", "actions.reminder",
    "actions.computer_settings", "actions.screen_processor",
    "actions.youtube_video", "actions.desktop", "actions.file_controller",
    "actions.code_helper", "actions.dev_agent", "actions.web_search",
    "actions.computer_control", "actions.game_updater", "actions.terminal",
    "core.logging_setup", "core.security", "core.tool_registry",
    "core.llm_provider", "core.observability", "core.mcp_manager",
    "core.tool_index",
]

_SAVED_MODS = {}
for _name in _STUBS:
    _SAVED_MODS[_name] = sys.modules.get(_name)
    sys.modules[_name] = MagicMock()

_SAVED_ARGV = sys.argv
sys.argv = ["main.py"]          # main.py runs argparse.parse_args() at import
import main  # noqa: E402
sys.argv = _SAVED_ARGV

for _name in _STUBS:
    if _SAVED_MODS[_name] is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _SAVED_MODS[_name]


def test_recall_conversation_handler(monkeypatch):
    # Never touch real Chroma in tests
    monkeypatch.setattr("memory.chroma_memory.search_turn_embeddings",
                        lambda q, n_results=6: [], raising=False)

    app = main.Orthos.__new__(main.Orthos)
    sid = cdb.create_session("recall handler test")
    cdb.save_turn("user", "remember the chai recipe with elaichi and adrak", sid)

    out = app._recall_conversation({"query": "chai recipe elaichi"})
    assert "PAST CONVERSATIONS" in out
    assert "chai recipe" in out.lower()
    assert "recall handler test" in out

    # full_session expansion includes the whole session block
    out2 = app._recall_conversation({"query": "chai recipe", "full_session": True})
    assert "FULL SESSION" in out2

    # empty query → error string
    assert "Error" in app._recall_conversation({})


def test_recall_handler_no_match_returns_recent(monkeypatch):
    monkeypatch.setattr("memory.chroma_memory.search_turn_embeddings",
                        lambda q, n_results=6: [], raising=False)
    app = main.Orthos.__new__(main.Orthos)
    sid = cdb.create_session("recent fallback")
    cdb.save_turn("user", "the fallback probe message", sid)
    out = app._recall_conversation({"query": "zzzqqqxx impossible string"})
    assert "PAST CONVERSATIONS" in out  # recent-turn fallback, not an error
