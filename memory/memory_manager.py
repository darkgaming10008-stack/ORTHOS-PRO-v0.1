from datetime import datetime, timedelta
from threading import Lock
from pathlib import Path
import sys

from memory.conversation_db import (
    init_db,
    save_conversation_db,
    format_recent_conversations,
    get_all_summaries_text,
    is_migrated,
    migrate_json_sessions,
    get_total_turn_count,
    # ── Session-aware imports ──
    create_session,
    list_sessions,
    get_session,
    rename_session,
    delete_session,
    close_and_summarize,
    get_turns_by_session,
    get_session_recent_turns,
    get_session_context,
    set_session_project,
    compress_oldest_turns,
    estimate_session_tokens,
    save_summary,
    summarize_session,
    # ── Session memory ──
    set_session_memory,
    get_session_memory,
    format_session_memory,
    log_timeline_event,
    format_timeline,
    auto_detect_timeline_events,
    search_timeline,
    # ── Branches / Tree ──
    create_branch,
    get_branches,
    get_branch,
    fork_session,
    get_turns_up_to,
    get_turns_by_branch,
    # ── Memory items ──
    add_memory_item,
    get_memory_items,
    get_memory_item,
    search_memory_items,
    update_memory_item,
    delete_memory_item,
    format_memory_items_for_prompt,
    # ── FTS / Search ──
    rebuild_fts_index,
    hybrid_search,
    invalidate_search_cache,
    # ── Project ──
    _ensure_default_hierarchy,
    list_projects,
    create_project,
    get_project,
    # ── Memory versioning ──
    save_memory_version,
    get_memory_versions,
    # ── Tool memory ──
    set_tool_memory,
    get_tool_memory,
    # ── Logging ──
    log_prompt,
    log_retrieval,
    # ── Archive ──
    _compress_to_file,
    _read_from_file,
)


def get_entity_graph(query: str) -> str:
    from memory.entity_store import get_entity_graph as _kg
    return _kg(query)


def graph_retrieve(query: str, max_hops: int = 2) -> list[dict]:
    from memory.entity_store import graph_retrieve as _gr
    return _gr(query, max_hops=max_hops)

def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR          = get_base_dir()
CONVERSATIONS_DIR = BASE_DIR / "memory" / "conversations"
_lock             = Lock()

# ── Initialise SQLite DB + migrate old JSON sessions ─────────────────────
init_db()
if not is_migrated():
    migrated = migrate_json_sessions()
    if migrated:
        print(f"[Memory] Migration complete — {migrated} turns imported.")

# Migrate long_term.json → memory_items table (idempotent — now source of truth)
try:
    from memory.conversation_db import (
        migrate_long_term_to_sqlite, dedup_memory_items, rebuild_memory_fts,
    )
    count = migrate_long_term_to_sqlite()
    if count:
        print(f"[Memory] long_term.json → memory_items: {count} entries migrated")
    removed = dedup_memory_items()
    if removed:
        print(f"[Memory] Deduped {removed} duplicate memory items")
    fts_n = rebuild_memory_fts()
    if fts_n:
        print(f"[Memory] FTS5 index built: {fts_n} entries")
except Exception:
    pass

# Bootstrap the raw-conversation FTS index (idempotent): every saved turn
# becomes searchable. Pre-existing turns are backfilled once at startup.
try:
    from memory.conversation_db import backfill_turns_fts as _backfill_turns_fts
    _bf_n = _backfill_turns_fts()
    if _bf_n:
        print(f"[Memory] turns_fts bootstrap: {_bf_n} raw turns now searchable")
except Exception as e:
    print(f"[Memory] turns_fts bootstrap skipped: {e}")


def _empty_memory() -> dict:
    return {
        "identity":      {},
        "preferences":   {},
        "projects":      {},
        "relationships": {},
        "wishes":        {},
        "notes":         {},
        "procedures":    {},
    }


def load_memory() -> dict:
    """Load all active memory from SQLite memory_items and return as dict (backward compat).
    Archived items are excluded (they live in the archive, not active memory)."""
    try:
        from memory.conversation_db import get_memory_items as _gmi
        items = _gmi(include_archived=False, limit=99999)
    except Exception:
        return _empty_memory()
    result = _empty_memory()
    for item in items:
        cat = item.get("type", "notes")
        if cat not in result:
            continue
        key = item.get("key") or "item_" + str(item.get("id", ""))
        val = item.get("content", "")
        importance = float(item.get("importance", 1.0))
        source = item.get("source", "manual")
        updated = item.get("updated_at", datetime.now().strftime("%Y-%m-%d"))
        acc_count = int(item.get("access_count", 0))
        expires = item.get("expires_at")
        meta = item.get("metadata", {})

        entry = {
            "value": str(val),
            "updated": updated[:10] if updated else datetime.now().strftime("%Y-%m-%d"),
            "history": [],
            "source": source,
            "score": importance,
            "access_count": acc_count,
        }
        if expires:
            entry["expires"] = expires

        # Try to get history from memory_versions
        try:
            from memory.conversation_db import get_memory_item_history
            key_id = item.get("id")
            history_versions = get_memory_item_history(key_id) if key_id else []
            if history_versions:
                entry["history"] = [
                    {"value": h.get("content", ""), "updated": h.get("created_at", "?")}
                    for h in history_versions[:20]
                ]
        except Exception:
            pass

        result[cat][key] = entry
    return result


def _sync_chroma(category: str, key: str, value: str) -> None:
    """Write-through: keep the ChromaDB copy in sync with SQLite on every
    write path (save/update/remember/dreaming). Chroma upserts by doc_id,
    so repeated syncs are idempotent."""
    try:
        from memory.chroma_memory import _get_chroma
        chroma = _get_chroma()
        if chroma:
            chroma.add_memory(category, key, str(value)[:2000])
    except Exception:
        pass


def save_memory(memory: dict, skip_git: bool = False,
                session_id: str | None = None,
                project_id: str | None = None) -> None:
    if not isinstance(memory, dict):
        return
    from memory.conversation_db import (
        add_memory_item as _ami,
        memory_items_exist as _mie,
        update_memory_item_by_key as _umik,
        get_memory_items as _gmi,
    )

    for cat, entries in memory.items():
        if not isinstance(entries, dict):
            continue
        for key, entry in entries.items():
            if isinstance(entry, dict):
                value = str(entry.get("value", ""))
                score = float(entry.get("score", 1.0))
                source = entry.get("source", "manual")
                expires = entry.get("expires")
                acc = int(entry.get("access_count", 0))
            else:
                value = str(entry)
                score = 1.0
                source = "manual"
                expires = None
                acc = 0

            try:
                if _mie(cat, key):
                    _umik(cat, key, value[:2000], importance=score,
                          source=source, expires_at=expires,
                          bump_access=True)
                else:
                    _ami(session_id, cat, str(value)[:2000], key=key,
                         importance=score, source=source,
                         expires_at=expires, access_count=acc,
                         project_id=project_id)
                _sync_chroma(cat, key, value)
            except Exception as _e:
                print(f"[Memory] ERROR saving '{cat}/{key}': {_e}")

    if not skip_git:
        try:
            from memory.git_memory import commit_memory
            commit_memory("Auto-save memory state")
        except Exception as e:
            print(f"[Memory] Git commit error: {e}")


SCORE_INITIAL = 1.0
SCORE_DECAY_DAYS = 30  # days until score halves
SCORE_BOOST_ON_ACCESS = 0.1


def _new_entry(value: str, source: str = "manual") -> dict:
    return {
        "value": value,
        "updated": datetime.now().strftime("%Y-%m-%d"),
        "history": [],
        "source": source,
        "score": SCORE_INITIAL,
        "access_count": 0,
    }


def _decay_score(entry: dict) -> float:
    """Return time-decayed score. Scores halve every SCORE_DECAY_DAYS without access."""
    score = float(entry.get("score", SCORE_INITIAL))
    updated = entry.get("updated", "")
    if not updated:
        return score
    try:
        days_since = (datetime.now() - datetime.fromisoformat(updated)).days
    except Exception:
        return score
    if days_since <= 0:
        return score
    halves = days_since / SCORE_DECAY_DAYS
    return score * (0.5 ** halves)


def _bump_score(entry: dict, memory: dict | None = None, cat: str = "", key: str = "") -> dict:
    """Increment score and access_count on retrieval, then save."""
    now = datetime.now().strftime("%Y-%m-%d")
    entry["score"] = _decay_score(entry) + SCORE_BOOST_ON_ACCESS
    entry["access_count"] = entry.get("access_count", 0) + 1
    entry["updated"] = now
    if memory is not None and cat and key:
        if cat in memory and key in memory[cat]:
            memory[cat][key] = entry
    return entry


CONTRADICTION_THRESHOLD = 0.7  # only flag contradictions for high-importance facts


def _predict_ttl(value: str, category: str) -> str | None:
    """Heuristic TTL prediction. Returns expiry date string or None."""
    val_lower = value.lower()

    # Time-bound patterns
    import re
    patterns = [
        (r"(?:next|this|coming)\s+(week|month|year)", 30),
        (r"(?:going|traveling|heading)\s+to\s+\w+\s+in\s+(january|february|march|april|may|june|july|august|september|october|november|december)", 60),
        (r"(?:in|within)\s+(\d+)\s+(day|week|month)s?", lambda m: int(m.group(1)) * (7 if m.group(2) == "week" else 30 if m.group(2) == "month" else 1)),
        (r"(?:until|till|through)\s+(\w+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})", 90),
        (r"tomorrow", 1),
        (r"next\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", 7),
    ]
    today = datetime.now()
    for pattern, days in patterns:
        m = re.search(pattern, val_lower)
        if m:
            if callable(days):
                try:
                    d = days(m)
                except Exception:
                    d = 30
            else:
                d = days
            expiry = today + timedelta(days=d)
            return expiry.strftime("%Y-%m-%d")

    # Low-importance facts in transient categories get 30-day default
    if category in ("notes", "wishes") and len(value) > 10:
        expiry = today + timedelta(days=30)
        return expiry.strftime("%Y-%m-%d")

    return None


def _recursive_update(target: dict, updates: dict) -> bool:
    changed = False
    for key, value in updates.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, dict) and "value" not in value:
            if key not in target or not isinstance(target[key], dict):
                target[key] = {}
                changed = True
            if _recursive_update(target[key], value):
                changed = True
        else:
            new_val  = str(value["value"] if isinstance(value, dict) else value)
            today    = datetime.now().strftime("%Y-%m-%d")
            existing = target.get(key, {})
            old_val  = existing.get("value") if isinstance(existing, dict) else None
            old_score = float(existing.get("score", SCORE_INITIAL)) if isinstance(existing, dict) else SCORE_INITIAL
            old_access = existing.get("access_count", 0) if isinstance(existing, dict) else 0
            if old_val is not None and old_val != new_val:
                # Contradiction detection: flag if old value was high-importance
                contradictions = existing.get("contradictions", [])
                if old_score >= CONTRADICTION_THRESHOLD:
                    contradictions.append({
                        "old_value": old_val,
                        "new_value": new_val,
                        "old_score": old_score,
                        "detected": today,
                    })
                    contradictions = contradictions[-10:]  # cap at 10

                # ADD-only: push old to history instead of overwriting
                history = existing.get("history", [])
                history.append({"value": old_val, "updated": existing.get("updated", "?")})
                if len(history) > 20:
                    history = history[-20:]
                entry = {
                    "value": new_val,
                    "updated": today,
                    "history": history,
                    "source": existing.get("source", "manual"),
                    "score": old_score + SCORE_BOOST_ON_ACCESS,
                    "access_count": old_access + 1,
                    "contradictions": contradictions,
                }
                # Auto-TTL on update
                ttl = _predict_ttl(new_val, "")
                if ttl:
                    entry["expires"] = ttl
                target[key] = entry
                changed = True
            elif old_val is None:
                entry = _new_entry(new_val)
                ttl = _predict_ttl(new_val, "")
                if ttl:
                    entry["expires"] = ttl
                target[key] = entry
                changed = True
    return changed


def update_memory(memory_update: dict,
                  session_id: str | None = None,
                  project_id: str | None = None) -> dict:
    if not isinstance(memory_update, dict) or not memory_update:
        return load_memory()
    memory = load_memory()
    if _recursive_update(memory, memory_update):
        save_memory(memory, session_id=session_id, project_id=project_id)
        print(f"[Memory] Saved: {list(memory_update.keys())}")
    return memory


def format_memory_for_prompt(memories: list[dict] | None = None) -> str:
    """Pure formatter — takes memory items (list[dict]) and formats as prompt block.
    Has zero knowledge of retrieval engines (ChromaDB, BM25, etc.)."""
    if not memories:
        return ""
    return _format_memories(memories)


def _format_memories(memories: list[dict]) -> str:
    lines = []
    for m in memories:
        cat = m.get("category", "")
        key = m.get("key", "")
        val = m.get("value", "")
        if not val:
            continue
        if cat == "session_summary":
            continue
        if cat == "identity":
            lines.append(f"{key.replace('_', ' ').title()}: {val}")
        elif cat == "preferences":
            lines.append(f"  Preference - {key.replace('_', ' ').title()}: {val}")
        elif cat == "projects":
            lines.append(f"  Project - {key.replace('_', ' ').title()}: {val}")
        elif cat == "relationships":
            lines.append(f"  {key.replace('_', ' ').title()}: {val}")
        elif cat == "wishes":
            lines.append(f"  Wish - {key.replace('_', ' ').title()}: {val}")
        elif cat == "notes":
            lines.append(f"  Note - {key}: {val}")
        elif cat == "procedures":
            val_str = val[:200] + "..." if len(str(val)) > 200 else val
            lines.append(f"  Procedure - {key}: {val_str}")
        else:
            lines.append(f"  {cat}/{key}: {val}")

    if not lines:
        return ""
    return "[CORE MEMORY — always-known facts about the user]\n" + "\n".join(lines) + "\n"


def _format_json_memory(memory: dict) -> str:
    lines = []
    for cat in ("identity", "preferences", "projects", "relationships", "wishes", "notes", "procedures"):
        entries = memory.get(cat, {})
        for key, entry in entries.items():
            val = entry.get("value") if isinstance(entry, dict) else entry
            if not val:
                continue
            if cat == "identity":
                lines.append(f"{key.replace('_', ' ').title()}: {val}")
            elif cat == "preferences":
                lines.append(f"  Preference - {key.replace('_', ' ').title()}: {val}")
            elif cat == "projects":
                lines.append(f"  Project - {key.replace('_', ' ').title()}: {val}")
            elif cat == "relationships":
                lines.append(f"  {key.replace('_', ' ').title()}: {val}")
            elif cat == "wishes":
                lines.append(f"  Wish - {key.replace('_', ' ').title()}: {val}")
            elif cat == "notes":
                lines.append(f"  Note - {key}: {val}")
            elif cat == "procedures":
                val_str = val[:200] + "..." if len(str(val)) > 200 else val
                lines.append(f"  Procedure - {key}: {val_str}")
    if not lines:
        return ""
    return "[CORE MEMORY — always-known facts about the user]\n" + "\n".join(lines) + "\n"


def format_memory_items_from_sqlite() -> str:
    """Format memory_items table content as a prompt block (replaces JSON-based format)."""
    try:
        from memory.conversation_db import get_memory_items
        items = get_memory_items(limit=20)
    except Exception:
        return ""
    if not items:
        return ""
    lines = ["[CORE MEMORY — always-known facts about the user]"]
    for item in items:
        t = item["type"].replace("_", " ").title()
        content = (item.get("content") or "")[:300]
        lines.append(f"  [{t}] {content}")
    return "\n".join(lines) + "\n"


def remember(key: str, value: str, category: str = "notes",
             session_id: str | None = None, project_id: str | None = None) -> str:
    valid = {"identity", "preferences", "projects", "relationships", "wishes", "notes", "procedures"}
    if category not in valid:
        category = "notes"
    from memory.conversation_db import (
        add_memory_item as _ami,
        update_memory_item_by_key as _umik,
        memory_items_exist as _mie,
    )
    try:
        if _mie(category, key):
            _umik(category, key, str(value)[:2000], source="manual",
                  source_quality="manual", bump_access=True)
        else:
            _ami(session_id, category, str(value)[:2000], key=key,
                 importance=1.0, source="manual", source_quality="manual",
                 project_id=project_id)
    except Exception:
        pass
    # Also add to knowledge graph
    try:
        from memory.entity_store import link_fact
        link_fact(category, key, value)
    except Exception:
        pass
    # Also add to ChromaDB for vector search
    try:
        from memory.chroma_memory import _get_chroma
        chroma = _get_chroma()
        if chroma:
            chroma.add_memory(category, key, str(value)[:2000])
    except Exception:
        pass
    return f"Remembered: {category}/{key} = {value}"


def forget(key: str, category: str = "notes") -> str:
    from memory.conversation_db import delete_memory_item_by_key as _dmik
    if _dmik(category, key):
        # Also remove from ChromaDB
        try:
            from memory.chroma_memory import _get_chroma
            chroma = _get_chroma()
            if chroma:
                chroma.delete_memory(category, key)
        except Exception:
            pass
        return f"Forgotten: {category}/{key}"
    return f"Not found: {category}/{key}"


forget_memory = forget


# ── Cross-session conversation history (SQLite-backed) ────────────────

_save_counter = 0
FULL_REBUILD_INTERVAL = 100  # full rebuild every 100 saves (incremental is sufficient)


def save_conversation(turns: list[dict], session_id: str | None = None) -> None:
    """Save turns to SQLite under a session.
    Also triggers background dreaming for automatic fact extraction.
    Uses incremental indexing; full rebuild every FULL_REBUILD_INTERVAL saves.
    """
    global _save_counter
    if session_id is None:
        session_id = create_session("Legacy Chat")
    save_conversation_db(turns, session_id)
    # Compression is handled internally by save_conversation_db (token-based)
    try:
        from memory.dreaming import mark_active, trigger_dream
        mark_active()
        trigger_dream()
    except Exception as e:
        print(f"[Memory] Dreaming trigger: {e}")

    _save_counter += 1
    do_full_rebuild = (_save_counter % FULL_REBUILD_INTERVAL) == 0

    # ── Slow indexing off the chat thread ──
    # Vector/BM25/Chroma work goes to the background index worker; the chat
    # thread only queues the task (microseconds) and returns. SQLite rows and
    # the FTS index were already committed synchronously in
    # save_conversation_db, so keyword search is live immediately.
    task: dict = {"turns": turns, "full_rebuild": do_full_rebuild,
                  "session_id": session_id}
    try:
        from memory.index_worker import submit as _iw_submit
        if not _iw_submit(task):
            # Sync mode (ORT_SYNC_INDEXING=1) or spawn failure — run inline
            # exactly like the worker would.
            from memory.index_worker import run_tasks
            run_tasks([task])
    except Exception as e:
        print(f"[Memory] index queue error: {e}")


def get_recent_turns_for_vector(limit: int = 50) -> list[dict]:
    from memory.conversation_db import get_recent_turns as _grt
    return _grt(limit)


# ── Session-aware utilities ──────────────────────────────────────────────

def load_session_turns(session_id: str) -> list[dict]:
    """Load full raw turns for a specific session."""
    return get_turns_by_session(session_id)


def load_all_summaries(exclude_session: str | None = None) -> str:
    """Return formatted summaries of all closed sessions (optionally excluding one)."""
    return get_session_context(exclude_session) if exclude_session else get_all_summaries_text()


def summarize_session_via_llm(session_id: str) -> str | None:
    """Generate and save a summary for a session."""
    return summarize_session(session_id)


def auto_title_session(session_id: str) -> str:
    """Auto-generate a title from the first user message in a session."""
    turns = get_turns_by_session(session_id)
    for t in turns:
        if t.get("role") == "user":
            content = (t.get("content") or "").strip()
            words = content.split()[:6]
            title = " ".join(words)
            if len(title) > 60:
                title = title[:57] + "..."
            rename_session(session_id, title)
            return title
    return "New Chat"


SEARCH_IDENTITY_KEYWORDS = {"who", "me", "user", "name", "identity", "myself", "brother", "friend", "i am", "call me"}


def _has_identity_keywords(query: str) -> bool:
    q = query.lower()
    for kw in SEARCH_IDENTITY_KEYWORDS:
        if kw in q:
            return True
    return False


def _temporal_decay(updated_at: str | None, now: datetime | None = None) -> float:
    """Phase 4 — search-time temporal decay (Mem0/GraphRAG DRIFT style).
    Recent facts keep full weight; older facts decay so stale info sinks.
    """
    if not updated_at:
        return 1.0
    if now is None:
        now = datetime.now()
    try:
        ts = datetime.fromisoformat(str(updated_at)[:19])
    except Exception:
        return 1.0
    days = (now - ts).days
    if days < 7:
        return 1.0
    elif days < 30:
        return 0.9
    elif days < 90:
        return 0.7
    elif days < 180:
        return 0.4
    else:
        return 0.2


def search_all_memories(query: str, limit: int = 10) -> str:
    """Multi-signal search: ChromaDB vector + BM25 + entity graph.
    Uses weighted score fusion. Industry standard hybrid search.
    """
    import os
    from memory.bm25_search import search_bm25
    from memory.chroma_memory import ChromaMemory, CHROMA_VECTOR_SEARCH_ENABLED
    from memory.entity_store import get_entity_graph
    from concurrent.futures import ThreadPoolExecutor

    HYDE_ENABLED = os.environ.get("ALEX_HYDE", "1") != "0"
    RERANKER_ENABLED = os.environ.get("ALEX_RERANKER", "1") != "0"

    BM25_WEIGHT = 0.30
    VECTOR_WEIGHT = 0.50
    ENTITY_WEIGHT = 0.20
    FTS_WEIGHT = 0.25      # Phase 1: SQLite FTS5 signal
    TURN_WEIGHT = 0.35     # Phase 6: raw conversation turns signal
    RERANK_WEIGHT = 0.15   # Phase 5: access-frequency rerank boost

    fusion = []

    # Signal 1: BM25 keyword search with query expansion
    try:
        from memory.query_expansion import search_with_expansion
        bm25_results = search_with_expansion(query, search_bm25, top_k=limit)
        for text, meta, bm25_score in bm25_results:
            normalized = min(1.0, bm25_score / 5.0)
            fusion.append((text, meta, "bm25", normalized * BM25_WEIGHT))
    except Exception as e:
        print(f"[Search] BM25 error: {e}")

    # Signal 1c: raw conversation turns (FTS over turns_fts) — every past
    # chat message is searchable, not just distilled facts.
    try:
        from memory.conversation_db import search_turns_fts
        turn_hits = search_turns_fts(query, limit=limit)
        for t in turn_hits:
            text = f"{t.get('session_title', '')} ({t.get('role', '')}): {t.get('snippet') or t.get('content', '')}"
            meta = dict(t)
            meta["type"] = "turn"
            score = 0.7 * TURN_WEIGHT * _temporal_decay(t.get("timestamp"))
            fusion.append((text, meta, "turns", score))
    except Exception as e:
        print(f"[Search] Turns FTS error: {e}")

    # Signal 1b: SQLite FTS5 full-text search (Phase 1)
    try:
        from memory.conversation_db import search_memory_items
        fts_results = search_memory_items(query, limit=limit)
        for item in fts_results:
            text = f"{item.get('type', '')}/{item.get('key', '')}: {item.get('content', '')}"
            boost = item.get("fts_boost", 1.0)
            decay = _temporal_decay(item.get("updated_at"))
            score = 0.7 * boost * FTS_WEIGHT * decay
            fusion.append((text, item, "fts", score))
    except Exception as e:
        print(f"[Search] FTS error: {e}")

    def _entity_work():
        try:
            eg = get_entity_graph(query)
            if eg:
                return eg
        except Exception as e:
            print(f"[Search] Entity error: {e}")
        return None

    def _chroma_work():
        try:
            if not CHROMA_VECTOR_SEARCH_ENABLED:
                return []
            # HyDE (phase 5, 2026): embed a hypothetical document instead of
            # the bare query — markedly better recall for short/typo-y queries.
            vec_query = query
            if HYDE_ENABLED:
                try:
                    from memory.query_expansion import generate_hyde_query
                    vec_query = generate_hyde_query(query) or query
                except Exception:
                    pass
            chroma = ChromaMemory()
            results = chroma.search_memory(vec_query, n_results=limit)
            seen = set()
            out = []
            for r in results:
                doc_id = r.get("id", "")
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                base = r.get("score", 0.5) * VECTOR_WEIGHT
                decay = _temporal_decay(r.get("updated_at"))
                score = base * decay
                text = f"{r.get('category', '')}/{r.get('key', '')}: {r.get('value', '')}"
                out.append((text, r, score))
            return out
        except Exception as e:
            print(f"[Search] ChromaDB error: {e}")
            return []

    with ThreadPoolExecutor(max_workers=2) as ex:
        ef = ex.submit(_entity_work)
        cf = ex.submit(_chroma_work)
        entity_graph = ef.result()
        chroma_results = cf.result()

    # Signal 2: ChromaDB vector search
    for text, meta, weighted_score in chroma_results:
        fusion.append((text, meta, "chroma", weighted_score))

    # Signal 3: Entity links
    if entity_graph:
        fusion.append((entity_graph, {"type": "entity"}, "entity", 1.0 * ENTITY_WEIGHT))

    if not fusion:
        return f"No matches found for: {query}"

    # Phase 5: rerank — access-frequency + exact-key + identity boosts,
    # properly weighted by RERANK_WEIGHT, then cross-signal dedup + floor.
    MIN_SCORE = 0.04
    q_low = query.lower()
    reranked = []
    for text, meta, signal, score in fusion:
        access = int(meta.get("access_count", 0) or 0)
        freq_norm = min(1.0, access / 20.0)
        boost = 1.0 + RERANK_WEIGHT * freq_norm
        key = str(meta.get("key", "")).lower()
        if key and key in q_low:
            boost += RERANK_WEIGHT
        cat = str(meta.get("category", "")).lower()
        if cat == "identity":
            boost += RERANK_WEIGHT * 0.5
        reranked.append((text, meta, signal, score * boost))
    fusion = reranked

    # Dedup across signals: the same fact can be surfaced by BM25 + FTS5 +
    # Chroma together — keep only the highest-scoring copy per fact.
    best_by_sig: dict = {}
    for text, meta, signal, score in fusion:
        cat = str(meta.get("category", ""))
        key = str(meta.get("key", ""))
        if cat and key:
            dedup_key = f"fact:{cat}/{key}"
        else:
            dedup_key = f"{signal}:{str(text)[:120]}"
        cur = best_by_sig.get(dedup_key)
        if cur is None or score > cur[3]:
            best_by_sig[dedup_key] = (text, meta, signal, score)
    fusion = list(best_by_sig.values())

    # Hard floor: drop low-relevance hits before they reach the prompt
    fusion = [x for x in fusion if x[3] >= MIN_SCORE]
    if not fusion:
        return f"No matches found for: {query}"

    # Phase 5 (2026): true cross-encoder rerank of the top candidates —
    # model-grounded relevance ordering on top of the heuristic fusion.
    # The model lives in the memory service (preloaded at boot); the app
    # asks the service for the re-ranked order via HTTP.
    if RERANKER_ENABLED and len(fusion) > 2:
        try:
            from memory.memory_client import memory_rerank
            texts = [str(c[0]) for c in fusion[:12]]
            order = memory_rerank(query, texts, top_k=limit)
            fusion = [fusion[i] for i in order if 0 <= i < len(fusion)]
            fusion = fusion[:limit]
        except Exception as e:
            print(f"[Search] Reranker error: {e}")
            fusion.sort(key=lambda x: x[3], reverse=True)
            fusion = fusion[:limit]
    else:
        fusion.sort(key=lambda x: x[3], reverse=True)
        fusion = fusion[:limit]

    by_signal = {"bm25": [], "chroma": [], "entity": [], "fts": [], "turns": []}
    for text, meta, signal, score in fusion:
        by_signal.setdefault(signal, []).append((text, meta, score))

    parts = []
    if by_signal["bm25"]:
        bm25_lines = ["[KEYWORD MATCHES — BM25 ranked]"]
        for text, meta, score in by_signal["bm25"]:
            meta_type = meta.get("type", "")
            if meta_type == "fact":
                bm25_lines.append(f"  [{score:.2f}] FACT: {meta.get('category')}/{meta.get('key')}: {meta.get('value')}")
            else:
                bm25_lines.append(f"  [{score:.2f}] TURN: {text[:200]}")
        parts.append("\n".join(bm25_lines))

    if by_signal["fts"]:
        fts_lines = ["[FULL-TEXT MATCHES — SQLite FTS5]"]
        for text, meta, score in by_signal["fts"]:
            fts_lines.append(f"  [{score:.2f}] {meta.get('type', '?')}/{meta.get('key', '?')}: {meta.get('content', text)[:200]}")
        parts.append("\n".join(fts_lines))

    if by_signal["chroma"]:
        chroma_lines = ["[SEMANTIC MATCHES — ChromaDB vector search]"]
        for text, meta, score in by_signal["chroma"]:
            chroma_lines.append(f"  [{score:.2f}] {meta.get('category', '?')}/{meta.get('key', '?')}: {meta.get('value', text)[:200]}")
        parts.append("\n".join(chroma_lines))

    if by_signal["entity"]:
        entity_lines = ["[ENTITY LINKS — connected facts]"]
        for text, _meta, score in by_signal["entity"]:
            entity_lines.extend(text.split("\n")[1:])
        parts.append("\n".join(entity_lines))

    if by_signal["turns"]:
        turn_lines = ["[RAW CONVERSATION MATCHES — past chat, quoted verbatim]"]
        for text, meta, score in by_signal["turns"]:
            ts = str(meta.get("timestamp", ""))[:10]
            title = meta.get("session_title", "(untitled)")
            role = meta.get("role", "?")
            snip = (meta.get("snippet") or meta.get("content") or text)[:300]
            turn_lines.append(f"  [{score:.2f}] {ts} — session \"{title}\" ({role}): {snip}")
        parts.append("\n".join(turn_lines))

    return "\n\n---\n\n".join(parts)


def format_memory_summary() -> str:
    """Return a ChatGPT-style memory summary page of all known facts.
    Includes expiry information for time-bound memories.
    """
    from memory.conversation_db import get_memory_items as _gmi, count_memory_items
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")

    sections = []
    total = count_memory_items()

    CAT_LABELS = {
        "identity": "Identity", "preferences": "Preferences",
        "projects": "Active Projects", "relationships": "Relationships",
        "wishes": "Wishes & Goals", "procedures": "Procedures",
        "notes": "Other Notes",
    }

    for cat, label in CAT_LABELS.items():
        items = _gmi(type=cat, limit=999)
        if not items:
            continue
        lines = [f"  [{label}]"]
        count = 0
        for item in items:
            key = item.get("key", "?")
            val = item.get("content", "")
            expires = item.get("expires_at")
            source = item.get("source", "manual")
            if not val:
                continue
            if expires and expires < today:
                continue
            expiry_str = f" (expires: {expires})" if expires else ""
            lines.append(f"    {key}: {val}{expiry_str} [source: {source}]")
            count += 1
        if count > 0:
            sections.append("\n".join(lines))

    header = f"[MEMORY SUMMARY]\nI know {total} things about you. Last updated: {now.strftime('%B %d, %Y')}.\n"
    if total == 0:
        header += "Nothing yet — I'll learn as we talk.\n"

    body = "\n\n".join(sections) if sections else "  (nothing stored yet)"
    return header + "\n" + body


def clear_expired_memories() -> int:
    """Remove expired memories via SQLite. Returns count of items removed."""
    from memory.conversation_db import clear_expired_memory_items
    removed = clear_expired_memory_items()
    if removed > 0:
        print(f"[Memory] Cleared {removed} expired memories")
    return removed


def distill_memory() -> int:
    """Phase 3 — ADD-only consolidation: merge exact-duplicate content within a
    category into a single item (keeping the highest-importance / most recent),
    and drop empty/near-empty items. Never discards contradictory distinct facts.
    Returns number of items removed.
    """
    from memory.conversation_db import (
        get_memory_items as _gmi,
        delete_memory_item_by_key as _dmik,
        update_memory_item_by_key as _umik,
    )
    removed = 0
    for cat in ("identity", "preferences", "projects", "relationships",
                "wishes", "notes", "procedures"):
        items = _gmi(type=cat, limit=9999)
        seen: dict[str, dict] = {}
        for item in items:
            content = (item.get("content") or "").strip()
            if not content or len(content) < 2:
                if _dmik(cat, item.get("key", "")):
                    removed += 1
                continue
            norm = content.lower()
            if norm in seen:
                # Merge: keep the higher-importance row, drop the other
                keeper = seen[norm]
                dup = item if float(item.get("importance", 0)) <= float(keeper.get("importance", 0)) else keeper
                drop = item if dup is keeper else keeper
                if _dmik(cat, drop.get("key", "")):
                    removed += 1
                    seen[norm] = dup
            else:
                seen[norm] = item
    if removed:
        print(f"[Memory] Distilled {removed} duplicate/empty items")
    return removed


MAX_MEMORY_ENTRIES = 200


def _count_entries(memory: dict) -> int:
    total = 0
    for cat, entries in memory.items():
        if isinstance(entries, dict):
            total += len(entries)
    return total


def enforce_memory_cap() -> int:
    """Cap memory at MAX_MEMORY_ENTRIES using SQLite archive."""
    from memory.conversation_db import (
        count_memory_items,
        get_memory_items as _gmi,
        archive_memory_item,
    )
    total = count_memory_items()
    if total <= MAX_MEMORY_ENTRIES:
        return 0

    # Get oldest items beyond the cap
    excess = total - MAX_MEMORY_ENTRIES
    try:
        from memory import conversation_db as _cdb
        with __import__("sqlite3").connect(str(_cdb.DB_PATH)) as conn:
            conn.row_factory = __import__("sqlite3").Row
            rows = conn.execute(
                "SELECT id FROM memory_items WHERE archived = 0 "
                "ORDER BY updated_at ASC LIMIT ?",
                (excess,),
            ).fetchall()
            archived_count = 0
            for row in rows:
                archive_memory_item(row[0])
                archived_count += 1
            print(f"[Memory] Archived {archived_count} entries (cap: {MAX_MEMORY_ENTRIES})")
            return archived_count
    except Exception:
        return 0


def get_archived_count() -> int:
    try:
        from memory.conversation_db import get_archived_memory_items
        return len(get_archived_memory_items(limit=9999))
    except Exception:
        return 0


# ── Self-directed memory management tools (MemGPT-style) ──────────────

def core_memory_append(key: str, value: str,
                       session_id: str | None = None,
                       project_id: str | None = None) -> str:
    """Append a fact to core memory (identity category). MemGPT-style."""
    if not key or not value:
        return "Error: both key and value required"
    from memory.conversation_db import add_memory_item, get_memory_item_by_key, update_memory_item_by_key

    existing = get_memory_item_by_key("identity", key)
    if existing:
        return f"Key '{key}' already exists in core memory (use core_memory_replace to update)"
    add_memory_item(
        session_id, "identity", value, key=key,
        source="core_memory", expires_at=None,
        project_id=project_id,
    )
    try:
        from memory.chroma_memory import ChromaMemory
        ChromaMemory().add_memory("identity", key, value)
    except Exception:
        pass
    try:
        from memory.entity_store import link_fact
        link_fact("identity", key, value)
    except Exception:
        pass
    print(f"[Memory][Core] Appended {key}: {value[:60]}")
    return f"Core memory updated: {key} = {value}"


def core_memory_replace(key: str, value: str,
                        session_id: str | None = None,
                        project_id: str | None = None) -> str:
    """Replace a fact in core memory (identity category). MemGPT-style."""
    if not key or not value:
        return "Error: both key and value required"
    from memory.conversation_db import add_memory_item, get_memory_item_by_key, update_memory_item_by_key

    existing = get_memory_item_by_key("identity", key)
    old_val = existing.get("content") if existing else None
    if old_val:
        update_memory_item_by_key("identity", key, value)
    else:
        add_memory_item(
            session_id, "identity", value, key=key,
            source="core_memory", expires_at=None,
            project_id=project_id,
        )
    try:
        from memory.chroma_memory import ChromaMemory
        cm = ChromaMemory()
        cm.delete_memory("identity", key)
        cm.add_memory("identity", key, value)
    except Exception:
        pass
    try:
        from memory.entity_store import link_fact
        link_fact("identity", key, value)
    except Exception:
        pass
    print(f"[Memory][Core] Replaced {key}: {old_val} -> {value[:60]}")
    return f"Core memory replaced: {key} = {value}"


def archival_memory_search(query: str, top_k: int = 10) -> str:
    """Search archived memory items (from SQLite archived flag)."""
    from memory.conversation_db import search_memory_items
    q = query.lower()
    results = []
    try:
        # M4 fix: pass include_archived=True — the loop below only keeps
        # archived=1 rows, but the default search filtered those OUT, so the
        # "search archived memory" tool could never return anything.
        archived_items = search_memory_items(q, limit=top_k * 2, include_archived=True)
    except Exception:
        return "Error reading archive."
    for item in archived_items:
        if item.get("archived", 0) != 1:
            continue
        cat = item.get("type", "notes")
        key = item.get("key", "?")
        val = item.get("content", "")
        updated = item.get("updated_at", "?")
        if not val:
            continue
        if q in val.lower() or q in key.lower() or q in cat.lower():
            results.append({
                "category": cat,
                "key": key,
                "value": str(val)[:200],
                "updated": updated[:10] if updated else "?",
            })
    if not results:
        return f"No archived results for: {query}"
    lines = ["[ARCHIVED MEMORY — old facts preserved from archive]"]
    for r in sorted(results, key=lambda x: x.get("updated", ""), reverse=True)[:top_k]:
        lines.append(f"  {r['category']}/{r['key']} ({r['updated']}): {r['value']}")
    return "\n".join(lines)


def context_status() -> str:
    """MemGPT-style context status: memory usage, token estimates, index health."""
    from memory.conversation_db import get_memory_items as _gmi, count_memory_items
    total_entries = count_memory_items()
    archived = get_archived_count()

    memory = load_memory()
    # Estimate tokens (rough: 4 chars per token)
    total_chars = sum(
        len(str(entry.get("value", "")))
        for cat, entries in memory.items()
        if isinstance(entries, dict)
        for key, entry in entries.items()
    )
    estimated_tokens = total_chars // 4

    # Per-category breakdown
    cat_counts = {}
    for cat in ("identity", "preferences", "projects", "relationships", "wishes", "notes", "procedures"):
        entries = memory.get(cat, {})
        if isinstance(entries, dict):
            cat_counts[cat] = len(entries)

    # Index health
    bm25_ok = False
    vec_ok = False
    try:
        from memory.bm25_search import get_index
        bm25_ok = get_index() is not None and get_index().num_docs > 0
    except Exception:
        pass
    try:
        from memory.vector_index import get_index as get_vec
        vec_ok = get_vec() is not None and len(get_vec().texts) > 0
    except Exception:
        pass

    lines = [
        "[CONTEXT STATUS — memory usage and health]",
        f"  Memory: {total_entries}/{MAX_MEMORY_ENTRIES} entries ({MAX_MEMORY_ENTRIES - total_entries} free)",
        f"  Archived entries: {archived}",
        f"  Estimated tokens stored: ~{estimated_tokens}",
        f"  BM25 index: {'healthy' if bm25_ok else 'empty'}",
        f"  Vector index: {'healthy' if vec_ok else 'empty'}",
        "  Per-category:",
    ]
    for cat, count in sorted(cat_counts.items()):
        if count > 0:
            lines.append(f"    {cat}: {count}")
    lines.append(f"  Core memory (identity): {cat_counts.get('identity', 0)} entries")
    return "\n".join(lines)


def format_core_memory() -> str:
    """MemGPT-style core memory block — always-in-context identity summary.
    Retrieves via search_memory (ChromaDB), then formats via format_memory_for_prompt.
    """
    try:
        from memory.chroma_memory import get_core_memories
        memories = get_core_memories()
    except Exception:
        memories = None
    if not memories:
        return "[CORE MEMORY — always-known facts about the user]\n  (no permanent facts yet — learning as we talk)\n"
    return format_memory_for_prompt(memories)


def load_recent_conversations(max_sessions: int = 10, session_id: str | None = None) -> str:
    """Session-aware conversation context.
    - If session_id is provided: returns current session's full raw turns
      + summaries of all other closed sessions.
    - If session_id is None: falls back to old behavior (last 10 raw + all summaries).
    """
    if session_id is not None:
        return get_session_context(session_id)
    # Legacy fallback
    parts = []
    recent = format_recent_conversations(limit=10)
    if recent:
        parts.append(recent)
    summaries = get_all_summaries_text()
    if summaries:
        parts.append(summaries)
    return "\n\n".join(parts) if parts else ""
