import json
import hashlib
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
import re
import uuid
import gzip
import msgpack
import sentencepiece as spm
from functools import lru_cache

DB_PATH = Path(__file__).parent / "conversations.db"
COMPRESSED_DIR = Path(__file__).parent / "compressed"
_lock = threading.Lock()

# Memory revision counter — bumped on every memory WRITE so cached prompt
# segments (main.py static cache) can detect "something changed" cheaply.
_MEM_REV = 0


def get_memory_rev() -> int:
    return _MEM_REV

# ── SentencePiece tokenizer (Gemma 4, exact counting) ───────────────────
_TOKENIZER = None

def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            model_path = Path(__file__).parent.parent / "models" / "tokenizer.model"
            _TOKENIZER = spm.SentencePieceProcessor(model_file=str(model_path))
        except Exception as e:
            print(f"[Tokenizer] SentencePiece load failed: {e} — using fallback char/4 estimator")
            _TOKENIZER = None  # Will trigger fallback in _estimate_tokens
    return _TOKENIZER

# ── Token budget (replaces turn-count limit) ────────────────────────────
MAX_TOKENS_PER_SESSION = 100000  # summarize/compress when a session exceeds this
_COMPRESS_KEEP_TOKENS = 50000    # keep this many raw tokens after compression
_CONTEXT_MAX_TOKENS = 100000     # fallback: total context sent to LLM (raw + summaries)
_CONTEXT_RAW_BUDGET = 50000      # fallback: max tokens for raw turns within context

# ── Context window cache ─────────────────────────────────────────────
CTX_CACHE_PATH = Path(__file__).parent / "ctx_cache.json"
_ctx_cache: dict[str, int] = {}
_ctx_cache_lock = threading.Lock()

def _load_ctx_cache() -> dict[str, int]:
    try:
        return json.loads(CTX_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save_ctx_cache() -> None:
    try:
        CTX_CACHE_PATH.write_text(json.dumps(_ctx_cache, indent=2), encoding="utf-8")
    except Exception:
        pass

def _get_config() -> dict:
    path = Path(__file__).resolve().parent.parent / "config" / "api_keys.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

# ── Provider API queries ─────────────────────────────────────────────

def _query_gemini_context(model_name: str, api_key: str) -> int | None:
    try:
        import requests
        url = "https://generativelanguage.googleapis.com/v1beta/models"
        resp = requests.get(url, params={"key": api_key}, timeout=10)
        if resp.ok:
            for m in resp.json().get("models", []):
                mid = m.get("name", "").removeprefix("models/")
                if mid == model_name:
                    return m.get("inputTokenLimit")
    except Exception:
        pass
    return None

def _query_ollama_context(model_name: str, base_url: str) -> int | None:
    try:
        import requests
        url = base_url.rstrip("/") + "/api/show"
        resp = requests.post(url, json={"model": model_name}, timeout=10)
        if resp.ok:
            info = resp.json().get("model_info", {})
            for key, val in info.items():
                if key.endswith(".context_length") and isinstance(val, (int, float)):
                    return int(val)
    except Exception:
        pass
    return None

def _query_openrouter_context(model_name: str, api_key: str) -> int | None:
    try:
        import requests
        search = model_name.split("/")[-1].replace(":free", "").replace("@", "")
        url = f"https://openrouter.ai/api/v1/models?q={requests.utils.quote(search)}&limit=10"
        resp = requests.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
        if resp.ok:
            for m in resp.json().get("data", []):
                mid = m.get("id", "").lower()
                model_lower = model_name.lower()
                # Only accept if OpenRouter model ID is a reasonable match
                search_in_mid = search.lower().replace("-", "").replace(".", "")
                mid_clean = mid.replace("-", "").replace(".", "").replace(":free", "")
                if search_in_mid in mid_clean or mid_clean in search_in_mid:
                    ctx = m.get("context_length") or (m.get("top_provider") or {}).get("context_length")
                    if isinstance(ctx, int) and ctx > 0 and ctx <= 10_000_000:
                        return ctx
    except Exception:
        pass
    return None

def _query_anthropic_context(model_name: str, api_key: str) -> int | None:
    try:
        import requests
        slug = model_name.split("/")[-1]
        url = f"https://api.anthropic.com/v1/models/{slug}"
        resp = requests.get(url, headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }, timeout=10)
        if resp.ok:
            return resp.json().get("max_input_tokens")
    except Exception:
        pass
    return None

# ── Resolution chain ─────────────────────────────────────────────────
# Provider-specific patterns: MORE SPECIFIC FIRST to avoid wrong matches
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    # All providers standardized to 128K context window
    "llama-4-scout-17b":      128_000,
    "gemini-3":               128_000,
    "gemini-2.5":             128_000,
    "gemini_live":            128_000,
    "gemini":                 128_000,
    "mistral-small-3.1":      128_000,
    "mistral-small":          128_000,
    "ministral-3":            128_000,
    "ministral":              128_000,
    "mistral":                128_000,
    "llama-3.3":              128_000,
    "llama-3.2":              128_000,
    "llama-4-maverick":       128_000,
    "llama-4-scout":          128_000,
    "llama-4":                128_000,
    "llama4":                 128_000,
    "gemma-4":                128_000,
    "gemma-3":                128_000,
    "gemma":                  128_000,
    "qwen-3.5":               128_000,
    "qwen-3":                 128_000,
    "qwen2.5":                128_000,
    "qwen":                   128_000,
    "claude-sonnet-4":        128_000,
    "claude-opus-4":          128_000,
    "claude-4":               128_000,
    "claude-3.5":             128_000,
    "claude":                 128_000,
    "deepseek-v4":            128_000,
    "deepseek-v3":            128_000,
    "deepseek-r1":            128_000,
    "deepseek":               128_000,
    "gpt-4o":                 128_000,
    "gpt-5":                  128_000,
    "gpt-4":                  128_000,
    "o4":                     128_000,
    "o3":                     128_000,
    "openrouter":             128_000,
    "nvidia":                 128_000,
    "cloudflare":             128_000,
    "ollama":                 128_000,
}

# Provider-level defaults (when model name doesn't match any pattern)
PROVIDER_CONTEXT_DEFAULTS: dict[str, int] = {
    "gemini":         128_000,
    "gemini_live":    128_000,
    "ollama":         128_000,
    "openai":         128_000,
    "openrouter":     128_000,
    "groq":           128_000,
    "cloudflare":     128_000,
    "nvidia":         128_000,
    "anthropic":      128_000,
    "kaggle":         60_000,
}

# ── Provider query dispatch ──────────────────────────────────────────
# NOTE (2026-09): legacy — get_model_context_window() now pins values
# directly (PINNED_CONTEXT_WINDOWS) and never reaches this code.

_PROVIDER_API_HANDLERS: dict[str, callable] = {
    "gemini":   lambda m, p: _query_gemini_context(m, _get_config().get("gemini_api_key", "")),
    "ollama":   lambda m, p: _query_ollama_context(m, _get_ollama_config_url()),
    "kaggle":   lambda m, p: _query_ollama_context(m, _get_config().get("kaggle_url", "")),
    "anthropic": lambda m, p: _query_anthropic_context(m, _get_config().get("anthropic_api_key", "")),
}


def _get_ollama_config_url() -> str:
    """Configured Ollama URL honouring the active endpoint profile (if any)."""
    url = _get_config().get("llm_url", "http://localhost:11434")
    try:
        from core.llm_profiles import resolve_profile_url_model
        url, _ = resolve_profile_url_model(url, "")
    except Exception:
        pass
    return url

# Providers whose /v1/models does NOT expose context_length
# → skip directly to pattern matching
_PROVIDERS_WITHOUT_CTX_API = {"groq", "openai", "nvidia", "cloudflare", "openrouter"}


# ── Pinned context windows (Sept 2026 budget policy) ────────────────
# Hard-pinned per provider so every budget calculation is deterministic:
# NO live /v1/models query, NO pattern match, NO disk cache.
# (kaggle previously resolved to 128k via cache/patterns while the real
#  server window is 60k; gemini reported 1M — both broke budgets.)
PINNED_CONTEXT_WINDOWS: dict[str, int] = {
    "kaggle": 60_000,
}
DEFAULT_PINNED_CONTEXT_WINDOW = 100_000


def get_model_context_window(model_name: str, provider: str = "") -> int:
    """Pinned context window for a provider (see PINNED_CONTEXT_WINDOWS).

    Kept signature-compatible with the old resolution chain
    (model_name ignored — budgets must not vary by model-name pattern).
    """
    pname = provider.lower().strip() if provider else ""
    if not pname:
        return DEFAULT_PINNED_CONTEXT_WINDOW
    return PINNED_CONTEXT_WINDOWS.get(pname, DEFAULT_PINNED_CONTEXT_WINDOW)


def _estimate_tokens(text: str) -> int:
    tok = _get_tokenizer()
    if tok is None:
        # Fallback: ~4 chars per token (rough but safe)
        return max(1, len(text) // 4)
    try:
        return len(tok.Encode(text))
    except Exception:
        return max(1, len(text) // 4)


# ── Archive helpers (msgpack + zstd) ────────────────────────────────────

def _compress_to_file(session_id: str, chunk_id: int, data: dict) -> str:
    """Serialize data as msgpack, compress with zstd, write to file. Returns relative path."""
    import zstandard
    packed = msgpack.packb(data)
    compressed = zstandard.compress(packed)

    project_dir = COMPRESSED_DIR / session_id[:8]
    project_dir.mkdir(parents=True, exist_ok=True)
    fname = f"chunk_{chunk_id:04d}.msgpack.zst"
    path = project_dir / fname
    path.write_bytes(compressed)
    return str(path.relative_to(COMPRESSED_DIR))


def _read_from_file(rel_path: str) -> dict:
    """Read compressed archive file back."""
    import zstandard
    full = COMPRESSED_DIR / rel_path
    compressed = full.read_bytes()
    decompressed = zstandard.decompress(compressed)
    return msgpack.unpackb(decompressed)


def _delete_archive(rel_path: str) -> None:
    full = COMPRESSED_DIR / rel_path
    try:
        full.unlink(missing_ok=True)
    except Exception:
        pass


# ── DB init ─────────────────────────────────────────────────────────────

def init_db() -> None:
    with _lock:
        conn = sqlite3.connect(str(DB_PATH), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
        c = conn.cursor()

        # ── Hierarchy tables ──
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL DEFAULT 'default',
                created_at TEXT NOT NULL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS workspaces (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL DEFAULT '',
                name TEXT NOT NULL DEFAULT 'default',
                created_at TEXT NOT NULL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL DEFAULT '',
                name TEXT NOT NULL DEFAULT 'default',
                created_at TEXT NOT NULL
            )
        """)

        # ── Sessions (now has project_id) ──
        c.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT 'New Chat',
                status TEXT NOT NULL DEFAULT 'active',
                project_id TEXT DEFAULT NULL,
                parent_session_id TEXT DEFAULT NULL,
                fork_turn_id INTEGER DEFAULT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        # ── Branches ──
        c.execute("""
            CREATE TABLE IF NOT EXISTS branches (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                parent_branch_id TEXT DEFAULT NULL,
                fork_turn_id INTEGER DEFAULT NULL,
                title TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )
        """)

        # ── Turns (linked list) ──
        c.execute("""
            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL DEFAULT '',
                tool_calls TEXT DEFAULT NULL,
                tool_call_id TEXT DEFAULT NULL,
                tool_result TEXT DEFAULT NULL,
                session_id TEXT NOT NULL DEFAULT '',
                branch_id TEXT NOT NULL DEFAULT '',
                archived INTEGER NOT NULL DEFAULT 0,
                parent_turn_id INTEGER DEFAULT NULL,
                previous_turn_id INTEGER DEFAULT NULL,
                next_turn_id INTEGER DEFAULT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        # Migration: add tool_call_id if missing (pre-v0.3 databases)
        try:
            c.execute("ALTER TABLE turns ADD COLUMN tool_call_id TEXT DEFAULT NULL")
        except sqlite3.OperationalError:
            pass  # column already exists

        # ── Summaries (no embedding BLOB — ChromaDB only) ──
        c.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL,
                turn_count INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        # ── Compressed turns (filepath-based, no raw_text in SQLite) ──
        c.execute("""
            CREATE TABLE IF NOT EXISTS compressed_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                branch_id TEXT NOT NULL DEFAULT '',
                turn_ids TEXT NOT NULL DEFAULT '[]',
                summary TEXT NOT NULL DEFAULT '',
                archive_path TEXT NOT NULL DEFAULT '',
                token_count INTEGER NOT NULL DEFAULT 0,
                compressed_at TEXT NOT NULL
            )
        """)

        # ── Memory items (no embedding BLOB) ──
        c.execute("""
            CREATE TABLE IF NOT EXISTS memory_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT DEFAULT NULL,
                project_id TEXT DEFAULT NULL,
                type TEXT NOT NULL DEFAULT 'note',
                content TEXT NOT NULL,
                importance REAL NOT NULL DEFAULT 1.0,
                confidence REAL NOT NULL DEFAULT 0.5,
                source_quality TEXT NOT NULL DEFAULT 'system_inferred',
                decay_score REAL NOT NULL DEFAULT 1.0,
                version INTEGER NOT NULL DEFAULT 1,
                source TEXT NOT NULL DEFAULT 'manual',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_access TEXT NOT NULL DEFAULT '',
                access_count INTEGER NOT NULL DEFAULT 0,
                metadata TEXT NOT NULL DEFAULT '{}'
            )
        """)

        # ── Memory versions ──
        c.execute("""
            CREATE TABLE IF NOT EXISTS memory_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_item_id INTEGER NOT NULL,
                version INTEGER NOT NULL,
                content TEXT NOT NULL DEFAULT '',
                importance REAL NOT NULL DEFAULT 1.0,
                source_quality TEXT NOT NULL DEFAULT 'system_inferred',
                superseded_by INTEGER DEFAULT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (memory_item_id) REFERENCES memory_items(id)
            )
        """)

        # ── FTS5 full-text index over memory_items (Phase 1) ──
        c.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
            USING fts5(content, type, key, tokenize='unicode61')
        """)

        # ── Tool memory ──
        c.execute("""
            CREATE TABLE IF NOT EXISTS tool_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_name TEXT NOT NULL,
                session_id TEXT NOT NULL DEFAULT '',
                key TEXT NOT NULL DEFAULT '',
                value TEXT NOT NULL DEFAULT '',
                version INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        # ── Prompt log (quality tracking) ──
        c.execute("""
            CREATE TABLE IF NOT EXISTS prompt_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_hash TEXT NOT NULL DEFAULT '',
                prompt_text TEXT NOT NULL DEFAULT '',
                retrieved_ids TEXT NOT NULL DEFAULT '[]',
                response_text TEXT NOT NULL DEFAULT '',
                quality_score REAL DEFAULT NULL,
                latency_ms INTEGER DEFAULT 0,
                accepted INTEGER DEFAULT 0,
                timestamp TEXT NOT NULL
            )
        """)

        # ── Retrieval log (analytics) ──
        c.execute("""
            CREATE TABLE IF NOT EXISTS retrieval_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT NOT NULL DEFAULT '',
                retrieved_ids TEXT NOT NULL DEFAULT '[]',
                latency_ms INTEGER DEFAULT 0,
                hit_rate REAL DEFAULT 0.0,
                accepted INTEGER DEFAULT 0,
                timestamp TEXT NOT NULL
            )
        """)

        # ── Existing tables (idempotent) ──
        c.execute("""
            CREATE TABLE IF NOT EXISTS session_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL,
                UNIQUE(session_id, key)
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS timeline (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL DEFAULT '',
                event_type TEXT NOT NULL DEFAULT 'note',
                description TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                auto INTEGER NOT NULL DEFAULT 0,
                importance REAL NOT NULL DEFAULT 0.5,
                project_id TEXT NOT NULL DEFAULT '',
                keywords TEXT NOT NULL DEFAULT ''
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        # ── Migration: add columns to existing tables ──
        _add_column(c, "turns", "branch_id",           "TEXT NOT NULL DEFAULT ''")
        _add_column(c, "turns", "archived",            "INTEGER NOT NULL DEFAULT 0")
        _add_column(c, "turns", "parent_turn_id",      "INTEGER DEFAULT NULL")
        _add_column(c, "turns", "previous_turn_id",    "INTEGER DEFAULT NULL")
        _add_column(c, "turns", "next_turn_id",        "INTEGER DEFAULT NULL")
        _add_column(c, "sessions", "project_id",        "TEXT DEFAULT NULL")
        _add_column(c, "sessions", "parent_session_id", "TEXT DEFAULT NULL")
        _add_column(c, "sessions", "fork_turn_id",      "INTEGER DEFAULT NULL")
        _add_column(c, "memory_items", "project_id",    "TEXT DEFAULT NULL")
        _add_column(c, "memory_items", "confidence",    "REAL NOT NULL DEFAULT 0.5")
        _add_column(c, "memory_items", "source_quality","TEXT NOT NULL DEFAULT 'system_inferred'")
        _add_column(c, "memory_items", "decay_score",   "REAL NOT NULL DEFAULT 1.0")
        _add_column(c, "memory_items", "version",       "INTEGER NOT NULL DEFAULT 1")
        _add_column(c, "memory_items", "last_access",   "TEXT NOT NULL DEFAULT ''")
        _add_column(c, "memory_items", "key",           "TEXT DEFAULT NULL")
        _add_column(c, "memory_items", "archived",      "INTEGER NOT NULL DEFAULT 0")
        _add_column(c, "memory_items", "expires_at",    "TEXT DEFAULT NULL")
        _add_column(c, "summaries", "session_id",       "TEXT NOT NULL DEFAULT ''")
        _add_column(c, "sessions", "pinned",            "INTEGER NOT NULL DEFAULT 0")
        _add_column(c, "sessions", "archived",          "INTEGER NOT NULL DEFAULT 0")
        _add_column(c, "timeline", "importance",         "REAL NOT NULL DEFAULT 0.5")
        _add_column(c, "timeline", "project_id",         "TEXT NOT NULL DEFAULT ''")
        _add_column(c, "timeline", "keywords",           "TEXT NOT NULL DEFAULT ''")

        # ── FTS5 ──
        _init_fts(conn)

        conn.commit()
        conn.close()


def _add_column(conn, table: str, column: str, col_type: str) -> None:
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
    except sqlite3.OperationalError:
        pass


def _init_fts(conn) -> None:
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(content, session_id UNINDEXED, tokenize='porter unicode61')")
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS summaries_fts USING fts5(summary, session_id UNINDEXED, tokenize='porter unicode61')")
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS timeline_fts USING fts5(description, keywords UNINDEXED, tokenize='porter unicode61')")
    except sqlite3.OperationalError:
        pass
    except Exception:
        pass


def is_migrated() -> bool:
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'json_migrated'"
        ).fetchone()
        conn.close()
    return row is not None and row[0] == "1"


def mark_migrated() -> None:
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ("json_migrated", "1"),
        )
        conn.commit()
        conn.close()


# ── Session CRUD ─────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now().isoformat()


def _generate_session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def create_session(title: str = "New Chat", project_id: str | None = None) -> str:
    session_id = _generate_session_id()
    now = _now()
    pid = project_id or W_DEFAULT_PROJECT
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute(
            "INSERT INTO sessions (id, title, status, project_id, created_at, updated_at) "
            "VALUES (?, ?, 'active', ?, ?, ?)",
            (session_id, title, pid, now, now),
        )
        conn.commit()
        conn.close()
    print(f"[Session] Created: {session_id} — {title}")
    return session_id


def list_sessions(include_closed: bool = True, project_id: str | None = None) -> list[dict]:
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        order = "pinned DESC, updated_at DESC"
        if project_id:
            if include_closed:
                rows = conn.execute(
                    f"SELECT * FROM sessions WHERE project_id = ? ORDER BY {order}",
                    (project_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT * FROM sessions WHERE project_id = ? AND status='active' ORDER BY {order}",
                    (project_id,),
                ).fetchall()
        else:
            if include_closed:
                rows = conn.execute(
                    f"SELECT * FROM sessions ORDER BY {order}"
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT * FROM sessions WHERE status='active' ORDER BY {order}"
                ).fetchall()
        conn.close()
    return [dict(r) for r in rows]


@lru_cache(maxsize=4096)
def _cached_session_row(session_id: str) -> tuple[tuple[str, ...], tuple] | None:
    """Cached (columns, values) tuple — kills the N+1 in summary building.

    maxsize=4096 (was 256): the machine had ~471 sessions — the old cap
    evicted entries mid-build, so every rebuild re-queried the same rows.
    """
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        cols = tuple(row.keys()) if row else ()
        conn.close()
    return (cols, tuple(row)) if row else None


def get_session(session_id: str) -> dict | None:
    """LRU-cached (sessions rarely change; rename/delete invalidate)."""
    cached = _cached_session_row(session_id)
    if not cached:
        return None
    cols, vals = cached
    return dict(zip(cols, vals))


def invalidate_session_cache(session_id: str | None = None) -> None:
    if session_id:
        _cached_session_row.cache_clear()   # cheap; full clear is fine
    else:
        _cached_session_row.cache_clear()


def search_sessions(
    query: str,
    include_closed: bool = True,
    project_id: str | None = None,
) -> list[dict]:
    """Search sessions by title OR by any turn's content.

    Matching is case-insensitive. Title matches are ranked first; content
    matches bring up sessions whose conversation mentions the query.
    Results keep the same ordering contract as ``list_sessions``
    (pinned DESC, updated_at DESC) so the sidebar renders them in place.
    """
    q = (query or "").strip()
    if not q:
        return list_sessions(include_closed=include_closed, project_id=project_id)
    like = f"%{q}%"
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        if project_id:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE project_id = ? "
                "AND (title LIKE ? OR id IN (SELECT DISTINCT session_id FROM turns "
                "WHERE content LIKE ?)) "
                "ORDER BY pinned DESC, updated_at DESC",
                (project_id, like, like),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE title LIKE ? "
                "OR id IN (SELECT DISTINCT session_id FROM turns WHERE content LIKE ?) "
                "ORDER BY pinned DESC, updated_at DESC",
                (like, like),
            ).fetchall()
        conn.close()
    return [dict(r) for r in rows]


def rename_session(session_id: str, title: str) -> None:
    invalidate_session_cache()
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
            (title, _now(), session_id),
        )
        conn.commit()
        conn.close()


def set_session_pinned(session_id: str, pinned: bool = True) -> None:
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute(
            "UPDATE sessions SET pinned = ?, updated_at = ? WHERE id = ?",
            (1 if pinned else 0, _now(), session_id),
        )
        conn.commit()
        conn.close()


def delete_session(session_id: str) -> None:
    invalidate_session_cache()
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
        try:
            conn.execute("DELETE FROM turns_fts WHERE session_id = ?", (session_id,))
        except sqlite3.OperationalError:
            pass
        conn.execute("DELETE FROM summaries WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
        conn.close()
    _drop_token_cache(session_id)
    invalidate_search_cache()
    print(f"[Session] Deleted: {session_id}")


def close_session(session_id: str) -> None:
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute(
            "UPDATE sessions SET status = 'closed', updated_at = ? WHERE id = ?",
            (_now(), session_id),
        )
        conn.commit()
        conn.close()
    print(f"[Session] Closed: {session_id}")


def set_session_project(session_id: str, project_id: str) -> None:
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute(
            "UPDATE sessions SET project_id = ?, updated_at = ? WHERE id = ?",
            (project_id, _now(), session_id),
        )
        conn.commit()
        conn.close()


# ── Branch CRUD (Conversation Tree) ────────────────────────────────────

def _generate_branch_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def create_branch(session_id: str, parent_branch_id: str | None = None,
                  fork_turn_id: int | None = None, title: str = "") -> str:
    branch_id = _generate_branch_id()
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute(
            "INSERT INTO branches (id, session_id, parent_branch_id, fork_turn_id, title, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (branch_id, session_id, parent_branch_id, fork_turn_id, title, _now()),
        )
        conn.commit()
        conn.close()
    return branch_id


def get_branches(session_id: str) -> list[dict]:
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM branches WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        ).fetchall()
        conn.close()
    return [dict(r) for r in rows]


def get_branch(branch_id: str) -> dict | None:
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM branches WHERE id = ?", (branch_id,)
        ).fetchone()
        conn.close()
    return dict(row) if row else None


def fork_session(session_id: str, from_turn_id: int, title: str = "Fork",
                 embed: bool = True) -> str:
    invalidate_session_cache()
    """Create a new session as a fork of an existing one at a specific turn.

    ``from_turn_id`` of 0 (or any value ≤ 0) means "fork up to the newest
    active turn" — the sidebar used to call this with 0, which matched
    ``WHERE id <= 0`` and copied nothing (empty forks).
    """
    if from_turn_id <= 0:
        with _lock:
            conn = sqlite3.connect(str(DB_PATH))
            last = conn.execute(
                "SELECT MAX(id) FROM turns WHERE session_id = ? AND archived = 0",
                (session_id,),
            ).fetchone()[0]
            conn.close()
        from_turn_id = last or 0
    from_turns = get_turns_up_to(session_id, from_turn_id)
    new_session_id = create_session(title)
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute(
            "UPDATE sessions SET parent_session_id = ?, fork_turn_id = ? WHERE id = ?",
            (session_id, from_turn_id, new_session_id),
        )
        conn.commit()
        conn.close()
    save_conversation_db(from_turns, new_session_id, embed=embed)
    return new_session_id


def get_turns_up_to(session_id: str, turn_id: int) -> list[dict]:
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT role, content, tool_calls, tool_call_id, tool_result, timestamp FROM turns "
            "WHERE session_id = ? AND id <= ? AND archived = 0 ORDER BY id ASC",
            (session_id, turn_id),
        ).fetchall()
        conn.close()
    result = []
    for r in rows:
        d = dict(r)
        if d.get("tool_call_id") is None:
            d.pop("tool_call_id", None)
        if d["tool_calls"]:
            try:
                d["tool_calls"] = json.loads(d["tool_calls"])
            except Exception:
                d["tool_calls"] = []
        result.append(d)
    return result


def get_turns_by_branch(session_id: str, branch_id: str) -> list[dict]:
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT role, content, tool_calls, tool_call_id, tool_result, timestamp FROM turns "            
            "WHERE session_id = ? AND branch_id = ? ORDER BY id ASC",
            (session_id, branch_id),
        ).fetchall()
        conn.close()
    result = []
    for r in rows:
        d = dict(r)
        if d.get("tool_call_id") is None:
            d.pop("tool_call_id", None)
        if d["tool_calls"]:
            try:
                d["tool_calls"] = json.loads(d["tool_calls"])
            except Exception:
                d["tool_calls"] = []
        result.append(d)
    return result

def set_session_memory(session_id: str, key: str, value: str) -> None:
    global _MEM_REV
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute(
            "INSERT OR REPLACE INTO session_memory (session_id, key, value, updated_at) VALUES (?, ?, ?, ?)",
            (session_id, key, value, _now()),
        )
        conn.commit()
        conn.close()
    _MEM_REV += 1


def get_session_memory(session_id: str, key: str | None = None) -> dict[str, str]:
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        if key:
            rows = conn.execute(
                "SELECT key, value FROM session_memory WHERE session_id = ? AND key = ?",
                (session_id, key),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT key, value FROM session_memory WHERE session_id = ? ORDER BY updated_at DESC",
                (session_id,),
            ).fetchall()
        conn.close()
    return {r["key"]: r["value"] for r in rows}


def delete_session_memory(session_id: str, key: str | None = None) -> None:
    global _MEM_REV
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        if key:
            conn.execute("DELETE FROM session_memory WHERE session_id = ? AND key = ?", (session_id, key))
        else:
            conn.execute("DELETE FROM session_memory WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()
    _MEM_REV += 1


def format_session_memory(session_id: str) -> str:
    """Return formatted session memory block for prompt injection.
    Excludes _summary_* and _rolling_summary keys — those are added
    separately by _build_system_prompt() to avoid 5× duplication."""
    mem = get_session_memory(session_id)
    if not mem:
        return ""
    lines = ["[SESSION MEMORY — current goals, issues, code, files]"]
    for key, value in mem.items():
        if key.startswith("_summary") or key == "_rolling_summary":
            continue
        lines.append(f"  {key.replace('_', ' ').title()}: {value}")
    if len(lines) == 1:
        return ""
    return "\n".join(lines)

# ── Memory Items (replaces long_term.json) ─────────────────────────────

def add_memory_item(session_id: str | None, type: str, content: str,
                    importance: float = 1.0, source: str = "manual",
                    project_id: str | None = None,
                    confidence: float = 0.5,
                    source_quality: str = "system_inferred",
                    metadata: dict | None = None,
                    key: str | None = None,
                    expires_at: str | None = None,
                    access_count: int = 0) -> int:
    now = _now()
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.execute(
            "INSERT INTO memory_items (session_id, project_id, type, key, content, importance, "
            "confidence, source_quality, source, created_at, updated_at, last_access, "
            "access_count, expires_at, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, project_id, type, key, content, importance,
             confidence, source_quality, source, now, now, now,
             access_count, expires_at,
             json.dumps(metadata or {})),
        )
        conn.commit()
        item_id = c.lastrowid
        conn.close()
        _fts_sync(item_id, content, type, key or "")
    invalidate_search_cache()
    return item_id


def get_memory_items(type: str | None = None, limit: int = 50,
                     include_archived: bool = False) -> list[dict]:
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        if type:
            clause = "WHERE type = ?"
            params = [type]
            if not include_archived:
                clause += " AND archived = 0"
            rows = conn.execute(
                "SELECT * FROM memory_items "
                f"{clause} ORDER BY importance DESC, updated_at DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
        else:
            clause = ""
            params = []
            if not include_archived:
                clause = "WHERE archived = 0"
            rows = conn.execute(
                "SELECT * FROM memory_items "
                f"{clause} ORDER BY importance DESC, updated_at DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
        conn.close()
    result = []
    for r in rows:
        d = dict(r)
        try:
            d["metadata"] = json.loads(d["metadata"]) if isinstance(d["metadata"], str) else d["metadata"]
        except Exception:
            d["metadata"] = {}
        result.append(d)
    return result


def get_memory_item(item_id: int) -> dict | None:
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM memory_items WHERE id = ?", (item_id,)
        ).fetchone()
        conn.close()
    if row:
        d = dict(row)
        try:
            d["metadata"] = json.loads(d["metadata"]) if isinstance(d["metadata"], str) else d["metadata"]
        except Exception:
            d["metadata"] = {}
        return d
    return None


def get_memory_item_by_key(type: str, key: str) -> dict | None:
    """Get a memory item by type+key (replaces JSON category/key lookup)."""
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM memory_items WHERE type = ? AND key = ?",
            (type, key),
        ).fetchone()
        conn.close()
    if row:
        d = dict(row)
        try:
            d["metadata"] = json.loads(d["metadata"]) if isinstance(d["metadata"], str) else d["metadata"]
        except Exception:
            d["metadata"] = {}
        return d
    return None


def _fts_sync(item_id: int, content: str, type_: str, key: str) -> None:
    """Insert/replace the FTS5 row for a memory item."""
    try:
        conn = sqlite3.connect(str(DB_PATH), timeout=30)
        conn.execute("DELETE FROM memory_fts WHERE rowid = ?", (item_id,))
        conn.execute(
            "INSERT INTO memory_fts (rowid, content, type, key) VALUES (?, ?, ?, ?)",
            (item_id, content, type_, key),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _fts_delete(item_id: int) -> None:
    try:
        conn = sqlite3.connect(str(DB_PATH), timeout=30)
        conn.execute("DELETE FROM memory_fts WHERE rowid = ?", (item_id,))
        conn.commit()
        conn.close()
    except Exception:
        pass


def search_memory_items(query: str, limit: int = 10,
                        type_filter: str | None = None,
                        include_archived: bool = False) -> list[dict]:
    """FTS5 full-text search across memory_items (Phase 1).

    Falls back to a LIKE scan if FTS5 is unavailable. Boosts rows whose
    key matches the query (field-weighted, Mem0-style).
    """
    q = (query or "").strip()
    if not q:
        return []
    terms = [t for t in q.lower().split() if len(t) > 1]
    if not terms:
        return []

    with _lock:
        conn = sqlite3.connect(str(DB_PATH), timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            match = " OR ".join(terms)
            sql = (
                "SELECT m.* FROM memory_fts f "
                "JOIN memory_items m ON m.id = f.rowid "
                "WHERE memory_fts MATCH ?"
            )
            params: list = [match]
            if type_filter:
                sql += " AND m.type = ?"
                params.append(type_filter)
            if not include_archived:
                sql += " AND m.archived = 0"
            sql += " ORDER BY rank LIMIT ?"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
        except Exception:
            # Fallback: LIKE scan
            like = f"%{q}%"
            if type_filter:
                rows = conn.execute(
                    "SELECT * FROM memory_items WHERE type = ? AND content LIKE ? "
                    "ORDER BY importance DESC LIMIT ?",
                    (type_filter, like, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM memory_items WHERE content LIKE ? "
                    "ORDER BY importance DESC LIMIT ?",
                    (like, limit),
                ).fetchall()
        conn.close()

    result = []
    ql = q.lower()
    for r in rows:
        d = dict(r)
        try:
            d["metadata"] = json.loads(d["metadata"]) if isinstance(d["metadata"], str) else d["metadata"]
        except Exception:
            d["metadata"] = {}
        # Field-weighting: exact key match ranks above body match
        if d.get("key") and ql in str(d.get("key", "")).lower():
            d["fts_boost"] = 1.5
        result.append(d)
    return result



def update_memory_item(item_id: int, content: str | None = None,
                       importance: float | None = None, metadata: dict | None = None) -> bool:
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        sets = []
        params = []
        if content is not None:
            sets.append("content = ?")
            params.append(content)
        if importance is not None:
            sets.append("importance = ?")
            params.append(importance)
        if metadata is not None:
            sets.append("metadata = ?")
            params.append(json.dumps(metadata))
        if not sets:
            conn.close()
            return False
        sets.append("updated_at = ?")
        params.append(_now())
        params.append(item_id)
        conn.execute(f"UPDATE memory_items SET {', '.join(sets)} WHERE id = ?", params)
        conn.commit()
        conn.close()
    return True


def update_memory_item_by_key(type: str, key: str, content: str,
                              importance: float | None = None,
                              source: str = "manual",
                              source_quality: str | None = None,
                              metadata: dict | None = None,
                              expires_at: str | None = None,
                              bump_access: bool = False) -> bool:
    """Update a memory item by type+key.

    ``source_quality=None`` preserves the existing value (M2): the old
    default of ``"system_inferred"`` silently demoted every user-explicit
    fact on re-save.  A version snapshot is only written when the CONTENT
    actually changes (M1).
    """
    now = _now()
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM memory_items WHERE type = ? AND key = ?",
            (type, key),
        ).fetchone()
        if not row:
            conn.close()
            return False
        item_id = row["id"]
        old_content = row["content"]
        old_importance = row["importance"]
        current_version = row["version"]
        old_metadata = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row.get("metadata", {})

        # M2 fix: preserve the existing source_quality unless the caller
        # explicitly wants to override it (default is the current value, not
        # a hard reset to "system_inferred" which halves retrieval weight).
        if source_quality is None:
            source_quality = row["source_quality"]

        # Is the CONTENT actually changing?  M1 fix: only record a version
        # row when the value changes — otherwise every update_memory() call
        # spammed one version row PER surrounding memory item forever.
        content_changed = (content != old_content)

        # Save old version to memory_versions ONLY when content changes.
        if content_changed:
            conn.execute(
                "INSERT INTO memory_versions (memory_item_id, version, content, importance, "
                "source_quality, superseded_by, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (item_id, current_version, old_content, old_importance,
                 row["source_quality"], None, now),
            )

        new_version = current_version + 1 if content_changed else current_version
        sets = [
            "content = ?", "version = ?", "updated_at = ?", "source = ?",
            "source_quality = ?",
        ]
        params = [content, new_version, now, source, source_quality]
        if importance is not None:
            sets.append("importance = ?")
            params.append(importance)
        if expires_at is not None:
            sets.append("expires_at = ?")
            params.append(expires_at)
        if metadata is not None:
            merged_meta = {**old_metadata, **metadata}
            sets.append("metadata = ?")
            params.append(json.dumps(merged_meta))
        if bump_access:
            new_count = row["access_count"] + 1
            sets.append("access_count = ?")
            params.append(new_count)
            sets.append("last_access = ?")
            params.append(now)
        params.append(item_id)
        conn.execute(
            f"UPDATE memory_items SET {', '.join(sets)} WHERE id = ?",
            params,
        )
        conn.commit()
        conn.close()
        _fts_sync(item_id, content, type, key)
    invalidate_search_cache()
    return True


def delete_memory_item_by_key(type: str, key: str) -> bool:
    """Delete a memory item by type+key."""
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.execute(
            "SELECT id FROM memory_items WHERE type = ? AND key = ?",
            (type, key),
        )
        row = cursor.fetchone()
        if not row:
            conn.close()
            return False
        item_id = row[0]
        conn.execute("DELETE FROM memory_versions WHERE memory_item_id = ?", (item_id,))
        _fts_delete(item_id)
        conn.execute("DELETE FROM memory_items WHERE id = ?", (item_id,))
        conn.commit()
        conn.close()
    invalidate_search_cache()
    return True


def count_memory_items(type: str | None = None) -> int:
    """Count memory items, optionally filtered by type."""
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        if type:
            row = conn.execute(
                "SELECT COUNT(*) FROM memory_items WHERE type = ? AND archived = 0", (type,)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) FROM memory_items WHERE archived = 0"
            ).fetchone()
        conn.close()
    return row[0] if row else 0


def archive_memory_item(item_id: int) -> bool:
    """Mark a memory item as archived (soft delete for cap enforcement)."""
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute(
            "UPDATE memory_items SET archived = 1, updated_at = ? WHERE id = ?",
            (_now(), item_id),
        )
        conn.commit()
        conn.close()
    invalidate_search_cache()
    return True


def get_archived_memory_items(limit: int = 50) -> list[dict]:
    """Get archived memory items."""
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM memory_items WHERE archived = 1 "
            "ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
    result = []
    for r in rows:
        d = dict(r)
        try:
            d["metadata"] = json.loads(d["metadata"]) if isinstance(d["metadata"], str) else d["metadata"]
        except Exception:
            d["metadata"] = {}
        result.append(d)
    return result


def get_memory_item_history(memory_item_id: int) -> list[dict]:
    """Get version history for a memory item."""
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM memory_versions WHERE memory_item_id = ? "
            "ORDER BY version DESC LIMIT 20",
            (memory_item_id,),
        ).fetchall()
        conn.close()
    return [dict(r) for r in rows]


def clear_expired_memory_items() -> int:
    """Remove expired memory items. Returns count removed."""
    today = _now()[:10]  # YYYY-MM-DD
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        rows = conn.execute(
            "SELECT id FROM memory_items WHERE expires_at IS NOT NULL AND expires_at < ?",
            (today,),
        ).fetchall()
        ids = [r[0] for r in rows]
        for item_id in ids:
            conn.execute("DELETE FROM memory_versions WHERE memory_item_id = ?", (item_id,))
            conn.execute("DELETE FROM memory_items WHERE id = ?", (item_id,))
        conn.commit()
        conn.close()
    return len(ids)


def memory_items_exist(type: str, key: str) -> bool:
    """Check if a memory item exists by type+key."""
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        row = conn.execute(
            "SELECT 1 FROM memory_items WHERE type = ? AND key = ?",
            (type, key),
        ).fetchone()
        conn.close()
    return row is not None


def delete_memory_item(item_id: int) -> bool:
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("DELETE FROM memory_items WHERE id = ?", (item_id,))
        conn.commit()
        conn.close()
    return True


def format_memory_items_for_prompt(limit: int = 10) -> str:
    """Format top memory items as a prompt block."""
    items = get_memory_items(limit=limit)
    if not items:
        return ""
    lines = ["[LONG-TERM MEMORY — important facts about the user]"]
    for item in items:
        t = item["type"].replace("_", " ").title()
        lines.append(f"  [{t}] {item['content']}")
    return "\n".join(lines)


# ── Timeline ────────────────────────────────────────────────────────────

def _extract_keywords(text: str, max_keywords: int = 8) -> str:
    """Extract topical keywords (capitalized tech terms, proper nouns) from text."""
    words = re.findall(r'\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*\b', text)
    seen = set()
    unique = []
    for w in words:
        lower = w.lower()
        if lower not in seen and len(w) > 2:
            seen.add(lower)
            unique.append(w)
        if len(unique) >= max_keywords:
            break
    return ", ".join(unique) if unique else ""


def log_timeline_event(session_id: str, event_type: str, description: str,
                       auto: bool = False, importance: float = 0.5,
                       project_id: str = "", keywords: str = "") -> None:
    if not keywords:
        keywords = _extract_keywords(description)
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute(
            "INSERT INTO timeline (session_id, event_type, description, timestamp, auto, importance, project_id, keywords) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, event_type, description, _now(), 1 if auto else 0,
             importance, project_id, keywords),
        )
        # Sync to FTS5
        try:
            conn.execute(
                "INSERT INTO timeline_fts (rowid, description, keywords) VALUES (?, ?, ?)",
                (conn.execute("SELECT last_insert_rowid()").fetchone()[0], description, keywords),
            )
        except Exception:
            pass
        conn.commit()
        conn.close()


def get_timeline(limit: int = 50, event_type: str = "",
                 project_id: str = "", min_importance: float = 0.0) -> list[dict]:
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        clauses = ["1=1"]
        params = []
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        if project_id:
            clauses.append("project_id = ?")
            params.append(project_id)
        if min_importance > 0:
            clauses.append("importance >= ?")
            params.append(min_importance)
        rows = conn.execute(
            f"SELECT * FROM timeline WHERE {' AND '.join(clauses)} ORDER BY timestamp DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
        conn.close()
    return [dict(r) for r in rows]


def search_timeline(query: str, limit: int = 10, event_type: str = "",
                    project_id: str = "", days_back: int = 0) -> list[dict]:
    """Search timeline using FTS5 semantic + temporal filters."""
    results = []
    try:
        with _lock:
            conn = sqlite3.connect(str(DB_PATH))
            try:
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='timeline_fts'"
                )
                if cursor.fetchone():
                    clauses = ["timeline_fts MATCH ?"]
                    params = [query]
                    if event_type:
                        clauses.append("t.event_type = ?")
                        params.append(event_type)
                    if project_id:
                        clauses.append("t.project_id = ?")
                        params.append(project_id)
                    if days_back > 0:
                        from datetime import timedelta
                        cutoff = (datetime.now() - timedelta(days=days_back)).isoformat()
                        clauses.append("t.timestamp >= ?")
                        params.append(cutoff)
                    limit_param = limit
                    params.append(limit_param)
                    rows = conn.execute(
                        f"SELECT t.* FROM timeline_fts f JOIN timeline t ON f.rowid = t.id "
                        f"WHERE {' AND '.join(clauses)} ORDER BY t.importance DESC, t.timestamp DESC LIMIT ?",
                        params,
                    ).fetchall()
                    conn.row_factory = sqlite3.Row
                    for r in rows:
                        results.append(dict(r) if hasattr(r, 'keys') else {
                            "id": r[0], "session_id": r[1], "event_type": r[2],
                            "description": r[3], "timestamp": r[4], "auto": r[5],
                            "importance": r[6], "project_id": r[7], "keywords": r[8],
                        })
            except Exception:
                pass
            conn.close()
    except Exception:
        pass

    # Fallback: keyword search
    if not results:
        all_events = get_timeline(limit=200, event_type=event_type, project_id=project_id)
        q = query.lower()
        for ev in all_events:
            if q in ev.get("description", "").lower() or q in ev.get("keywords", "").lower():
                results.append(ev)
        results = results[:limit]

    return results


def auto_detect_timeline_events(session_id: str, turns: list[dict],
                                project_id: str = "") -> None:
    """Scan recent turns for notable events and log them."""
    import re
    notable_patterns = [
        (r"(?:added|integrated|set up|configured)\s+(\w[\w\s]*)", "integration"),
        (r"(?:installed|downloaded|set up)\s+(\w[\w\s]*)", "installation"),
        (r"(?:benchmark|test|evaluate|measure)\s+(\w[\w\s]*)", "benchmark"),
        (r"(?:switched|changed|migrated)\s+to\s+(\w[\w\s]*)", "migration"),
        (r"(?:created|built|wrote|developed)\s+(?:a|an|the)?\s*(\w[\w\s]*)", "creation"),
        (r"(?:fixed|resolved|solved)\s+(\w[\w\s]*)", "fix"),
        (r"(?:deployed|released|launched)\s+(\w[\w\s]*)", "deployment"),
    ]
    for t in turns:
        content = (t.get("content") or "").lower()
        for pattern, event_type in notable_patterns:
            m = re.search(pattern, content)
            if m:
                desc = f"{event_type}: {m.group(0).capitalize()}"
                importance = 0.7 if event_type in ("deployment", "migration", "creation") else 0.5
                log_timeline_event(session_id, event_type, desc, auto=True,
                                   importance=importance, project_id=project_id)
                break


def format_timeline(limit: int = 20, event_type: str = "",
                    project_id: str = "") -> str:
    events = get_timeline(limit, event_type=event_type, project_id=project_id)
    if not events:
        return ""
    lines = ["[TIMELINE — project history and milestones]"]
    for ev in events:
        ts = ev['timestamp'][:10]
        desc = ev['description']
        imp = ev.get('importance', 0.5)
        marker = " ★" if imp >= 0.8 else ""
        lines.append(f"  {ts} — {desc}{marker}")
    return "\n".join(lines)


def rebuild_timeline_fts() -> None:
    try:
        with _lock:
            conn = sqlite3.connect(str(DB_PATH))
            conn.execute("INSERT INTO timeline_fts(timeline_fts) VALUES('rebuild')")
            rows = conn.execute("SELECT id, description, keywords FROM timeline").fetchall()
            for row_id, desc, kw in rows:
                conn.execute(
                    "INSERT OR REPLACE INTO timeline_fts (rowid, description, keywords) VALUES (?, ?, ?)",
                    (row_id, desc, kw),
                )
            conn.commit()
            conn.close()
    except Exception:
        pass


# ── Turn management ──────────────────────────────────────────────────────

# Incremental token cache: session_id -> estimated total tokens.
# Updated on every insert so the live UI counter never needs a full
# content re-scan (which was O(all turns) per refresh).
_token_count_cache: dict[str, int] = {}


def _bump_token_cache(session_id: str, content: str) -> None:
    if not session_id:
        return
    _token_count_cache[session_id] = (
        _token_count_cache.get(session_id, 0) + _estimate_tokens(content)
    )


def _drop_token_cache(session_id: str) -> None:
    _token_count_cache.pop(session_id, None)


def save_turn(
    role: str,
    content: str,
    session_id: str = "",
    tool_calls: list | None = None,
    tool_call_id: str | None = None,
    tool_result: str | None = None,
    branch_id: str = "",
) -> None:
    sid = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        cur = conn.execute(
            "INSERT INTO turns (role, content, tool_calls, tool_call_id, tool_result, session_id, branch_id, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                role,
                content,
                json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None,
                tool_call_id,
                tool_result,
                sid,
                branch_id,
                datetime.now().isoformat(),
            ),
        )
        row_id = cur.lastrowid
        # ── Live FTS index for raw conversation search ──
        # Every saved turn is immediately searchable (turns_fts). Without this
        # insert the FTS index stays empty and raw chat is unsearchable.
        try:
            if (content or "").strip():
                conn.execute(
                    "INSERT INTO turns_fts(rowid, content, session_id) VALUES (?, ?, ?)",
                    (row_id, content, sid),
                )
        except sqlite3.OperationalError:
            pass  # FTS virtual table missing (shouldn't happen post-init)
        conn.commit()
        conn.close()
    _bump_token_cache(sid, content or "")
    invalidate_search_cache()
    return row_id


def save_conversation_db(turns: list[dict], session_id: str | None = None,
                         embed: bool = True) -> None:
    """Save all turns from a conversation exchange to SQLite under a session.

    embed=False skips the (background-queued) vector indexing — used by
    session forks/copies whose content is already indexed under the source.
    """
    chroma_rows: list[dict] = []  # (id, role, content, timestamp) for vector indexing
    if session_id is None:
        session_id = create_session("Imported Chat")
    for turn in turns:
        role = turn.get("role", "")
        content = turn.get("content", "") or ""
        tool_calls = turn.get("tool_calls")
        tool_call_id = turn.get("tool_call_id")
        branch_id = turn.get("branch_id", "")
        if role == "user" and not content.strip():
            continue
        row_id = None
        if role == "tool":
            row_id = save_turn("tool", content, session_id, tool_call_id=tool_call_id, tool_result=content, branch_id=branch_id)
        elif role == "assistant":
            row_id = save_turn("assistant", content, session_id, tool_calls=tool_calls, branch_id=branch_id)
        else:
            row_id = save_turn(role, content, session_id, branch_id=branch_id)
        if row_id and content.strip() and len(content) > 10:
            chroma_rows.append({
                "id": row_id,
                "role": role,
                "content": content,
                "timestamp": turn.get("timestamp", "") or datetime.now().isoformat(),
            })
    # Update session timestamp
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (_now(), session_id),
        )
        conn.commit()
        conn.close()
    # Token-based archive check
    total_tokens = estimate_session_tokens(session_id)
    print(f"[Session] Saved {len(turns)} turns to {session_id} (~{total_tokens} tokens)")
    # Semantic index for raw turns — queued to the background index worker
    # (chat thread must not block on embedding HTTP/CPU). FTS + SQLite rows
    # above are already committed synchronously, so keyword search is live
    # immediately; only the slow vector copy trails a few seconds.
    if chroma_rows and embed:
        try:
            from memory.index_worker import submit as _iw_submit
            if not _iw_submit({"chroma_rows": chroma_rows}):
                # Sync mode (ORT_SYNC_INDEXING=1) or spawn failure — run inline.
                from memory.chroma_memory import add_turn_embeddings as _ate
                _ate(chroma_rows)
        except Exception as e:
            print(f"[Memory] Turn embedding skipped: {e}")
    if total_tokens > MAX_TOKENS_PER_SESSION:
        compress_oldest_turns(session_id)
    invalidate_search_cache()


def estimate_session_tokens(session_id: str) -> int:
    """Estimate total token count for a session's turns.

    Uses the incremental insert-time cache when available; falls back to a
    full scan (and primes the cache) for sessions first seen after restart.
    """
    cached = _token_count_cache.get(session_id)
    if cached is not None:
        return cached
    turns = get_turns_by_session(session_id)
    total = sum(_estimate_tokens(t.get("content", "")) for t in turns)
    _token_count_cache[session_id] = total
    return total


def get_session_turn_count(session_id: str) -> int:
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        count = conn.execute(
            "SELECT COUNT(*) FROM turns WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
        conn.close()
    return count


def get_turns_by_session(session_id: str, include_archived: bool = False) -> list[dict]:
    where = "session_id = ?" if include_archived else "session_id = ? AND archived = 0"
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT role, content, tool_calls, tool_call_id, tool_result, timestamp, archived, branch_id "
            f"FROM turns WHERE {where} ORDER BY id ASC",
            (session_id,),
        ).fetchall()
        conn.close()
    result = []
    for r in rows:
        d = dict(r)
        if d.get("tool_call_id") is None:
            d.pop("tool_call_id", None)
        if d["tool_calls"]:
            try:
                d["tool_calls"] = json.loads(d["tool_calls"])
            except Exception:
                d["tool_calls"] = []
        result.append(d)
    return result


def get_total_turn_count() -> int:
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        count = conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
        conn.close()
    return count


def get_recent_turns(limit: int = 10) -> list[dict]:
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT role, content, tool_calls, tool_call_id, tool_result, timestamp "
            "FROM turns ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
    result = []
    for r in reversed(rows):
        d = dict(r)
        if d.get("tool_call_id") is None:
            d.pop("tool_call_id", None)
        if d["tool_calls"]:
            try:
                d["tool_calls"] = json.loads(d["tool_calls"])
            except Exception:
                d["tool_calls"] = []
        result.append(d)
    return result


def get_session_recent_turns(session_id: str, limit: int = 10) -> list[dict]:
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT role, content, tool_calls, tool_call_id, tool_result, timestamp "
            "FROM turns WHERE session_id = ? AND archived = 0 ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        conn.close()
    result = []
    for r in reversed(rows):
        d = dict(r)
        if d.get("tool_call_id") is None:
            d.pop("tool_call_id", None)
        if d["tool_calls"]:
            try:
                d["tool_calls"] = json.loads(d["tool_calls"])
            except Exception:
                d["tool_calls"] = []
        result.append(d)
    return result


# ── Session-aware context building ───────────────────────────────────────

def get_session_context(current_session_id: str, include_summaries: bool = True, include_raw_turns: bool = True, max_tokens: int = 100000) -> str:
    """Build prompt context for the current session.
    - Current session: full raw turns (capped at 50K tokens)
    - Other closed sessions: summaries only (total capped at max_tokens)
    """
    parts = []
    # 1. Current session raw turns — cap at 50K tokens
    if include_raw_turns:
        current_turns = get_turns_by_session(current_session_id)
        active_turns = [t for t in current_turns if not t.get("archived")]
        if active_turns:
            turn_lines = ["[CURRENT CONVERSATION]"]
            for t in active_turns:
                formatted = format_turn_for_prompt(t)
                if formatted:
                    turn_lines.append(formatted)
            raw_text = "\n".join(turn_lines)
            raw_tokens = _estimate_tokens(raw_text)
            if raw_tokens > _CONTEXT_RAW_BUDGET:
                while len(turn_lines) > 1:
                    turn_lines.pop(1)
                    raw_text = "\n".join(turn_lines)
                    if _estimate_tokens(raw_text) <= _CONTEXT_RAW_BUDGET:
                        break
            parts.append(raw_text)
        else:
            parts.append("[CURRENT CONVERSATION]\n(New session — no messages yet)")

    # 2. Past session summaries — cap total at max_tokens
    raw_tokens = _estimate_tokens(parts[0]) if parts else 0
    remaining = max_tokens - raw_tokens

    past_summaries = ""
    if include_summaries and remaining > 200:
        past_summaries = _get_all_summaries_text(exclude_session=current_session_id)
        if past_summaries:
            summary_tokens = _estimate_tokens(past_summaries)
            if summary_tokens > remaining:
                # Budget gate: summaries only hit the LLM when they don't fit
                # (same input → sha256 cache hit → NO repeated hidden calls).
                # block_ok=False: a cold cache miss must NEVER block message
                # send — cheap truncation runs now, the LLM condense warms
                # the disk cache in the background for the next message.
                condensed = _condense_past_summaries(past_summaries, remaining, block_ok=False)
                if condensed and _estimate_tokens(condensed) <= remaining:
                    past_summaries = condensed
                else:
                    # LLM failed / didn't fit → safety valve: keep the newest
                    # sessions, drop oldest sections (never mid-section).
                    sections = past_summaries.split("\n--- Session:")
                    kept = [sections[0]]
                    for sec in sections[1:]:
                        section_text = "--- Session:" + sec
                        candidate = "\n".join(kept) + "\n" + section_text
                        if _estimate_tokens(candidate) <= remaining:
                            kept.append(section_text)
                        else:
                            break
                    past_summaries = "\n".join(kept) if len(kept) > 1 else sections[0]
    if past_summaries:
        parts.append(past_summaries)

    return "\n\n".join(parts)


# ── Past-summary condense cache (sha256 + TTL) ──────────────────────────
# Keyed by content+target: identical summaries within the TTL return the
# cached condense instantly — the same 65K chars no longer trigger a
# hidden Gemini call on EVERY message (503-spam bug, Sept 2026).
_past_condense_cache: dict[str, dict] = {}   # sha256 -> {"text", "built_at"}
_PAST_CONDENSE_TTL = 600.0                   # seconds (successful condense)
_PAST_CONDENSE_FAIL_TTL = 60.0               # seconds (recent failure — don't re-hammer)
# Disk persistence: a restart must not resurrect the 15-30s hidden LLM call
# on the first over-budget message (a memory-only cache is wiped on start).
_CONDENSE_CACHE_PATH = Path(__file__).parent / ".condense_cache.json"
_condense_disk_lock = threading.Lock()
_CONDENSE_DISK_MAX = 64                      # keep the newest entries on disk


def _load_condense_disk() -> None:
    try:
        raw = json.loads(_CONDENSE_CACHE_PATH.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            _past_condense_cache.update(raw)
    except Exception:
        pass


def _save_condense_disk() -> None:
    """Write-through persist (atomic replace; keeps the newest entries)."""
    try:
        with _condense_disk_lock:
            if len(_past_condense_cache) > _CONDENSE_DISK_MAX:
                for k in sorted(
                    _past_condense_cache,
                    key=lambda k: _past_condense_cache[k].get("built_at", 0.0),
                )[:-_CONDENSE_DISK_MAX]:
                    _past_condense_cache.pop(k, None)
            tmp = _CONDENSE_CACHE_PATH.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(_past_condense_cache, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(_CONDENSE_CACHE_PATH)
    except Exception:
        pass


_load_condense_disk()


def _condense_past_summaries(raw: str, target_tokens: int, block_ok: bool = False) -> str | None:
    """LLM-condense over-budget past summaries, with sha256+TTL caching.

    Failures are negative-cached briefly too: a 503 must not turn into a
    hidden LLM call on EVERY message (the old bug).

    block_ok=False (hot path): on a cache miss NEVER wait on the LLM —
    return None immediately (the caller falls back to cheap truncation) and
    kick off a background condense that warms the disk cache for the NEXT
    message.
    """
    import time as _time
    try:
        key = hashlib.sha256(f"{raw}|{target_tokens}".encode("utf-8")).hexdigest()
    except Exception:
        key = None
    now = _time.time()
    if key:
        hit = _past_condense_cache.get(key)
        if hit is not None:
            ttl = _PAST_CONDENSE_TTL if hit["text"] else _PAST_CONDENSE_FAIL_TTL
            if (now - hit["built_at"]) < ttl:
                return hit["text"]   # None = recent failure, skip retry
    if not block_ok:
        # Hot path: never block message send on a background summarizer.
        # (A recent failure stays negative-cached, so no retry-spam.)
        threading.Thread(
            target=_condense_past_summaries,
            args=(raw, target_tokens, True),
            daemon=True,
            name="past-summary-condense",
        ).start()
        return None
    condensed = _summarize_with_llm(
        f"Compress these session summaries into key points. "
        f"Keep under {target_tokens} tokens. Preserve user preferences, "
        f"decisions, tasks, and important context. "
        f"Output only the summary:\n{raw[:25000]}",
        timeout_s=30
    )
    if key:
        _past_condense_cache[key] = {"text": condensed, "built_at": now}
        _save_condense_disk()
    return condensed


def format_turn_for_prompt(t: dict) -> str:
    role = t.get("role", "")
    content = (t.get("content") or "").strip()
    tc = t.get("tool_calls") or []
    tr = (t.get("tool_result") or "").strip()

    if role == "user":
        return f"User: {content}" if content else ""
    elif role == "assistant":
        lines = []
        if content:
            lines.append(f"Orthos: {content}")
        for call in tc:
            fn = call.get("function", {})
            name = fn.get("name", "?")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                args = args[:120]
            else:
                args = json.dumps(args, ensure_ascii=False)[:120]
            lines.append(f"  [Tool Call: {name} ({args})]")
        return "\n".join(lines)
    elif role == "tool":
        return f"  [Result: {tr[:200]}]" if tr else ""
    return ""


def format_recent_conversations(limit: int = 10) -> str:
    turns = get_recent_turns(limit)
    if not turns:
        return ""
    lines = ["[RECENT CONVERSATIONS]"]
    for t in turns:
        formatted = format_turn_for_prompt(t)
        if formatted:
            lines.append(formatted)
    return "\n".join(lines)


# ── Session Summarization ────────────────────────────────────────────────

def save_summary(session_id: str, summary: str, turn_count: int) -> None:
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute(
            "INSERT INTO summaries (session_id, summary, turn_count, created_at) VALUES (?, ?, ?, ?)",
            (session_id, summary, turn_count, datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()
    print(f"[Session] Summary saved for {session_id} ({turn_count} turns)")
    _summaries_cache.clear()   # new summary row → cached texts are stale
    invalidate_search_cache()
    try:
        from memory.chroma_memory import embed_session_summary
        embed_session_summary(session_id, summary)
    except Exception:
        pass


def get_session_summary(session_id: str) -> str | None:
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        row = conn.execute(
            "SELECT summary FROM summaries WHERE session_id = ? ORDER BY id DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        conn.close()
    return row[0] if row else None


def get_all_summaries_text() -> str:
    return _get_all_summaries_text()


_summaries_cache: dict[str, dict] = {}   # key: exclude_session or "" -> {"text", "built_at"}
_SUMMARIES_TTL = 30.0   # seconds - summaries barely change during a chat


def _get_all_summaries_text(exclude_session: str | None = None) -> str:
    """All past-session summaries (30s TTL cache, keyed by exclude_session).

    Two perf fixes (Sept 2026):
    - Titles come from ONE SQL LEFT JOIN. The old code called get_session()
      once per summary row — 471 rows = 471 queries per build.
    - The cache is keyed by exclude_session. The hot path (main loop passes
      the current session id) previously bypassed the cache entirely,
      because only the exclude_session=None variant was ever stored.
    """
    import time as _time
    now = _time.time()
    key = exclude_session or ""
    entry = _summaries_cache.get(key)
    if entry is not None and (now - entry["built_at"]) < _SUMMARIES_TTL:
        return entry["text"]
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        sql = (
            "SELECT summaries.session_id, summaries.summary, summaries.created_at, "
            "COALESCE(sessions.title, summaries.session_id) AS title "
            "FROM summaries "
            "LEFT JOIN sessions ON sessions.id = summaries.session_id "
        )
        params: tuple = ()
        if exclude_session:
            sql += "WHERE summaries.session_id != ? "
            params = (exclude_session,)
        sql += "ORDER BY summaries.id ASC"
        rows = conn.execute(sql, params).fetchall()
        conn.close()
    if not rows:
        _summaries_cache[key] = {"text": "", "built_at": now}
        return ""
    lines_out = ["[PREVIOUS CONVERSATION SUMMARIES - historical context]"]
    for _sid, summary, created_at, title in rows:
        lines_out.append(f"\n--- Session: {title or _sid} ({(created_at or '')[:10]}) ---")
        lines_out.append(summary)
    text = "\n".join(lines_out)
    _summaries_cache[key] = {"text": text, "built_at": now}
    # High-water trim: one entry per session-id key can grow unbounded.
    if len(_summaries_cache) > 64:
        for k in sorted(_summaries_cache, key=lambda k: _summaries_cache[k]["built_at"])[:-32]:
            _summaries_cache.pop(k, None)
    return text



def summarize_session(session_id: str) -> str | None:
    """Generate & save summary for a closed session. Returns summary text."""
    turns = get_turns_by_session(session_id)
    if not turns:
        return None
    turn_count = len(turns)
    all_text = _turns_to_text(turns)
    summary = _summarize_with_llm(all_text)
    if summary:
        save_summary(session_id, summary, turn_count)
    return summary


def close_and_summarize(session_id: str) -> str | None:
    """Mark session as closed and generate its summary."""
    summary = summarize_session(session_id)
    close_session(session_id)
    return summary


def estimate_messages_tokens(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        text = (m.get("content") or "")
        if isinstance(text, str):
            total += _estimate_tokens(text)
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function", {})
            fn_text = fn.get("name", "") + json.dumps(fn.get("arguments", {}), ensure_ascii=False)
            total += _estimate_tokens(fn_text)
        total += _estimate_tokens(m.get("tool_result") or "")
        total += 20
    return total


def summarize_turns(turns: list[dict], target_tokens: int = 50000) -> str | None:
    """LLM-summarize turns to <= target_tokens. NO word-drop fallback.

    Returns None when the LLM fails — the caller decides (auto-compact
    sends raw rather than butchering the text into a word soup).
    """
    text = "\n".join(format_turn_for_prompt(t) for t in turns if format_turn_for_prompt(t))
    if not text.strip():
        return None
    if _estimate_tokens(text) <= target_tokens:
        return text
    summary = _summarize_with_llm(
        text, timeout_s=30,
        target_note=f"Hard limit: under {target_tokens} tokens.",
    )
    if not summary:
        return None
    if _estimate_tokens(summary) > target_tokens:
        # One condense retry on the summary itself (still no word-drop).
        condensed = _summarize_with_llm(
            f"Condense this summary further to under {target_tokens} tokens. "
            f"Output only the summary:\n{summary}",
            timeout_s=30,
        )
        if condensed and _estimate_tokens(condensed) <= target_tokens:
            return condensed
    return summary   # best effort (may be slightly over cap)


def compress_oldest_turns(session_id: str) -> None:
    """When a session exceeds MAX_TOKENS_PER_SESSION, compress oldest turns:
    store raw text in compressed .msgpack.zst files, mark turns as archived (don't delete)."""
    turns = get_turns_by_session(session_id, include_archived=True)
    # Only ACTIVE turns count for the trigger — archived turns are already
    # compressed into files, so re-summing them keeps the session "over budget"
    # forever and triggers an infinite re-compression loop on every save (T1).
    active = [t for t in turns if not t.get("archived")]
    total_tokens = sum(_estimate_tokens(t.get("content", "")) for t in active)
    if total_tokens <= MAX_TOKENS_PER_SESSION:
        return

    keep_tokens = 0
    compress_idx = 0
    for i, t in enumerate(reversed(active)):
        keep_tokens += _estimate_tokens(t.get("content", ""))
        compress_idx = len(active) - i - 1
        if keep_tokens >= _COMPRESS_KEEP_TOKENS:
            break

    archive_turns = active[:compress_idx]
    archive_token_count = sum(_estimate_tokens(t.get("content", "")) for t in archive_turns)

    # Map back to raw DB rows (ids) for the archived subset.  ``active`` came
    # from non-archived turns only, so these ids are exactly the rows to mark.
    archive_raw = [t for t in turns if not t.get("archived")][:compress_idx]
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id FROM turns WHERE session_id = ? AND archived = 0 "
            "ORDER BY id ASC LIMIT ?",
            (session_id, compress_idx),
        ).fetchall()
        archive_ids = [r["id"] for r in rows]
        conn.close()
    if not archive_ids:
        return
    archive_turns = archive_raw

    archive_text = json.dumps(archive_turns, ensure_ascii=False, default=str)

    # Generate summary for compressed portion
    summary_text = ""
    try:
        summary_text = _summarize_with_llm(
            f"[COMPRESSED PORTION — old context preserved as compressed raw]\n{archive_text[:8000]}"
        )
    except Exception as e:
        print(f"[Session] Compression summarization skipped: {e}")
        summary_text = f"[Auto-compressed: {len(archive_turns)} turns]"

    if archive_ids:
        # Store to file instead of raw_text in SQLite
        chunk_id = int(datetime.now().timestamp())
        rel_path = _compress_to_file(session_id, chunk_id, {
            "session_id": session_id,
            "turn_ids": archive_ids,
            "turns": archive_turns,
            "summary": summary_text,
            "compressed_at": _now(),
        })

        with _lock:
            conn = sqlite3.connect(str(DB_PATH))
            conn.execute(
                "INSERT INTO compressed_turns (session_id, turn_ids, summary, archive_path, token_count, compressed_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    json.dumps(archive_ids),
                    summary_text,
                    rel_path,
                    archive_token_count,
                    _now(),
                ),
            )
            placeholders = ",".join("?" * len(archive_ids))
            conn.execute(
                f"UPDATE turns SET archived = 1 WHERE id IN ({placeholders})", archive_ids
            )
            conn.commit()
            conn.close()
        # T1 fix: subtract the archived tokens from the live cache so the
        # session is no longer counted as over-budget (stops the re-compress
        # loop).  A cold cache re-primes from non-archived turns anyway.
        if session_id in _token_count_cache:
            _token_count_cache[session_id] = max(
                0, _token_count_cache[session_id] - archive_token_count)
        print(f"[Session] Compressed {len(archive_ids)} old turns for {session_id} (~{archive_token_count} tokens -> {rel_path})")
        if summary_text:
            save_summary(session_id, f"[Compressed] {summary_text}", len(archive_ids))


# ── Search ───────────────────────────────────────────────────────────────

def _split_query(query: str) -> list[str]:
    words = query.strip().split()
    return [w for w in words if len(w) > 1]


def search_conversations(query: str, limit: int = 10) -> str:
    keywords = _split_query(query)
    if not keywords:
        keywords = [query]

    with _lock:
        conn = sqlite3.connect(str(DB_PATH))

        turn_rows = []
        summary_rows = []
        seen_turn_ids = set()
        seen_summary_ids = set()

        for kw in keywords:
            pattern = f"%{kw}%"
            rows = conn.execute(
                "SELECT id, role, content, timestamp FROM turns WHERE content LIKE ? ORDER BY id DESC LIMIT ?",
                (pattern, limit),
            ).fetchall()
            for row in rows:
                if row[0] not in seen_turn_ids:
                    seen_turn_ids.add(row[0])
                    turn_rows.append(row[1:])

            srows = conn.execute(
                "SELECT id, summary, created_at FROM summaries WHERE summary LIKE ? ORDER BY id DESC LIMIT ?",
                (pattern, limit),
            ).fetchall()
            for srow in srows:
                if srow[0] not in seen_summary_ids:
                    seen_summary_ids.add(srow[0])
                    summary_rows.append(srow[1:])

        conn.close()

    turn_rows = turn_rows[:limit]
    summary_rows = summary_rows[:limit]

    parts = []
    if turn_rows:
        lines = ["[CONVERSATION HISTORY MATCHES]"]
        for role, content, ts in reversed(turn_rows):
            content_short = (content or "")[:300]
            lines.append(f"[{ts}] {role}: {content_short}")
        parts.append("\n".join(lines))

    if summary_rows:
        lines = ["[SUMMARY MATCHES]"]
        for summary, ts in reversed(summary_rows):
            summary_short = summary[:300]
            lines.append(f"[{ts}] {summary_short}")
        parts.append("\n".join(lines))

    if not parts:
        return f"No matches found for: {query}"
    return "\n\n".join(parts)


def rebuild_fts_index() -> None:
    """Rebuild FTS5 index from turns and summaries."""
    try:
        with _lock:
            conn = sqlite3.connect(str(DB_PATH))
            # Check if FTS tables exist
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='turns_fts'")
            if cursor.fetchone():
                conn.execute("INSERT INTO turns_fts(turns_fts) VALUES('rebuild')")
                conn.execute("INSERT INTO summaries_fts(summaries_fts) VALUES('rebuild')")
            conn.close()
    except Exception:
        pass


# [REMOVED] Old hybrid_search — replaced by cached version at line ~2473


# ── Raw conversation (turns) FTS search ─────────────────────────────────
# The turns_fts virtual table indexes every raw chat turn so that past
# conversations are searchable — not just distilled facts. The backfill
# below bootstraps the index for pre-existing turns (turns_fts shipped as
# a dead table: created but never written to or queried).

_FTS_STOPWORDS = {"the", "a", "an", "of", "and", "or", "to", "in", "is", "are", "was", "were"}


def _sanitize_fts_query(query: str) -> str:
    """Make an arbitrary user query safe for FTS5 MATCH.

    Strips FTS operators/quotes, splits into terms, drops stopwords/short
    tokens, and OR-joins each term as a quoted phrase. This keeps Hinglish,
    punctuation-heavy, and operator-laden queries from crashing MATCH.
    """
    q = (query or "").strip()
    if not q:
        return ""
    cleaned = re.sub(r'["\'\(\)\*\^:\-,;]', " ", q)
    terms = []
    seen: set[str] = set()
    for t in cleaned.split():
        t = t.strip()
        low = t.lower()
        if len(t) < 2 or low in _FTS_STOPWORDS or low in seen:
            continue
        seen.add(low)
        terms.append(f'"{t}"')
    return " OR ".join(terms)


def backfill_turns_fts(batch_size: int = 2000) -> int:
    """Index turns that are missing from turns_fts (idempotent bootstrap).

    Returns the number of newly indexed rows. Safe to call repeatedly —
    only rows absent from the FTS index are inserted.
    """
    try:
        with _lock:
            conn = sqlite3.connect(str(DB_PATH), timeout=30)
            try:
                cur = conn.execute(
                    "INSERT INTO turns_fts(rowid, content, session_id) "
                    "SELECT id, content, session_id FROM turns "
                    "WHERE content IS NOT NULL AND TRIM(content) != '' "
                    "AND id NOT IN (SELECT rowid FROM turns_fts)"
                )
                conn.commit()
                n = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
            except sqlite3.OperationalError:
                n = 0  # FTS table missing — schema predates turns_fts
            finally:
                conn.close()
        if n:
            print(f"[FTS] turns_fts backfill complete: {n} turns indexed")
        return n
    except Exception as e:
        print(f"[FTS] backfill error: {e}")
        return 0


def search_turns_fts(query: str, limit: int = 8, session_id: str = "") -> list[dict]:
    """Full-text search over raw conversation turns (turns_fts).

    Returns matches with session title + highlighted snippet so the model
    can quote past conversations exactly. Falls back to LIKE scan if FTS5
    is unavailable.
    """
    q = (query or "").strip()
    if not q:
        return []
    match = _sanitize_fts_query(q)
    if not match:
        return []
    with _lock:
        conn = sqlite3.connect(str(DB_PATH), timeout=30)
        conn.row_factory = sqlite3.Row
        rows: list = []
        try:
            sql = (
                "SELECT t.id, t.role, t.content, t.timestamp, t.session_id, "
                "s.title AS session_title, "
                "snippet(turns_fts, 0, '>>', '<<', '…', 14) AS snip "
                "FROM turns_fts f "
                "JOIN turns t ON t.id = f.rowid "
                "LEFT JOIN sessions s ON s.id = t.session_id "
                "WHERE turns_fts MATCH ? AND t.archived = 0"
            )
            params: list = [match]
            if session_id:
                sql += " AND t.session_id = ?"
                params.append(session_id)
            sql += " ORDER BY rank LIMIT ?"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            # Fallback: LIKE scan (FTS unavailable)
            like = f"%{q}%"
            try:
                if session_id:
                    rows = conn.execute(
                        "SELECT t.id, t.role, t.content, t.timestamp, t.session_id, "
                        "s.title AS session_title, '' AS snip "
                        "FROM turns t LEFT JOIN sessions s ON s.id = t.session_id "
                        "WHERE t.content LIKE ? AND t.session_id = ? AND t.archived = 0 "
                        "ORDER BY t.id DESC LIMIT ?",
                        (like, session_id, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT t.id, t.role, t.content, t.timestamp, t.session_id, "
                        "s.title AS session_title, '' AS snip "
                        "FROM turns t LEFT JOIN sessions s ON s.id = t.session_id "
                        "WHERE t.content LIKE ? AND t.archived = 0 "
                        "ORDER BY t.id DESC LIMIT ?",
                        (like, limit),
                    ).fetchall()
            except Exception:
                rows = []
        except Exception:
            rows = []
        conn.close()
    out = []
    for r in rows:
        d = dict(r)
        out.append({
            "type": "turn",
            "id": d.get("id", 0),
            "session_id": d.get("session_id", ""),
            "session_title": d.get("session_title") or "(untitled)",
            "role": d.get("role", ""),
            "content": d.get("content", ""),
            "snippet": d.get("snip", "") or (d.get("content", "") or "")[:200],
            "timestamp": d.get("timestamp", ""),
        })
    return out


def get_all_turns_text() -> str:
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        rows = conn.execute(
            "SELECT role, content, tool_calls, tool_call_id, tool_result FROM turns ORDER BY id ASC"
        ).fetchall()
        conn.close()
    return _format_turns_list(rows)


def _format_turns_list(rows: list) -> str:
    parts = []
    for role, content, tool_calls_json, tool_call_id, tool_result in rows:
        tc = json.loads(tool_calls_json) if tool_calls_json else []
        tr = (tool_result or "").strip()
        if role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            if content:
                parts.append(f"Orthos: {content}")
            for call in tc:
                fn = call.get("function", {})
                name = fn.get("name", "?")
                args = fn.get("arguments", {})
                parts.append(f"  [Tool: {name} args: {json.dumps(args, ensure_ascii=False)[:200]}]")
        elif role == "tool" and tr:
            parts.append(f"  [Result: {tr[:200]}]")
    return "\n".join(parts)


def _turns_to_text(turns: list[dict]) -> str:
    parts = []
    for t in turns:
        role = t.get("role", "")
        content = (t.get("content") or "").strip()
        tc = t.get("tool_calls") or []
        tr = (t.get("tool_result") or "").strip()
        if role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            if content:
                parts.append(f"Orthos: {content}")
            for call in tc:
                fn = call.get("function", {})
                name = fn.get("name", "?")
                args = fn.get("arguments", {})
                parts.append(f"  [Tool: {name}]")
        elif role == "tool" and tr:
            parts.append(f"  [Result: {tr[:200]}]")
    return "\n".join(parts)


# ── LLM summarization call ──────────────────────────────────────────────

# ── Summarization LLM ───────────────────────────────────────────────────
# Input cap per call: 60k chars (~15k tokens). Larger inputs are CHUNKED
# (summarize parts → merge) instead of silently truncated — the old
# 12k-char cut discarded ~80% of a 65k-char past-summary corpus.
_MAX_SUMMARIZE_INPUT_CHARS = 60_000
_MAX_SUMMARIZE_CHUNKS = 6


def _summarize_with_llm(text: str, timeout_s: int = 15, target_note: str | None = None) -> str | None:
    cfg = _get_config()
    summarization_provider = (cfg.get("summarization_provider", "") or "").lower().strip()
    summarization_model = cfg.get("summarization_model", "")
    summarization_api_key = cfg.get("summarization_api_key", "")

    def _make_prompt(body: str) -> str:
        note = f"\n{target_note}" if target_note else ""
        return (
            "Summarize the following conversation into key points. "
            "Include: user preferences, facts learned about the user, tasks completed, "
            "important decisions, ongoing projects, and any context the AI must remember. "
            "Be concise but comprehensive. Output only the summary, no preamble."
            f"{note}\n\n"
            f"{body}"
        )

    def _once(prompt: str) -> str | None:
        # Dedicated Ollama summarization model (local, separate from main provider)
        if summarization_provider == "ollama" and summarization_model:
            try:
                import requests
                url = (cfg.get("summarization_url", "http://localhost:11434") or "http://localhost:11434").rstrip("/")
                payload = {
                    "model": summarization_model,
                    "messages": [
                        {"role": "system", "content": "You are a conversation summarizer."},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.3},
                }
                resp = requests.post(f"{url}/api/chat", json=payload, timeout=max(timeout_s, 60))
                resp.raise_for_status()
                data = resp.json()
                content = (data.get("message", {}).get("content") or "").strip()
                if content:
                    return content
            except Exception as e:
                print(f"[Session] Ollama summarization error ({summarization_model}): {e}")

        # Dedicated Gemini summarization model
        if summarization_model and summarization_api_key:
            try:
                from google import genai
                client = genai.Client(api_key=summarization_api_key)
                result = client.models.generate_content(
                    model=summarization_model,
                    contents=prompt,
                    config={"system_instruction": "You are a conversation summarizer."}
                )
                return result.text.strip() if result.text.strip() else None
            except Exception as e:
                print(f"[Session] Dedicated summarization LLM error: {e}")
                return None

        from core.llm_client import call_llm_text
        try:
            result = call_llm_text(prompt, system="You are a conversation summarizer.", timeout=timeout_s)
            return result.strip() if result.strip() else None
        except Exception as e:
            print(f"[Session] Summarization LLM error: {e}")
            return None

    if len(text) <= _MAX_SUMMARIZE_INPUT_CHARS:
        return _once(_make_prompt(text))

    # Chunked two-pass: summarize parts, then merge if still oversized.
    cap = _MAX_SUMMARIZE_INPUT_CHARS
    if len(text) > cap * _MAX_SUMMARIZE_CHUNKS:
        print(f"[Session] Summarize input {len(text):,} chars exceeds "
              f"{cap * _MAX_SUMMARIZE_CHUNKS:,} — analyzing first "
              f"{cap * _MAX_SUMMARIZE_CHUNKS:,} chars only")
    pieces: list[str] = []
    for i in range(_MAX_SUMMARIZE_CHUNKS):
        chunk = text[i * cap:(i + 1) * cap]
        if not chunk:
            break
        piece = _once(_make_prompt(chunk))
        if piece:
            pieces.append(piece)
    if not pieces:
        return None
    merged = "\n\n".join(pieces)
    if len(merged) > cap:
        final = _once(_make_prompt(merged))
        return final if final else merged[:cap]
    return merged


# ── Legacy JSON migration (unchanged) ────────────────────────────────────

def migrate_json_sessions() -> int:
    from memory.memory_manager import CONVERSATIONS_DIR
    if not CONVERSATIONS_DIR.exists():
        return 0
    files = sorted(CONVERSATIONS_DIR.glob("session_*.json"))
    if not files:
        return 0
    total_turns = 0
    for fpath in files:
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
            turns = data.get("turns", [])
            stamp = data.get("timestamp", fpath.stem)
            # Create a session for this JSON file
            session_id = create_session(fpath.stem.replace("session_", "")[:50])
            for turn in turns:
                role = turn.get("role", "")
                content = turn.get("content", "") or ""
                tool_calls = turn.get("tool_calls")
                tool_call_id = turn.get("tool_call_id")
                if role == "user" and not content.strip():
                    continue
                if role == "tool":
                    save_turn("tool", content, session_id, tool_call_id=tool_call_id, tool_result=content)
                elif role == "assistant":
                    save_turn("assistant", content, session_id, tool_calls=tool_calls)
                else:
                    save_turn(role, content, session_id)
                total_turns += 1
            close_session(session_id)
            try:
                save_summary(session_id, f"[Imported] {len(turns)} turns", len(turns))
            except Exception:
                pass
        except Exception as e:
            print(f"[Memory] Skipping {fpath.name}: {e}")
    if total_turns > 0:
        mark_migrated()
        print(f"[Memory] Migrated {total_turns} turns from {len(files)} JSON session files")
    return total_turns


# ── Migrate long_term.json → memory_items table ─────────────────────────

def migrate_long_term_to_sqlite() -> int:
    """Read long_term.json (legacy) and import all entries into memory_items table."""
    json_path = Path(__file__).parent / "long_term.json"
    if not json_path.exists():
        return 0
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        print("[Memory] Could not read long_term.json for migration")
        return 0

    count = 0
    for category, entries in data.items():
        if not isinstance(entries, dict):
            continue
        for key, entry in entries.items():
            # Idempotent: skip keys already present in SQLite
            if memory_items_exist(category, key):
                continue
            if isinstance(entry, dict):
                value = entry.get("value", str(entry))
                score = entry.get("score", 1.0)
                source = entry.get("source", "manual")
                acc = entry.get("access_count", 0)
                expires = entry.get("expires", None)
            else:
                value = str(entry)
                score = 1.0
                source = "manual"
                acc = 0
                expires = None
            try:
                add_memory_item(None, category, str(value)[:2000],
                                key=key, importance=float(score),
                                source=source, access_count=acc,
                                expires_at=expires)
                count += 1
            except Exception:
                pass

    print(f"[Memory] Migrated {count} long_term.json entries into memory_items table")
    return count


def dedup_memory_items() -> int:
    """Remove duplicate (type, key) rows, keeping the most recently updated.
    Returns number of rows deleted.
    """
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, type, key, updated_at FROM memory_items ORDER BY type, key, updated_at DESC"
        ).fetchall()
        seen = set()
        to_delete = []
        for r in rows:
            ident = (r["type"], r["key"])
            if ident in seen:
                to_delete.append(r["id"])
            else:
                seen.add(ident)
        if to_delete:
            conn.executemany("DELETE FROM memory_items WHERE id = ?", [(i,) for i in to_delete])
            conn.commit()
        conn.close()
    return len(to_delete)


def rebuild_memory_fts() -> int:
    """(Re)build the FTS5 index from memory_items. Idempotent.
    Returns number of rows indexed.
    """
    try:
        conn = sqlite3.connect(str(DB_PATH), timeout=30)
        conn.execute("DELETE FROM memory_fts")
        conn.execute(
            "INSERT INTO memory_fts (rowid, content, type, key) "
            "SELECT id, content, type, key FROM memory_items"
        )
        conn.commit()
        n = conn.execute("SELECT count(*) FROM memory_fts").fetchone()[0]
        conn.close()
        return n
    except Exception as e:
        print(f"[Memory] FTS rebuild failed: {e}")
        return 0



# ── Project CRUD (User → Workspace → Project) ──────────────────────────

W_DEFAULT_USER = "default_user"
W_DEFAULT_WORKSPACE = "default_workspace"
W_DEFAULT_PROJECT = "default_project"


def _ensure_default_hierarchy() -> tuple[str, str, str]:
    """Create default user → workspace → project if missing. Returns (user_id, workspace_id, project_id)."""
    user_id = W_DEFAULT_USER
    workspace_id = W_DEFAULT_WORKSPACE
    project_id = W_DEFAULT_PROJECT
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("INSERT OR IGNORE INTO users (id, name, created_at) VALUES (?, ?, ?)",
                     (user_id, "default", _now()))
        conn.execute("INSERT OR IGNORE INTO workspaces (id, user_id, name, created_at) VALUES (?, ?, ?, ?)",
                     (workspace_id, user_id, "default", _now()))
        conn.execute("INSERT OR IGNORE INTO projects (id, workspace_id, name, created_at) VALUES (?, ?, ?, ?)",
                     (project_id, workspace_id, "default", _now()))
        conn.commit()
        conn.close()
    return user_id, workspace_id, project_id


def list_projects() -> list[dict]:
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM projects ORDER BY created_at DESC").fetchall()
        conn.close()
    return [dict(r) for r in rows]


def create_project(name: str, workspace_id: str = W_DEFAULT_WORKSPACE) -> str:
    pid = uuid.uuid4().hex[:12]
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("INSERT INTO projects (id, workspace_id, name, created_at) VALUES (?, ?, ?, ?)",
                     (pid, workspace_id, name, _now()))
        conn.commit()
        conn.close()
    return pid


def get_project(project_id: str) -> dict | None:
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        conn.close()
    return dict(row) if row else None


# ── Weighted retrieval (hybrid + weights) ───────────────────────────────

_RETRIEVAL_WEIGHTS = {
    "cosine": 0.30,
    "importance": 0.20,
    "recency": 0.15,
    "frequency": 0.10,
    "source_quality": 0.10,
    "project_match": 0.10,
    "session_match": 0.05,
}

@lru_cache(maxsize=128)
def _cached_hybrid_search(query: str, limit: int = 5, project_id: str = "",
                          session_id: str = "") -> list[dict]:
    """Cached hybrid search with weighted scoring."""
    results = []
    now_ts = datetime.now()

    # 1. FTS5 search on summaries
    try:
        with _lock:
            conn = sqlite3.connect(str(DB_PATH))
            try:
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='summaries_fts'")
                if cursor.fetchone():
                    fts_rows = conn.execute(
                        "SELECT s.id, s.session_id, s.summary, s.turn_count, s.created_at "
                        "FROM summaries_fts f JOIN summaries s ON f.rowid = s.id "
                        "WHERE summaries_fts MATCH ? LIMIT ?",
                        (query, limit * 2),
                    ).fetchall()
                    for r in fts_rows:
                        results.append({
                            "type": "summary",
                            "id": r[0],
                            "session_id": r[1],
                            "content": r[2],
                            "turn_count": r[3] if len(r) > 3 else 0,
                            "created_at": r[4] if len(r) > 4 else "",
                            "cosine_sim": 0.5,
                            "importance": 0.8,
                            "source_quality": "system_inferred",
                        })
            except Exception:
                pass
            conn.close()
    except Exception:
        pass

    # 2. Keyword search on memory_items
    try:
        mem_results = search_memory_items(query, limit * 2)
        for m in mem_results:
            created = m.get("created_at", _now())
            days_old = (now_ts - datetime.fromisoformat(created)).days if created else 365
            results.append({
                "type": "memory",
                "id": m.get("id", 0),
                "session_id": m.get("session_id", ""),
                "content": m.get("content", ""),
                "cosine_sim": 0.4,
                "importance": m.get("importance", 0.5),
                "project_id": m.get("project_id", ""),
                "access_count": m.get("access_count", 0),
                "source_quality": m.get("source_quality", "system_inferred"),
                "days_old": days_old,
            })
    except Exception:
        pass

    # 2b. Raw conversation turns (FTS over turns_fts) — past chat is
    #     searchable alongside distilled facts.
    try:
        turn_results = search_turns_fts(query, limit * 2)
        for t in turn_results:
            ts = t.get("timestamp", "")
            days_old = 365
            if ts:
                try:
                    days_old = max(0, (now_ts - datetime.fromisoformat(str(ts)[:19])).days)
                except Exception:
                    pass
            results.append({
                "type": "turn",
                "id": t.get("id", 0),
                "session_id": t.get("session_id", ""),
                "session_title": t.get("session_title", ""),
                "role": t.get("role", ""),
                "content": t.get("snippet") or t.get("content", ""),
                "cosine_sim": 0.45,
                "importance": 0.5,
                "source_quality": "raw_conversation",
                "days_old": days_old,
            })
    except Exception:
        pass

    # 3. Score each result with weighted formula
    scored = []
    for r in results:
        cos = r.get("cosine_sim", 0.0)
        imp = r.get("importance", 0.5)
        days = r.get("days_old", 365)
        freq = min(r.get("access_count", 0) / 20.0, 1.0)
        sq_map = {"user_explicit": 1.0, "reflection": 0.8, "system_inferred": 0.5, "manual": 0.7, "tool": 0.6}
        sq = sq_map.get(r.get("source_quality", "system_inferred"), 0.5)
        pm = 1.0 if project_id and r.get("project_id") == project_id else 0.5
        sm = 1.0 if session_id and r.get("session_id") == session_id else 0.3
        recency = max(0.0, 1.0 - days / 365.0)

        score = (
            _RETRIEVAL_WEIGHTS["cosine"] * cos +
            _RETRIEVAL_WEIGHTS["importance"] * imp +
            _RETRIEVAL_WEIGHTS["recency"] * recency +
            _RETRIEVAL_WEIGHTS["frequency"] * freq +
            _RETRIEVAL_WEIGHTS["source_quality"] * sq +
            _RETRIEVAL_WEIGHTS["project_match"] * pm +
            _RETRIEVAL_WEIGHTS["session_match"] * sm
        )
        r["score"] = round(score, 4)
        scored.append(r)

    scored.sort(key=lambda x: x["score"], reverse=True)
    seen = set()
    deduped = []
    for r in scored:
        key = f"{r['type']}_{r['id']}"
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped[:limit]


def hybrid_search(query: str, limit: int = 5, project_id: str = "",
                  session_id: str = "") -> list[dict]:
    """Hybrid search with weighted scoring. Uses LRU cache."""
    return _cached_hybrid_search(query, limit, project_id, session_id)


def invalidate_search_cache() -> None:
    """Call after any memory save to invalidate retrieval cache."""
    _cached_hybrid_search.cache_clear()


# ── Prompt log ──────────────────────────────────────────────────────────

def log_prompt(prompt_hash: str, prompt_text: str, retrieved_ids: list,
               response_text: str, quality_score: float | None = None,
               latency_ms: int = 0, accepted: bool = False) -> int:
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.execute(
            "INSERT INTO prompt_log (prompt_hash, prompt_text, retrieved_ids, response_text, "
            "quality_score, latency_ms, accepted, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (prompt_hash, prompt_text, json.dumps(retrieved_ids), response_text,
             quality_score, latency_ms, 1 if accepted else 0, _now()),
        )
        conn.commit()
        pid = c.lastrowid
        conn.close()
    return pid


# ── Retrieval log ───────────────────────────────────────────────────────

def log_retrieval(query_text: str, retrieved_ids: list, latency_ms: int = 0,
                  hit_rate: float = 0.0, accepted: bool = False) -> int:
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.execute(
            "INSERT INTO retrieval_log (query_text, retrieved_ids, latency_ms, hit_rate, accepted, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (query_text, json.dumps(retrieved_ids), latency_ms, hit_rate, 1 if accepted else 0, _now()),
        )
        conn.commit()
        rid = c.lastrowid
        conn.close()
    return rid


# ── Memory versioning ───────────────────────────────────────────────────

def save_memory_version(memory_item_id: int, content: str, importance: float = 1.0,
                        source_quality: str = "system_inferred") -> int:
    """Create a new version of a memory item. Increments version counter."""
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        row = conn.execute("SELECT version FROM memory_items WHERE id = ?", (memory_item_id,)).fetchone()
        old_version = row[0] if row else 0
        new_version = old_version + 1
        conn.execute(
            "INSERT INTO memory_versions (memory_item_id, version, content, importance, source_quality, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (memory_item_id, new_version, content, importance, source_quality, _now()),
        )
        # Bump the main item's version
        conn.execute("UPDATE memory_items SET version = ?, updated_at = ? WHERE id = ?",
                     (new_version, _now(), memory_item_id))
        conn.commit()
        vid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.close()
    return vid


def get_memory_versions(memory_item_id: int) -> list[dict]:
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM memory_versions WHERE memory_item_id = ? ORDER BY version DESC",
            (memory_item_id,),
        ).fetchall()
        conn.close()
    return [dict(r) for r in rows]


# ── Tool memory ─────────────────────────────────────────────────────────

def set_tool_memory(tool_name: str, session_id: str, key: str, value: str) -> None:
    global _MEM_REV
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        row = conn.execute(
            "SELECT version FROM tool_memory WHERE tool_name = ? AND session_id = ? AND key = ?",
            (tool_name, session_id, key),
        ).fetchone()
        ver = (row[0] + 1) if row else 1
        conn.execute(
            "INSERT OR REPLACE INTO tool_memory (tool_name, session_id, key, value, version, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM tool_memory WHERE tool_name=? AND session_id=? AND key=?), ?), ?)",
            (tool_name, session_id, key, value, ver,
             tool_name, session_id, key, _now(), _now()),
        )
        conn.commit()
        conn.close()
    _MEM_REV += 1


def get_tool_memory(tool_name: str, session_id: str) -> dict[str, str]:
    with _lock:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT key, value FROM tool_memory WHERE tool_name = ? AND session_id = ? ORDER BY id ASC",
            (tool_name, session_id),
        ).fetchall()
        conn.close()
    return {r["key"]: r["value"] for r in rows}
