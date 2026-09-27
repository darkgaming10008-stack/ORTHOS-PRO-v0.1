"""
ChromaDB-based vector memory layer over the SQLite memory_items store.

Provides:
  - ChromaMemory:   direct ChromaDB PersistentClient wrapper
  - MemoryBridge:   unified interface between SQLite + ChromaDB
  - Compatibility:  existing callers of load_memory(), save_memory(),
                    search_all_memories() continue to work.
"""

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ── Configuration flag ─────────────────────────────────────────────────────

CHROMA_VECTOR_SEARCH_ENABLED = True

# Allow override via api_keys.json config at module load time
try:
    from memory.config_manager import load_api_keys
    _cfg = load_api_keys()
    if "CHROMA_VECTOR_SEARCH_ENABLED" in _cfg:
        CHROMA_VECTOR_SEARCH_ENABLED = bool(_cfg["CHROMA_VECTOR_SEARCH_ENABLED"])
except Exception:
    pass


def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR = get_base_dir()
CHROMA_DB_PATH = str(BASE_DIR / "memory" / "chroma_db")

_EMBED_MODEL = None
_EMBED_LOCK = Lock()

# ── Embedding helpers ──────────────────────────────────────────────────

_EMBED_FN_INSTANCE = None


def _get_embed_fn():
    """Lazily build the local SentenceTransformer embedding function.

    This constructor loads the model AND checks the HuggingFace hub —
    it measured 47.5 SECONDS on the target i3 (offline, firewall). It was
    module-level, so EVERY app start (even pure cloud mode) paid it.
    Now only actual local-embedding fallbacks pay it.
    """
    global _EMBED_FN_INSTANCE
    if _EMBED_FN_INSTANCE is None:
        with _EMBED_LOCK:
            if _EMBED_FN_INSTANCE is None:
                _EMBED_FN_INSTANCE = SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2",
                )
    return _EMBED_FN_INSTANCE



def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts for ChromaDB.

    Prefers the memory service's /embed endpoint (the MiniLM L6 v2 model is
    already preloaded at memory service startup) — this avoids loading a
    duplicate SentenceTransformer model inside the app process, which was the
    main cause of slow first-message latency. Falls back to the local
    CHROMA_EMBED_FN only when the service is unavailable.
    """
    try:
        from memory.memory_client import _is_healthy, memory_embed
        if _is_healthy():
            vecs = memory_embed(texts)
            return [v.tolist() for v in vecs]
    except Exception:
        pass
    return _get_embed_fn()(texts)


def _is_remote_memory() -> bool:
    """True when the active memory service is NOT on this machine.

    In that (cloud) mode the server owns the vector index: writes are routed
    to the service so embeddings are computed on the fast machine and the
    remote index stays coherent. Reads stay on the same service via the
    existing client helpers.
    """
    try:
        from memory.memory_client import get_service_url, service_url, _LOCAL_URL
        return service_url() != _LOCAL_URL
    except Exception:
        return False


def _make_metadata(
    category: str,
    key: str,
    value: str,
    extra: Optional[dict] = None,
) -> dict:
    """Build a flat metadata dict from an entry dict.
    Lists/dicts (history, contradictions) are serialised to JSON strings
    so ChromaDB can store them.
    """
    meta = {
        "category": category,
        "key": key,
        "value": value,
    }
    if extra:
        for k, v in extra.items():
            if isinstance(v, (str, int, float, bool)):
                meta[k] = v
            elif v is None:
                meta[k] = ""
            else:
                meta[k] = json.dumps(v, ensure_ascii=False)
    return meta


def _parse_extra_metadata(meta: dict) -> dict:
    """Reverse of _make_metadata — deserialise JSON fields."""
    result = {}
    for k, v in meta.items():
        if k in ("category", "key", "value"):
            continue
        if isinstance(v, str) and v.startswith("["):
            try:
                result[k] = json.loads(v)
            except Exception:
                result[k] = v
        elif isinstance(v, str) and v.startswith("{"):
            try:
                result[k] = json.loads(v)
            except Exception:
                result[k] = v
        else:
            result[k] = v
    return result


# ── ChromaMemory ───────────────────────────────────────────────────────────

class ChromaMemory:
    """ChromaDB-backed persistent vector storage for long-term memories."""

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: str = "long_term_memory",
    ):
        self.persist_dir = persist_dir or CHROMA_DB_PATH
        self.collection_name = collection_name
        self._lock = Lock()
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection = None
        self._ready = False
        self._init()

    # ── internal ────────────────────────────────────────────────────────

    def _init(self) -> None:
        # Cloud mode: the service owns the index — never open the local
        # PersistentClient (wasted disk handles, and local reads would be
        # stale anyway). Reads route through _remote_search, writes are
        # already POSTed to the service.
        if _is_remote_memory():
            self._ready = False
            return
        try:
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=_get_embed_fn(),
                metadata={"hnsw:space": "cosine"},
            )
            self._ready = True
        except Exception as e:
            print(f"[ChromaMemory] Init error: {e}")
            self._ready = False

    def _ensure_ready(self) -> None:
        if not self._ready:
            self._init()

    def _doc_id(self, category: str, key: str) -> str:
        return f"{category}/{key}"

    def _split_doc_id(self, doc_id: str):
        parts = doc_id.split("/", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return "notes", parts[0]

    # ── public API ──────────────────────────────────────────────────────

    def health(self) -> dict:
        if _is_remote_memory():
            try:
                from memory.memory_client import health as _svc_health
                svc = _svc_health()
                if svc:
                    return {"status": "healthy", "mode": "cloud", "service": svc}
                return {"status": "unhealthy", "mode": "cloud",
                        "error": "cloud service unreachable"}
            except Exception as e:
                return {"status": "unhealthy", "mode": "cloud", "error": str(e)}
        self._ensure_ready()
        if not self._ready:
            return {"status": "unhealthy", "error": "ChromaDB not initialised"}
        try:
            count = self._collection.count()
            return {
                "status": "healthy",
                "collection": self.collection_name,
                "count": count,
                "persist_dir": self.persist_dir,
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def add_memory(
        self,
        category: str,
        key: str,
        value: str,
        metadata: Optional[dict] = None,
    ) -> bool:
        # Cloud mode: the server owns the index — route the write there.
        if _is_remote_memory():
            try:
                from memory.memory_client import memory_upsert_memory
                return memory_upsert_memory({
                    "category": category, "key": key, "value": value,
                })
            except Exception:
                return False
        self._ensure_ready()
        if not self._ready:
            return False
        doc_id = self._doc_id(category, key)
        meta = _make_metadata(category, key, value, metadata)
        with self._lock:
            try:
                # H1 fix: use upsert — `add` raises on an existing doc_id, so
                # every re-save of an existing fact silently kept the OLD
                # embedding/value in Chroma while SQLite had the new one.
                self._collection.upsert(
                    documents=[value],
                    embeddings=_embed_texts([value]),
                    metadatas=[meta],
                    ids=[doc_id],
                )
                return True
            except Exception as e:
                print(f"[ChromaMemory] add_memory error: {e}")
                return False

    # ── Cloud-mode read routing ─────────────────────────────────────────────

    def _remote_search(self, query: str, n_results: int) -> list[dict] | None:
        """Query the cloud service's vector index; None when cloud is down.

        Results are formatted identically to _format_results() so callers
        (prompt builder, MemoryBridge, memory_manager fusion) can't tell the
        difference between local and cloud reads.
        """
        try:
            from memory.memory_client import _is_healthy, memory_search_vector
            if not _is_healthy():
                return None
            out = []
            for text, source, score in memory_search_vector(query, top_k=n_results):
                if not isinstance(source, dict):
                    source = {"type": str(source)}
                entry = {
                    "id": f"{source.get('category', '')}/{source.get('key', '')}",
                    "value": text,
                    "score": float(score),
                }
                if source.get("category"):
                    entry["category"] = source.get("category")
                if source.get("key"):
                    entry["key"] = source.get("key")
                entry.update(_parse_extra_metadata(source))
                out.append(entry)
            return out
        except Exception as e:
            print(f"[ChromaMemory] cloud search failed: {e}")
            return None

    def search_memory(self, query: str, n_results: int = 10) -> list[dict]:
        """Vector search — routes to the cloud service in cloud mode.

        Cloud-first design: when a remote service is configured and healthy,
        reads go there (single source of truth). Local ChromaDB is only used
        when the service is genuinely local.
        """
        if _is_remote_memory():
            remote = self._remote_search(query, n_results)
            if remote is not None:
                return remote
            return []   # cloud-only mode: no stale local reads
        self._ensure_ready()
        if not self._ready:
            return []
        with self._lock:
            try:
                results = self._collection.query(
                    query_embeddings=_embed_texts([query]),
                    n_results=n_results,
                )
                return self._format_results(results)
            except Exception as e:
                print(f"[ChromaMemory] search_memory error: {e}")
                return []

    def search_by_category(
        self,
        category: str,
        query: str,
        n_results: int = 5,
    ) -> list[dict]:
        if _is_remote_memory():
            remote = self._remote_search(query, n_results)
            if remote is None:
                return []
            return [r for r in remote if r.get("category", "") == category]
        self._ensure_ready()
        if not self._ready:
            return []
        with self._lock:
            try:
                results = self._collection.query(
                    query_embeddings=_embed_texts([query]),
                    n_results=n_results,
                    where={"category": category},
                )
                return self._format_results(results)
            except Exception as e:
                print(f"[ChromaMemory] search_by_category error: {e}")
                return []

    def get_all_memories(self) -> list[dict]:
        if self.collection_name == TURNS_COLLECTION:
            self._ensure_ready()
            if not self._ready:
                return []
            with self._lock:
                try:
                    all_data = self._collection.get()
                    return self._format_get_result(all_data)
                except Exception as e:
                    print(f"[ChromaMemory] get_all_memories error: {e}")
                    return []
        # Facts collection: SQLite memory_items is the source-of-truth in
        # BOTH modes (local mirror can lag; cloud never mirrors back).
        return _load_all_memory_entries(self.collection_name)

    def delete_memory(self, category: str, key: str) -> bool:
        """Cloud mode: DELETE goes to the service too — forgetting must work
        on the authoritative index, not just the local mirror."""
        if self.collection_name != TURNS_COLLECTION and _is_remote_memory():
            try:
                from memory.memory_client import memory_delete_memory
                return memory_delete_memory(category, key)
            except Exception:
                return False
        self._ensure_ready()
        if not self._ready:
            return False
        doc_id = self._doc_id(category, key)
        with self._lock:
            try:
                self._collection.delete(ids=[doc_id])
                return True
            except Exception:
                return False

    def update_memory(
        self,
        category: str,
        key: str,
        new_value: str,
        metadata: Optional[dict] = None,
    ) -> bool:
        """Cloud mode: an update is just an upsert on the service-owned store."""
        if self.collection_name != TURNS_COLLECTION and _is_remote_memory():
            try:
                from memory.memory_client import memory_upsert_memory
                return memory_upsert_memory({
                    "category": category, "key": key, "value": str(new_value)[:2000],
                })
            except Exception:
                return False
        self._ensure_ready()
        if not self._ready:
            return False
        doc_id = self._doc_id(category, key)
        meta = _make_metadata(category, key, new_value, metadata)
        with self._lock:
            try:
                self._collection.update(
                    documents=[new_value],
                    metadatas=[meta],
                    ids=[doc_id],
                )
                return True
            except Exception as e:
                print(f"[ChromaMemory] update_memory error: {e}")
                try:
                    self._collection.add(
                        documents=[new_value],
                        metadatas=[meta],
                        ids=[doc_id],
                    )
                    return True
                except Exception:
                    return False

    def count(self) -> int:
        if self.collection_name != TURNS_COLLECTION:
            return len(_load_all_memory_entries(self.collection_name))
        self._ensure_ready()
        if not self._ready:
            return 0
        with self._lock:
            try:
                return self._collection.count()
            except Exception:
                return 0

    def get_collection_stats(self) -> dict:
        if _is_remote_memory() and self.collection_name != TURNS_COLLECTION:
            entries = _load_all_memory_entries(self.collection_name)
            categories: dict = {}
            for e in entries:
                cat = e.get("category", "unknown")
                categories[cat] = categories.get(cat, 0) + 1
            return {
                "total_count": len(entries),
                "categories": categories,
                "mode": "cloud",
                "collection": self.collection_name,
                "embedding_model": "all-MiniLM-L6-v2",
            }
        self._ensure_ready()
        if not self._ready:
            return {"status": "not_ready"}
        with self._lock:
            try:
                cnt = self._collection.count()
                all_meta = self._collection.get(limit=cnt) if cnt > 0 else {"metadatas": []}
                categories = {}
                for m in (all_meta.get("metadatas") or []):
                    cat = m.get("category", "unknown") if m else "unknown"
                    categories[cat] = categories.get(cat, 0) + 1
                return {
                    "total_count": cnt,
                    "categories": categories,
                    "persist_dir": self.persist_dir,
                    "collection": self.collection_name,
                    "embedding_model": "all-MiniLM-L6-v2",
                }
            except Exception as e:
                return {"status": "error", "error": str(e)}

    # ── result formatters ───────────────────────────────────────────────

    @staticmethod
    def _format_results(results) -> list[dict]:
        out = []
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        documents = results.get("documents", [[]])[0]
        for i in range(len(ids)):
            meta = metadatas[i] if i < len(metadatas) else {}
            doc = documents[i] if i < len(documents) else ""
            entry = {
                "id": ids[i],
                "value": doc,
                "score": 1.0 - (distances[i] if i < len(distances) else 0.0),
            }
            if meta:
                entry["category"] = meta.get("category", "")
                entry["key"] = meta.get("key", "")
                entry.update(_parse_extra_metadata(meta))
            out.append(entry)
        return out

    @staticmethod
    def _format_get_result(data) -> list[dict]:
        out = []
        ids = data.get("ids", [])
        metadatas = data.get("metadatas", [])
        documents = data.get("documents", [])
        for i in range(len(ids)):
            meta = metadatas[i] if i < len(metadatas) else {}
            doc = documents[i] if i < len(documents) else ""
            entry = {
                "id": ids[i],
                "value": doc,
            }
            if meta:
                entry["category"] = meta.get("category", "")
                entry["key"] = meta.get("key", "")
                entry.update(_parse_extra_metadata(meta))
            out.append(entry)
        return out


# ── MemoryBridge ───────────────────────────────────────────────────────────

# Shared singleton
_chroma_instance: Optional[ChromaMemory] = None
_bridge_lock = Lock()


def _get_chroma() -> ChromaMemory:
    global _chroma_instance
    if _chroma_instance is None:
        with _bridge_lock:
            if _chroma_instance is None:
                _chroma_instance = ChromaMemory()
    return _chroma_instance


class MemoryBridge:
    """Unified interface: keeps existing JSON memory working while adding
    ChromaDB as a faster vector-search alternative."""

    def __init__(self):
        self.chroma = _get_chroma()

    # ── search ──────────────────────────────────────────────────────────

    def search_all(self, query: str, n_results: int = 10, mode: str = "summary",
                   project_id: str = "") -> list[dict]:
        """Search ChromaDB first; fall back to JSON-based search if ChromaDB
        is unavailable or returns no results.
        
        mode="summary" — return ChromaDB results as-is (summaries stay as text).
        mode="expand"  — resolve session_summary pointers to raw SQLite turns.
        project_id     — scope results to a specific project (filters metadata).
        """
        if not CHROMA_VECTOR_SEARCH_ENABLED:
            return self._search_json(query, n_results)

        results = self.chroma.search_memory(query, n_results=n_results)
        if results:
            # Filter by project_id if specified
            if project_id:
                results = [r for r in results if r.get("project_id", "") == project_id or not r.get("project_id")]
            if mode == "expand":
                results = resolve_summaries(results)
            return results

        # Fallback to JSON-based keyword search
        return self._search_json(query, n_results)

    def _search_json(self, query: str, n_results: int = 10) -> list[dict]:
        """Simple keyword search over memory as fallback."""
        try:
            from memory.memory_manager import load_memory
            memory = load_memory()
        except Exception:
            return []

        q = query.lower()
        results = []
        for category, entries in memory.items():
            if not isinstance(entries, dict):
                continue
            for key, entry in entries.items():
                val = entry.get("value") if isinstance(entry, dict) else entry
                if not val:
                    continue
                if q in str(val).lower() or q in key.lower() or q in category.lower():
                    results.append({
                        "id": f"{category}/{key}",
                        "category": category,
                        "key": key,
                        "value": str(val),
                        "score": float(entry.get("score", 1.0)) if isinstance(entry, dict) else 1.0,
                    })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:n_results]

    # ── migration ───────────────────────────────────────────────────────

    def migrate_from_json(self) -> dict:
        """Import all memory data into ChromaDB.
        Returns stats about what was migrated.
        """
        try:
            from memory.memory_manager import load_memory
            memory = load_memory()
        except Exception:
            memory = {}

        stats = {"added": 0, "skipped": 0, "errors": 0, "categories": {}}
        for category, entries in memory.items():
            if not isinstance(entries, dict):
                continue
            cat_count = 0
            for key, entry in entries.items():
                val = entry.get("value") if isinstance(entry, dict) else entry
                if not val:
                    stats["skipped"] += 1
                    continue
                extra = entry if isinstance(entry, dict) else None
                ok = self.chroma.add_memory(category, key, str(val), metadata=extra)
                if ok:
                    stats["added"] += 1
                    cat_count += 1
                else:
                    stats["errors"] += 1
            if cat_count > 0:
                stats["categories"][category] = cat_count

        stats["total_before"] = sum(stats["categories"].values())
        print(f"[MemoryBridge] Migrated entries from JSON -> ChromaDB")
        return stats

    def count(self) -> int:
        return self.chroma.count()

    def health(self) -> dict:
        return self.chroma.health()

    def stats(self) -> dict:
        return self.chroma.get_collection_stats()


# ── Compatibility shim ─────────────────────────────────────────────────────
# Existing code calling load_memory(), save_memory(), search_all_memories()
# continues to work unchanged.

from memory import memory_manager as _mm

load_memory = _mm.load_memory
save_memory = _mm.save_memory
update_memory = _mm.update_memory
remember = _mm.remember
forget = _mm.forget
forget_memory = _mm.forget_memory
format_memory_for_prompt = _mm.format_memory_for_prompt
format_memory_summary = _mm.format_memory_summary
format_core_memory = _mm.format_core_memory
clear_expired_memories = _mm.clear_expired_memories
enforce_memory_cap = _mm.enforce_memory_cap
context_status = _mm.context_status
archival_memory_search = _mm.archival_memory_search
core_memory_append = _mm.core_memory_append
core_memory_replace = _mm.core_memory_replace
load_recent_conversations = _mm.load_recent_conversations
save_conversation = _mm.save_conversation


def search_all_memories(query: str, limit: int = 10, mode: str = "expand",
                        project_id: str = "") -> str:
    """Overloaded search: always runs the true multi-signal fusion
    (BM25 + SQLite FTS5 + ChromaDB vector + entity graph, weighted + reranked).

    Previously this short-circuited to Chroma-only when Chroma was enabled,
    silently bypassing the keyword/FTS/entity signals. The fusion in
    memory_manager.search_all_memories includes the Chroma signal itself,
    so delegating always gives the full 5-signal result.

    mode / project_id are kept for API compatibility; summary-only rendering
    (mode="summary") is applied on top of the fused results.
    """
    fused = _mm.search_all_memories(query, limit=limit)

    # mode="summary" — keep the fused result but strip raw conversation
    # expansions that resolve_summaries() used to inject (token bomb guard).
    if mode == "summary":
        return fused

    return fused


# ── Session summary embedding / resolution ─────────────────────────────


def _extract_keywords(text: str, max_keywords: int = 10) -> str:
    """Extract topical keywords (capitalized tech terms, proper nouns) from summary text."""
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


def embed_session_summary(session_id: str, summary: str) -> bool:
    """Store session summary in ChromaDB as a pointer to raw SQLite turns.
    Uses upsert so per-session duplicate summaries update rather than collide."""
    try:
        from memory.conversation_db import get_session
        session = get_session(session_id)
        title = session["title"] if session else session_id
        project_id = session.get("project_id", "") if session else ""
        chroma = _get_chroma()
        keywords = _extract_keywords(summary)
        metadata = {
            "type": "session_summary",
            "session_id": session_id,
            "title": title,
            "project_id": project_id,
            "keywords": keywords,
            "created_at": datetime.now().isoformat(),
        }
        return chroma.update_memory("session_summary", session_id, summary, metadata=metadata)
    except Exception as e:
        print(f"[ChromaMemory] embed_session_summary error: {e}")
        return False


def resolve_summaries(results: list[dict]) -> list[dict]:
    """Post-process ChromaDB results: resolve session_summary pointers to raw conversation turns.
    Returns results with 'value' replaced by the full formatted conversation."""
    try:
        from memory.conversation_db import get_turns_by_session, format_turn_for_prompt
    except Exception:
        return results
    out = []
    for r in results:
        if r.get("category") == "session_summary":
            session_id = r.get("session_id") or r.get("key", "")
            turns = get_turns_by_session(session_id)
            if turns:
                conv_lines = ["[RAW CONVERSATION — loaded from session]"]
                for t in turns:
                    formatted = format_turn_for_prompt(t)
                    if formatted:
                        conv_lines.append(formatted)
                r["value"] = "\n".join(conv_lines)
                r["resolved"] = True
                r["raw_turns"] = len(turns)
        out.append(r)
    return out


def migrate_summaries_to_chroma() -> dict:
    """One-time migration: embed all existing SQLite session summaries into ChromaDB.
    Uses batched ChromaDB add for performance. Returns stats."""
    import sqlite3
    from memory.conversation_db import DB_PATH, get_session
    chroma = _get_chroma()
    if not chroma._ready:
        return {"total": 0, "embedded": 0, "skipped": 0, "errors": 0, "error": "ChromaDB not ready"}

    try:
        conn = sqlite3.connect(str(DB_PATH))
        rows = conn.execute(
            "SELECT id, session_id, summary FROM summaries ORDER BY id ASC"
        ).fetchall()
        conn.close()
    except Exception as e:
        return {"total": 0, "embedded": 0, "skipped": 0, "errors": 0, "error": str(e)}

    stats = {"total": len(rows), "embedded": 0, "skipped": 0, "errors": 0}

    # Deduplicate: per session_id, keep the latest summary (max id)
    latest_per_session = {}
    for row_id, session_id, summary in rows:
        if session_id not in latest_per_session or row_id > latest_per_session[session_id][0]:
            latest_per_session[session_id] = (row_id, summary)

    batch_docs = []
    batch_metas = []
    batch_ids = []

    for session_id, (row_id, summary) in latest_per_session.items():
        if not summary or not summary.strip():
            stats["skipped"] += 1
            continue
        session = get_session(session_id)
        title = session["title"] if session else session_id
        project_id = session.get("project_id", "") if session else ""
        keywords = _extract_keywords(summary)

        doc_id = f"session_summary/{session_id}"
        meta = _make_metadata("session_summary", session_id, summary, {
            "type": "session_summary",
            "session_id": session_id,
            "title": title,
            "project_id": project_id,
            "keywords": keywords,
            "created_at": str(datetime.now().isoformat()),
        })

        batch_docs.append(summary)
        batch_metas.append(meta)
        batch_ids.append(doc_id)
        stats["embedded"] += 1

    stats["total"] = len(latest_per_session)

    if batch_docs:
        try:
            chroma._collection.upsert(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids,
            )
            print(f"[ChromaMemory] Summary migration: {stats['embedded']}/{stats['total']} embedded "
                  f"({stats['skipped']} skipped, {stats['errors']} errors)")
        except Exception as e:
            stats["errors"] = len(batch_docs)
            print(f"[ChromaMemory] Summary migration error: {e}")
    else:
        print("[ChromaMemory] No summaries to migrate")

    return stats


def _load_all_memory_entries(collection_name: str = "long_term_memory") -> list[dict]:
    """All memory entries as [{id, category, key, value}] — collection-aware.

    Default collection: read the SQLite memory_items table (the durable
    source-of-truth; every write path — remember(), core_memory_append,
    upsert_memory — lands there first, locally or on the cloud). The local
    ChromaDB copy is only a mirror, so SQLite is correct in BOTH modes.
    conversation_turns collection: turns live only in the vector mirror.
    """
    if collection_name == TURNS_COLLECTION:
        try:
            chroma = _get_turns_chroma()
            if not chroma._ready:
                return []
            with chroma._lock:
                all_data = chroma._collection.get()
            return ChromaMemory._format_get_result(all_data)
        except Exception:
            return []
    try:
        from memory.conversation_db import get_memory_items
        items = get_memory_items(limit=500)
        out = []
        for it in items:
            out.append({
                "id": f"{it.get('type', '')}/{it.get('key', '')}",
                "category": it.get("type", ""),
                "key": it.get("key", ""),
                "value": it.get("content", ""),
            })
        return out
    except Exception as e:
        print(f"[ChromaMemory] _load_all_memory_entries error: {e}")
        return []


def get_core_memories() -> list[dict]:
    """Get all core memories (identity, preferences, projects, etc.) from ChromaDB.
    Excludes session_summary entries — those are for search, not core memory injection."""
    chroma = _get_chroma()
    all_mem = chroma.get_all_memories()
    return [m for m in all_mem if m.get("category") != "session_summary"]


# ── Module-level convenience ───────────────────────────────────────────────

def get_bridge() -> MemoryBridge:
    return MemoryBridge()


# ── Raw conversation turn vectors (semantic search over past chat) ──────────
# A dedicated collection so raw turns never pollute the fact store. Doc ids
# are deterministic ("turn/<id>") so upserts are idempotent.

TURNS_COLLECTION = "conversation_turns"

_turns_instance: Optional[ChromaMemory] = None
_turns_lock = Lock()


def _get_turns_chroma() -> ChromaMemory:
    global _turns_instance
    if _turns_instance is None:
        with _turns_lock:
            if _turns_instance is None:
                _turns_instance = ChromaMemory(collection_name=TURNS_COLLECTION)
    return _turns_instance


def add_turn_embeddings(rows: list[dict]) -> int:
    """Batch-upsert raw turns into the conversation_turns collection.

    rows: [{id, role, content, session_id, timestamp}]. Returns embedded count.
    Cloud mode: the rows are POSTed to the service (server-owned index).
    """
    if not rows or not CHROMA_VECTOR_SEARCH_ENABLED:
        return 0
    if _is_remote_memory():
        try:
            from memory.memory_client import memory_add_turn_embeddings
            return memory_add_turn_embeddings(rows)
        except Exception:
            return 0
    try:
        chroma = _get_turns_chroma()
        if not chroma._ready:
            return 0
        docs, ids, metas = [], [], []
        for r in rows:
            content = (r.get("content") or "").strip()
            turn_id = str(r.get("id", ""))
            if not content or not turn_id:
                continue
            docs.append(content[:4000])
            ids.append(f"turn/{turn_id}")
            metas.append({
                "turn_id": int(turn_id),
                "session_id": str(r.get("session_id", "")),
                "role": str(r.get("role", "")),
                "timestamp": str(r.get("timestamp", "")),
            })
        if not docs:
            return 0
        with chroma._lock:
            chroma._collection.upsert(
                documents=docs,
                embeddings=_embed_texts(docs),
                metadatas=metas,
                ids=ids,
            )
        return len(docs)
    except Exception as e:
        print(f"[ChromaMemory] add_turn_embeddings error: {e}")
        return 0


def search_turn_embeddings(query: str, n_results: int = 6) -> list[dict]:
    """Semantic search over raw conversation turns (meaning-based match).

    Cloud mode: the service's FAISS turn index is authoritative — query it
    and map results back into the local shape.
    """
    if not CHROMA_VECTOR_SEARCH_ENABLED:
        return []
    if _is_remote_memory():
        try:
            from memory.memory_client import _is_healthy, memory_search_vector
            if not _is_healthy():
                return []
            out = []
            for text, source, score in memory_search_vector(query, top_k=n_results):
                if not isinstance(source, dict) or source.get("type") != "turn":
                    continue
                out.append({
                    "type": "turn",
                    "id": 0,
                    "session_id": "",
                    "role": str(source.get("role", "")),
                    "content": text,
                    "timestamp": str(source.get("timestamp", "")),
                    "score": max(0.0, float(score)),
                })
            return out
        except Exception as e:
            print(f"[ChromaMemory] cloud turn search failed: {e}")
            return []
    try:
        chroma = _get_turns_chroma()
        if not chroma._ready:
            return []
        with chroma._lock:
            res = chroma._collection.query(
                query_embeddings=_embed_texts([query]),
                n_results=n_results,
            )
        out = []
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        for i, doc_id in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            dist = dists[i] if i < len(dists) else 1.0
            try:
                turn_id = int(meta.get("turn_id", 0) or 0)
            except Exception:
                turn_id = 0
            out.append({
                "type": "turn",
                "id": turn_id,
                "session_id": meta.get("session_id", ""),
                "role": meta.get("role", ""),
                "content": docs[i] if i < len(docs) else "",
                "timestamp": meta.get("timestamp", ""),
                "score": max(0.0, 1.0 - float(dist)),
            })
        return out
    except Exception as e:
        print(f"[ChromaMemory] search_turn_embeddings error: {e}")
        return []


def backfill_turn_embeddings(batch_size: int = 256) -> int:
    """One-time bootstrap: embed existing raw turns into conversation_turns.

    Idempotent — per batch, turns already present (by doc id) are skipped.
    Returns the number of turns embedded. Designed to run in a background
    thread at startup; does not block chat.
    """
    if not CHROMA_VECTOR_SEARCH_ENABLED:
        return 0
    import sqlite3
    from memory.conversation_db import DB_PATH as _DB
    try:
        chroma = _get_turns_chroma()
        if not chroma._ready:
            print("[TurnVectors] ChromaDB not ready — backfill skipped")
            return 0
        conn = sqlite3.connect(f"file:{_DB}?mode=ro", uri=True, timeout=30)
        rows = conn.execute(
            "SELECT id, role, content, session_id, timestamp FROM turns "
            "WHERE content IS NOT NULL AND TRIM(content) != '' AND archived = 0 "
            "ORDER BY id ASC"
        ).fetchall()
        conn.close()
    except Exception as e:
        print(f"[TurnVectors] backfill error: {e}")
        return 0
    if not rows:
        return 0
    total = 0
    pending: list[dict] = []

    def _flush(batch: list[dict]) -> int:
        chroma2 = _get_turns_chroma()
        if not chroma2._ready:
            return 0
        batch_ids = [f"turn/{r['id']}" for r in batch]
        try:
            with chroma2._lock:
                found = set(chroma2._collection.get(ids=batch_ids).get("ids") or [])
            fresh = [r for r, bid in zip(batch, batch_ids) if bid not in found]
        except Exception:
            fresh = batch
        if not fresh:
            return 0
        return add_turn_embeddings(fresh)

    for turn_id, role, content, session_id, ts in rows:
        pending.append({"id": turn_id, "role": role, "content": content,
                        "session_id": session_id or "", "timestamp": ts or ""})
        if len(pending) >= batch_size:
            total += _flush(pending)
            pending = []
            print(f"[TurnVectors] backfill progress: {total} turns embedded")
    if pending:
        total += _flush(pending)
    print(f"[TurnVectors] backfill complete: {total} turns embedded")
    return total
