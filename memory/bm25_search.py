import math
import sqlite3
import threading
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path(__file__).parent / "conversations.db"
_k1 = 1.5
_b = 0.75
_rebuild_lock = threading.Lock()

_index = None


class BM25Index:
    def __init__(self):
        self.doc_texts: list[str] = []
        self.doc_metadata: list[dict] = []
        self.doc_freqs: list[Counter] = []
        self.idf: dict[str, float] = {}
        self.avgdl: float = 0
        self.num_docs: int = 0
        self._all_terms: set = set()

    def add_document(self, text: str, metadata: dict) -> None:
        tokens = text.lower().split()
        self.doc_texts.append(text)
        self.doc_metadata.append(metadata)
        self.doc_freqs.append(Counter(tokens))
        for t in tokens:
            self._all_terms.add(t)

    def add_document_incremental(self, text: str, metadata: dict) -> None:
        """Incremental add — approximate IDF update without full rebuild."""
        tokens = text.lower().split()
        self.doc_texts.append(text)
        self.doc_metadata.append(metadata)
        self.doc_freqs.append(Counter(tokens))
        for t in tokens:
            self._all_terms.add(t)

        old_n = self.num_docs
        self.num_docs += 1
        old_total = self.avgdl * old_n if old_n > 0 else 0
        doc_len = len(tokens)
        self.avgdl = (old_total + doc_len) / self.num_docs if self.num_docs > 0 else 0

        # Incremental IDF: boost IDF for new terms, approximate for existing
        for term in set(tokens):
            if term not in self.idf:
                self.idf[term] = math.log(1 + (self.num_docs - 1 + 0.5) / (1 + 0.5))
            # Existing terms keep their IDF (off by at most 1 doc, negligible for large indexes)

    def build(self) -> None:
        self.num_docs = len(self.doc_texts)
        if self.num_docs == 0:
            return
        total_len = sum(sum(f.values()) for f in self.doc_freqs)
        self.avgdl = total_len / self.num_docs

        doc_count = Counter()
        for freq in self.doc_freqs:
            for term in freq:
                doc_count[term] += 1

        self.idf = {}
        for term in self._all_terms:
            n_t = doc_count.get(term, 0)
            # BM25 IDF with smoothing
            self.idf[term] = math.log(1 + (self.num_docs - n_t + 0.5) / (n_t + 0.5))

    def score_document(self, query_terms: list[str], doc_idx: int) -> float:
        if self.num_docs == 0:
            return 0.0
        freq = self.doc_freqs[doc_idx]
        doc_len = sum(freq.values())
        score = 0.0
        for term in query_terms:
            if term not in self.idf:
                continue
            tf = freq.get(term, 0)
            if tf == 0:
                continue
            numerator = tf * (_k1 + 1)
            denominator = tf + _k1 * (1 - _b + _b * doc_len / self.avgdl)
            score += self.idf[term] * numerator / denominator
        return score

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, dict, float]]:
        query_terms = [w for w in query.lower().split() if len(w) > 1]
        if not query_terms:
            return []
        scored = []
        for i in range(self.num_docs):
            s = self.score_document(query_terms, i)
            if s > 0:
                scored.append((self.doc_texts[i], self.doc_metadata[i], s))
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_k]


def load_turns_for_index(limit: int = 200) -> list[dict]:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT role, content, timestamp, id FROM turns ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def load_summaries_for_index() -> list[dict]:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT summary, created_at FROM summaries ORDER BY id ASC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def load_facts_for_index() -> dict:
    """Load all memory facts grouped by category from SQLite (memory_items)."""
    from memory.conversation_db import get_memory_items as _gmi
    try:
        items = _gmi(limit=99999)
    except Exception:
        return {}
    facts: dict[str, dict] = {}
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    for item in items:
        cat = item.get("type", "notes")
        key = item.get("key", "?")
        val = item.get("content", "")
        expires = item.get("expires_at")
        if expires and expires < today:
            continue
        if not val:
            continue
        facts.setdefault(cat, {})[key] = {
            "value": val,
            "updated": (item.get("updated_at") or "")[:10],
            "source": item.get("source", "manual"),
        }
    return facts


def _temporal_decay(timestamp_str: str, now: datetime = None) -> float:
    if not timestamp_str:
        return 1.0
    if now is None:
        now = datetime.now()
    try:
        ts = datetime.fromisoformat(timestamp_str)
    except Exception:
        return 1.0
    days_ago = (now - ts).days
    if days_ago < 7:
        return 1.0
    elif days_ago < 30:
        return 0.9
    elif days_ago < 90:
        return 0.7
    elif days_ago < 180:
        return 0.4
    else:
        return 0.2


def rebuild_bm25_index() -> BM25Index:
    global _index
    bm25 = BM25Index()

    turns = load_turns_for_index(200)
    for t in turns:
        content = (t.get("content") or "").strip()
        if len(content) > 10:
            bm25.add_document(
                f"{t['role']}: {content}",
                {"type": "turn", "role": t["role"], "content": content,
                 "timestamp": t.get("timestamp", ""), "id": t.get("id", 0)},
            )

    summaries = load_summaries_for_index()
    for s in summaries:
        text = (s.get("summary") or "").strip()
        if len(text) > 10:
            bm25.add_document(
                text,
                {"type": "summary", "content": text, "timestamp": s.get("created_at", "")},
            )

    facts = load_facts_for_index()
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    for cat, entries in facts.items():
        if not isinstance(entries, dict):
            continue
        for key, entry in entries.items():
            val = entry.get("value") if isinstance(entry, dict) else entry
            expires = entry.get("expires") if isinstance(entry, dict) else None
            if expires and expires < today:
                continue
            if val and isinstance(val, str) and len(val) > 3:
                meta = {"type": "fact", "category": cat, "key": key, "value": val,
                        "updated": entry.get("updated", ""), "timestamp": entry.get("updated", "")}
                bm25.add_document(f"{cat}/{key}: {val}", meta)

    bm25.build()
    with _rebuild_lock:
        _index = bm25
    print(f"[BM25] Index rebuilt: {bm25.num_docs} docs")
    return bm25


def get_index() -> BM25Index:
    global _index
    if _index is None:
        rebuild_bm25_index()
    return _index


def search_bm25(query: str, top_k: int = 10) -> list[tuple[str, dict, float]]:
    # NOTE: intentionally does NOT proxy to the memory service. The service's
    # BM25 index is frozen at boot; incremental adds land in the main
    # process's index, so search always runs against the live index.
    idx = get_index()
    raw_results = idx.search(query, top_k=top_k * 2)
    now = datetime.now()

    scored = []
    for text, meta, bm25_score in raw_results:
        ts = meta.get("timestamp", "")
        decay = _temporal_decay(ts, now)
        final_score = bm25_score * decay
        scored.append((text, meta, bm25_score, decay, final_score))

    scored.sort(key=lambda x: x[4], reverse=True)
    return [(t, m, s) for t, m, _, _, s in scored[:top_k]]


def add_to_bm25_index(text: str, metadata: dict) -> None:
    """Incremental add a single document to the BM25 index."""
    idx = get_index()
    with _rebuild_lock:
        idx.add_document_incremental(text, metadata)
    print(f"[BM25] Incremental add: {len(text[:60])} chars")
