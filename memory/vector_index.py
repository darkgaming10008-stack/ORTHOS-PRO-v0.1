import json
import hashlib
import threading
import time
import numpy as np
from pathlib import Path
from typing import List, Tuple

MODEL_NAME = "all-MiniLM-L6-v2"
_INDEX_STORE_PATH = Path(__file__).parent / ".vector_index.faiss"
_META_STORE_PATH = Path(__file__).parent / ".vector_meta.json"
_model = None
_model_lock = threading.Lock()
_index_lock = threading.Lock()

DB_PATH = Path(__file__).parent / "conversations.db"


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer(MODEL_NAME)
                print(f"[VectorIndex] Loaded {MODEL_NAME}")
    return _model


def embed(texts: List[str]) -> np.ndarray:
    try:
        from memory.memory_client import _is_healthy, memory_embed
        if _is_healthy():
            return memory_embed(texts)
    except Exception:
        pass
    model = _get_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


class VectorIndex:
    def __init__(self):
        self.texts: List[str] = []
        self.vectors: np.ndarray = np.empty((0, 384), dtype=np.float32)
        self.sources: List[dict] = []
        self._dirty = True

    def add(self, text: str, source: dict) -> None:
        """Incremental add: embed single text and append to index.

        Cloud mode: the service's index is authoritative — mirror the row
        there (POST /add_turn_embeddings) and skip the local index entirely,
        so server-side search stays live with every incremental save.
        """
        try:
            from memory.memory_client import (
                _IS_MEMORY_SERVICE, _is_remote_memory, _is_healthy,
                memory_add_turn_embeddings,
            )
            if not _IS_MEMORY_SERVICE and _is_remote_memory() and _is_healthy():
                try:
                    row_id = source.get("id")
                    if not row_id:
                        row_id = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16)
                    memory_add_turn_embeddings([{
                        "id": int(row_id),
                        "role": str(source.get("role", "")),
                        "content": text,
                        "session_id": str(source.get("session_id", "")),
                        "timestamp": str(source.get("timestamp", "")),
                    }])
                    return   # mirrored to cloud — no local index work
                except Exception as e:
                    print(f"[VectorIndex] cloud mirror add failed: {e}")
        except Exception:
            pass
        vec = embed([text])
        with _index_lock:
            self.texts.append(text)
            self.sources.append(source)
            self.vectors = np.vstack([self.vectors, vec]) if self.vectors.shape[0] > 0 else vec
            self._dirty = True
        print(f"[VectorIndex] Incremental add: {len(text[:60])} chars")

    def save(self) -> None:
        if not self.texts or self.vectors.shape[0] == 0:
            return
        try:
            npy_path = _INDEX_STORE_PATH.with_suffix(".npy")
            np.save(str(npy_path), self.vectors)
            meta = {"texts": self.texts, "sources": self.sources}
            _META_STORE_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"[VectorIndex] Saved to disk: {len(self.texts)} entries")
        except Exception as e:
            print(f"[VectorIndex] Save error: {e}")

    def load(self) -> bool:
        npy_path = _INDEX_STORE_PATH.with_suffix(".npy")
        if not npy_path.exists() or not _META_STORE_PATH.exists():
            return False
        try:
            self.vectors = np.load(str(npy_path))
            meta = json.loads(_META_STORE_PATH.read_text(encoding="utf-8"))
            self.texts = meta.get("texts", [])
            self.sources = meta.get("sources", [])
            self._dirty = False
            print(f"[VectorIndex] Loaded from disk: {len(self.texts)} entries")
            return self.vectors.shape[0] > 0
        except Exception as e:
            print(f"[VectorIndex] Load error: {e}")
            return False

    def rebuild(self, turns: List[dict], facts: dict) -> None:
        with _index_lock:
            texts = []
            sources = []

            for t in turns:
                role = t.get("role", "")
                content = (t.get("content") or "").strip()
                ts = t.get("timestamp", "")
                if content and len(content) > 10:
                    texts.append(f"[{ts}] {role}: {content}")
                    sources.append({"type": "turn", "role": role, "content": content, "timestamp": ts})

            if isinstance(facts, dict):
                for category, entries in facts.items():
                    if not isinstance(entries, dict):
                        continue
                    for key, entry in entries.items():
                        val = entry.get("value") if isinstance(entry, dict) else entry
                        if val and isinstance(val, str) and len(val) > 3:
                            texts.append(f"[FACT] {category}/{key}: {val}")
                            sources.append({"type": "fact", "category": category, "key": key, "value": val})

            if texts:
                self.vectors = embed(texts)
                self.texts = texts
                self.sources = sources
            else:
                self.vectors = np.empty((0, 384), dtype=np.float32)
                self.texts = []
                self.sources = []
            self._dirty = False
            print(f"[VectorIndex] Rebuilt: {len(self.texts)} entries")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, dict, float]]:
        if not self.texts or self.vectors.shape[0] == 0:
            return []
        q_vec = embed([query])[0]
        scores = self.vectors @ q_vec
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0.2:
                results.append((self.texts[idx], self.sources[idx], float(scores[idx])))
        return results


def preload_model():
    """Preload MiniLM model during startup (blocking call)."""
    _get_model()


_index_instance = None


def get_index() -> VectorIndex:
    global _index_instance
    if _index_instance is None:
        _index_instance = VectorIndex()
        _index_instance.load()  # try disk first
    return _index_instance


def rebuild_index(turns: List[dict], facts: dict) -> None:
    idx = get_index()
    idx.rebuild(turns, facts)
    idx.save()


def search_vector(query: str, top_k: int = 5) -> List[Tuple[str, dict, float]]:
    """Vector search over the conversation/fact index.

    Cloud mode (default): the service's index is authoritative — query it.
    The old note said the server index was "frozen at boot", but that predates
    the remote-write endpoints (/add_turn_embeddings, /upsert_memory) which
    keep the server index live; every local write is mirrored there.

    On the service process itself (_IS_MEMORY_SERVICE) this always runs the
    local index — a server must never proxy to itself.
    """
    try:
        from memory.memory_client import (
            _IS_MEMORY_SERVICE, _is_remote_memory, _is_healthy,
            memory_search_vector,
        )
        if not _IS_MEMORY_SERVICE and _is_remote_memory():
            if not _is_healthy():
                return []
            return [
                (t, s if isinstance(s, dict) else {"type": str(s)}, float(sc))
                for t, s, sc in memory_search_vector(query, top_k=top_k)
            ]
    except Exception as e:
        print(f"[VectorIndex] cloud search failed: {e}")
        return []
    return get_index().search(query, top_k)
