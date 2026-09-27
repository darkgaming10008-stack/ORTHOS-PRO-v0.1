"""Optional local cross-encoder reranker (2026-standard retrieval step).

Reranks the fused candidates (top ~20) with a small cross-encoder so the
final top-k is relevance-ranked by the model itself, not just by the
weighted heuristic scores. Falls back gracefully if the model is missing.
"""
import os
import threading

_ENABLED = os.environ.get("ALEX_RERANKER", "1") != "0"
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model = None
_lock = threading.Lock()


def is_enabled() -> bool:
    return _ENABLED


def _get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import CrossEncoder
                _model = CrossEncoder(MODEL_NAME)
                print(f"[Reranker] Loaded {MODEL_NAME}")
    return _model


def rerank(query: str, candidates: list, top_k: int | None = None) -> list:
    """Rerank candidates (each item is (text, meta, signal, score)) by a
    cross-encoder. Returns the same tuples re-ordered (score untouched)."""
    if not candidates or not _ENABLED:
        return candidates[:top_k] if top_k else candidates
    try:
        model = _get_model()
    except Exception as e:
        print(f"[Reranker] Unavailable ({e}) — skipping rerank")
        return candidates[:top_k] if top_k else candidates
    try:
        pairs = [(query, str(c[0])) for c in candidates]
        scores = model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        out = [c for c, _s in ranked]
        return out[:top_k] if top_k else out
    except Exception as e:
        print(f"[Reranker] Error: {e}")
        return candidates[:top_k] if top_k else candidates
