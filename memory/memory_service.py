import sys
import os
# ── Mark this process as the memory service ────────────────────────────────
os.environ["MEMORY_SERVICE_PROCESS"] = "1"
# ── Migrate from deprecated TRANSFORMERS_CACHE to HF_HOME ──────────────────
_hf_cache = os.environ.pop("TRANSFORMERS_CACHE", None)
if _hf_cache:
    os.environ.setdefault("HF_HOME", _hf_cache)
# ───────────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np

_model = None
_ner = None
_reranker_ready = False

class EmbedRequest(BaseModel):
    texts: list[str]

class EmbedResponse(BaseModel):
    vectors: list[list[float]]
    dim: int
    elapsed_ms: float

class NERRequest(BaseModel):
    text: str

class NERResponse(BaseModel):
    entities: list[str]

class RerankRequest(BaseModel):
    query: str
    texts: list[str]
    top_k: int | None = None

class RerankResponse(BaseModel):
    order: list[int]

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class SearchResponse(BaseModel):
    results: list[dict]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _ner, _reranker_ready
    print("[Memory Service] Loading MiniLM L6 v2…")
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer("all-MiniLM-L6-v2")
    print("[Memory Service] MiniLM ready.")

    print("[Memory Service] Loading spaCy NER…")
    try:
        import spacy
        _ner = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer"])
        print("[Memory Service] spaCy NER ready.")
    except Exception as e:
        print(f"[Memory Service] spaCy NER unavailable: {e}")

    print("[Memory Service] Pre-loading cross-encoder reranker…")
    try:
        from memory.reranker import _get_model as _get_reranker, is_enabled
        if is_enabled():
            _get_reranker()
            _reranker_ready = True
            print("[Memory Service] Reranker ready.")
        else:
            print("[Memory Service] Reranker disabled (ALEX_RERANKER=0).")
    except Exception as e:
        print(f"[Memory Service] Reranker unavailable: {e}")

    print("[Memory Service] Pre-loading vector index…")
    try:
        from memory.vector_index import get_index
        get_index()
    except Exception as e:
        print(f"[Memory Service] Vector index load: {e}")

    print("[Memory Service] Pre-loading BM25 index…")
    try:
        from memory.bm25_search import get_index as get_bm25
        get_bm25()
    except Exception as e:
        print(f"[Memory Service] BM25 load: {e}")

    print(f"[Memory Service] Ready on http://127.0.0.1:9876")
    yield

app = FastAPI(docs_url=None, redoc_url=None, lifespan=lifespan)

# ── Optional bearer-token auth (for cloud/Tailscale deployments) ────────────
# Set MEMORY_TOKEN env (or config memory_service_token on the server side) to
# require 'Authorization: Bearer <token>' on every request except /health.
import os as _os
_SERVICE_TOKEN = _os.environ.get("MEMORY_TOKEN", "").strip()

if _SERVICE_TOKEN:
    from fastapi import Request
    from fastapi.responses import JSONResponse

    @app.middleware("http")
    async def _auth_middleware(request: Request, call_next):
        if request.url.path == "/health":
            return await call_next(request)
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {_SERVICE_TOKEN}":
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
        return await call_next(request)
    print("[Memory Service] Bearer-token auth ENABLED")

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None, "ner_loaded": _ner is not None,
            "reranker_loaded": _reranker_ready}

@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    import time
    t0 = time.perf_counter()
    vecs = _model.encode(req.texts, normalize_embeddings=True, show_progress_bar=False)
    elapsed = (time.perf_counter() - t0) * 1000
    return EmbedResponse(vectors=vecs.tolist(), dim=vecs.shape[1], elapsed_ms=round(elapsed, 1))

def _ner_typed(text: str) -> list[dict]:
    """Typed NER via the preloaded spaCy model: [{name, type}]."""
    if _ner is None:
        return []
    try:
        doc = _ner(text)
        SPACY_TYPE_MAP = {
            "PERSON": "Person", "ORG": "Organization", "GPE": "Location",
            "LOC": "Location", "DATE": "Date", "EVENT": "Event",
            "PRODUCT": "Object", "WORK_OF_ART": "Concept", "NORP": "Concept",
        }
        seen: dict[str, str] = {}
        for ent in doc.ents:
            if ent.label_ in SPACY_TYPE_MAP and len(ent.text.strip()) > 1:
                seen.setdefault(ent.text.strip(), SPACY_TYPE_MAP[ent.label_])
        return [{"name": n, "type": t} for n, t in seen.items()]
    except Exception:
        return []

@app.post("/ner", response_model=NERResponse)
async def ner(req: NERRequest):
    return NERResponse(entities=[e["name"] for e in _ner_typed(req.text)])

class NERTypedResponse(BaseModel):
    entities: list[dict]

@app.post("/ner_typed", response_model=NERTypedResponse)
async def ner_typed(req: NERRequest):
    return NERTypedResponse(entities=_ner_typed(req.text))

@app.post("/search_vector", response_model=SearchResponse)
async def search_vector(req: SearchRequest):
    try:
        from memory.vector_index import search_vector
        results = search_vector(req.query, req.top_k)
        return SearchResponse(results=[
            {"text": t, "source": s, "score": sc} for t, s, sc in results
        ])
    except Exception as e:
        return SearchResponse(results=[])

@app.post("/search_bm25", response_model=SearchResponse)
async def search_bm25(req: SearchRequest):
    try:
        from memory.bm25_search import search_bm25
        results = search_bm25(req.query, req.top_k)
        return SearchResponse(results=[
            {"text": t, "source": s, "score": sc} for t, s, sc in results
        ])
    except Exception as e:
        return SearchResponse(results=[])

@app.post("/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest):
    n = len(req.texts)
    if not req.texts:
        return RerankResponse(order=[])
    try:
        from memory.reranker import rerank as _rerank
        candidates = [(t, {"_idx": i}, "rerank", 0.0) for i, t in enumerate(req.texts)]
        ranked = _rerank(req.query, candidates, top_k=req.top_k)
        return RerankResponse(order=[m["_idx"] for _, m, _, _ in ranked])
    except Exception:
        return RerankResponse(order=list(range(n)))


# ── Remote write endpoints (cloud mode) ────────────────────────────────────
# In cloud mode the SERVER owns the vector index. The i3 PC mirrors nothing:
# it POSTs new memories/turns here, so embeddings are computed on the fast
# machine and the index stays coherent for subsequent searches.

from pydantic import BaseModel as _BM

class MemoryEntry(_BM):
    category: str
    key: str
    value: str

class MemoryUpsertResponse(_BM):
    ok: bool

@app.post("/upsert_memory", response_model=MemoryUpsertResponse)
async def upsert_memory(entry: MemoryEntry):
    """Add/replace a memory item in the server-owned vector store."""
    try:
        from memory.chroma_memory import _get_chroma
        chroma = _get_chroma()
        if not chroma or not chroma._ready:
            return MemoryUpsertResponse(ok=False)
        chroma.add_memory(entry.category, entry.key, str(entry.value)[:2000])
        return MemoryUpsertResponse(ok=True)
    except Exception as e:
        print(f"[Memory Service] upsert_memory error: {e}")
        return MemoryUpsertResponse(ok=False)

class DeleteMemoryRequest(_BM):
    category: str
    key: str

class DeleteMemoryResponse(_BM):
    ok: bool

@app.post("/delete_memory", response_model=DeleteMemoryResponse)
async def delete_memory(req: DeleteMemoryRequest):
    """Remove a memory item from the server-owned vector store (cloud mode)."""
    try:
        from memory.chroma_memory import _get_chroma
        chroma = _get_chroma()
        if not chroma:
            return DeleteMemoryResponse(ok=False)
        return DeleteMemoryResponse(ok=bool(chroma.delete_memory(req.category, req.key)))
    except Exception as e:
        print(f"[Memory Service] delete_memory error: {e}")
        return DeleteMemoryResponse(ok=False)

class TurnRows(_BM):
    rows: list[dict]

class TurnEmbedResponse(_BM):
    added: int

@app.post("/add_turn_embeddings", response_model=TurnEmbedResponse)
async def add_turn_embeddings(req: TurnRows):
    """Batch-upsert conversation turns into the server-owned turn index."""
    try:
        from memory.chroma_memory import add_turn_embeddings as _local_add
        added = _local_add(req.rows)
        return TurnEmbedResponse(added=added)
    except Exception as e:
        print(f"[Memory Service] add_turn_embeddings error: {e}")
        return TurnEmbedResponse(added=0)

if __name__ == "__main__":
    _host = _os.environ.get("MEMORY_HOST", "127.0.0.1")
    _port = int(_os.environ.get("MEMORY_PORT", "9876"))
    print(f"[Memory Service] Listening on {_host}:{_port}")
    uvicorn.run(app, host=_host, port=_port, log_level="info", workers=1)
