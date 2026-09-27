import os
import time
import threading
import numpy as np

_IS_MEMORY_SERVICE = os.environ.get("MEMORY_SERVICE_PROCESS") == "1"

from pathlib import Path as _Path
BASE_DIR = _Path(__file__).resolve().parent.parent
CONFIG_FILE = BASE_DIR / "config" / "api_keys.json"

_LOCAL_URL = "http://127.0.0.1:9876"
_HEALTH_TTL = 30.0          # seconds a positive health verdict stays cached
_SESSION = None             # lazily-created pooled requests.Session
_session_lock = threading.Lock()

_healthy: bool | None = None
_health_checked_at: float = 0.0
_active_url: str = ""       # URL the current health verdict belongs to


def _load_config() -> dict:
    """Read config/api_keys.json without importing heavy modules."""
    import json

    try:
        raw = CONFIG_FILE.read_bytes()
        if raw.startswith(b"\xef\xbb\xbf"):      # tolerate Windows PowerShell BOM
            raw = raw[3:]
            try:
                CONFIG_FILE.write_bytes(raw)
            except Exception:
                pass
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return {}


def get_service_url() -> str:
    """Configured memory-service URL (normalized).

    Priority: env ORT_MEMORY_SERVICE_URL > config memory_service_url > local default.
    An empty/missing value means the LOCAL service (127.0.0.1:9876) is used,
    which keeps today's behavior when the cloud option is never configured.

    Normalization: a bare IP/hostname ("100.69.108.124") gets a scheme and
    the default port appended -> http://100.69.108.124:9876.
    """
    env = os.environ.get("ORT_MEMORY_SERVICE_URL", "").strip()
    if env:
        return _normalize_url(env)
    cfg_url = str(_load_config().get("memory_service_url", "") or "").strip()
    if cfg_url:
        return _normalize_url(cfg_url)
    return _LOCAL_URL


def _normalize_url(url: str) -> str:
    """Normalize the configured service URL.

    Bare inputs (no scheme) get http:// and, when no port was given, the
    default :9876 — so '100.69.108.124' becomes 'http://100.69.108.124:9876'.
    Full URLs (with a scheme) are trusted as-is.
    """
    original = url.strip().rstrip("/")
    if "://" in original:
        return original                      # user typed a full URL — trust it
    url = "http://" + original
    if ":" not in original:                  # bare host/IP, no port
        url += ":9876"
    return url


def get_service_token() -> str:
    env = os.environ.get("ORT_MEMORY_SERVICE_TOKEN", "").strip()
    if env:
        return env
    return str(_load_config().get("memory_service_token", "") or "").strip()


def _session():
    """Pooled HTTP session (TCP/TLS reuse — matters over a tunnel)."""
    global _SESSION
    with _session_lock:
        if _SESSION is None:
            import requests
            _SESSION = requests.Session()
        return _SESSION


def _headers() -> dict:
    token = get_service_token()
    return {"Authorization": f"Bearer {token}"} if token else {}


def service_url() -> str:
    """URL of the service the CURRENT health verdict was checked against."""
    return _active_url or get_service_url()


def _probe(url: str, timeout: float = 2.0) -> bool:
    try:
        resp = _session().get(f"{url}/health", headers=_headers(), timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def health() -> dict | None:
    try:
        resp = _session().get(
            f"{get_service_url()}/health", headers=_headers(), timeout=3
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def _cloud_only() -> bool:
    """True when cloud memory is the ONLY source of truth (default).

    Precedence: env ORT_MEMORY_CLOUD_ONLY > config "memory_cloud_only"
    > default True. When the configured cloud service is unreachable,
    calls fail softly (callers return empty results) instead of silently
    reading a stale local index — the user asked for cloud to be the
    default, always. Opt back into the old local-fallback behavior with
    ORT_MEMORY_CLOUD_ONLY=false or config "memory_cloud_only": false.
    """
    env = os.environ.get("ORT_MEMORY_CLOUD_ONLY", "").strip().lower()
    if env:
        return env not in ("0", "false", "no", "off")
    try:
        return bool(_load_config().get("memory_cloud_only", True))
    except Exception:
        return True


def _is_healthy() -> bool:
    """Health with TTL + (optional) local fallback.

    The verdict is cached for _HEALTH_TTL seconds against the URL it was
    probed. When cloud-only mode is OFF and the configured (possibly remote)
    service is down and a LOCAL service is reachable, calls transparently
    use the local one. In cloud-only mode (default) there is NO local
    fallback — offline means empty results, never stale local data.
    """
    if _IS_MEMORY_SERVICE:
        return False

    global _healthy, _health_checked_at, _active_url

    url = get_service_url()
    now = time.time()
    if _healthy is not None and _active_url == url and (now - _health_checked_at) < _HEALTH_TTL:
        return _healthy

    healthy = _probe(url)
    if not healthy and url != _LOCAL_URL and not _cloud_only():
        # Remote down -> fall back to a local service if one is running.
        healthy = _probe(_LOCAL_URL, timeout=1.5)
        if healthy:
            url = _LOCAL_URL
            print("[MemoryClient] Remote service down — falling back to local service")
    _healthy = healthy
    _active_url = url
    _health_checked_at = now
    return healthy


def get_cloud_only() -> bool:
    """Public accessor (callers/tests may want the mode explicitly)."""
    return _cloud_only()


def invalidate_cache():
    global _healthy
    _healthy = None


def memory_embed(texts: list[str]) -> np.ndarray:
    resp = _session().post(
        f"{service_url()}/embed",
        json={"texts": texts},
        headers=_headers(),
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return np.array(data["vectors"], dtype=np.float32)


def memory_ner(text: str) -> list[str]:
    resp = _session().post(
        f"{service_url()}/ner",
        json={"text": text},
        headers=_headers(),
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("entities", [])


def memory_ner_typed(text: str) -> list[dict]:
    """Ask the memory service for typed entities: [{name, type}]."""
    resp = _session().post(
        f"{service_url()}/ner_typed",
        json={"text": text},
        headers=_headers(),
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json().get("entities", [])


def memory_search_vector(query: str, top_k: int = 5) -> list[tuple[str, dict, float]]:
    resp = _session().post(
        f"{service_url()}/search_vector",
        json={"query": query, "top_k": top_k},
        headers=_headers(),
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return [
        (r["text"], r["source"], r["score"])
        for r in data.get("results", [])
    ]


def memory_search_bm25(query: str, top_k: int = 10) -> list[tuple[str, dict, float]]:
    resp = _session().post(
        f"{service_url()}/search_bm25",
        json={"query": query, "top_k": top_k},
        headers=_headers(),
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return [
        (r["text"], r["source"], r["score"])
        for r in data.get("results", [])
    ]


def memory_rerank(query: str, texts: list[str], top_k: int | None = None) -> list[int]:
    """Ask the memory service to cross-encoder rerank candidate texts.

    Returns the candidate indices in re-ranked order (service-side model).
    """
    resp = _session().post(
        f"{service_url()}/rerank",
        json={"query": query, "texts": texts, "top_k": top_k},
        headers=_headers(),
    )
    resp.raise_for_status()
    data = resp.json()
    order = data.get("order")
    if order is None:
        raise RuntimeError("memory service /rerank returned no order")
    return order


def memory_upsert_memory(entry: dict) -> bool:
    """Remote write: add a memory item + its embedding (cloud mode).

    Returns True when the service accepted it; False when unavailable
    (caller should fall back to the local write path).
    """
    if not _is_healthy():
        return False
    try:
        resp = _session().post(
            f"{service_url()}/upsert_memory",
            json=entry,
            headers=_headers(),
            timeout=60,
        )
        resp.raise_for_status()
        return bool(resp.json().get("ok", False))
    except Exception as e:
        print(f"[MemoryClient] upsert_memory failed: {e}")
        return False


def memory_add_turn_embeddings(rows: list[dict]) -> int:
    """Remote write: index turn texts (cloud mode). Returns accepted count."""
    if not _is_healthy() or not rows:
        return 0
    try:
        resp = _session().post(
            f"{service_url()}/add_turn_embeddings",
            json={"rows": rows},
            headers=_headers(),
            timeout=120,
        )
        resp.raise_for_status()
        return int(resp.json().get("added", 0))
    except Exception as e:
        print(f"[MemoryClient] add_turn_embeddings failed: {e}")
        return 0


def memory_delete_memory(category: str, key: str) -> bool:
    """Remote write: remove a memory item from the cloud vector store."""
    if not _is_healthy():
        return False
    try:
        resp = _session().post(
            f"{service_url()}/delete_memory",
            json={"category": category, "key": key},
            headers=_headers(),
            timeout=30,
        )
        resp.raise_for_status()
        return bool(resp.json().get("ok", False))
    except Exception as e:
        print(f"[MemoryClient] delete_memory failed: {e}")
        return False
