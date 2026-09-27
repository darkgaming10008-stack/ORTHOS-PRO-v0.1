"""Background index worker — chat instant, indexing peechhe chupchap.

Chat thread sirf task submit karta hai (queue put = microseconds) aur turant
free ho jata hai. Ek single daemon worker task ko FIFO order me drain karta
hai aur slow kaam yahin karta hai:

  - MiniLM embeddings (local CPU ya remote memory-service HTTP)
  - BM25 incremental adds
  - ChromaDB conversation_turns upserts

Single worker = deterministic ordering + duplicate-free indexing. Agar app
drain ke beech band ho jaye to kuch bhi permanently lost nahi hota — SQLite
`turns` table source of truth hai, aur startup backfills
(backfill_turns_fts / backfill_turn_embeddings) missing derived entries
agli baar launch pe auto-recover kar dete hain.

Tests / deterministic mode: ORT_SYNC_INDEXING=1 set karne par submit()
False return karta hai aur caller indexing inline chalata hai.
"""
import os
import queue
import threading

_task_queue: "queue.Queue" = queue.Queue()
_started = False
_start_lock = threading.Lock()

MAX_DRAIN = 8  # ek wake-up me max itne tasks batch honge


def _sync_mode() -> bool:
    """ORT_SYNC_INDEXING=1 → inline indexing (tests / explicit opt-out)."""
    return os.environ.get("ORT_SYNC_INDEXING", "0") == "1"


def submit(task: dict) -> bool:
    """Queue one indexing task. Returns False → caller should run it inline."""
    global _started
    if _sync_mode():
        return False
    if not _started:
        with _start_lock:
            if not _started:
                try:
                    threading.Thread(
                        target=_worker, name="memory-index-worker", daemon=True
                    ).start()
                    _started = True
                    print("[IndexWorker] Background indexing started (async mode)")
                except Exception:
                    return False
    try:
        _task_queue.put(task)
        return True
    except Exception:
        return False


def pending() -> int:
    """Tasks waiting in the queue (observability)."""
    return _task_queue.qsize()


def warm_start() -> None:
    """Pre-start the worker thread at app boot (optional — submit() also
    lazy-starts). Keeps first-message behavior identical to steady state."""
    global _started
    if _sync_mode() or _started:
        return
    submit({"turns": []})  # empty task: cheap, just spins the thread up


def run_tasks(tasks: list[dict]) -> None:
    """Execute index tasks synchronously (worker body + sync-mode fallback).

    Task shapes:
      {"turns": [...], "full_rebuild": bool, "session_id": str}
      {"chroma_rows": [{id, role, content, session_id, timestamp}, ...]}
    """
    # 1. Vector/BM25 incremental adds + periodic full rebuild
    full_rebuild = any(t.get("full_rebuild") for t in tasks)
    flat_turns: list[dict] = []
    for t in tasks:
        flat_turns.extend(t.get("turns") or [])

    try:
        if full_rebuild:
            # Rebuild path re-reads recent turns from SQLite (which already
            # includes everything just saved) — incremental adds skip.
            from memory.memory_manager import get_recent_turns_for_vector, load_memory
            from memory.vector_index import rebuild_index
            rebuild_index(get_recent_turns_for_vector(50), load_memory())
            from memory.bm25_search import rebuild_bm25_index
            rebuild_bm25_index()
        else:
            from memory.vector_index import get_index
            from memory.bm25_search import add_to_bm25_index
            vec_idx = get_index()
            for t in flat_turns:
                role = t.get("role", "")
                content = (t.get("content") or "").strip()
                ts = t.get("timestamp", "")
                if content and len(content) > 10:
                    vec_idx.add(
                        f"[{ts}] {role}: {content}",
                        {"type": "turn", "role": role, "content": content,
                         "timestamp": ts},
                    )
                    add_to_bm25_index(
                        f"{role}: {content}",
                        {"type": "turn", "role": role, "content": content,
                         "timestamp": ts, "id": t.get("id", 0)},
                    )
    except Exception as e:
        print(f"[IndexWorker] vector/bm25 task error: {e}")

    # 2. Chroma turn embeddings (semantic recall over raw chat)
    rows: list[dict] = []
    for t in tasks:
        rows.extend(t.get("chroma_rows") or [])
    if rows:
        try:
            from memory.chroma_memory import add_turn_embeddings
            add_turn_embeddings(rows)
        except Exception as e:
            print(f"[IndexWorker] chroma task error: {e}")


def _worker() -> None:
    """Single consumer loop: block on queue, drain a small batch, process."""
    while True:
        try:
            first = _task_queue.get()
        except Exception:
            continue
        if first is None:
            break  # shutdown sentinel
        tasks = [first]
        while len(tasks) < MAX_DRAIN:
            try:
                tasks.append(_task_queue.get_nowait())
            except queue.Empty:
                break
        run_tasks(tasks)
