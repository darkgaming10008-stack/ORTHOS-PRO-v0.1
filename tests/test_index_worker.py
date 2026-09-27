"""Tests: background index worker (async indexing off the chat thread)."""
import time
from unittest.mock import MagicMock, patch

from memory import index_worker as iw


def test_sync_mode_submit_returns_false(monkeypatch):
    monkeypatch.setenv("ORT_SYNC_INDEXING", "1")
    assert iw.submit({"turns": []}) is False


def test_submit_queues_async_and_worker_drains(monkeypatch):
    monkeypatch.setenv("ORT_SYNC_INDEXING", "0")
    with patch("memory.index_worker.run_tasks") as fake_run:
        assert iw.submit({"turns": [{"role": "user", "content": "queue drain test"}]}) is True
        deadline = time.time() + 5
        while iw.pending() > 0 and time.time() < deadline:
            time.sleep(0.05)
        # give the worker a beat to invoke the patched run_tasks
        deadline = time.time() + 5
        while not fake_run.called and time.time() < deadline:
            time.sleep(0.05)
        assert fake_run.called, "worker must process queued tasks"


def test_run_tasks_drives_vector_and_bm25(monkeypatch):
    vec = MagicMock()
    bm25_add = MagicMock()
    monkeypatch.setattr("memory.vector_index.get_index", lambda: vec)
    monkeypatch.setattr("memory.bm25_search.add_to_bm25_index", bm25_add)

    iw.run_tasks([{"turns": [
        {"role": "user", "content": "mango shake recipe test message", "timestamp": "2026-09-27T10:00:00", "id": 42},
        {"role": "tool", "content": "", "timestamp": "", "id": 43},  # empty → skipped
    ]}])

    assert vec.add.called
    args = vec.add.call_args[0]
    assert "mango shake recipe" in args[0]
    assert args[1]["type"] == "turn"
    assert bm25_add.called


def test_run_tasks_full_rebuild_path(monkeypatch):
    monkeypatch.setattr("memory.memory_manager.get_recent_turns_for_vector", lambda limit=50: [])
    monkeypatch.setattr("memory.memory_manager.load_memory", lambda: {})
    monkeypatch.setattr("memory.vector_index.rebuild_index", MagicMock())
    monkeypatch.setattr("memory.bm25_search.rebuild_bm25_index", MagicMock())
    vec = MagicMock()
    monkeypatch.setattr("memory.vector_index.get_index", lambda: vec)

    iw.run_tasks([{"turns": [], "full_rebuild": True}])

    import memory.vector_index as vi
    assert vi.rebuild_index.called
    assert not vec.add.called, "rebuild path must skip incremental adds"


def test_run_tasks_chroma_rows(monkeypatch):
    seen = {}
    monkeypatch.setattr("memory.chroma_memory.add_turn_embeddings",
                        lambda rows: seen.setdefault("n", len(rows)))
    iw.run_tasks([{"chroma_rows": [
        {"id": 1, "role": "user", "content": "x" * 30, "session_id": "s", "timestamp": ""},
    ]}])
    assert seen.get("n") == 1
