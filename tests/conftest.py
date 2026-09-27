import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# ---- Ensure project root is on sys.path for imports ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---- Deterministic indexing in tests: run vector/bm25/chroma work inline ----
# (the background index worker would otherwise race with assertions)
os.environ.setdefault("ORT_SYNC_INDEXING", "1")

# ---- Redirect memory store to a throwaway SQLite DB (no long_term.json) ----
import memory.conversation_db as _cdb

_TMP_DB = Path(tempfile.mkdtemp(prefix="alex_mem_test_")) / "conversations.db"
_cdb.DB_PATH = _TMP_DB
_cdb.init_db()  # create schema in the temp DB

# ---- Module-level patches: applied before any test module is imported ----
# Prevent legacy migrations from reading real files / the real DB.
_patcher_ismig = patch("memory.conversation_db.is_migrated", return_value=True)
_patcher_migjs = patch("memory.conversation_db.migrate_json_sessions", return_value=0)
_patcher_miglt = patch("memory.conversation_db.migrate_long_term_to_sqlite", return_value=0)
_patcher_ismig.start()
_patcher_migjs.start()
_patcher_miglt.start()

# Patch the _HTTP session in llm_client early so imported modules don't use real HTTP
_patcher_http = patch("core.llm_client._HTTP", MagicMock())
_patcher_http.start()
# We'll re-create fresh mocks per test in test_llm_client.py via a fixture


# ---- Fixtures ----

@pytest.fixture(autouse=True)
def _clear_memory_items():
    """Empty the memory_items table between tests for full isolation."""
    yield
    try:
        import sqlite3
        with sqlite3.connect(str(_cdb.DB_PATH)) as conn:
            conn.execute("DELETE FROM memory_items")
            conn.execute("DELETE FROM memory_versions")
            conn.commit()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _auto_patch_entity_store():
    """Patch entity_store functions that are imported/used in memory_manager."""
    with patch("memory.entity_store.link_fact", return_value=None):
        yield


@pytest.fixture(autouse=True)
def _auto_patch_dreaming():
    """Patch dreaming module calls."""
    with patch("memory.dreaming.mark_active", return_value=None), \
         patch("memory.dreaming.trigger_dream", return_value=None):
        yield


@pytest.fixture(autouse=True)
def _auto_patch_vector_bm25():
    """Patch vector_index and bm25_search calls to prevent real index operations."""
    mock_vec_index = MagicMock()
    mock_vec_index.texts = []
    with \
        patch("memory.vector_index.rebuild_index", return_value=None), \
        patch("memory.bm25_search.rebuild_bm25_index", return_value=None), \
        patch("memory.bm25_search.search_bm25", return_value=[]), \
        patch("memory.vector_index.search_vector", return_value=[]), \
        patch("memory.bm25_search.add_to_bm25_index", return_value=None), \
        patch("memory.vector_index.get_index", return_value=mock_vec_index), \
        patch("memory.vector_index.embed", return_value=[0.1] * 128), \
        patch("memory.bm25_search.get_index"):
        yield


@pytest.fixture(autouse=True)
def _auto_patch_query_expansion():
    """Patch query_expansion to return empty expansions."""
    with \
        patch("memory.query_expansion.search_with_expansion", return_value=[]), \
        patch("memory.query_expansion.expand_query", return_value=[""]):
        yield


@pytest.fixture(autouse=True)
def _auto_patch_entity_graph():
    """Patch entity_store.get_entity_graph to return None."""
    with patch("memory.entity_store.get_entity_graph", return_value=None):
        yield


@pytest.fixture(autouse=True)
def _auto_patch_git_memory():
    """Patch git_memory calls to prevent real git operations."""
    with patch("memory.git_memory.commit_memory", return_value=None):
        yield


# ---- Mock response builders ----

@pytest.fixture
def mock_ollama_response():
    """Return a function that produces a mock Ollama /api/chat response."""
    def _make(content: str = "Hello!", tool_calls: list | None = None):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "message": {
                "content": content,
                "tool_calls": tool_calls or [],
            },
        }
        return resp
    return _make


# ---- Temporary directories ----

@pytest.fixture
def tmp_memory_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for memory/ test files."""
    d = tmp_path / "memory"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---- Test data ----

@pytest.fixture
def sample_memory() -> dict:
    return {
        "identity": {
            "name": {"value": "Alice", "updated": "2025-01-01", "score": 1.0, "access_count": 1, "source": "manual"},
        },
        "preferences": {
            "color": {"value": "blue", "updated": "2025-01-15", "score": 0.8, "access_count": 0, "source": "manual"},
        },
        "projects": {},
        "relationships": {},
        "wishes": {},
        "notes": {
            "todo": {"value": "buy milk", "updated": "2025-06-01", "score": 0.5, "access_count": 0, "source": "manual"},
        },
        "procedures": {},
    }


@pytest.fixture
def sample_memory_json(sample_memory: dict) -> str:
    return json.dumps(sample_memory, indent=2, ensure_ascii=False)
