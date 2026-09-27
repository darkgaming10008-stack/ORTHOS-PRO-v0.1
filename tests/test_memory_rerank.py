"""Tests for service-side reranker preload + HTTP /rerank flow.

Covers: memory_client.memory_rerank (HTTP contract + service-down failure),
memory_manager.search_all_memories re-ordering via the service (and sort
fallback when the service is down), and the memory service /rerank handler.
"""
import asyncio
import os

import pytest


@pytest.fixture(autouse=True)
def _patch_rerank_service_signals(monkeypatch):
    """Deterministic search signals: only BM25 returns 3 items."""
    monkeypatch.setattr(
        "memory.query_expansion.search_with_expansion",
        lambda query, search_bm25, top_k=10: [
            ("alpha", {"key": "a"}, 4.0),
            ("beta", {"key": "b"}, 3.0),
            ("gamma", {"key": "c"}, 2.0),
        ],
    )
    monkeypatch.setattr("memory.memory_manager.search_memory_items", lambda *a, **k: [])
    monkeypatch.setattr("memory.chroma_memory.CHROMA_VECTOR_SEARCH_ENABLED", False)
    monkeypatch.setenv("ALEX_RERANKER", "1")


class TestMemoryRerankClient:

    @pytest.fixture(autouse=True)
    def _force_local_url(self, monkeypatch):
        """Tests must not depend on the developer's real config file."""
        import memory.memory_client as mc
        monkeypatch.setattr(mc, "get_service_url", lambda: "http://127.0.0.1:9876")
        monkeypatch.setattr(mc, "service_url", lambda: "http://127.0.0.1:9876")

    def test_returns_order_from_service(self, monkeypatch):
        import memory.memory_client as mc

        class FakeResp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"order": [2, 0, 1]}

        captured = {}

        def fake_post(url, json=None, timeout=None, headers=None, **kw):
            captured["url"] = url
            captured["json"] = json
            return FakeResp()

        class FakeSession:
            post = staticmethod(fake_post)

        monkeypatch.setattr(mc, "_session", lambda: FakeSession())
        order = mc.memory_rerank("q", ["a", "b", "c"], top_k=2)
        assert order == [2, 0, 1]
        assert captured["url"] == "http://127.0.0.1:9876/rerank"
        assert captured["json"] == {"query": "q", "texts": ["a", "b", "c"], "top_k": 2}

    def test_top_k_none_passed_through(self, monkeypatch):
        import memory.memory_client as mc

        captured = {}

        class FakeResp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"order": [0, 1]}

        class FakeResp2:
            def raise_for_status(self):
                pass

            def json(self):
                return {"order": [0, 1]}

        def fake_post2(url, json=None, timeout=None, headers=None, **kw):
            captured["json"] = json
            return FakeResp2()

        class FakeSession2:
            post = staticmethod(fake_post2)

        monkeypatch.setattr(mc, "_session", lambda: FakeSession2())
        mc.memory_rerank("q", ["a", "b"])
        assert captured["json"]["top_k"] is None

    def test_raises_when_service_down(self, monkeypatch):
        import memory.memory_client as mc

        class DeadSession:
            def post(self, *a, **k):
                raise ConnectionError("service down")

        monkeypatch.setattr(mc, "_session", lambda: DeadSession())
        with pytest.raises(ConnectionError):
            mc.memory_rerank("q", ["a", "b"], top_k=2)


class TestMemoryManagerUsesService:
    def test_rerank_order_applied(self, monkeypatch):
        from memory import memory_manager as mm

        captured = {}

        def fake_rerank(query, texts, top_k=None):
            captured["query"] = query
            captured["texts"] = texts
            captured["top_k"] = top_k
            return [len(texts) - 1 - i for i in range(top_k or 1)]

        monkeypatch.setattr("memory.memory_client.memory_rerank", fake_rerank)
        result = mm.search_all_memories("espresso", limit=2)
        assert captured["query"] == "espresso"
        assert captured["texts"] == ["alpha", "beta", "gamma"]
        assert captured["top_k"] == 2
        assert result.index("TURN: gamma") < result.index("TURN: beta")
        assert "TURN: alpha" not in result

    def test_sort_fallback_when_service_down(self, monkeypatch):
        from memory import memory_manager as mm

        def fail(*a, **k):
            raise ConnectionError("service down")

        monkeypatch.setattr("memory.memory_client.memory_rerank", fail)
        result = mm.search_all_memories("espresso", limit=2)
        assert result.index("TURN: alpha") < result.index("TURN: beta")
        assert "TURN: gamma" not in result


class TestServiceRerankEndpoint:
    def _import_service(self):
        from memory import memory_service as svc

        os.environ.pop("MEMORY_SERVICE_PROCESS", None)
        return svc

    def test_rerank_endpoint_orders(self, monkeypatch):
        svc = self._import_service()

        def fake_rerank(query, candidates, top_k=None):
            return list(reversed(candidates))[: top_k or len(candidates)]

        monkeypatch.setattr("memory.reranker.rerank", fake_rerank)
        order = asyncio.run(svc.rerank(svc.RerankRequest(query="q", texts=["a", "b", "c"], top_k=2)))
        assert order.order == [2, 1]

    def test_rerank_endpoint_identity_on_error(self, monkeypatch):
        svc = self._import_service()

        def fail(*a, **k):
            raise RuntimeError("model gone")

        monkeypatch.setattr("memory.reranker.rerank", fail)
        order = asyncio.run(svc.rerank(svc.RerankRequest(query="q", texts=["a", "b", "c"])))
        assert order.order == [0, 1, 2]

    def test_rerank_endpoint_empty(self):
        svc = self._import_service()
        order = asyncio.run(svc.rerank(svc.RerankRequest(query="q", texts=[])))
        assert order.order == []

    def test_health_reports_reranker_loaded(self, monkeypatch):
        svc = self._import_service()
        monkeypatch.setattr(svc, "_reranker_ready", True)
        health = asyncio.run(svc.health())
        assert health["reranker_loaded"] is True