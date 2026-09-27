"""Tests for Option 3: typed NER via memory service (/ner_typed endpoint).

Covers: memory_client.memory_ner_typed (HTTP contract), service /ner_typed +
backward-compatible /ner (names only), and link_fact using the service type
map with a local spaCy fallback when the service is unreachable.
"""
import asyncio
import os

import pytest

# Captured at import time (before conftest autouse fixtures patch it).
import memory.entity_store as _es

_REAL_LINK_FACT = _es.link_fact


class _Ent:
    def __init__(self, label, text):
        self.label_ = label
        self.text = text


class _Doc:
    def __init__(self, ents):
        self.ents = ents


@pytest.fixture
def real_link_fact(monkeypatch, tmp_path):
    """Redirect KG to temp file and restore the real link_fact."""
    monkeypatch.setattr(_es, "KG_PATH", tmp_path / "kg.json")
    monkeypatch.setattr(_es, "link_fact", _REAL_LINK_FACT)
    return _REAL_LINK_FACT


class TestMemoryNerTypedClient:

    @pytest.fixture(autouse=True)
    def _force_local_url(self, monkeypatch):
        """Tests must not depend on the developer's real config file."""
        import memory.memory_client as mc
        monkeypatch.setattr(mc, "get_service_url", lambda: "http://127.0.0.1:9876")
        monkeypatch.setattr(mc, "service_url", lambda: "http://127.0.0.1:9876")
    def test_contract(self, monkeypatch):
        import memory.memory_client as mc

        class FakeResp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"entities": [{"name": "Harsh", "type": "Person"}]}

        captured = {}

        def fake_post(url, json=None, timeout=None, headers=None, **kw):
            captured["url"] = url
            captured["json"] = json
            return FakeResp()

        class FakeSession:
            post = staticmethod(fake_post)

        monkeypatch.setattr(mc, "_session", lambda: FakeSession())
        result = mc.memory_ner_typed("My name is Harsh")
        assert result == [{"name": "Harsh", "type": "Person"}]
        assert captured["url"] == "http://127.0.0.1:9876/ner_typed"
        assert captured["json"] == {"text": "My name is Harsh"}

    def test_raises_when_service_down(self, monkeypatch):
        import memory.memory_client as mc

        class DeadSession:
            def post(self, *a, **k):
                raise ConnectionError("service down")

        monkeypatch.setattr(mc, "_session", lambda: DeadSession())
        with pytest.raises(ConnectionError):
            mc.memory_ner_typed("text")


class TestServiceNerEndpoints:
    def _import_service(self):
        from memory import memory_service as svc

        os.environ.pop("MEMORY_SERVICE_PROCESS", None)
        return svc

    def test_ner_typed_returns_types(self, monkeypatch):
        svc = self._import_service()
        monkeypatch.setattr(
            svc, "_ner",
            lambda text: _Doc([_Ent("PERSON", "Harsh"), _Ent("ORG", "Google")]),
        )
        resp = asyncio.run(svc.ner_typed(svc.NERRequest(text="My name is Harsh and I work at Google")))
        assert resp.entities == [
            {"name": "Harsh", "type": "Person"},
            {"name": "Google", "type": "Organization"},
        ]

    def test_ner_backward_compat_names_only(self, monkeypatch):
        svc = self._import_service()
        monkeypatch.setattr(
            svc, "_ner",
            lambda text: _Doc([_Ent("PERSON", "Harsh"), _Ent("ORG", "Google")]),
        )
        resp = asyncio.run(svc.ner(svc.NERRequest(text="My name is Harsh and I work at Google")))
        assert resp.entities == ["Harsh", "Google"]

    def test_ner_typed_empty_without_model(self, monkeypatch):
        svc = self._import_service()
        monkeypatch.setattr(svc, "_ner", None)
        resp = asyncio.run(svc.ner_typed(svc.NERRequest(text="anything")))
        assert resp.entities == []


class TestLinkFactTypedTypes:
    def test_type_from_service(self, monkeypatch, real_link_fact):
        import memory.memory_client as mc

        monkeypatch.setattr(mc, "_is_healthy", lambda: True)
        monkeypatch.setattr(mc, "memory_ner", lambda text: ["Harsh", "Google"])
        monkeypatch.setattr(mc, "memory_ner_typed", lambda text: [
            {"name": "Harsh", "type": "Person"},
            {"name": "Google", "type": "Organization"},
        ])
        real_link_fact("identity", "name", "My name is Harsh and I work at Google")
        graph = _es._load()
        types = {e["name"]: e["type"] for e in graph["entities"].values()}
        assert types.get("Harsh") == "Person"
        assert types.get("Google") == "Organization"

    def test_local_spacy_fallback_when_service_down(self, monkeypatch, real_link_fact):
        import memory.memory_client as mc
        from memory import entity_store as es

        monkeypatch.setattr(mc, "_is_healthy", lambda: False)
        called = {}

        def fake_ner(text):
            return _Doc([_Ent("PERSON", "Harsh")])

        monkeypatch.setattr(es, "_get_ner", lambda: (called.__setitem__("called", True), fake_ner)[1])
        real_link_fact("identity", "name", "My name is Harsh and I work at Google")
        assert called.get("called") is True
        graph = _es._load()
        types = {e["name"]: e["type"] for e in graph["entities"].values()}
        assert types.get("Harsh") == "Person"
