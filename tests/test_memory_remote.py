"""Remote/cloud memory client tests: URL resolution, token auth, health-TTL
fallback to local, and remote write routing. Network fully mocked."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory import memory_client as mc  # noqa: E402


@pytest.fixture(autouse=True)
def _reset(tmp_path):
    """Fresh client state + config file per test."""
    cfg = tmp_path / "api_keys.json"
    cfg.write_text("{}", encoding="utf-8")
    mc._healthy = None
    mc._health_checked_at = 0.0
    mc._active_url = ""
    mc._SESSION = None
    env = {
        "ORT_MEMORY_SERVICE_URL": "",
        "ORT_MEMORY_SERVICE_TOKEN": "",
    }
    with patch.object(mc, "CONFIG_FILE", cfg, create=True), \
         patch.dict(mc.os.environ, env):
        yield cfg


def _set_cfg(cfg_file, d: dict):
    cfg_file.write_text(json.dumps(d), encoding="utf-8")


# ── URL resolution ───────────────────────────────────────────────────────────

class TestUrlResolution:
    def test_default_local(self, _reset):
        assert mc.get_service_url() == "http://127.0.0.1:9876"

    def test_config_url(self, _reset):
        _set_cfg(_reset, {"memory_service_url": "http://100.64.0.9:9876/"})
        assert mc.get_service_url() == "http://100.64.0.9:9876"  # trailing / stripped

    def test_env_overrides_config(self, _reset):
        _set_cfg(_reset, {"memory_service_url": "http://from-config:9876"})
        with patch.dict(mc.os.environ, {"ORT_MEMORY_SERVICE_URL": "http://from-env:9876"}):
            assert mc.get_service_url() == "http://from-env:9876"

    def test_local_url_matches_default(self, _reset):
        assert mc.get_service_url() == mc._LOCAL_URL


class TestToken:
    def test_token_from_config(self, _reset):
        _set_cfg(_reset, {"memory_service_token": "sekrit"})
        assert mc.get_service_token() == "sekrit"
        assert mc._headers() == {"Authorization": "Bearer sekrit"}

    def test_env_token(self, _reset):
        with patch.dict(mc.os.environ, {"ORT_MEMORY_SERVICE_TOKEN": "envtok"}):
            assert mc.get_service_token() == "envtok"

    def test_no_token_no_header(self, _reset):
        assert mc._headers() == {}


# ── Health TTL + fallback ────────────────────────────────────────────────────

class TestHealthFallback:
    def test_remote_up(self, _reset):
        _set_cfg(_reset, {"memory_service_url": "http://remote:9876"})
        with patch.object(mc, "_probe", return_value=True) as probe:
            assert mc._is_healthy() is True
        assert probe.call_args[0][0] == "http://remote:9876"
        assert mc.service_url() == "http://remote:9876"

    def test_remote_down_falls_back_local(self, _reset):
        # cloud_only=False: legacy behavior — fall back to a local service.
        _set_cfg(_reset, {"memory_service_url": "http://remote:9876",
                          "memory_cloud_only": False})
        calls = []
        def probe(url, timeout=2.0):
            calls.append(url)
            return url == mc._LOCAL_URL
        with patch.object(mc, "_probe", side_effect=probe):
            assert mc._is_healthy() is True
        assert calls[0] == "http://remote:9876"
        assert mc.service_url() == mc._LOCAL_URL  # fallback active

    def test_cloud_only_no_local_fallback(self, _reset):
        # DEFAULT cloud-only mode: remote down → unhealthy, NO local probe.
        _set_cfg(_reset, {"memory_service_url": "http://remote:9876"})
        assert mc.get_cloud_only() is True, "cloud-only must be the default"
        calls = []
        def probe(url, timeout=2.0):
            calls.append(url)
            return url == mc._LOCAL_URL
        with patch.object(mc, "_probe", side_effect=probe):
            assert mc._is_healthy() is False
        assert calls == ["http://remote:9876"], \
            "cloud-only mode must never probe the local service"
        assert mc.service_url() != mc._LOCAL_URL

    def test_all_down_is_unhealthy(self, _reset):
        _set_cfg(_reset, {"memory_service_url": "http://remote:9876"})
        with patch.object(mc, "_probe", return_value=False):
            assert mc._is_healthy() is False

    def test_ttl_caches_verdict(self, _reset):
        _set_cfg(_reset, {"memory_service_url": "http://remote:9876"})
        with patch.object(mc, "_probe", return_value=True) as probe:
            mc._is_healthy()
            mc._is_healthy()
            assert probe.call_count == 1  # TTL cache prevents re-probe

    def test_ttl_expiry_reprobes(self, _reset):
        _set_cfg(_reset, {"memory_service_url": "http://remote:9876"})
        with patch.object(mc, "_probe", return_value=True) as probe:
            mc._is_healthy()
            mc._health_checked_at -= 61  # force expiry
            mc._is_healthy()
            assert probe.call_count == 2

    def test_url_change_invalidates(self, _reset):
        with patch.object(mc, "_probe", return_value=True):
            mc._is_healthy()  # local verdict cached
        _set_cfg(_reset, {"memory_service_url": "http://other:9876"})
        with patch.object(mc, "_probe", return_value=True) as probe:
            mc._is_healthy()
            assert probe.called  # URL changed -> fresh probe, no stale cache


# ── Calls carry token + use service_url() ────────────────────────────────────

class TestCalls:
    def test_embed_uses_active_url_and_token(self, _reset):
        _set_cfg(_reset, {"memory_service_url": "http://remote:9876",
                          "memory_service_token": "tok"})
        with patch.object(mc, "_is_healthy", return_value=True), \
             patch.object(mc, "service_url", return_value="http://remote:9876"):
            resp = MagicMock()
            resp.json.return_value = {"vectors": [[0.1, 0.2]], "dim": 2}
            resp.raise_for_status = lambda: None
            sess = MagicMock()
            sess.post.return_value = resp
            with patch.object(mc, "_SESSION", sess):
                vecs = mc.memory_embed(["hi"])
        assert vecs.shape == (1, 2)
        args, kwargs = sess.post.call_args
        assert args[0] == "http://remote:9876/embed"
        assert kwargs["headers"] == {"Authorization": "Bearer tok"}

    def test_upsert_memory_unhealthy_returns_false(self, _reset):
        with patch.object(mc, "_is_healthy", return_value=False):
            assert mc.memory_upsert_memory({"category": "notes", "key": "k", "value": "v"}) is False

    def test_add_turns_empty_returns_zero(self, _reset):
        assert mc.memory_add_turn_embeddings([]) == 0


# ── Cloud-only mode default ────────────────────────────────────────────────

class TestCloudOnlyMode:
    def test_default_is_cloud_only(self, _reset):
        assert mc.get_cloud_only() is True, "missing key must default to True"

    def test_env_cloud_only_false(self, _reset):
        with patch.dict(mc.os.environ, {"ORT_MEMORY_CLOUD_ONLY": "false"}):
            assert mc.get_cloud_only() is False

    def test_env_cloud_only_true(self, _reset):
        _set_cfg(_reset, {"memory_cloud_only": False})
        with patch.dict(mc.os.environ, {"ORT_MEMORY_CLOUD_ONLY": "1"}):
            assert mc.get_cloud_only() is True

    def test_config_cloud_only_false(self, _reset):
        _set_cfg(_reset, {"memory_cloud_only": False})
        assert mc.get_cloud_only() is False
