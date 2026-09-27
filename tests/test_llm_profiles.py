"""Tests for core/llm_profiles.py — endpoint profile store and switching.

Network access is fully mocked; config I/O is redirected to a temp file.
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import llm_profiles as lp  # noqa: E402


@pytest.fixture(autouse=True)
def _tmp_config(tmp_path):
    """Point llm_profiles' CONFIG_FILE at a throwaway file per test."""
    cfg_file = tmp_path / "api_keys.json"
    cfg_file.write_text("{}", encoding="utf-8")
    with patch.object(lp, "CONFIG_FILE", cfg_file):
        yield cfg_file


def _write_cfg(d: dict, cfg_file) -> None:
    cfg_file.write_text(json.dumps(d), encoding="utf-8")


def _read_cfg(cfg_file) -> dict:
    return json.loads(cfg_file.read_text(encoding="utf-8"))


# ── get_profiles / get_active_profile ────────────────────────────────────────

class TestGetProfiles:
    def test_empty_when_missing(self, _tmp_config):
        assert lp.get_profiles() == {}
        assert lp.get_active_profile() == ""

    def test_reads_saved_profiles(self, _tmp_config):
        _write_cfg({
            "llm_profiles": {
                "kaggle": {"llm_url": "https://x.trycloudflare.com", "llm_model": "qwen:27b"},
            },
        }, _tmp_config)
        assert lp.get_profiles()["kaggle"]["llm_url"] == "https://x.trycloudflare.com"

    def test_skips_corrupt_entries(self, _tmp_config):
        _write_cfg({"llm_profiles": {"bad": "not-a-dict"}}, _tmp_config)
        assert lp.get_profiles() == {}

    def test_is_profile_active(self, _tmp_config):
        _write_cfg({
            "llm_profiles": {"kaggle": {"llm_url": "https://x", "llm_model": "m"}},
            "llm_active_profile": "kaggle",
        }, _tmp_config)
        assert lp.is_profile_active() is True
        assert lp.get_active_profile() == "kaggle"


# ── save / delete / switch ───────────────────────────────────────────────────

class TestSaveProfile:
    def test_save_normalizes_name(self, _tmp_config):
        stored = lp.save_profile("My Kaggle!", "https://x.trycloudflare.com/", "m")
        assert stored == "my_kaggle"
        assert _read_cfg(_tmp_config)["llm_profiles"]["my_kaggle"]["llm_url"] == "https://x.trycloudflare.com"

    def test_save_rejects_empty(self, _tmp_config):
        with pytest.raises(ValueError):
            lp.save_profile("  ", "http://x", "m")

    def test_save_preserves_other_keys(self, _tmp_config):
        _write_cfg({"llm_url": "http://localhost:11434", "gemini_api_key": "keep-me"}, _tmp_config)
        lp.save_profile("kaggle", "https://x", "m")
        cfg = _read_cfg(_tmp_config)
        assert cfg["llm_url"] == "http://localhost:11434"
        assert cfg["gemini_api_key"] == "keep-me"


class TestDeleteProfile:
    def test_delete_removes(self, _tmp_config):
        lp.save_profile("kaggle", "https://x", "m")
        lp.delete_profile("kaggle")
        assert "kaggle" not in lp.get_profiles()

    def test_delete_active_resets_to_local(self, _tmp_config):
        lp.save_profile("kaggle", "https://x", "m")
        lp.switch_profile("kaggle")
        lp.delete_profile("kaggle")
        assert lp.get_active_profile() == ""

    def test_delete_missing_is_noop(self, _tmp_config):
        lp.delete_profile("ghost")  # must not raise


class TestSwitchProfile:
    def test_switch_to_saved(self, _tmp_config):
        lp.save_profile("kaggle", "https://x", "m")
        assert lp.switch_profile("kaggle") is True
        assert lp.get_active_profile() == "kaggle"

    def test_switch_to_unknown_fails(self, _tmp_config):
        assert lp.switch_profile("ghost") is False

    def test_switch_to_local_deactivates(self, _tmp_config):
        lp.save_profile("kaggle", "https://x", "m")
        lp.switch_profile("kaggle")
        assert lp.switch_profile("local") is True
        assert lp.get_active_profile() == ""


# ── resolve_profile_url_model ────────────────────────────────────────────────

class TestResolve:
    DEFAULTS = ("http://localhost:11434", "gemma:7b")

    def test_no_active_returns_defaults(self, _tmp_config):
        assert lp.resolve_profile_url_model(*self.DEFAULTS) == self.DEFAULTS

    def test_active_remote_overrides(self, _tmp_config):
        lp.save_profile("kaggle", "https://tunnel.example.com/", "qwen:27b")
        lp.switch_profile("kaggle")
        url, model = lp.resolve_profile_url_model(*self.DEFAULTS)
        assert url == "https://tunnel.example.com"      # trailing / stripped
        assert model == "qwen:27b"

    def test_local_profile_always_defaults(self, _tmp_config):
        lp.save_profile("local", "https://stale.example.com", "wrong-model")
        lp.switch_profile("local")
        assert lp.resolve_profile_url_model(*self.DEFAULTS) == self.DEFAULTS

    def test_dangling_active_falls_back(self, _tmp_config):
        _write_cfg({"llm_active_profile": "ghost"}, _tmp_config)
        assert lp.resolve_profile_url_model(*self.DEFAULTS) == self.DEFAULTS


# ── test_endpoint (network mocked) ───────────────────────────────────────────

class TestTestEndpoint:
    def test_ok(self):
        resp = MagicMock(status_code=200, json=lambda: {"models": [{"name": "qwen:27b"}]})
        with patch.object(lp.requests, "get", return_value=resp):
            ok, msg = lp.test_endpoint("https://x.trycloudflare.com", "qwen:27b")
        assert ok is True

    def test_model_missing_warns_but_ok(self):
        resp = MagicMock(status_code=200, json=lambda: {"models": [{"name": "other:7b"}]})
        with patch.object(lp.requests, "get", return_value=resp):
            ok, msg = lp.test_endpoint("https://x", "qwen:27b")
        assert ok is True
        assert "not pulled" in msg

    def test_connection_error_unreachable(self):
        import requests as _rq
        with patch.object(lp.requests, "get", side_effect=_rq.exceptions.ConnectionError("refused")):
            ok, msg = lp.test_endpoint("https://x")
        assert ok is False

    def test_timeout(self):
        import requests as _rq
        with patch.object(lp.requests, "get", side_effect=_rq.exceptions.Timeout()):
            ok, msg = lp.test_endpoint("https://x")
        assert ok is False
        assert "Timed out" in msg

    def test_empty_url(self):
        ok, msg = lp.test_endpoint("")
        assert ok is False

    def test_scheme_added(self):
        resp = MagicMock(status_code=200, json=lambda: {"models": []})
        with patch.object(lp.requests, "get", return_value=resp) as mget:
            lp.test_endpoint("x.trycloudflare.com")
        assert mget.call_args[0][0] == "http://x.trycloudflare.com/api/tags"


# ── _is_remote_url ───────────────────────────────────────────────────────────

class TestIsRemoteUrl:
    def test_localhost_variants(self):
        for u in ("http://localhost:11434", "http://127.0.0.1:11434", ""):
            assert lp._is_remote_url(u) is False

    def test_remote(self):
        assert lp._is_remote_url("https://abc.trycloudflare.com") is True
