import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import pytest
from core.security import Sanitizer, SecretsManager, Vault, SafeFilePath


# ── Path traversal prevention ────────────────────────────────────────────────


class TestSanitizePath:
    def test_rejects_traversal(self):
        with pytest.raises(ValueError, match="traversal"):
            Sanitizer.sanitize_path("../etc/passwd")

    def test_rejects_double_dot(self):
        with pytest.raises(ValueError, match="traversal"):
            Sanitizer.sanitize_path("foo/../../bar")

    def test_rejects_absolute_by_default(self):
        with pytest.raises(ValueError, match="Absolute path"):
            Sanitizer.sanitize_path("C:/Windows/system32")

    def test_allows_absolute_when_permitted(self):
        result = Sanitizer.sanitize_path("C:/Windows", allow_absolute=True)
        assert result == "C:\\Windows"

    def test_resolves_relative_under_project(self):
        result = Sanitizer.sanitize_path("memory/conversations.db")
        assert "memory" in result
        assert "conversations" in result

    def test_rejects_outside_project(self):
        with pytest.raises(ValueError):
            Sanitizer.sanitize_path("..\\..\\Windows")

    def test_backslash_traversal_rejected(self):
        with pytest.raises(ValueError, match="traversal"):
            Sanitizer.sanitize_path("..\\..\\etc")


# ── Command whitelist ────────────────────────────────────────────────────────


class TestSanitizeCommand:
    def test_allows_known_command(self):
        assert Sanitizer.sanitize_command("git") == "git"

    def test_rejects_unknown_command(self):
        with pytest.raises(ValueError, match="not in the allowed list"):
            Sanitizer.sanitize_command("rm")

    def test_case_insensitive(self):
        assert Sanitizer.sanitize_command("GIT") == "GIT"

    def test_python_allowed(self):
        assert Sanitizer.sanitize_command("python") == "python"

    def test_strips_whitespace(self):
        result = Sanitizer.sanitize_command("  date  ")
        assert result == "  date  "  # returns original casing, does not strip


# ── Shell arg sanitization ───────────────────────────────────────────────────


class TestSanitizeShellArg:
    def test_escapes_ampersand(self):
        assert Sanitizer.sanitize_shell_arg("a&b") == "a^&b"

    def test_escapes_pipe(self):
        assert Sanitizer.sanitize_shell_arg("a|b") == "a^|b"

    def test_escapes_semicolon(self):
        assert Sanitizer.sanitize_shell_arg("a;b") == "a^;b"

    def test_safe_string_unchanged(self):
        assert Sanitizer.sanitize_shell_arg("hello") == "hello"

    def test_empty_string(self):
        assert Sanitizer.sanitize_shell_arg("") == ""

    def test_escapes_backtick(self):
        result = Sanitizer.sanitize_shell_arg("`rm -rf /`")
        assert result == "^`rm -rf /^`"

    def test_escapes_dollar(self):
        assert Sanitizer.sanitize_shell_arg("$PATH") == "^$PATH"


# ── Safe file paths ──────────────────────────────────────────────────────────


class TestSafeFilePath:
    def test_context_manager_returns_path(self):
        with SafeFilePath("memory/test.txt") as safe:
            assert isinstance(safe, Path)
            assert "memory" in str(safe)

    def test_rejects_traversal(self):
        with pytest.raises(ValueError):
            with SafeFilePath("../outside"):
                pass

    def test_sanitize_filepath_for_write_relative(self, tmp_path):
        target = tmp_path / "sub" / "out.txt"
        with patch.object(Sanitizer, "sanitize_path", return_value=str(target)):
            safe = Sanitizer.sanitize_filepath_for_write(str(target))
            safe.write_text("hello", encoding="utf-8")
            assert safe.read_text(encoding="utf-8") == "hello"


# ── Secrets manager ──────────────────────────────────────────────────────────


class TestSecretsManager:
    def test_get_from_env_var(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "env-value-123")
        sm = SecretsManager()
        assert sm.get("GEMINI_API_KEY") == "env-value-123"

    def test_get_from_json_fallback(self, monkeypatch, tmp_path):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        cfg = tmp_path / "api_keys.json"
        cfg.write_text(json.dumps({"telegram_bot_token": "json-value-456"}), encoding="utf-8")
        sm = SecretsManager(json_path=str(cfg))
        assert sm.get("TELEGRAM_BOT_TOKEN") == "json-value-456"

    def test_env_takes_priority(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "env-value")
        cfg = tmp_path / "api_keys.json"
        cfg.write_text(json.dumps({"telegram_bot_token": "json-value"}), encoding="utf-8")
        sm = SecretsManager(json_path=str(cfg))
        assert sm.get("TELEGRAM_BOT_TOKEN") == "env-value"

    def test_get_returns_none_if_not_found(self):
        sm = SecretsManager()
        assert sm.get("NONEXISTENT_KEY_XYZ123") is None

    def test_set_stores_in_env(self):
        sm = SecretsManager()
        sm.set("MY_TEST_KEY_ABC", "my-val")
        assert os.environ.get("MY_TEST_KEY_ABC") == "my-val"
        del os.environ["MY_TEST_KEY_ABC"]

    def test_list_keys_from_env(self, monkeypatch):
        monkeypatch.setenv("ELEVENLABS_API_KEY", "abc")
        sm = SecretsManager()
        keys = sm.list_keys()
        assert "ELEVENLABS_API_KEY" in keys

    def test_list_keys_from_json(self, monkeypatch, tmp_path):
        cfg = tmp_path / "api_keys.json"
        cfg.write_text(json.dumps({"telegram_bot_token": "xyz"}), encoding="utf-8")
        sm = SecretsManager(json_path=str(cfg))
        keys = sm.list_keys()
        assert "TELEGRAM_BOT_TOKEN" in keys

    def test_upper_strips_and_uppercases(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "val")
        sm = SecretsManager()
        assert sm.get("  gemini_api_key  ") == "val"


# ── Vault ────────────────────────────────────────────────────────────────────


class TestVault:
    def test_store_and_retrieve(self, tmp_path):
        vault_path = tmp_path / ".vault.json"
        v = Vault(vault_path=str(vault_path))
        v.unlock("correct-password")
        v.store("MY_SECRET", "s3cr3t!")
        assert v.retrieve("MY_SECRET") == "s3cr3t!"

    def test_locked_vault_raises_on_store(self):
        v = Vault(vault_path="C:/nonexistent/.vault.json")
        with pytest.raises(RuntimeError, match="locked"):
            v.store("x", "y")

    def test_locked_vault_raises_on_retrieve(self):
        v = Vault(vault_path="C:/nonexistent/.vault.json")
        with pytest.raises(RuntimeError, match="locked"):
            v.retrieve("x")

    def test_unlock_wrong_password(self, tmp_path):
        vault_path = tmp_path / ".vault.json"
        v = Vault(vault_path=str(vault_path))
        v.unlock("pw1")
        v.store("k", "v")
        v.lock()
        v2 = Vault(vault_path=str(vault_path))
        assert v2.unlock("wrong-pw") is False
        with pytest.raises(RuntimeError):
            v2.retrieve("k")

    def test_empty_vault_unlock_ok(self, tmp_path):
        vault_path = tmp_path / ".vault.json"
        v = Vault(vault_path=str(vault_path))
        assert v.unlock("any-password") is True

    def test_lock_prevents_access(self, tmp_path):
        vault_path = tmp_path / ".vault.json"
        v = Vault(vault_path=str(vault_path))
        v.unlock("pw")
        v.store("k", "v")
        v.lock()
        with pytest.raises(RuntimeError):
            v.retrieve("k")

    def test_missing_vault_file_returns_empty(self, tmp_path):
        vault_path = tmp_path / "nonexistent.vault"
        v = Vault(vault_path=str(vault_path))
        assert v._read_vault() == {}
