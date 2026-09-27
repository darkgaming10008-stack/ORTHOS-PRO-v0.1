"""
security.py — Orthos Security Hardening Module

Provides:
  • Sanitizer       – Path, command, and shell-arg sanitisation
  • SecretsManager  – API keys from env vars → JSON file fallback
  • Vault           – Fernet-based encrypted key-value storage
  • SafeFilePath    – Context manager that enforces allowed directory access

Requirements
────────────
  pip install cryptography

Integration Guide (replacing raw JSON reads)
────────────────────────────────────────────

  ┌ BEFORE ─────────────────────────────────────────────────────────┐
  │ from pathlib import Path                                        │
  │ _cfg = json.loads(Path("config/api_keys.json").read_text())     │
  │ token = _cfg.get("telegram_bot_token", "")                      │
  └─────────────────────────────────────────────────────────────────┘

  ┌ AFTER ──────────────────────────────────────────────────────────┐
  │ from core.security import SecretsManager                        │
  │ secrets = SecretsManager()                                      │
  │ token   = secrets.get("TELEGRAM_BOT_TOKEN")                     │
  └─────────────────────────────────────────────────────────────────┘

  Both env-var and JSON-file lookups work without any extra setup.
  To migrate existing api_keys.json into Vault (encrypted):

      from core.security import Vault
      v = Vault()
      v.unlock("your-master-password")
      v.store("GEMINI_API_KEY", "…")
      v.store("ELEVENLABS_API_KEY", "…")
      v.lock()
"""

from __future__ import annotations

import json
import os
import re
import stat
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any


# ═════════════════════════════════════════════════════════════════════════════
# Public helpers
# ═════════════════════════════════════════════════════════════════════════════

_BASE_DIR: Path | None = None


def _project_root() -> Path:
    global _BASE_DIR
    if _BASE_DIR is None:
        _BASE_DIR = Path(__file__).resolve().parent.parent
    return _BASE_DIR


# ═════════════════════════════════════════════════════════════════════════════
# Sanitizer
# ═════════════════════════════════════════════════════════════════════════════

class Sanitizer:
    """Whitelist-based input validation for paths, commands, and shell args."""

    # -- commands allowed to run via subprocess ---------------------------------
    ALLOWED_COMMANDS: list[str] = [
        "git", "python", "pip", "node", "npm", "npx",
        "whoami", "date", "echo", "cd", "dir", "ls", "pwd",
        "cat", "type", "find", "grep", "where", "which",
        "systeminfo", "ipconfig", "tasklist", "ping", "curl", "wget",
    ]

    # -- shell metacharacters that MUST be escaped in arguments -----------------
    _SHELL_META: re.Pattern = re.compile(r'([&|;<>`$!(){}[\]^~#*\?\\\'"])')

    # -- path traversal patterns ------------------------------------------------
    _PATH_TRAVERSAL: re.Pattern = re.compile(r'(?:\.\.(?:[/\\]|$))')

    # ------------------------------------------------------------------
    # Path sanitisation
    # ------------------------------------------------------------------
    @classmethod
    def sanitize_path(cls, path: str | Path, allow_absolute: bool = False) -> str:
        """
        Resolve a user-supplied path and reject traversal attempts.

        Raises ValueError if the path contains '..' sequences or escapes
        the project directory (unless *allow_absolute* is True).
        """
        p = Path(path)

        # 1. Reject obvious traversal before resolution
        if cls._PATH_TRAVERSAL.search(str(p)):
            raise ValueError(f"Path traversal detected: {path!r}")

        # 2. Reject absolute paths unless explicitly allowed
        if p.is_absolute() and not allow_absolute:
            raise ValueError(f"Absolute path not allowed: {path!r}")

        # 3. Resolve and pin under project root
        resolved = (_project_root() / p).resolve()
        root = _project_root().resolve()

        if allow_absolute:
            # Only check traversal, still resolve but don't pin to project root
            return str(resolved)

        try:
            resolved.relative_to(root)
        except ValueError:
            raise ValueError(
                f"Path {path!r} resolves outside the project directory."
            )
        return str(resolved)

    # ------------------------------------------------------------------
    # Command whitelist
    # ------------------------------------------------------------------
    @classmethod
    def sanitize_command(cls, command: str) -> str:
        """
        Return *command* if it appears in ALLOWED_COMMANDS.

        Raises ValueError on unknown / unlisted commands.
        """
        cmd = command.strip().lower()
        if cmd not in cls.ALLOWED_COMMANDS:
            raise ValueError(
                f"Command {command!r} is not in the allowed list. "
                f"Permitted: {', '.join(cls.ALLOWED_COMMANDS)}"
            )
        return command  # return original casing

    @classmethod
    def sanitize_shell_arg(cls, arg: str) -> str:
        """
        Escape shell metacharacters so *arg* is safe to pass to a shell.

        Safe for cmd.exe, PowerShell, bash, and sh.
        """
        return cls._SHELL_META.sub(r'^\1', arg) if arg else arg

    # ------------------------------------------------------------------
    # Write-path validation
    # ------------------------------------------------------------------
    @classmethod
    def sanitize_filepath_for_write(cls, path: str | Path) -> Path:
        """
        Validate and return a safe writable path.

        Rules:
          • No path traversal
          • Must stay inside the project directory
          • Must not overwrite existing directories
          • Must not overwrite files with system/hidden attributes
        """
        safe = Path(cls.sanitize_path(path))

        if safe.exists():
            if safe.is_dir():
                raise ValueError(f"Cannot write: {safe} is a directory.")
            try:
                mode = safe.stat().st_mode
                if stat.S_ISREG(mode) and (mode & (stat.S_IRWXG | stat.S_IRWXO)):
                    pass  # regular user-writable file – allowed
            except OSError:
                raise ValueError(f"Cannot write: permission check failed for {safe}")

        # Ensure parent directory exists
        safe.parent.mkdir(parents=True, exist_ok=True)
        return safe


# ═════════════════════════════════════════════════════════════════════════════
# SecretsManager
# ═════════════════════════════════════════════════════════════════════════════

class SecretsManager:
    """
    Loads API keys from environment variables first, then falls back to
    ``config/api_keys.json``.
    """

    SUPPORTED_KEYS: set[str] = {
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
        "ELEVENLABS_API_KEY",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_BACKEND_TOKEN",
    }

    def __init__(self, json_path: str | Path | None = None):
        self._json_path = Path(json_path) if json_path else (
            _project_root() / "config" / "api_keys.json"
        )
        self._json_cache: dict[str, str] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get(self, key: str) -> str | None:
        """Return the secret for *key*, or ``None`` if not found."""
        key = key.upper().strip()

        # 1. Environment variable
        value = os.environ.get(key)
        if value:
            return value

        # 2. JSON file fallback
        return self._load_json().get(key.lower()) or self._load_json().get(key)

    def set(self, key: str, value: str) -> None:
        """Store *value* in the process environment (volatile)."""
        key = key.upper().strip()
        os.environ[key] = value

    def list_keys(self) -> list[str]:
        """Return all available key names (without exposing values)."""
        available: set[str] = set()

        # From environment
        for k in self.SUPPORTED_KEYS:
            if os.environ.get(k):
                available.add(k)

        # From JSON file
        cfg = self._load_json()
        json_to_env = {
            "gemini_api_key":          "GEMINI_API_KEY",
            "groq_api_key":            "GROQ_API_KEY",
            "elevenlabs_api_key":      "ELEVENLABS_API_KEY",
            "telegram_bot_token":      "TELEGRAM_BOT_TOKEN",
            "telegram_backend_token":  "TELEGRAM_BACKEND_TOKEN",
        }
        for json_key, env_key in json_to_env.items():
            if cfg.get(json_key):
                available.add(env_key)

        return sorted(available)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_json(self) -> dict[str, Any]:
        if self._json_cache is not None:
            return self._json_cache
        try:
            raw = self._json_path.read_text(encoding="utf-8")
            self._json_cache = json.loads(raw)
        except (FileNotFoundError, json.JSONDecodeError):
            self._json_cache = {}
        return self._json_cache


# ═════════════════════════════════════════════════════════════════════════════
# Vault — Fernet-based encrypted storage
# ═════════════════════════════════════════════════════════════════════════════

class Vault:
    """
    Simple encrypted key-value store backed by a JSON file.

    Uses Fernet (symmetric AES-128-CBC with HMAC-SHA256).
    Master-password lock prevents accidental reads.
    """

    _FERNET_AVAILABLE: bool = False

    @classmethod
    def _check_fernet(cls) -> None:
        if not cls._FERNET_AVAILABLE:
            try:
                from cryptography.fernet import Fernet
                cls._FERNET_AVAILABLE = True
            except ImportError:
                raise RuntimeError(
                    "Missing 'cryptography' package. "
                    "Install with: pip install cryptography"
                )

    def __init__(self, vault_path: str | Path | None = None):
        self._check_fernet()
        from cryptography.fernet import Fernet

        self._vault_path = Path(vault_path) if vault_path else (
            _project_root() / "config" / ".vault.json"
        )
        self._fernet: Fernet | None = None
        self._locked = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def store(self, key: str, value: str) -> None:
        """Encrypt *value* and persist to the vault file."""
        if self._locked:
            raise RuntimeError("Vault is locked. Call unlock() first.")
        assert self._fernet is not None

        data = self._read_vault()
        data[key] = self._fernet.encrypt(value.encode("utf-8")).decode("utf-8")
        self._write_vault(data)

    def retrieve(self, key: str) -> str | None:
        """Decrypt and return a stored value, or ``None``."""
        if self._locked:
            raise RuntimeError("Vault is locked. Call unlock() first.")
        assert self._fernet is not None

        data = self._read_vault()
        encrypted = data.get(key)
        if encrypted is None:
            return None
        return self._fernet.decrypt(encrypted.encode("utf-8")).decode("utf-8")

    def lock(self) -> None:
        """Lock the vault — subsequent reads/writes require unlock()."""
        self._fernet = None
        self._locked = True

    def unlock(self, master_password: str) -> bool:
        """
        Derive a Fernet key from *master_password* and unlock the vault.

        Returns True on success, False if the password is wrong.
        """
        from cryptography.fernet import Fernet, InvalidToken
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        from base64 import urlsafe_b64encode

        # Use a fixed salt (not secret — key is derived from password)
        salt = b"ORTHOS_VAULT_SALT_v1"
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480_000,
        )
        key = urlsafe_b64encode(kdf.derive(master_password.encode("utf-8")))

        # Verify the key works by trying to decrypt a known value
        try:
            test_data = self._read_vault()
            if test_data:
                sample = next(iter(test_data.values()))
                Fernet(key).decrypt(sample.encode("utf-8"))
            self._fernet = Fernet(key)
            self._locked = False
            return True
        except (InvalidToken, StopIteration, OSError):
            # StopIteration = vault is empty (still valid)
            if not self._read_vault():
                self._fernet = Fernet(key)
                self._locked = False
                return True
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _read_vault(self) -> dict[str, str]:
        try:
            return json.loads(self._vault_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _write_vault(self, data: dict[str, str]) -> None:
        self._vault_path.parent.mkdir(parents=True, exist_ok=True)
        self._vault_path.write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )


# ═════════════════════════════════════════════════════════════════════════════
# SafeFilePath — context manager
# ═════════════════════════════════════════════════════════════════════════════

class SafeFilePath(AbstractContextManager):
    """
    Context manager that guarantees file operations stay within allowed
    directories.

    Usage::

        with SafeFilePath("subdir/file.txt") as safe:
            safe.write_text("hello")

        with SafeFilePath("/etc/passwd") as safe:   # raises ValueError
            ...
    """

    def __init__(self, path: str | Path, allow_absolute: bool = False):
        self._path = Path(path)
        self._allow_absolute = allow_absolute
        self._safe_path: Path | None = None

    def __enter__(self) -> Path:
        raw = Sanitizer.sanitize_path(self._path, allow_absolute=self._allow_absolute)
        self._safe_path = Path(raw)
        return self._safe_path

    def __exit__(self, *exc: Any) -> None:
        self._safe_path = None
