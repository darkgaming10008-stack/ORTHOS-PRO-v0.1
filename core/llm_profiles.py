"""
LLM Endpoint Profiles for Orthos.

Lets the user save named LLM endpoint profiles (e.g. "local" = the machine's
own Ollama, "kaggle" = a remote Kaggle+Cloudflare-tunnel Ollama) and switch
between them with one click from the Settings UI — no manual config editing.

Storage lives inside the existing config/api_keys.json (no new files):

    {
      "llm_profiles": {
        "local":  {"llm_url": "http://localhost:11434",
                    "llm_model": "ministral-3:14b-cloud"},
        "kaggle": {"llm_url": "https://xxxx.trycloudflare.com",
                    "llm_model": "qwen3.8-27b-uncensored-mtp"}
      },
      "llm_active_profile": "kaggle"
    }

Resolution contract (see core/llm_client.py):
  - The active profile's llm_url / llm_model OVERRIDE the top-level
    config keys at provider-config time, so switching is live (no restart).
  - "local" is a special-cased profile: it always resolves to the built-in
    defaults (localhost:11434), so a stale saved URL can never break it.
  - If there is no active profile (or it is missing/corrupt), the top-level
    llm_url / llm_model are used exactly as before — full backward compat.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from urllib.parse import urlparse

import requests

from memory.config_manager import CONFIG_FILE

PROFILES_KEY = "llm_profiles"
ACTIVE_KEY = "llm_active_profile"
LOCAL_PROFILE = "local"

_HEALTH_TIMEOUT = 6
# /api/tags can be slow the very first time a huge model is being loaded;
# probe both endpoints with a short timeout and fall back to plain TCP reachability.
_MODEL_LIST_TIMEOUT = 25


# ── raw config access ────────────────────────────────────────────────────────
def _read_raw_config() -> dict:
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


def _patch_config(key: str, value) -> None:
    """Read-modify-write a single top-level key in api_keys.json."""
    cfg = _read_raw_config()
    cfg[key] = value
    try:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[LLM-Profiles] Failed to write config: {e}")


def get_profiles() -> dict[str, dict]:
    """Return {name: {"llm_url": ..., "llm_model": ...}} — never None."""
    raw = _read_raw_config().get(PROFILES_KEY)
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict] = {}
    for name, data in raw.items():
        if isinstance(data, dict) and (data.get("llm_url") or data.get("llm_model")):
            out[str(name)] = {
                "llm_url":   str(data.get("llm_url") or ""),
                "llm_model": str(data.get("llm_model") or ""),
            }
    return out


def get_active_profile() -> str:
    """Name of the active profile, or '' when no profile is active."""
    raw = _read_raw_config().get(ACTIVE_KEY)
    return str(raw) if isinstance(raw, str) and raw else ""


def is_profile_active() -> bool:
    return bool(get_active_profile()) and get_active_profile() in get_profiles()


def resolve_profile_url_model(default_url: str, default_model: str) -> tuple[str, str]:
    """Return the effective (llm_url, llm_model) honouring the active profile.

    The "local" profile is special-cased: it *always* resolves to the given
    defaults (which the caller derives from the built-in localhost default),
    so a stale saved value can never break local switching.
    """
    active = get_active_profile()
    profiles = get_profiles()
    if not active or active not in profiles:
        return default_url, default_model
    if active == LOCAL_PROFILE:
        return default_url, default_model
    p = profiles[active]
    url = p.get("llm_url") or default_url
    model = p.get("llm_model") or default_model
    return url.rstrip("/"), model


# ── mutations ────────────────────────────────────────────────────────────────
def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_ -]", "", str(name)).strip().lower().replace(" ", "_")


def save_profile(name: str, url: str, model: str) -> str:
    """Create/update a profile. Returns the normalized stored name."""
    clean = _normalize_name(name)
    if not clean:
        raise ValueError("Profile name cannot be empty")
    cfg = _read_raw_config()
    profiles = cfg.get(PROFILES_KEY) if isinstance(cfg.get(PROFILES_KEY), dict) else {}
    profiles = dict(profiles)
    profiles[clean] = {
        "llm_url":   str(url or "").strip().rstrip("/"),
        "llm_model": str(model or "").strip(),
    }
    _patch_config(PROFILES_KEY, profiles)
    return clean


def delete_profile(name: str) -> None:
    cfg = _read_raw_config()
    profiles = cfg.get(PROFILES_KEY)
    if not isinstance(profiles, dict) or name not in profiles:
        return
    profiles = dict(profiles)
    profiles.pop(name, None)
    _patch_config(PROFILES_KEY, profiles)
    if get_active_profile() == name:
        # Switch back to top-level (local) behaviour when the active profile
        # is removed.
        _patch_config(ACTIVE_KEY, "")


def switch_profile(name: str) -> bool:
    """Activate a profile. Empty name / 'local' deactivates profile overlay."""
    name = _normalize_name(name or "")
    if name and name != LOCAL_PROFILE and name not in get_profiles():
        return False
    _patch_config(ACTIVE_KEY, "" if name in ("", LOCAL_PROFILE) else name)
    return True


# ── network helpers ──────────────────────────────────────────────────────────
def test_endpoint(url: str, model: str = "") -> tuple[bool, str]:
    """Health-check an Ollama endpoint. Returns (ok, human message).

    Checks, in order: reachability of /api/tags, then (optionally) whether
    *model* is present in the server's model list.
    """
    url = str(url or "").strip().rstrip("/")
    if not url:
        return False, "No URL"
    if not re.match(r"^https?://", url):
        url = "http://" + url

    try:
        resp = requests.get(f"{url}/api/tags", timeout=_HEALTH_TIMEOUT)
    except requests.exceptions.SSLError:
        return False, "SSL error — tunnel/certificate problem?"
    except requests.exceptions.ConnectionError:
        return _probe_tcp(url)
    except requests.exceptions.Timeout:
        return False, "Timed out — server busy or slow tunnel"
    except Exception as e:  # noqa: BLE001 — report anything unexpected
        return False, f"Error: {e}"

    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code} from /api/tags"

    model = (model or "").strip()
    if model:
        try:
            names = {m.get("name", "") for m in resp.json().get("models", [])}
        except Exception:
            names = set()
        base = model.split(":")[0]
        if not any(n == model or n.startswith(base + ":") for n in names):
            return True, f"Reachable, but model '{model}' not pulled on server"

    return True, "OK"


def _probe_tcp(url: str) -> tuple[bool, str]:
    """ConnectionError fallback: is the host:port at least open?"""
    try:
        parsed = urlparse(url)
        host, port = parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
        import socket
        socket.create_connection((host, port), timeout=3).close()
        return True, "Port open, HTTP failed — server starting?"
    except Exception:
        return False, "Unreachable — is the tunnel/Kaggle session running?"


def test_profile(name: str) -> tuple[bool, str]:
    """Health-check a stored profile by name ('' = current local default)."""
    if name == LOCAL_PROFILE or not name:
        from core.llm_client import get_llm_settings
        url, model = get_llm_settings()
        return test_endpoint(url, model)

    profiles = get_profiles()
    p = profiles.get(name)
    if not p:
        return False, f"Profile '{name}' not found"
    return test_endpoint(p.get("llm_url", ""), p.get("llm_model", ""))


# ── remote detection ─────────────────────────────────────────────────────────
def _is_remote_url(url: str) -> bool:
    """True when the URL is NOT this machine (i.e. don't auto-launch ollama)."""
    try:
        host = (urlparse(str(url or "")).hostname or "").lower()
    except Exception:
        return False
    return host not in ("localhost", "127.0.0.1", "::1", "", "0.0.0.0")


def launch_local_ollama() -> bool:
    """Start `ollama serve` if installed and not already up. Best-effort."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        if resp.status_code == 200:
            return True  # already running
    except Exception:
        pass
    if not shutil.which("ollama"):
        return False
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return False
    deadline = time.time() + 15
    while time.time() < deadline:
        time.sleep(1.0)
        try:
            if requests.get("http://localhost:11434/api/tags", timeout=2).status_code == 200:
                return True
        except Exception:
            continue
    return False
