"""
Unified LLM client for Orthos.

Supports six backends — selected via "llm_provider" in config/api_keys.json:

  "llm_provider": "ollama"   (default)
        Uses Ollama's native /api/chat endpoint.
        Download: https://ollama.com
        Default port: 11434

  "llm_provider": "openai"
        Uses any OpenAI-compatible server: LM Studio, Jan, LocalAI,
        llama.cpp server, vLLM, etc.

  "llm_provider": "openrouter"
        Uses OpenRouter API (cloud).
        https://openrouter.ai

  "llm_provider": "cloudflare"
        Uses Cloudflare Workers AI (cloud).
        https://developers.cloudflare.com/workers-ai/

  "llm_provider": "groq"
        Uses Groq LPU API (cloud).
        https://console.groq.com

  "llm_provider": "gemini"
        Uses Google Gemini API (cloud) via native generateContent endpoint.
        https://ai.google.dev/gemini-api/docs

  "llm_provider": "nvidia"
        Uses NVIDIA NIM API (cloud).
        https://build.nvidia.com/explore/discover

  "llm_provider": "kilo"
        Uses Kilo Code Gateway API (cloud, free kilo-auto/free).
        https://kilo.ai/docs/gateway

  "llm_provider": "kaggle"
        Uses a remote Ollama running on a free Kaggle GPU session
        (2x T4), reached over a private tunnel (Tailscale/Cloudflare).
        Config keys: kaggle_url (tunnel base URL), kaggle_model.
        Same native Ollama /api/chat protocol as the "ollama" provider.

  "llm_provider": "lmstudio"  (alias -> "openai")
  "llm_provider": "localai"   (alias -> "openai")
  "llm_provider": "jan"       (alias -> "openai")
  "llm_provider": "llamacpp"  (alias -> "openai")
"""
import ast
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Generator

import requests

from core.llm_provider import _convert_to_openai_tools

_HTTP = requests.Session()
VERBOSE = os.environ.get("LLM_VERBOSE", "0") == "1"
_DUMP_PAYLOAD = os.environ.get("LLM_DUMP_PAYLOAD", "0") == "1"

def _log_v(fmt: str, *args: object) -> None:
    if VERBOSE:
        print(f"[LLM-V] {fmt.format(*args)}")


def _http_error_body(e) -> str:
    """Best-effort extraction of the full response body from an HTTPError."""
    body = ""
    try:
        raw = e.response.content or b""
        body = raw[:2000].decode("utf-8", errors="replace")
    except Exception:
        pass
    if not body.strip():
        try:
            body = e.response.text[:2000]
        except Exception:
            body = ""
    return body if body.strip() else "(empty response body)"

def _dump_ollama_payload(payload: dict) -> None:
    if _DUMP_PAYLOAD:
        print("[LLM-DUMP] === OLLAMA PAYLOAD (full) ===")
        print(json.dumps(payload, indent=2, default=str))
        print("[LLM-DUMP] === END ===")

def _mask_key(text: str) -> str:
    if len(text) > 8:
        return text[:4] + "****" + text[-4:]
    return "****"

_SENT_END = re.compile(r'(?<=[.!?])\s+|(?<=\n)\s*\n')

# Voice mode: when a streaming TTS engine is active, long unbroken runs are
# also split at commas/semicolons/colons so playback starts after ~1 s.
_VOICE_MODE = False


def set_voice_mode(enabled: bool) -> None:
    """Enable clause-level splitting for streaming TTS (Gemini voice)."""
    global _VOICE_MODE
    _VOICE_MODE = bool(enabled)


def _next_utterance(buf: str, min_len: int = 80) -> tuple[str, str] | None:
    """Find the next voice-friendly utterance boundary.

    Sentences (. ! ? or blank lines) always split. In voice mode, long
    unbroken runs also split at , ; : so TTS playback can start early.
    """
    m = _SENT_END.search(buf)
    if m:
        return buf[: m.start() + 1].strip(), buf[m.end():]
    if _VOICE_MODE and len(buf) >= min_len:
        cm = re.search(r'(?<=[,;:])\s', buf)
        if cm:
            return buf[: cm.start() + 1].strip(), buf[cm.end():]
    return None

_PROVIDER_ALIASES = {
    "lmstudio": "openai", "localai": "openai",
    "jan": "openai", "llamacpp": "openai",
}


def _ensure_ollama_messages(messages: list) -> None:
    """Ensure messages are in Ollama native /api/chat format.

    Ollama's native API expects:
      - content is always a plain string (NOT an array)
      - images are a separate field on the message (not inline in content)
      - tool_calls have NO id/type fields (Ollama ignores them but they bloat the payload)
      - tool result messages (role=tool) use tool_name field, not tool_call_id
    """
    _pending_tool_names: list[str] = []
    for msg in messages:
        # ── content: flatten array back to string ──
        content = msg.get("content")
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    text_parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    text_parts.append(part)
            msg["content"] = " ".join(tp for tp in text_parts if tp).strip() or ""
        elif content is None:
            msg["content"] = ""

        # ── images: keep as separate list field (Ollama native API supports this) ──
        images = msg.get("images")
        if images and isinstance(content, list) and all(isinstance(i, str) for i in images):
            pass

        # ── assistant: queue tool call names for pairing with tool results ──
        if msg.get("role") == "assistant":
            for _tc in msg.get("tool_calls") or []:
                _fn = _tc.get("function") if isinstance(_tc, dict) else None
                if isinstance(_fn, dict) and _fn.get("name"):
                    _pending_tool_names.append(_fn["name"])

        # ── tool results: Ollama native format is {role, tool_name, content} only ──
        # (extra keys like tool_result/tool_calls/tool_call_id make Ollama Cloud's
        #  backend return 500 Internal Server Error on image-bearing payloads)
        if msg.get("role") == "tool":
            _tname = msg.get("tool_name") or ""
            if not _tname and _pending_tool_names:
                _tname = _pending_tool_names.pop(0)
            if not _tname and msg.get("tool_call_id"):
                _tname = msg["tool_call_id"]
            msg["tool_name"] = _tname or "tool_result"
            msg.pop("tool_call_id", None)
            msg.pop("tool_result", None)
            msg.pop("tool_calls", None)
            msg.pop("timestamp", None)
            msg.pop("archived", None)
            msg.pop("branch_id", None)
            continue

        # ── tool_call_id: only valid on role=tool messages ──
        if msg.get("role") != "tool":
            msg.pop("tool_call_id", None)

        # ── tool_calls: strip OpenAI-specific fields ──
        for tc in msg.get("tool_calls") or []:
            tc.pop("id", None)
            tc.pop("type", None)
            fn = tc.get("function")
            if not isinstance(fn, dict):
                continue
            args = fn.get("arguments")
            if isinstance(args, str):
                parsed = _parse_args_to_dict(args)
                if parsed is not None:
                    fn["arguments"] = parsed
                else:
                    print(f"[LLM] WARNING: Could not parse arguments for Ollama: {args[:200]}")
            fn.pop("index", None)

def _parse_args_to_dict(args: str) -> dict | None:
    """Try multiple strategies to parse a string into a dict."""
    s = args.strip()

    # Strategy 1: Valid JSON
    try:
        result = json.loads(s)
        if isinstance(result, dict):
            return result
        # Handle double-encoded JSON: '{"key": "value"}' -> {"key": "value"}
        if isinstance(result, str):
            try:
                inner = json.loads(result)
                if isinstance(inner, dict):
                    return inner
            except Exception:
                pass
    except Exception:
        pass

    # Strategy 2: ast.literal_eval (handles Python-style single-quoted dicts)
    try:
        result = ast.literal_eval(s)
        if isinstance(result, dict):
            return result
    except Exception:
        pass

    # Strategy 3: Replace single quotes with double quotes (careful, handle escaped quotes)
    try:
        # Only replace quotes that are likely JSON delimiters, not inside strings
        # This is a best-effort approach
        fixed = s.replace("'", '"')
        result = json.loads(fixed)
        if isinstance(result, dict):
            return result
    except Exception:
        pass

    # Strategy 4: Handle cases like "{'key': 'value'}" (wrapped in outer quotes)
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        try:
            inner = s[1:-1]
            result = json.loads(inner)
            if isinstance(result, dict):
                return result
        except Exception:
            pass

    # Strategy 5: Last-resort fixes for truncated/malformed JSON
    if s.startswith("{"):
        # 5a: Try adding a missing closing brace (truncated JSON edge case)
        if not s.endswith("}"):
            try:
                result = json.loads(s + "}")
                if isinstance(result, dict):
                    return result
            except Exception:
                pass
        # 5b: Wrap the entire raw string in a dict so Ollama at least gets {}
        return {"raw_arguments": s}

    return None


def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR    = get_base_dir()
CONFIG_PATH = BASE_DIR / "config" / "api_keys.json"

_DEFAULTS = {
    "llm_url":           "http://localhost:11434",
    "llm_model":         "ministral-3:14b-cloud",
    "llm_provider":      "ollama",
    "or_model":          "openrouter/free",
    "cf_model":          "@cf/google/gemma-4-26b-a4b-it",
    "groq_llm_model":    "meta-llama/llama-4-scout-17b-16e-instruct",
    "gemini_model":      "gemini-2.5-flash",  # stable, vision, free tier 1,500 RPD
    "nvidia_model":      "qwen/qwen3.5-397b-a17b",
    "kilo_model":        "kilo-auto/free",
    "kaggle_model":      "qwen3.8-27b-uncensored-mtp",
}


def _load_config() -> dict:
    try:
        raw = CONFIG_PATH.read_bytes()
        if raw.startswith(b"\xef\xbb\xbf"):      # tolerate Windows PowerShell BOM
            raw = raw[3:]
            try:
                CONFIG_PATH.write_bytes(raw)
            except Exception:
                pass
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return {}


def get_llm_provider() -> str:
    raw = _load_config().get("llm_provider", "ollama").strip().lower()
    return _PROVIDER_ALIASES.get(raw, raw)


def _get_provider_config() -> dict:
    cfg      = _load_config()
    provider = get_llm_provider()

    _log_v("Provider={} model={}", provider, cfg.get("llm_model", "") or cfg.get(f"{provider}_model", ""))

    if provider == "kaggle":
        url   = cfg.get("kaggle_url", "http://localhost:11434").rstrip("/")
        model = cfg.get("kaggle_model", _DEFAULTS["kaggle_model"])
        return {
            "chat_endpoint": f"{url}/api/chat",
            "headers":       {},
            "model":         model,
            "provider":      provider,
            "base_url":      url,
            "health_url":    f"{url}/api/tags",
        }

    if provider == "ollama":
        url   = cfg.get("llm_url", _DEFAULTS["llm_url"]).rstrip("/")
        model = cfg.get("llm_model", _DEFAULTS["llm_model"])
        # Active endpoint profile ("local" <-> "kaggle" tunnel etc.) overrides
        # the top-level llm_url/llm_model.  Config is re-read on every call,
        # so switching profiles is live — no restart needed.
        try:
            from core.llm_profiles import resolve_profile_url_model
            url, model = resolve_profile_url_model(url, model)
        except Exception:
            pass
        return {
            "chat_endpoint": f"{url}/api/chat",
            "headers":       {},
            "model":         model,
            "provider":      provider,
            "base_url":      url,
            "health_url":    f"{url}/api/tags",
        }

    if provider == "openai":
        url   = cfg.get("llm_url", "http://localhost:1234").rstrip("/")
        model = cfg.get("llm_model", "")
        return {
            "chat_endpoint": f"{url}/v1/chat/completions",
            "headers":       {},
            "model":         model,
            "provider":      provider,
            "base_url":      url,
            "health_url":    f"{url}/v1/models",
        }

    if provider == "openrouter":
        key   = cfg.get("or_api_key", "")
        model = cfg.get("or_model", _DEFAULTS["or_model"])
        base  = "https://openrouter.ai/api/v1"
        return {
            "chat_endpoint": f"{base}/chat/completions",
            "headers":       {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            "model":         model,
            "provider":      provider,
            "base_url":      base,
            "health_url":    f"{base}/models",
        }

    if provider == "cloudflare":
        token = cfg.get("cf_api_token", "")
        aid   = cfg.get("cf_account_id", "")
        model = cfg.get("cf_model", _DEFAULTS["cf_model"])
        base  = f"https://api.cloudflare.com/client/v4/accounts/{aid}/ai/v1"
        return {
            "chat_endpoint": f"{base}/chat/completions",
            "headers":       {"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            "model":         model,
            "provider":      provider,
            "base_url":      base,
            "health_url":    f"{base}/models",
        }

    if provider == "groq":
        key   = cfg.get("groq_llm_key", "")
        model = cfg.get("groq_llm_model", _DEFAULTS["groq_llm_model"])
        base  = "https://api.groq.com/openai/v1"
        return {
            "chat_endpoint": f"{base}/chat/completions",
            "headers":       {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            "model":         model,
            "provider":      provider,
            "base_url":      base,
            "health_url":    f"{base}/models",
        }

    if provider == "gemini":
        key   = cfg.get("gemini_api_key", "")
        model = cfg.get("gemini_model", _DEFAULTS["gemini_model"])
        base  = "https://generativelanguage.googleapis.com/v1beta"
        return {
            "chat_endpoint": f"{base}/models/{model}:generateContent",
            "headers":       {"x-goog-api-key": key, "Content-Type": "application/json"},
            "model":         model,
            "provider":      provider,
            "base_url":      base,
            "health_url":    f"{base}/models",
        }

    if provider == "gemini_live":
        key   = cfg.get("gemini_api_key", "")
        model = cfg.get("gemini_live_model", "gemini-2.5-flash-native-audio-preview-12-2025")
        return {
            "chat_endpoint": "",
            "headers":       {},
            "model":         model,
            "provider":      provider,
            "base_url":      "",
            "health_url":    "",
            "api_key":       key,
        }

    if provider == "nvidia":
        key   = cfg.get("nvidia_api_key", "")
        model = cfg.get("nvidia_model", _DEFAULTS["nvidia_model"])
        base  = "https://integrate.api.nvidia.com/v1"
        return {
            "chat_endpoint": f"{base}/chat/completions",
            "headers":       {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            "model":         model,
            "provider":      provider,
            "base_url":      base,
            "health_url":    f"{base}/models",
        }

    if provider == "kilo":
        key   = cfg.get("kilo_api_key", "")
        model = cfg.get("kilo_model", _DEFAULTS["kilo_model"])
        base  = "https://api.kilo.ai/api/gateway"
        return {
            "chat_endpoint": f"{base}/chat/completions",
            "headers":       {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            "model":         model,
            "provider":      provider,
            "base_url":      base,
            "health_url":    f"{base}/models",
        }

    raise RuntimeError(f"Unknown LLM provider: {provider}")


def get_llm_settings() -> tuple[str, str]:
    cfg = _get_provider_config()
    return cfg["base_url"], cfg["model"]


def ensure_ollama_running(timeout: int = 15) -> bool:
    cfg = _get_provider_config()
    provider = cfg["provider"]

    if provider in ("ollama", "kaggle"):
        health = cfg["health_url"]
        def _is_up() -> bool:
            try:
                return _HTTP.get(health, timeout=3).status_code == 200
            except Exception:
                return False
        if _is_up():
            return True
        # Remote endpoints: never try to launch a local ollama — just report
        # reachability with a helpful message.
        try:
            from core.llm_profiles import _is_remote_url
            if provider == "kaggle" or _is_remote_url(cfg.get("base_url", "")):
                print("[LLM] Remote LLM endpoint unreachable — is the tunnel/Kaggle session running?")
                return False
        except Exception:
            pass
        print("[LLM] Ollama not running — launching 'ollama serve'…")
        try:
            kwargs: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            subprocess.Popen(["ollama", "serve"], **kwargs)
        except FileNotFoundError:
            print("[LLM] 'ollama' command not found. Install Ollama from https://ollama.com")
            return False
        except Exception as e:
            print(f"[LLM] Could not launch Ollama: {e}")
            return False
        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(1.0)
            if _is_up():
                print("[LLM] Ollama started successfully.")
                return True
        print("[LLM] Ollama did not respond within the timeout.")
        return False

    if provider in ("gemini", "gemini_live", "cloudflare"):
        try:
            ok = _HTTP.get(cfg["health_url"], headers=cfg["headers"], timeout=5).status_code == 200
            if ok:
                print(f"[LLM] {provider} reachable at {cfg['base_url']}")
            return ok
        except Exception:
            return True

    try:
        ok = _HTTP.get(cfg["health_url"], headers=cfg["headers"], timeout=5).status_code == 200
        if ok:
            print(f"[LLM] {provider} reachable at {cfg['base_url']}")
        else:
            print(f"[LLM] {provider} at {cfg['base_url']} returned non-200.")
        return ok
    except Exception as e:
        print(f"[LLM] Cannot reach {provider} at {cfg['base_url']}: {e}")
        return False


def warmup_model(system_prompt: str | None = None) -> bool:
    cfg = _get_provider_config()
    provider = cfg["provider"]
    model = cfg["model"]
    print(f"[LLM] Warming up '{model}' ({provider})…")

    # Gemini Live uses a persistent WebSocket session — use the shared
    # registry instance so main.py callbacks stay wired to the same session.
    if provider == "gemini_live":
        from core.llm_provider import get_registry
        p = get_registry().get_provider(
            "gemini_live",
            model=model,
            api_key=cfg.get("api_key", ""),
        )
        p.set_system_prompt(system_prompt)
        return p.warmup(system_prompt)

    if provider == "groq":
        from core.llm_provider import get_registry
        p = get_registry().get_provider("groq", model=model)
        return p.warmup(system_prompt)

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": "hi"})

    payload = _build_payload(messages, tools=None, stream=False, model=model, provider=provider)
    if provider == "ollama":
        payload["keep_alive"] = -1
        payload["options"]    = payload.get("options", {})
        payload["options"]["num_predict"] = 1

    try:
        resp = _HTTP.post(cfg["chat_endpoint"], headers=cfg["headers"], json=payload, timeout=180)
        resp.raise_for_status()
        print(f"[LLM] '{model}' ready ({provider}).")
        return True
    except Exception as e:
        print(f"[LLM] Warmup failed (non-fatal): {e}")
        return False


def _convert_messages_for_openai(messages: list) -> list:
    out = []
    _img_count = 0
    for msg in messages:
        msg = dict(msg)
        msg.pop("image_files", None)
        if msg.get("tool_call_id") is None:
            msg.pop("tool_call_id", None)
        images = msg.pop("images", None)
        if images:
            content: list[dict] = []
            text = msg.get("content") or ""
            if text:
                content.append({"type": "text", "text": text})
            for b64 in images:
                if _img_count >= 10:
                    break
                _img_count += 1
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
            if content:
                msg["content"] = content
        if msg.get("tool_calls"):
            new_tc = []
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                args = fn.get("arguments")
                new_fn = dict(fn)
                if isinstance(args, dict):
                    new_fn["arguments"] = json.dumps(args)
                elif isinstance(args, str):
                    parsed = _parse_args_to_dict(args)
                    if parsed is not None:
                        new_fn["arguments"] = json.dumps(parsed)
                new_tc.append({
                    "id": tc.get("id", ""),
                    "type": tc.get("type") if tc.get("type") else "function",
                    "function": new_fn,
                })
            msg["tool_calls"] = new_tc
        out.append(msg)
    return out


def _prepare_cloudflare_messages(messages: list) -> list:
    out: list[dict] = []
    _img_count = 0
    for msg in messages:
        msg = dict(msg)
        msg.pop("image_files", None)
        if msg.get("tool_call_id") is None:
            msg.pop("tool_call_id", None)
        images = msg.pop("images", None)
        if images:
            content: list[dict] = []
            text = msg.get("content") or ""
            if text:
                content.append({"type": "text", "text": text})
            for b64 in images[:10]:
                _img_count += 1
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
            if content:
                msg["content"] = content
        if msg.get("tool_calls"):
            new_tc = []
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                args = fn.get("arguments")
                new_fn = dict(fn)
                if isinstance(args, dict):
                    new_fn["arguments"] = json.dumps(args)
                elif isinstance(args, str):
                    parsed = _parse_args_to_dict(args)
                    if parsed is not None:
                        new_fn["arguments"] = json.dumps(parsed)
                new_tc.append({
                    "id": tc.get("id", ""),
                    "type": tc.get("type") if tc.get("type") else "function",
                    "function": new_fn,
                })
            msg["tool_calls"] = new_tc
        out.append(msg)

    # Cloudflare/Mistral strictly enforces: tool → assistant → user sequence.
    # If a tool message is followed by user, insert a placeholder assistant response.
    fixed: list[dict] = []
    for i, msg in enumerate(out):
        fixed.append(msg)
        if msg.get("role") == "tool" and i + 1 < len(out) and out[i + 1].get("role") == "user":
            fixed.append({"role": "assistant", "content": "."})
    return fixed


def _build_payload(
    messages: list,
    tools: list | None = None,
    stream: bool = False,
    model: str = "",
    provider: str = "",
) -> dict:
    if provider == "gemini":
        return _build_gemini_payload(messages, tools, stream, model, provider)

    payload: dict = {
        "model":    model,
        "messages": messages,
        "stream":   stream,
    }

    # Dynamic max_tokens: 25% of ctx window (clamped to 32k max, 1024 min)
    try:
        from memory.conversation_db import get_model_context_window
        _ctx = get_model_context_window(model, provider)
        _output_budget = int(_ctx * 0.25)
        max_tokens = min(32000, max(1024, _output_budget))
    except Exception:
        max_tokens = 32000

    if provider in ("ollama", "kaggle"):
        _ensure_ollama_messages(messages)
        payload["keep_alive"] = -1
        payload["options"]    = {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "num_gpu": 99}
        if tools:
            payload["tools"] = tools
        payload["max_tokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens
        if tools and provider != "kaggle":
            tools = _select_tools_for_provider(tools, messages, provider)
            payload["tools"]       = _convert_to_openai_tools(tools)
            payload["tool_choice"] = "auto"

    return payload


# Per-provider tool quantity limits for OpenAI-compatible APIs.
# Providers not listed here send all tools without truncation.
_TOOL_LIMITS: dict[str, int] = {
    "openai": 128,
    "groq":   128,
}

# Tool names ALWAYS included when dynamic selection is active. These cover
# memory, timeline, projects, and system operations so the LLM can always
# read/write its core state regardless of what the user asks.
_TOOL_ESSENTIAL_NAMES: set[str] = {
    "save_memory", "search_memory", "recall_conversation", "list_memories", "forget_memory",
    "core_memory_append", "core_memory_replace", "archival_memory_search",
    "context_status", "save_procedure", "list_procedures",
    "search_timeline", "save_timeline_event",
    "save_tool_state", "get_tool_state",
    "search_knowledge_graph",
    "list_mcp_servers", "shutdown_orthos",
    "list_projects", "create_project", "set_active_project",
    "get_project_memories",
}


def _tool_name(t: dict) -> str:
    """Extract the tool name from either Ollama or OpenAI format."""
    fn = t.get("function")
    if isinstance(fn, dict):
        return fn.get("name", "")
    return t.get("name", "")


def _select_tools_for_provider(
    tools: list,
    messages: list,
    provider: str,
) -> list:
    """Dynamically select relevant tools by analyzing the last user message.

    For providers listed in * _TOOL_LIMITS (groq, openai) where a hard cap
    (128) exists, this function extracts keywords from the last user message,
    scores each tool by how many keyword tokens appear in its name, and
    returns:

      1. Essential system tools (always).
      2. Best-matching tools, highest score first.
      3. Remaining slots filled with unused tools (legacy before MCP).

    Providers without a limit receive *all* tools unchanged, so only groq
    and openai are affected.
    """
    limit = _TOOL_LIMITS.get(provider)
    if not limit or not tools or len(tools) <= limit:
        return tools

    # ---- extract keywords from the last user message ----
    user_text = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                user_text = content
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        user_text = part.get("text", "")
                        break
            break

    keywords: set[str] = set(re.findall(r"[a-zA-Z0-9_]+", user_text.lower()))

    # ---- split into essential vs. everything else ----
    essential: list[dict] = []
    others: list[tuple[str, dict, int]] = []          # (name, tool, score)

    for t in tools:
        # Supports both Ollama format ({"name": ..., ...}) and OpenAI format
        # ({"type": "function", "function": {"name": ..., ...}}).
        name = _tool_name(t)
        if name in _TOOL_ESSENTIAL_NAMES:
            essential.append(t)
        else:
            parts = name.lower().split("_")
            score = sum(1 for kw in keywords if kw in parts)
            others.append((name, t, score))

    # Sort: highest score first → non-MCP before MCP → alphabetical
    others.sort(key=lambda x: (-x[2], x[0].startswith("mcp_"), x[0]))

    # ---- build the final tool list ----
    result: list[dict] = list(essential)
    slots = limit - len(result)

    for name, t, _ in others:
        if slots <= 0:
            break
        result.append(t)
        slots -= 1

    orig = len(tools)
    if len(result) < orig:
        print(f"[LLM] Dynamic selection: {orig} -> {len(result)} tools ({provider})")

    return result


def _normalise_tool_calls(raw_tc: list) -> list:
    tc_list = []
    for t in raw_tc:
        args_raw = t["function"].get("arguments", {})
        if isinstance(args_raw, str):
            try:
                args = json.loads(args_raw)
            except Exception:
                parsed = _parse_args_to_dict(args_raw)
                args = parsed if parsed is not None else args_raw
        else:
            args = args_raw
        tc_list.append({
            "id":       t.get("id", ""),
            "type":     t.get("type") or "function",
            "function": {
                "name":      t["function"]["name"],
                "arguments": args,
            },
        })
    return tc_list


def _sanitize_gemini_schema(schema: dict) -> dict:
    """Strip JSON Schema features that Gemini's API doesn't accept.

    Keeps only: type (string, never array), properties, items, required,
    description, enum, default, nullable.
    """
    ALLOWED = {"type", "properties", "items", "required", "description",
               "enum", "default", "nullable", "prefixItems"}
    clean = {}
    for k, v in schema.items():
        if k not in ALLOWED:
            continue
        if k == "type":
            if isinstance(v, list):
                non_null = [t for t in v if t != "null"]
                clean["type"] = non_null[0] if non_null else "string"
                if "null" in v:
                    clean["nullable"] = True
            else:
                clean["type"] = v
        elif k in ("properties", "prefixItems"):
            if isinstance(v, dict):
                clean[k] = {sk: _sanitize_gemini_schema(sv)
                            for sk, sv in v.items() if isinstance(sv, dict)}
            elif isinstance(v, list):
                clean[k] = [_sanitize_gemini_schema(item)
                            for item in v if isinstance(item, dict)]
        elif k == "items":
            if isinstance(v, dict):
                clean["items"] = _sanitize_gemini_schema(v)
            elif isinstance(v, list):
                clean["prefixItems"] = [_sanitize_gemini_schema(item)
                                        for item in v if isinstance(item, dict)]
        elif k == "required" and isinstance(v, list):
            clean["required"] = v
        elif k in ("description", "default"):
            clean[k] = v
        elif k == "enum" and isinstance(v, list):
            clean["enum"] = [str(e) for e in v]
        elif k == "nullable" and isinstance(v, bool):
            clean["nullable"] = v
    return clean


def _lookup_last_fc_name(contents: list[dict]) -> str:
    for c in reversed(contents):
        for p in c.get("parts", []):
            fc = p.get("functionCall")
            if fc:
                return fc.get("name", "")
    return ""


def _build_gemini_payload(
    messages: list,
    tools: list | None = None,
    stream: bool = False,
    model: str = "",
    provider: str = "gemini",
) -> dict:
    contents: list[dict] = []
    system_parts: list[dict] = []
    _pending_fc_as_text: int = 0
    _img_count = 0

    for msg in messages:
        role = msg.get("role", "")
        parts: list[dict] = []
        images = msg.pop("images", None)

        if role == "system":
            text = msg.get("content", "")
            if text:
                system_parts.append({"text": text})
            continue

        if role == "assistant":
            gemini_role = "model"
            text = msg.get("content", "")
            if text:
                parts.append({"text": text})
            tc = msg.get("tool_calls")
            _had_any: bool = False
            _had_valid: bool = False
            if tc:
                _had_any = True
                _foreign_count = 0
                for call in tc:
                    fn = call.get("function", {})
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            pass
                    if "thoughtSignature" in call:
                        p_part = {"functionCall": {"name": fn.get("name", ""), "args": args}}
                        p_part["thoughtSignature"] = call["thoughtSignature"]
                        parts.append(p_part)
                        _had_valid = True
                    else:
                        _foreign_count += 1
                        parts.append({"text": f"[Called tool: {fn.get('name', '')}({json.dumps(args)[:200]})]"})
            _pending_fc_as_text = _foreign_count if (_had_any and not _had_valid) else 0

        elif role == "tool":
            gemini_role = "user"
            raw = msg.get("content", "{}")
            if isinstance(raw, str):
                try:
                    resp_data = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    resp_data = {"result": raw}
            else:
                resp_data = raw
            name = msg.get("name", "")
            if not name and msg.get("tool_call_id"):
                name = msg["tool_call_id"]
            if _pending_fc_as_text > 0:
                parts.append({"text": f"[Tool result: {json.dumps(resp_data)[:1000]}]"})
                _pending_fc_as_text -= 1
            else:
                if not name:
                    name = _lookup_last_fc_name(contents)
                parts.append({
                    "functionResponse": {
                        "name": name,
                        "response": resp_data,
                    }
                })

        else:
            gemini_role = "user"
            content = msg.get("content", "")
            if isinstance(content, str):
                if content:
                    parts.append({"text": content})
            elif isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        parts.append({"text": item["text"]})
                    elif item.get("type") == "image_url":
                        url = item["image_url"]["url"]
                        if url.startswith("data:"):
                            meta, b64 = url.split(",", 1)
                            mime = meta.split(":")[1].split(";")[0]
                            parts.append({"inlineData": {"mimeType": mime, "data": b64}})
            if images:
                for b64 in images:
                    if _img_count >= 10:
                        break
                    _img_count += 1
                    parts.append({"inlineData": {"mimeType": "image/jpeg", "data": b64}})
            _pending_fc_as_text = 0  # user message resets
            _pending_fc_as_text = None  # user message resets the flag

        if parts:
            contents.append({"role": gemini_role, "parts": parts})

    payload: dict = {"contents": contents}
    if system_parts:
        payload["systemInstruction"] = {"parts": system_parts}
    if tools:
        decls = []
        for t in tools:
            fd = dict(t["function"])
            if "parameters" in fd:
                fd["parameters"] = _sanitize_gemini_schema(fd["parameters"])
            decls.append(fd)
        payload["tools"] = [{"functionDeclarations": decls}]
    if not stream:
        try:
            from memory.conversation_db import get_model_context_window
            _ctx = get_model_context_window(model, provider)
            _output_budget = int(_ctx * 0.25)
            max_out = min(65536, max(32000, _output_budget))
        except Exception:
            max_out = 32000
        payload.setdefault("generationConfig", {})["maxOutputTokens"] = max_out
    return payload


def _parse_gemini_response(data: dict) -> dict:
    cand = data.get("candidates", [{}])[0]
    if not cand:
        return {"content": "", "tool_calls": []}
    parts = (cand.get("content") or {}).get("parts") or []
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for p in parts:
        if "text" in p:
            text_parts.append(p["text"])
        if "functionCall" in p:
            fc = p["functionCall"]
            tc = {
                "id": fc.get("name", ""),
                "function": {
                    "name": fc.get("name", ""),
                    "arguments": fc.get("args", {}),
                },
            }
            if "thoughtSignature" in p:
                tc["thoughtSignature"] = p["thoughtSignature"]
            tool_calls.append(tc)
    content = "".join(text_parts).strip()
    return {"content": content, "tool_calls": tool_calls}


def _parse_chat_response(data: dict, provider: str) -> dict:
    if provider in ("ollama", "kaggle"):
        msg = data.get("message", {})
        return {
            "content":    (msg.get("content") or "").strip(),
            "thinking":   (msg.get("thinking") or "").strip(),
            "tool_calls": msg.get("tool_calls") or [],
        }

    if provider == "gemini":
        return _parse_gemini_response(data)

    choice = data.get("choices", [{}])[0]
    msg    = choice.get("message", {})
    raw_tc = msg.get("tool_calls") or []
    return {
        "content":    (msg.get("content") or "").strip(),
        "tool_calls": _normalise_tool_calls(raw_tc),
    }


def call_llm(
    messages: list,
    tools:    list | None = None,
    timeout:  int = 3600,
) -> dict:
    cfg = _get_provider_config()

    if cfg["provider"] == "gemini_live":
        from core.llm_provider import get_registry
        provider = get_registry().get_provider(
            "gemini_live",
            api_key=cfg.get("api_key", ""),
            model=cfg.get("model", "gemini-2.5-flash-native-audio-preview-12-2025"),
        )
        return provider.chat(messages, tools, timeout)

    if cfg["provider"] == "groq":
        from core.llm_provider import get_registry
        provider = get_registry().get_provider(
            "groq",
            api_key=cfg.get("groq_llm_key"),
            model=cfg.get("groq_llm_model"),
        )
        return provider.chat(messages, tools, timeout)

    if cfg["provider"] == "cloudflare":
        messages = _prepare_cloudflare_messages(messages)
    elif cfg["provider"] not in ("ollama", "kaggle", "gemini"):
        messages = _convert_messages_for_openai(messages)
    if cfg["provider"] in ("ollama", "kaggle"):
        _ensure_ollama_messages(messages)
    payload = _build_payload(messages, tools, stream=False, model=cfg["model"], provider=cfg["provider"])

    _log_v("POST {} [{}] model={}", cfg["chat_endpoint"], cfg["provider"], cfg["model"])
    if VERBOSE:
        safe = {k: v for k, v in cfg["headers"].items()}
        if "Authorization" in safe:
            safe["Authorization"] = "Bearer " + _mask_key(safe["Authorization"].split("Bearer ")[-1])
        if "x-goog-api-key" in safe:
            safe["x-goog-api-key"] = _mask_key(safe["x-goog-api-key"])
        _log_v("Headers: {}", safe)
        _log_v("Payload: {}", json.dumps(payload, indent=2)[:500])

    try:
        _to = 600 if cfg["provider"] in ("nvidia", "kilo") else timeout
        if cfg["provider"] == "ollama":
            _dump_ollama_payload(payload)
        resp = _HTTP.post(cfg["chat_endpoint"], headers=cfg["headers"], json=payload, timeout=_to)
        resp.raise_for_status()
        _log_v("Response {} ({} bytes)", resp.status_code, len(resp.content))
        return _parse_chat_response(resp.json(), cfg["provider"])
    except requests.exceptions.ConnectionError as e:
        if cfg["provider"] in ("ollama", "kaggle"):
            print(f"[LLM] ConnectionError — trying to restart Ollama… ({e})")
            if cfg["provider"] == "kaggle":
                print("[LLM] Kaggle endpoint unreachable — is the tunnel/Kaggle session running?")
                raise RuntimeError("Cannot connect to the Kaggle endpoint. Check the tunnel URL in Settings.")
            if ensure_ollama_running():
                try:
                    resp = _HTTP.post(cfg["chat_endpoint"], headers=cfg["headers"], json=payload, timeout=timeout)
                    resp.raise_for_status()
                    return _parse_chat_response(resp.json(), cfg["provider"])
                except Exception:
                    pass
            raise RuntimeError("Cannot connect to Ollama. Make sure Ollama is installed and run: ollama serve")
        raise RuntimeError(f"Cannot reach {cfg['provider']} at {cfg['base_url']}")
    except requests.exceptions.Timeout:
        raise RuntimeError(f"{cfg['provider']} request timed out after {timeout}s.")
    except requests.exceptions.HTTPError as e:
        body = _http_error_body(e)
        print(f"[LLM] HTTPError: {e.response.status_code} — {body}")
        raise RuntimeError(f"{cfg['provider']} HTTP error: {e.response.status_code} — {body}")
    except Exception as e:
        print(f"[LLM] Unexpected error: {type(e).__name__}: {e}")
        raise RuntimeError(f"LLM call failed: {e}")


def call_llm_text(
    prompt:  str,
    system:  str | None = None,
    model:   str | None = None,
    timeout: int = 120,
) -> str:
    cfg = _get_provider_config()
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    if cfg["provider"] == "groq":
        from core.llm_provider import get_registry
        m = model or cfg["model"]
        p = get_registry().get_provider("groq", model=m)
        return p.chat(messages, tools=None, timeout=timeout).get("content", "")

    if cfg["provider"] == "gemini_live":
        # Live sessions are reserved for interactive voice turns.  Sending a
        # background summary through that persistent audio session competes
        # with microphone frames and can leave the turn waiting indefinitely.
        # Use Gemini's normal text endpoint directly for text-only helpers.
        from google import genai
        text_model = model or _load_config().get(
            "gemini_model", _DEFAULTS["gemini_model"]
        )
        client = genai.Client(
            api_key=cfg.get("api_key", ""),
            http_options={"api_version": "v1beta"},
        )
        result = client.models.generate_content(
            model=text_model,
            contents=prompt,
            config={"system_instruction": system} if system else None,
        )
        return (result.text or "").strip()

    m = model or cfg["model"]
    payload = _build_payload(messages, tools=None, stream=False, model=m, provider=cfg["provider"])

    _log_v("call_llm_text POST {} model={}", cfg["chat_endpoint"], m)
    try:
        _to = 600 if cfg["provider"] in ("nvidia", "kilo") else timeout
        resp = _HTTP.post(cfg["chat_endpoint"], headers=cfg["headers"], json=payload, timeout=_to)
        resp.raise_for_status()
        _log_v("call_llm_text response {} ({} bytes)", resp.status_code, len(resp.content))
        result = _parse_chat_response(resp.json(), cfg["provider"])
        return result["content"]
    except requests.exceptions.ConnectionError as e:
        if cfg["provider"] == "ollama":
            if ensure_ollama_running():
                try:
                    resp = _HTTP.post(cfg["chat_endpoint"], headers=cfg["headers"], json=payload, timeout=timeout)
                    resp.raise_for_status()
                    return _parse_chat_response(resp.json(), cfg["provider"])["content"]
                except Exception:
                    pass
            raise RuntimeError("Cannot connect to Ollama. Make sure Ollama is installed.")
        raise RuntimeError(f"Cannot reach {cfg['provider']}: {e}")
    except requests.exceptions.HTTPError as e:
        body = _http_error_body(e)
        raise RuntimeError(f"{cfg['provider']} HTTP error: {e.response.status_code} — {body}")
    except Exception as e:
        raise RuntimeError(f"LLM text call failed: {e}")


def _stream_openai(
    messages: list,
    tools:    list | None,
    timeout:  int,
    cancel_event: threading.Event | None = None,
) -> Generator[dict, None, None]:
    cfg = _get_provider_config()
    endpoint = cfg["chat_endpoint"]
    messages = _convert_messages_for_openai(messages)
    payload  = _build_payload(messages, tools, stream=True, model=cfg["model"], provider=cfg["provider"])
    # Reasoning models (DeepSeek-R1, QwQ, gpt-oss …) stream chain-of-thought
    # in delta.reasoning_content / delta.reasoning — surface it for the UI.
    _wants_thinking = cfg["provider"] in ("openai", "openrouter", "groq", "nvidia", "kilo")

    _log_v("_stream_openai POST {} model={}", endpoint, cfg["model"])
    try:
        _to = 600 if cfg["provider"] in ("nvidia", "kilo") else timeout
        with _HTTP.post(endpoint, headers=cfg["headers"], json=payload, timeout=_to, stream=True) as resp:
            resp.raise_for_status()
            _log_v("_stream_openai connected, status={}", resp.status_code)
            full_content = ""
            full_thinking = ""
            buf          = ""
            tc_fragments: dict[int, dict] = {}

            for raw in resp.iter_lines():
                if cancel_event and cancel_event.is_set():
                    resp.close()
                    if buf.strip():
                        yield {"type": "sentence", "text": buf.strip()}
                    tool_calls: list = []
                    for idx in sorted(tc_fragments):
                        frag = tc_fragments[idx]
                        args = frag["function"]["arguments"]
                        try:
                            args = json.loads(args)
                        except Exception:
                            pass
                        tool_calls.append({
                            "id":       frag["id"],
                            "type":     "function",
                            "function": {"name": frag["function"]["name"], "arguments": args},
                        })
                    yield {
                        "type":       "cancelled",
                        "content":    full_content.strip(),
                        "tool_calls": tool_calls,
                    }
                    return

                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                choices_list = chunk.get("choices", [])
                if not choices_list:
                    continue
                choice = choices_list[0]
                delta  = choice.get("delta", {})
                text   = delta.get("content") or ""
                if cfg["provider"] == "cloudflare" and not isinstance(text, str):
                    text = str(text)

                if _wants_thinking:
                    think_delta = (
                        delta.get("reasoning_content") or delta.get("reasoning") or ""
                    )
                    if isinstance(think_delta, str) and think_delta:
                        full_thinking += think_delta
                        yield {"type": "thinking", "text": think_delta}

                full_content += text
                buf          += text

                while True:
                    split = _next_utterance(buf)
                    if split is None:
                        break
                    sentence, buf = split
                    if sentence:
                        yield {"type": "sentence", "text": sentence}

                for tc in (delta.get("tool_calls") or []):
                    idx = tc.get("index", 0)
                    if idx not in tc_fragments:
                        tc_fragments[idx] = {"id": "", "function": {"name": "", "arguments": ""}}
                    frag = tc_fragments[idx]
                    frag["id"] = frag["id"] or tc.get("id", "")
                    fn = tc.get("function", {})
                    frag["function"]["name"]      += fn.get("name") or ""
                    frag["function"]["arguments"] += fn.get("arguments") or ""

                finish = choice.get("finish_reason")
                if finish in ("stop", "tool_calls", "length"):
                    break

            if buf.strip():
                yield {"type": "sentence", "text": buf.strip()}

            tool_calls: list = []
            for idx in sorted(tc_fragments):
                frag = tc_fragments[idx]
                args = frag["function"]["arguments"]
                try:
                    args = json.loads(args)
                except Exception:
                    pass
                tool_calls.append({
                    "id":       frag["id"],
                    "type":     "function",
                    "function": {"name": frag["function"]["name"], "arguments": args},
                })

            yield {
                "type":       "done",
                "content":    full_content.strip(),
                "thinking":   full_thinking.strip(),
                "tool_calls": tool_calls,
            }

    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Cannot reach {cfg['provider']} at {cfg['base_url']}.")
    except requests.exceptions.Timeout:
        raise RuntimeError(f"{cfg['provider']} stream timed out.")
    except requests.exceptions.HTTPError as e:
        body = _http_error_body(e)
        raise RuntimeError(f"{cfg['provider']} HTTP error: {e.response.status_code} — {body}")
    except Exception as e:
        raise RuntimeError(f"{cfg['provider']} stream failed: {e}")


def _stream_cloudflare(
    messages: list,
    tools:    list | None,
    timeout:  int,
    cancel_event: threading.Event | None = None,
) -> Generator[dict, None, None]:
    cfg = _get_provider_config()
    endpoint = cfg["chat_endpoint"]
    messages = _prepare_cloudflare_messages(messages)
    payload  = _build_payload(messages, tools, stream=True, model=cfg["model"], provider=cfg["provider"])

    _log_v("_stream_cloudflare POST {} model={}", endpoint, cfg["model"])
    try:
        _to = 600 if cfg["provider"] in ("nvidia", "kilo") else timeout
        with _HTTP.post(endpoint, headers=cfg["headers"], json=payload, timeout=_to, stream=True) as resp:
            resp.raise_for_status()
            _log_v("_stream_cloudflare connected, status={}", resp.status_code)
            full_content = ""
            buf          = ""
            tc_fragments: dict[int, dict] = {}

            for raw in resp.iter_lines():
                if cancel_event and cancel_event.is_set():
                    resp.close()
                    if buf.strip():
                        yield {"type": "sentence", "text": buf.strip()}
                    tool_calls: list = []
                    for idx in sorted(tc_fragments):
                        frag = tc_fragments[idx]
                        args = frag["function"]["arguments"]
                        try:
                            args = json.loads(args)
                        except Exception:
                            pass
                        tool_calls.append({
                            "id":       frag["id"],
                            "type":     "function",
                            "function": {"name": frag["function"]["name"], "arguments": args},
                        })
                    yield {
                        "type":       "cancelled",
                        "content":    full_content.strip(),
                        "tool_calls": tool_calls,
                    }
                    return

                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                choices_list = chunk.get("choices", [])
                if not choices_list:
                    continue
                choice = choices_list[0]
                delta  = choice.get("delta", {})
                text   = delta.get("content") or ""
                if not isinstance(text, str):
                    text = str(text)

                full_content += text
                buf          += text

                while True:
                    split = _next_utterance(buf)
                    if split is None:
                        break
                    sentence, buf = split
                    if sentence:
                        yield {"type": "sentence", "text": sentence}

                for tc in (delta.get("tool_calls") or []):
                    idx = tc.get("index", 0)
                    if idx not in tc_fragments:
                        tc_fragments[idx] = {"id": "", "function": {"name": "", "arguments": ""}}
                    frag = tc_fragments[idx]
                    frag["id"] = frag["id"] or tc.get("id", "")
                    fn = tc.get("function", {})
                    frag["function"]["name"]      += fn.get("name") or ""
                    frag["function"]["arguments"] += fn.get("arguments") or ""

                finish = choice.get("finish_reason")
                if finish in ("stop", "tool_calls", "length"):
                    break

            if buf.strip():
                yield {"type": "sentence", "text": buf.strip()}

            tool_calls: list = []
            for idx in sorted(tc_fragments):
                frag = tc_fragments[idx]
                args = frag["function"]["arguments"]
                try:
                    args = json.loads(args)
                except Exception:
                    pass
                tool_calls.append({
                    "id":       frag["id"],
                    "type":     "function",
                    "function": {"name": frag["function"]["name"], "arguments": args},
                })

            yield {
                "type":       "done",
                "content":    full_content.strip(),
                "tool_calls": tool_calls,
            }

    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Cannot reach Cloudflare at {cfg['base_url']}.")
    except requests.exceptions.Timeout:
        raise RuntimeError("Cloudflare stream timed out.")
    except requests.exceptions.HTTPError as e:
        body = _http_error_body(e)
        raise RuntimeError(f"Cloudflare HTTP error: {e.response.status_code} — {body}")
    except Exception as e:
        raise RuntimeError(f"Cloudflare stream failed: {e}")


def _stream_gemini(
    messages: list,
    tools:    list | None,
    timeout:  int,
    cancel_event: threading.Event | None = None,
) -> Generator[dict, None, None]:
    cfg = _get_provider_config()
    payload = _build_gemini_payload(messages, tools, stream=True)
    stream_url = cfg["chat_endpoint"].replace(":generateContent", ":streamGenerateContent?alt=sse")
    _log_v("_stream_gemini POST {} model={}", stream_url, cfg["model"])
    try:
        with _HTTP.post(stream_url, headers=cfg["headers"], json=payload, timeout=timeout, stream=True) as resp:
            resp.raise_for_status()
            _log_v("_stream_gemini connected, status={}", resp.status_code)
            full_content = ""
            tool_calls:  list[dict] = []
            buf          = ""

            for raw in resp.iter_lines():
                if cancel_event and cancel_event.is_set():
                    resp.close()
                    if buf.strip():
                        yield {"type": "sentence", "text": buf.strip()}
                    yield {
                        "type":       "cancelled",
                        "content":    full_content.strip(),
                        "tool_calls": tool_calls,
                    }
                    return

                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                cand = chunk.get("candidates", [{}])[0]
                if not cand:
                    continue
                parts = (cand.get("content") or {}).get("parts") or []
                for p in parts:
                    if "text" in p:
                        text = p["text"]
                        full_content += text
                        buf          += text
                        while True:
                            split = _next_utterance(buf)
                            if split is None:
                                break
                            sentence, buf = split
                            if sentence:
                                yield {"type": "sentence", "text": sentence}
                    if "functionCall" in p:
                        fc = p["functionCall"]
                        tc = {
                            "id": fc.get("name", ""),
                            "function": {
                                "name": fc.get("name", ""),
                                "arguments": fc.get("args", {}),
                            },
                        }
                        if "thoughtSignature" in p:
                            tc["thoughtSignature"] = p["thoughtSignature"]
                        if not any(t["id"] == tc["id"] for t in tool_calls):
                            tool_calls.append(tc)

                finish = cand.get("finishReason")
                if finish and finish != "FINISH_REASON_UNSPECIFIED":
                    if buf.strip():
                        yield {"type": "sentence", "text": buf.strip()}
                    yield {
                        "type":       "done",
                        "content":    full_content.strip(),
                        "tool_calls": tool_calls,
                    }
                    return

            if buf.strip():
                yield {"type": "sentence", "text": buf.strip()}
            yield {
                "type":       "done",
                "content":    full_content.strip(),
                "tool_calls": tool_calls,
            }
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Cannot reach Gemini at {cfg['base_url']}.")
    except requests.exceptions.Timeout:
        raise RuntimeError("Gemini stream timed out.")
    except requests.exceptions.HTTPError as e:
        body = _http_error_body(e)
        raise RuntimeError(f"Gemini HTTP error: {e.response.status_code} — {body}")
    except Exception as e:
        raise RuntimeError(f"Gemini stream failed: {e}")


def call_llm_stream(
    messages: list,
    tools:    list | None = None,
    timeout:  int = 3600,
    cancel_event: threading.Event | None = None,
) -> Generator[dict, None, None]:
    cfg = _get_provider_config()

    if cfg["provider"] == "gemini":
        yield from _stream_gemini(messages, tools, timeout, cancel_event)
        return

    if cfg["provider"] == "gemini_live":
        from core.llm_provider import get_registry
        provider = get_registry().get_provider(
            "gemini_live",
            api_key=cfg.get("api_key", ""),
            model=cfg.get("model", "gemini-2.5-flash-native-audio-preview-12-2025"),
        )
        yield from provider.chat_stream(messages, tools, timeout, cancel_event)
        return

    if cfg["provider"] == "groq":
        from core.llm_provider import get_registry
        provider = get_registry().get_provider(
            "groq",
            api_key=cfg.get("groq_llm_key"),
            model=cfg.get("groq_llm_model"),
        )
        yield from provider.chat_stream(messages, tools, timeout, cancel_event)
        return

    if cfg["provider"] == "cloudflare":
        yield from _stream_cloudflare(messages, tools, timeout, cancel_event)
        return

    if cfg["provider"] not in ("ollama", "kaggle"):
        yield from _stream_openai(messages, tools, timeout, cancel_event)
        return

    endpoint = cfg["chat_endpoint"]
    _ensure_ollama_messages(messages)
    payload  = _build_payload(messages, tools, stream=True, model=cfg["model"], provider="ollama")

    _log_v("call_llm_stream (ollama) POST {}", endpoint)
    _dump_ollama_payload(payload)

    def _do_stream() -> Generator[dict, None, None]:
        with _HTTP.post(endpoint, headers=cfg["headers"], json=payload, timeout=timeout, stream=True) as resp:
            resp.raise_for_status()
            _log_v("call_llm_stream connected, status={}", resp.status_code)
            full_content = ""
            full_thinking = ""
            tool_calls:  list = []
            buf          = ""

            for raw in resp.iter_lines():
                if cancel_event and cancel_event.is_set():
                    resp.close()
                    if buf.strip():
                        yield {"type": "sentence", "text": buf.strip()}
                    yield {
                        "type":       "cancelled",
                        "content":    full_content.strip(),
                        "tool_calls": tool_calls,
                    }
                    return

                if not raw:
                    continue
                try:
                    chunk = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                msg   = chunk.get("message", {})
                delta = msg.get("content") or ""

                # Reasoning models (Qwen3, DeepSeek-R1, GPT-OSS, gemma-4 …)
                # stream their chain-of-thought separately in `thinking`.
                # Surface it as a dedicated event so the UI can show a live
                # "Thinking…" card; it is NOT part of the spoken/visible reply.
                think_delta = msg.get("thinking") or ""
                if think_delta:
                    full_thinking += think_delta
                    yield {"type": "thinking", "text": think_delta}

                full_content += delta
                buf          += delta

                while True:
                    split = _next_utterance(buf)
                    if split is None:
                        break
                    sentence, buf = split
                    if sentence:
                        yield {"type": "sentence", "text": sentence}

                tc = msg.get("tool_calls")
                if tc:
                    tool_calls.extend(tc)

                if chunk.get("done"):
                    if buf.strip():
                        yield {"type": "sentence", "text": buf.strip()}
                    yield {
                        "type":       "done",
                        "content":    full_content.strip(),
                        "thinking":   full_thinking.strip(),
                        "tool_calls": tool_calls,
                    }
                    return

    _MAX_503_RETRIES = 3

    def _stream_with_retry():
        retries = 0
        while True:
            try:
                yield from _do_stream()
                break
            except requests.exceptions.ConnectionError as e:
                if cfg["provider"] == "kaggle":
                    print(f"[LLM] Kaggle stream ConnectionError ({e})")
                    raise RuntimeError("Cannot connect to the Kaggle endpoint. Is the tunnel/Kaggle session running?")
                print(f"[LLM] Stream ConnectionError — trying to restart Ollama… ({e})")
                if ensure_ollama_running():
                    yield from _do_stream()
                    break
                raise RuntimeError("Cannot connect to Ollama. Make sure Ollama is installed and run: ollama serve")
            except requests.exceptions.Timeout:
                raise RuntimeError("Ollama stream timed out.")
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 503 and retries < _MAX_503_RETRIES:
                    retries += 1
                    wait = 2 ** retries
                    print(f"[LLM] 503 — retry {retries}/{_MAX_503_RETRIES} in {wait}s")
                    import time
                    time.sleep(wait)
                    continue
                body = ""
                try:
                    raw = e.response.content or b""
                    body = raw[:2000].decode("utf-8", errors="replace") or e.response.text[:2000]
                except Exception:
                    try:
                        body = e.response.text[:2000] if e.response is not None else ""
                    except Exception:
                        body = ""
                if not body.strip():
                    body = "(empty response body)"
                _hdrs = {k: v for k, v in (e.response.headers or {}).items()
                         if k.lower() in ("retry-after", "content-type", "content-length",
                                          "server", "x-ratelimit-remaining", "x-ratelimit-reset")}
                print(f"[LLM] HTTPError: {e.response.status_code} — {body} | headers: {_hdrs}")
                raise RuntimeError(f"Ollama HTTP error: {e.response.status_code} — {body} | headers: {_hdrs}")
            except Exception as e:
                print(f"[LLM] Stream error: {type(e).__name__}: {e}")
                raise RuntimeError(f"LLM stream failed: {e}")

    yield from _stream_with_retry()
