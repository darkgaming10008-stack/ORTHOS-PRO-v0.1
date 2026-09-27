"""
Multi-model provider abstraction for Orthos.

Defines a uniform interface across Ollama, OpenAI-compatible servers,
Groq, and Anthropic Claude.  Every provider normalises its native output
to the following shape:

  chat()        -> {"content": str, "tool_calls": list}
  chat_stream() -> Generator yielding
       {"type": "sentence",  "text": str}
       {"type": "done",      "content": str, "tool_calls": list}
       {"type": "cancelled", "content": str, "tool_calls": list}
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generator

import requests


# ---------------------------------------------------------------------------
# Helpers shared by all providers
# ---------------------------------------------------------------------------

_HTTP = requests.Session()

_SENT_END = re.compile(r'(?<=[.!?])\s+|(?<=\n)\s*\n')


def _error_body(resp) -> str:
    """Best-effort extraction of the full response body for error reporting."""
    body = ""
    try:
        if getattr(resp, "content", None):
            body = resp.content[:2000].decode("utf-8", errors="replace")
        if not body.strip():
            body = resp.text[:2000]
    except Exception:
        try:
            body = resp.text[:2000]
        except Exception:
            body = ""
    return body if body.strip() else "(empty response body)"


def _error_headers(resp) -> str:
    """Subset of response headers useful for diagnosing HTTP errors."""
    try:
        picked = {k: v for k, v in (resp.headers or {}).items()
                  if k.lower() in ("retry-after", "content-type", "content-length",
                                   "server", "x-ratelimit-remaining", "x-ratelimit-reset")}
        return str(picked)
    except Exception:
        return ""


def _get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR    = _get_base_dir()
CONFIG_PATH = BASE_DIR / "config" / "api_keys.json"

_DEFAULTS = {
    "llm_url":           "http://localhost:11434",
    "llm_model":         "ministral-3:14b-cloud",
    "llm_provider":      "ollama",
    "cf_model":          "@cf/google/gemma-4-26b-a4b-it",
    "groq_llm_model":    "meta-llama/llama-4-scout-17b-16e-instruct",
    "gemini_model":      "gemini-2.5-flash",
    "nvidia_model":      "qwen/qwen3.5-397b-a17b",
    "kilo_model":        "kilo-auto/free",
}


def _load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _parse_arg_str_to_dict(args: str) -> dict | None:
    """Best-effort parse a string into a dict (JSON or Python literal)."""
    s = args.strip()
    try:
        r = json.loads(s)
        if isinstance(r, dict):
            return r
    except Exception:
        pass
    try:
        r = ast.literal_eval(s)
        if isinstance(r, dict):
            return r
    except Exception:
        pass
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        try:
            r = json.loads(s[1:-1])
            if isinstance(r, dict):
                return r
        except Exception:
            pass
    return None


def _flatten_openai_tool_calls(raw: list) -> list:
    """Normalise OpenAI-style tool-call chunks to a uniform list."""
    out: list = []
    for t in raw or []:
        args = t.get("function", {}).get("arguments", {})
        if isinstance(args, str):
            parsed = _parse_arg_str_to_dict(args)
            if parsed is not None:
                args = parsed
        out.append({
            "id":       t.get("id", ""),
            "type":     t.get("type") or "function",
            "function": {
                "name":      t["function"]["name"],
                "arguments": args,
            },
        })
    return out


def _lowercase_types(obj):
    """Recursively lowercase 'type' string values in tool schemas."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "type" and isinstance(v, str):
                out[k] = v.lower()
            else:
                out[k] = _lowercase_types(v)
        return out
    if isinstance(obj, list):
        return [_lowercase_types(i) for i in obj]
    return obj


def _convert_to_openai_tools(tools: list) -> list:
    """Convert Ollama-format tools to OpenAI/Groq-compatible format."""
    out = []
    for t in tools:
        if "type" in t and t["type"] == "function":
            out.append(t)
            continue
        fn = {
            "name": t.get("name", ""),
            "description": t.get("description", ""),
        }
        params = t.get("parameters", {})
        if isinstance(params, dict):
            params = _lowercase_types(params)
        fn["parameters"] = params
        out.append({"type": "function", "function": fn})
    return out


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    """Every LLM backend implements this contract."""

    @abstractmethod
    def chat(
        self,
        messages: list,
        tools: list | None = None,
        timeout: int = 120,
    ) -> dict:
        ...

    @abstractmethod
    def chat_stream(
        self,
        messages: list,
        tools: list | None = None,
        timeout: int = 120,
        cancel_event: threading.Event | None = None,
    ) -> Generator[dict, None, None]:
        ...

    @abstractmethod
    def warmup(self, system_prompt: str | None = None) -> bool:
        ...

    @abstractmethod
    def ensure_running(self) -> bool:
        ...

    @abstractmethod
    def get_settings(self) -> tuple[str, str]:
        """Return (base_url, model_name)."""
        ...

    def call_text(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        timeout: int = 120,
    ) -> str:
        """Convenience: text-only generation (no tools)."""
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        if model:
            old_url, old_model = self.get_settings()
            self._model = model
        result = self.chat(messages, tools=None, timeout=timeout)
        if model:
            self._model = old_model
        return result.get("content", "")


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

class OllamaProvider(LLMProvider):
    """Uses Ollama's native /api/chat endpoint."""

    def __init__(
        self,
        url: str | None = None,
        model: str | None = None,
    ):
        cfg = _load_config()
        self._url   = (url or cfg.get("llm_url", _DEFAULTS["llm_url"])).rstrip("/")
        self._model = model or cfg.get("llm_model", _DEFAULTS["llm_model"])

    # -- helpers ---------------------------------------------------------

    def get_settings(self) -> tuple[str, str]:
        return self._url, self._model

    def _build_payload(self, messages: list, tools: list | None, stream: bool) -> dict:
        payload: dict = {
            "model":      self._model,
            "messages":   messages,
            "stream":     stream,
            "keep_alive": -1,
            "options":    {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "num_gpu": 99},
        }
        if tools:
            payload["tools"] = tools
        return payload

    # -- public api -------------------------------------------------------

    def ensure_running(self) -> bool:
        health = f"{self._url}/api/tags"

        def _is_up() -> bool:
            try:
                return _HTTP.get(health, timeout=3).status_code == 200
            except Exception:
                return False

        if _is_up():
            return True

        print("[Ollama] Not running — launching 'ollama serve'…")
        try:
            kwargs: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            subprocess.Popen(["ollama", "serve"], **kwargs)
        except FileNotFoundError:
            print("[Ollama] 'ollama' command not found. Install from https://ollama.com")
            return False
        except Exception as e:
            print(f"[Ollama] Could not launch: {e}")
            return False

        deadline = time.time() + 15
        while time.time() < deadline:
            time.sleep(1.0)
            if _is_up():
                print("[Ollama] Started successfully.")
                return True

        print("[Ollama] Did not respond within the timeout.")
        return False

    def warmup(self, system_prompt: str | None = None) -> bool:
        print(f"[Ollama] Warming up '{self._model}'…")
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": "hi"})
        payload = {
            "model":      self._model,
            "messages":   messages,
            "stream":     False,
            "keep_alive": -1,
            "options":    {"num_predict": 1, "num_gpu": 99},
        }
        try:
            resp = _HTTP.post(f"{self._url}/api/chat", json=payload, timeout=180)
            resp.raise_for_status()
            print(f"[Ollama] '{self._model}' loaded and KV cache primed.")
            return True
        except Exception as e:
            print(f"[Ollama] Warmup failed (non-fatal): {e}")
            return False

    def chat(
        self,
        messages: list,
        tools: list | None = None,
        timeout: int = 120,
    ) -> dict:
        payload = self._build_payload(messages, tools, stream=False)
        endpoint = f"{self._url}/api/chat"

        _MAX_503_RETRIES = 3
        for attempt in range(_MAX_503_RETRIES + 1):
            try:
                resp = _HTTP.post(endpoint, json=payload, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()
                msg = data.get("message", {})
                return {
                    "content":    (msg.get("content") or "").strip(),
                    "tool_calls": msg.get("tool_calls") or [],
                }
            except requests.exceptions.ConnectionError as e:
                print(f"[Ollama] ConnectionError — trying restart … ({e})")
                if self.ensure_running():
                    try:
                        resp = _HTTP.post(endpoint, json=payload, timeout=timeout)
                        resp.raise_for_status()
                        data = resp.json()
                        msg = data.get("message", {})
                        return {
                            "content":    (msg.get("content") or "").strip(),
                            "tool_calls": msg.get("tool_calls") or [],
                        }
                    except Exception:
                        pass
                raise RuntimeError(
                    f"Cannot connect to Ollama at {self._url}. "
                    "Make sure Ollama is installed and run: ollama serve"
                )
            except requests.exceptions.Timeout:
                raise RuntimeError("Ollama request timed out.")
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 503 and attempt < _MAX_503_RETRIES:
                    wait = 2 ** (attempt + 1)
                    print(f"[Ollama] 503 — retry {attempt+1}/{_MAX_503_RETRIES} in {wait}s")
                    import time
                    time.sleep(wait)
                    continue
                body = _error_body(e.response)
                hdrs = _error_headers(e.response)
                print(f"[Ollama] HTTPError: {e.response.status_code} — {body} | headers: {hdrs}")
                raise RuntimeError(f"Ollama HTTP error: {e.response.status_code} — {body} | headers: {hdrs}")
            except Exception as e:
                raise RuntimeError(f"Ollama LLM call failed: {e}")

    def chat_stream(
        self,
        messages: list,
        tools: list | None = None,
        timeout: int = 120,
        cancel_event: threading.Event | None = None,
    ) -> Generator[dict, None, None]:
        payload = self._build_payload(messages, tools, stream=True)
        endpoint = f"{self._url}/api/chat"

        def _do_stream() -> Generator[dict, None, None]:
            with _HTTP.post(endpoint, json=payload, timeout=timeout, stream=True) as resp:
                resp.raise_for_status()
                full_content = ""
                tool_calls: list = []
                buf = ""

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

                    msg = chunk.get("message", {})
                    delta = msg.get("content") or ""

                    full_content += delta
                    buf += delta

                    while True:
                        m = _SENT_END.search(buf)
                        if not m:
                            break
                        sentence = buf[: m.start() + 1].strip()
                        buf = buf[m.end():]
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
                            "tool_calls": tool_calls,
                        }
                        return

        _MAX_503_RETRIES = 3
        for attempt in range(_MAX_503_RETRIES + 1):
            try:
                yield from _do_stream()
                break
            except requests.exceptions.ConnectionError as e:
                print(f"[Ollama] Stream ConnectionError — trying restart … ({e})")
                if self.ensure_running():
                    yield from _do_stream()
                    break
                raise RuntimeError(
                    f"Cannot connect to Ollama at {self._url}. "
                    "Make sure Ollama is installed and run: ollama serve"
                )
            except requests.exceptions.Timeout:
                raise RuntimeError("Ollama stream timed out.")
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 503 and attempt < _MAX_503_RETRIES:
                    wait = 2 ** (attempt + 1)
                    print(f"[Ollama] 503 — retry {attempt+1}/{_MAX_503_RETRIES} in {wait}s")
                    import time
                    time.sleep(wait)
                    continue
                body = _error_body(e.response)
                hdrs = _error_headers(e.response)
                print(f"[Ollama] HTTPError: {e.response.status_code} — {body} | headers: {hdrs}")
                raise RuntimeError(f"Ollama HTTP error: {e.response.status_code} — {body} | headers: {hdrs}")
            except Exception as e:
                raise RuntimeError(f"Ollama stream failed: {e}")


# ---------------------------------------------------------------------------
# Kaggle (remote Ollama-compatible server behind a cloudflare tunnel)
# ---------------------------------------------------------------------------

class KaggleProvider(OllamaProvider):
    """Kaggle-hosted Ollama-compatible endpoint (kaggle_url / kaggle_model).

    Same /api/chat protocol as OllamaProvider — only config keys and the
    health check differ: it is a REMOTE tunnel, so ensure_running() checks
    the tunnel and never tries to launch a local ``ollama serve``.
    """

    def __init__(self, url: str | None = None, model: str | None = None):
        cfg = _load_config()
        self._url = (
            url or cfg.get("kaggle_url", "") or cfg.get("llm_url", _DEFAULTS["llm_url"])
        ).rstrip("/")
        self._model = model or cfg.get("kaggle_model", "") or cfg.get("llm_model", _DEFAULTS["llm_model"])

    def ensure_running(self) -> bool:
        try:
            return _HTTP.get(f"{self._url}/api/tags", timeout=5).status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# OpenAI-compatible  (LM Studio, LocalAI, Jan, vLLM, llama.cpp …)
# ---------------------------------------------------------------------------

class OpenAIProvider(LLMProvider):
    """Any OpenAI-compatible chat completions server."""

    def __init__(
        self,
        url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ):
        cfg = _load_config()
        self._url     = (url or cfg.get("llm_url", _DEFAULTS["llm_url"])).rstrip("/")
        self._model   = model or cfg.get("llm_model", _DEFAULTS["llm_model"])
        self._api_key = api_key or cfg.get("openai_api_key", "")

    # -- helpers ---------------------------------------------------------

    def get_settings(self) -> tuple[str, str]:
        return self._url, self._model

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    def _build_payload(self, messages: list, tools: list | None, stream: bool) -> dict:
        payload: dict = {
            "model":    self._model,
            "messages": messages,
            "stream":   stream,
        }
        if tools:
            payload["tools"]       = tools
            payload["tool_choice"] = "auto"
        return payload

    # -- public api -------------------------------------------------------

    def ensure_running(self) -> bool:
        health = f"{self._url}/v1/models"
        try:
            ok = _HTTP.get(health, headers=self._headers(), timeout=5).status_code == 200
            if ok:
                print(f"[OpenAI] Server reachable at {self._url}")
            else:
                print(f"[OpenAI] Server at {self._url} returned non-200.")
            return ok
        except Exception as e:
            print(
                f"[OpenAI] Cannot reach server at {self._url}.\n"
                "      Make sure LM Studio / LocalAI / Jan is running."
            )
            return False

    def warmup(self, system_prompt: str | None = None) -> bool:
        print(f"[OpenAI] Warming up '{self._model}'…")
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": "hi"})
        payload = {
            "model":      self._model,
            "messages":   messages,
            "stream":     False,
            "max_tokens": 1,
        }
        try:
            resp = _HTTP.post(
                f"{self._url}/v1/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=180,
            )
            resp.raise_for_status()
            print(f"[OpenAI] '{self._model}' ready.")
            return True
        except Exception as e:
            print(f"[OpenAI] Warmup failed (non-fatal): {e}")
            return False

    def chat(
        self,
        messages: list,
        tools: list | None = None,
        timeout: int = 120,
    ) -> dict:
        payload = self._build_payload(messages, tools, stream=False)
        endpoint = f"{self._url}/v1/chat/completions"

        try:
            resp = _HTTP.post(endpoint, json=payload, headers=self._headers(), timeout=timeout)
            resp.raise_for_status()
            choice = resp.json().get("choices", [{}])[0]
            msg = choice.get("message", {})
            return {
                "content":    (msg.get("content") or "").strip(),
                "tool_calls": _flatten_openai_tool_calls(msg.get("tool_calls") or []),
            }
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot reach OpenAI-compatible server at {self._url}.\n"
                "Make sure LM Studio / LocalAI / Jan is running."
            )
        except requests.exceptions.Timeout:
            raise RuntimeError("OpenAI-compatible request timed out.")
        except requests.exceptions.HTTPError as e:
            body = _error_body(e.response)
            raise RuntimeError(f"OpenAI-compatible HTTP error: {e.response.status_code} — {body}")
        except Exception as e:
            raise RuntimeError(f"OpenAI-compatible LLM call failed: {e}")

    def chat_stream(
        self,
        messages: list,
        tools: list | None = None,
        timeout: int = 120,
        cancel_event: threading.Event | None = None,
    ) -> Generator[dict, None, None]:
        payload = self._build_payload(messages, tools, stream=True)
        endpoint = f"{self._url}/v1/chat/completions"

        try:
            with _HTTP.post(
                endpoint, json=payload, headers=self._headers(), timeout=timeout, stream=True
            ) as resp:
                resp.raise_for_status()
                full_content = ""
                buf = ""
                tc_fragments: dict[int, dict] = {}

                for raw in resp.iter_lines():
                    if cancel_event and cancel_event.is_set():
                        resp.close()
                        if buf.strip():
                            yield {"type": "sentence", "text": buf.strip()}
                        tool_calls = _flatten_openai_tool_calls(
                            self._finalise_tc_fragments(tc_fragments)
                        )
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

                    choice = chunk.get("choices", [{}])[0]
                    delta = choice.get("delta", {})
                    text = delta.get("content") or ""

                    full_content += text
                    buf += text

                    while True:
                        m = _SENT_END.search(buf)
                        if not m:
                            break
                        sentence = buf[: m.start() + 1].strip()
                        buf = buf[m.end():]
                        if sentence:
                            yield {"type": "sentence", "text": sentence}

                    for tc in delta.get("tool_calls") or []:
                        idx = tc.get("index", 0)
                        if idx not in tc_fragments:
                            tc_fragments[idx] = {
                                "id": "", "function": {"name": "", "arguments": ""},
                            }
                        frag = tc_fragments[idx]
                        frag["id"] = frag["id"] or tc.get("id", "")
                        fn = tc.get("function", {})
                        frag["function"]["name"] += fn.get("name") or ""
                        frag["function"]["arguments"] += fn.get("arguments") or ""

                    finish = choice.get("finish_reason")
                    if finish in ("stop", "tool_calls", "length"):
                        break

                if buf.strip():
                    yield {"type": "sentence", "text": buf.strip()}

                raw_tc = [tc_fragments[i] for i in sorted(tc_fragments)]
                tool_calls = _flatten_openai_tool_calls(raw_tc)

                yield {
                    "type":       "done",
                    "content":    full_content.strip(),
                    "tool_calls": tool_calls,
                }

        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot reach OpenAI-compatible server at {self._url}.\n"
                "Make sure LM Studio / LocalAI / Jan is running."
            )
        except requests.exceptions.Timeout:
            raise RuntimeError("OpenAI-compatible stream timed out.")
        except requests.exceptions.HTTPError as e:
            body = _error_body(e.response)
            raise RuntimeError(f"OpenAI-compatible HTTP error: {e.response.status_code} — {body}")
        except Exception as e:
            raise RuntimeError(f"OpenAI-compatible stream failed: {e}")

    @staticmethod
    def _finalise_tc_fragments(fragments: dict[int, dict]) -> list:
        out: list = []
        for idx in sorted(fragments):
            frag = fragments[idx]
            out.append({
                "id": frag["id"],
                "function": {
                    "name": frag["function"]["name"],
                    "arguments": frag["function"]["arguments"],
                },
            })
        return out


# ---------------------------------------------------------------------------
# Groq  (OpenAI-compatible wire format, different base URL + auth)
# ---------------------------------------------------------------------------

class GroqProvider(LLMProvider):
    """Groq API — standalone provider (not inheriting OpenAIProvider)."""

    GROQ_BASE = "https://api.groq.com/openai"

    def __init__(
        self,
        url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ):
        cfg = _load_config()
        self._url     = (url or self.GROQ_BASE).rstrip("/")
        self._model   = model or cfg.get("groq_llm_model", _DEFAULTS["groq_llm_model"])
        self._api_key = api_key or cfg.get("groq_llm_key", "")

    def get_settings(self) -> tuple[str, str]:
        return self._url, self._model

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    def _build_payload(self, messages: list, tools: list | None, stream: bool) -> dict:
        payload: dict = {
            "model":    self._model,
            "messages": messages,
            "stream":   stream,
            "max_tokens": 8192,
        }
        if tools:
            payload["tools"]       = _convert_to_openai_tools(tools)
            payload["tool_choice"] = "auto"
        return payload

    def ensure_running(self) -> bool:
        health = f"{self._url}/v1/models"
        try:
            ok = _HTTP.get(health, headers=self._headers(), timeout=10).status_code == 200
            if ok:
                print(f"[Groq] API reachable at {self._url}")
            else:
                print("[Groq] API returned non-200. Check API key.")
            return ok
        except Exception as e:
            print(f"[Groq] Cannot reach API: {e}")
            return False

    def warmup(self, system_prompt: str | None = None) -> bool:
        print(f"[Groq] Warming up '{self._model}'…")
        # Use a minimal prompt to avoid consuming limited TPM quota
        payload = {
            "model":      self._model,
            "messages":   [{"role": "user", "content": "hi"}],
            "stream":     False,
            "max_tokens": 1,
        }
        try:
            resp = _HTTP.post(
                f"{self._url}/v1/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=30,
            )
            resp.raise_for_status()
            print(f"[Groq] '{self._model}' ready.")
            return True
        except Exception as e:
            print(f"[Groq] Warmup failed (non-fatal): {e}")
            return False

    def chat(
        self,
        messages: list,
        tools: list | None = None,
        timeout: int = 120,
    ) -> dict:
        from core.llm_client import _convert_messages_for_openai, _select_tools_for_provider
        messages = _convert_messages_for_openai(messages)
        if tools:
            tools = _select_tools_for_provider(tools, messages, "groq")
        payload = self._build_payload(messages, tools, stream=False)
        endpoint = f"{self._url}/v1/chat/completions"
        try:
            resp = _HTTP.post(endpoint, json=payload, headers=self._headers(), timeout=timeout)
            resp.raise_for_status()
            choice = resp.json().get("choices", [{}])[0]
            msg = choice.get("message", {})
            return {
                "content":    (msg.get("content") or "").strip(),
                "tool_calls": _flatten_openai_tool_calls(msg.get("tool_calls") or []),
            }
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"Cannot reach Groq at {self._url}.")
        except requests.exceptions.Timeout:
            raise RuntimeError("Groq request timed out.")
        except requests.exceptions.HTTPError as e:
            body = _error_body(e.response)
            raise RuntimeError(f"Groq HTTP error: {e.response.status_code} — {body}")
        except Exception as e:
            raise RuntimeError(f"Groq LLM call failed: {e}")

    def chat_stream(
        self,
        messages: list,
        tools: list | None = None,
        timeout: int = 120,
        cancel_event: threading.Event | None = None,
    ) -> Generator[dict, None, None]:
        from core.llm_client import _convert_messages_for_openai, _select_tools_for_provider
        messages = _convert_messages_for_openai(messages)
        if tools:
            tools = _select_tools_for_provider(tools, messages, "groq")
        payload = self._build_payload(messages, tools, stream=True)
        endpoint = f"{self._url}/v1/chat/completions"

        try:
            with _HTTP.post(
                endpoint, json=payload, headers=self._headers(), timeout=timeout, stream=True
            ) as resp:
                resp.raise_for_status()
                full_content = ""
                buf = ""
                tc_fragments: dict[int, dict] = {}

                for raw in resp.iter_lines():
                    if cancel_event and cancel_event.is_set():
                        resp.close()
                        if buf.strip():
                            yield {"type": "sentence", "text": buf.strip()}
                        tool_calls = _flatten_openai_tool_calls(
                            [tc_fragments[i] for i in sorted(tc_fragments)]
                        )
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
                    delta = choice.get("delta", {})
                    text = delta.get("content") or ""

                    full_content += text
                    buf += text

                    while True:
                        m = _SENT_END.search(buf)
                        if not m:
                            break
                        sentence = buf[: m.start() + 1].strip()
                        buf = buf[m.end():]
                        if sentence:
                            yield {"type": "sentence", "text": sentence}

                    for tc in delta.get("tool_calls") or []:
                        idx = tc.get("index", 0)
                        if idx not in tc_fragments:
                            tc_fragments[idx] = {
                                "id": "", "function": {"name": "", "arguments": ""},
                            }
                        frag = tc_fragments[idx]
                        frag["id"] = frag["id"] or tc.get("id", "")
                        fn = tc.get("function", {})
                        frag["function"]["name"] += fn.get("name") or ""
                        frag["function"]["arguments"] += fn.get("arguments") or ""

                    finish = choice.get("finish_reason")
                    if finish in ("stop", "tool_calls", "length"):
                        break

                if buf.strip():
                    yield {"type": "sentence", "text": buf.strip()}

                raw_tc = [tc_fragments[i] for i in sorted(tc_fragments)]
                tool_calls = _flatten_openai_tool_calls(raw_tc)

                if not full_content.strip() and not tool_calls:
                    raise RuntimeError("Groq returned empty response — possible rate limit or overload")

                yield {
                    "type":       "done",
                    "content":    full_content.strip(),
                    "tool_calls": tool_calls,
                }

        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"Cannot reach Groq at {self._url}.")
        except requests.exceptions.Timeout:
            raise RuntimeError("Groq stream timed out.")
        except requests.exceptions.HTTPError as e:
            body = _error_body(e.response)
            raise RuntimeError(f"Groq HTTP error: {e.response.status_code} — {body}")
        except Exception as e:
            raise RuntimeError(f"Groq stream failed: {e}")


# ---------------------------------------------------------------------------
# Cloudflare  (OpenAI-compatible wire format, different base URL + auth)
# ---------------------------------------------------------------------------

class CloudflareProvider(OpenAIProvider):
    """Cloudflare Workers AI — thin wrapper over OpenAIProvider."""

    def __init__(
        self,
        url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ):
        cfg = _load_config()
        aid   = cfg.get("cf_account_id", "")
        self._url     = (url or f"https://api.cloudflare.com/client/v4/accounts/{aid}/ai").rstrip("/")
        self._model   = model or cfg.get("cf_model", _DEFAULTS["cf_model"])
        self._api_key = api_key or cfg.get("cf_api_token", "")

    def ensure_running(self) -> bool:
        try:
            ok = _HTTP.get(f"{self._url}/v1/models", headers=self._headers(), timeout=10).status_code == 200
            if ok:
                print(f"[Cloudflare] API reachable at {self._url}")
            return ok
        except Exception:
            return True

    def warmup(self, system_prompt: str | None = None) -> bool:
        print(f"[Cloudflare] Warming up '{self._model}'…")
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": "hi"})
        payload = {
            "model":      self._model,
            "messages":   messages,
            "stream":     False,
            "max_tokens": 1,
        }
        try:
            resp = _HTTP.post(
                f"{self._url}/v1/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=30,
            )
            resp.raise_for_status()
            print(f"[Cloudflare] '{self._model}' ready.")
            return True
        except Exception as e:
            print(f"[Cloudflare] Warmup failed (non-fatal): {e}")
            return False


# ---------------------------------------------------------------------------
# NVIDIA  (OpenAI-compatible)
# ---------------------------------------------------------------------------

class NVIDIAProvider(OpenAIProvider):
    """NVIDIA NIM API — OpenAI-compatible chat completions."""

    NVIDIA_BASE = "https://integrate.api.nvidia.com"

    def __init__(
        self,
        url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ):
        cfg = _load_config()
        self._url     = (url or self.NVIDIA_BASE).rstrip("/")
        self._model   = model or cfg.get("nvidia_model", _DEFAULTS.get("nvidia_model", "qwen/qwen3.5-397b-a17b"))
        self._api_key = api_key or cfg.get("nvidia_api_key", "")

    def ensure_running(self) -> bool:
        try:
            ok = _HTTP.get(f"{self._url}/v1/models", headers=self._headers(), timeout=10).status_code == 200
            if ok:
                print(f"[NVIDIA] API reachable at {self._url}")
            return ok
        except Exception:
            return True

    def warmup(self, system_prompt: str | None = None) -> bool:
        print(f"[NVIDIA] Warming up '{self._model}'…")
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": "hi"})
        payload = {
            "model":      self._model,
            "messages":   messages,
            "stream":     False,
            "max_tokens": 1,
        }
        try:
            resp = _HTTP.post(
                f"{self._url}/v1/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=30,
            )
            resp.raise_for_status()
            print(f"[NVIDIA] '{self._model}' ready.")
            return True
        except Exception as e:
            print(f"[NVIDIA] Warmup failed (non-fatal): {e}")
            return False


# ---------------------------------------------------------------------------
# Kilo Code Gateway  (OpenAI-compatible, free :free models)
# ---------------------------------------------------------------------------

class KiloProvider(OpenAIProvider):
    """Kilo Code Gateway — OpenAI-compatible chat completions."""

    KILO_BASE = "https://api.kilo.ai/api/gateway"

    def __init__(
        self,
        url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ):
        cfg = _load_config()
        self._url     = (url or self.KILO_BASE).rstrip("/")
        self._model   = model or cfg.get("kilo_model", _DEFAULTS.get("kilo_model", "kilo-auto/free"))
        self._api_key = api_key or cfg.get("kilo_api_key", "")

    def ensure_running(self) -> bool:
        try:
            ok = _HTTP.get(f"{self._url}/models", headers=self._headers(), timeout=10).status_code == 200
            if ok:
                print(f"[Kilo] API reachable at {self._url}")
            return ok
        except Exception:
            return True

    def warmup(self, system_prompt: str | None = None) -> bool:
        print(f"[Kilo] Warming up '{self._model}'…")
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": "hi"})
        payload = {
            "model":      self._model,
            "messages":   messages,
            "stream":     False,
            "max_tokens": 1,
        }
        try:
            resp = _HTTP.post(
                f"{self._url}/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=30,
            )
            resp.raise_for_status()
            print(f"[Kilo] '{self._model}' ready.")
            return True
        except Exception as e:
            print(f"[Kilo] Warmup failed (non-fatal): {e}")
            return False


# ---------------------------------------------------------------------------
# Gemini  (native generateContent endpoint — NOT OpenAI-compatible)
# ---------------------------------------------------------------------------

class GeminiProvider(LLMProvider):
    """Google Gemini API — native generateContent endpoint."""

    GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(
        self,
        url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ):
        cfg = _load_config()
        self._url     = (url or self.GEMINI_BASE).rstrip("/")
        self._model   = model or cfg.get("gemini_model", _DEFAULTS["gemini_model"])
        self._api_key = api_key or cfg.get("gemini_api_key", "")

    def get_settings(self) -> tuple[str, str]:
        return self._url, self._model

    def _headers(self) -> dict:
        return {"x-goog-api-key": self._api_key, "Content-Type": "application/json"}

    def ensure_running(self) -> bool:
        try:
            ok = _HTTP.get(f"{self._url}/models", headers=self._headers(), timeout=10).status_code == 200
            if ok:
                print(f"[Gemini] API reachable at {self._url}")
            return ok
        except Exception:
            return True

    def warmup(self, system_prompt: str | None = None) -> bool:
        print(f"[Gemini] Warming up '{self._model}'…")
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": "hi"})
        payload = {"contents": messages}
        try:
            resp = _HTTP.post(
                f"{self._url}/models/{self._model}:generateContent",
                json=payload,
                headers=self._headers(),
                timeout=30,
            )
            resp.raise_for_status()
            print(f"[Gemini] '{self._model}' ready.")
            return True
        except Exception as e:
            print(f"[Gemini] Warmup failed (non-fatal): {e}")
            return False

    def _convert_messages(self, messages: list) -> list:
        """Convert internal messages to Gemini native format."""
        from core.llm_client import _build_gemini_payload
        payload = _build_gemini_payload(messages, tools=None)
        return payload.get("contents", [])

    def _extract_system(self, messages: list) -> str:
        parts = [m["content"] for m in messages if m.get("role") == "system"]
        return "\n\n".join(parts)

    def chat(
        self,
        messages: list,
        tools: list | None = None,
        timeout: int = 120,
    ) -> dict:
        from core.llm_client import _build_gemini_payload, _parse_gemini_response
        payload = _build_gemini_payload(messages, tools, stream=False)
        try:
            resp = _HTTP.post(
                f"{self._url}/models/{self._model}:generateContent",
                json=payload,
                headers=self._headers(),
                timeout=timeout,
            )
            resp.raise_for_status()
            return _parse_gemini_response(resp.json())
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"Cannot reach Gemini API at {self._url}.")
        except requests.exceptions.Timeout:
            raise RuntimeError("Gemini request timed out.")
        except requests.exceptions.HTTPError as e:
            body = _error_body(e.response)
            raise RuntimeError(f"Gemini HTTP error: {e.response.status_code} — {body}")
        except Exception as e:
            raise RuntimeError(f"Gemini call failed: {e}")

    def chat_stream(
        self,
        messages: list,
        tools: list | None = None,
        timeout: int = 120,
        cancel_event: threading.Event | None = None,
    ) -> Generator[dict, None, None]:
        from core.llm_client import _stream_gemini
        yield from _stream_gemini(messages, tools, timeout, cancel_event)


# ---------------------------------------------------------------------------
# Anthropic Claude
# ---------------------------------------------------------------------------

class AnthropicProvider(LLMProvider):
    """Anthropic Messages API (Claude)."""

    ANTHROPIC_BASE = "https://api.anthropic.com/v1"

    def __init__(
        self,
        url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ):
        cfg = _load_config()
        self._url     = (url or self.ANTHROPIC_BASE).rstrip("/")
        self._model   = model or cfg.get("llm_model", "claude-sonnet-4-20250514")
        self._api_key = api_key or cfg.get("anthropic_api_key", "")

    # -- helpers ---------------------------------------------------------

    def get_settings(self) -> tuple[str, str]:
        return self._url, self._model

    def _headers(self) -> dict:
        return {
            "Content-Type":      "application/json",
            "x-api-key":         self._api_key,
            "anthropic-version": "2023-06-01",
        }

    def _convert_messages(self, messages: list) -> list:
        """Convert OpenAI-style messages to Anthropic format.

        Anthropic uses 'role': 'user' | 'assistant'.
        System messages are sent separately via the 'system' top-level field.
        """
        out: list = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "system":
                continue
            content: list = []
            text = msg.get("content") or ""
            if text:
                content.append({"type": "text", "text": text})
            for tc in msg.get("tool_calls") or []:
                content.append({
                    "type":  "tool_use",
                    "id":    tc.get("id", ""),
                    "name":  tc.get("function", {}).get("name", ""),
                    "input": tc.get("function", {}).get("arguments", {}),
                })
            out.append({"role": "assistant" if role == "assistant" else "user", "content": content})
        return out

    def _extract_system(self, messages: list) -> str:
        parts = [m["content"] for m in messages if m.get("role") == "system"]
        return "\n\n".join(parts)

    def _convert_tools(self, tools: list | None) -> list | None:
        """Convert to Anthropic tool format."""
        if not tools:
            return None
        out: list = []
        for t in tools:
            params = t.get("parameters", t.get("function", {}).get("parameters", {}))
            out.append({
                "name":        t.get("name", t.get("function", {}).get("name", "")),
                "description": t.get("description", ""),
                "input_schema": {
                    k.lower(): v
                    for k, v in params.items()
                },
            })
        return out

    def _parse_response(self, data: dict) -> tuple[str, list]:
        """Extract content & tool_calls from an Anthropic response."""
        content = ""
        tool_calls: list = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_calls.append({
                    "id":       block.get("id", ""),
                    "function": {
                        "name":      block.get("name", ""),
                        "arguments": block.get("input", {}),
                    },
                })
        return content.strip(), tool_calls

    def _parse_stream_event(self, event: dict, buffer: dict) -> bool:
        """Process one SSE event from Anthropic stream, mutating *buffer*.

        Returns True when the stream is complete.
        """
        e_type = event.get("type", "")
        if e_type == "message_start":
            buffer["message_id"] = event["message"]["id"]
            buffer["stop_reason"] = event["message"].get("stop_reason")
        elif e_type == "content_block_start":
            block = event.get("content_block", {})
            if block.get("type") == "tool_use":
                idx = event.get("index", 0)
                buffer.setdefault("tc_blocks", {})[idx] = {
                    "id":   block.get("id", ""),
                    "name": block.get("name", ""),
                    "input": "",
                }
        elif e_type == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta":
                buffer["content_delta"] += delta.get("text", "")
            elif delta.get("type") == "input_json_delta":
                idx = event.get("index", 0)
                block = buffer.setdefault("tc_blocks", {}).get(idx)
                if block:
                    block["input"] += delta.get("partial_json", "")
        elif e_type == "message_delta":
            buffer["stop_reason"] = event["delta"].get("stop_reason")
        elif e_type == "message_stop":
            return True
        return False

    def _build_tool_calls_from_buffer(self, buffer: dict) -> list:
        tc: list = []
        for idx in sorted(buffer.get("tc_blocks", {})):
            b = buffer["tc_blocks"][idx]
            try:
                inp = json.loads(b["input"])
            except Exception:
                inp = b["input"]
            tc.append({
                "id":       b["id"],
                "function": {"name": b["name"], "arguments": inp},
            })
        return tc

    # -- public api -------------------------------------------------------

    def ensure_running(self) -> bool:
        try:
            resp = _HTTP.get(
                f"{self._url}/models",
                headers=self._headers(),
                timeout=10,
            )
            ok = resp.status_code == 200
            if ok:
                print(f"[Anthropic] API reachable at {self._url}")
            else:
                print("[Anthropic] API returned non-200. Check API key.")
            return ok
        except Exception as e:
            print(f"[Anthropic] Cannot reach API: {e}")
            return False

    def warmup(self, system_prompt: str | None = None) -> bool:
        print(f"[Anthropic] Warming up '{self._model}'…")
        messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        payload: dict = {
            "model":      self._model,
            "messages":   messages,
            "max_tokens": 1,
        }
        if system_prompt:
            payload["system"] = system_prompt
        try:
            resp = _HTTP.post(
                f"{self._url}/messages",
                json=payload,
                headers=self._headers(),
                timeout=60,
            )
            resp.raise_for_status()
            print(f"[Anthropic] '{self._model}' ready.")
            return True
        except Exception as e:
            print(f"[Anthropic] Warmup failed (non-fatal): {e}")
            return False

    def chat(
        self,
        messages: list,
        tools: list | None = None,
        timeout: int = 120,
    ) -> dict:
        system = self._extract_system(messages)
        converted = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools)
        payload: dict = {
            "model":      self._model,
            "messages":   converted,
            "max_tokens": 4096,
        }
        if system:
            payload["system"] = system
        if anthropic_tools:
            payload["tools"] = anthropic_tools

        try:
            resp = _HTTP.post(
                f"{self._url}/messages",
                json=payload,
                headers=self._headers(),
                timeout=timeout,
            )
            resp.raise_for_status()
            content, tool_calls = self._parse_response(resp.json())
            return {"content": content, "tool_calls": tool_calls}
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"Cannot reach Anthropic API at {self._url}.")
        except requests.exceptions.Timeout:
            raise RuntimeError("Anthropic request timed out.")
        except requests.exceptions.HTTPError as e:
            body = _error_body(e.response)
            raise RuntimeError(f"Anthropic HTTP error: {e.response.status_code} — {body}")
        except Exception as e:
            raise RuntimeError(f"Anthropic call failed: {e}")

    def chat_stream(
        self,
        messages: list,
        tools: list | None = None,
        timeout: int = 120,
        cancel_event: threading.Event | None = None,
    ) -> Generator[dict, None, None]:
        system = self._extract_system(messages)
        converted = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools)
        payload: dict = {
            "model":      self._model,
            "messages":   converted,
            "max_tokens": 4096,
            "stream":     True,
        }
        if system:
            payload["system"] = system
        if anthropic_tools:
            payload["tools"] = anthropic_tools

        try:
            with _HTTP.post(
                f"{self._url}/messages",
                json=payload,
                headers=self._headers(),
                timeout=timeout,
                stream=True,
            ) as resp:
                resp.raise_for_status()
                full_content = ""
                buf = ""
                buffer: dict = {"content_delta": "", "tc_blocks": {}, "stop_reason": None}

                for raw in resp.iter_lines():
                    if cancel_event and cancel_event.is_set():
                        resp.close()
                        if buf.strip():
                            yield {"type": "sentence", "text": buf.strip()}
                        tool_calls = self._build_tool_calls_from_buffer(buffer)
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
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    done = self._parse_stream_event(event, buffer)
                    delta = buffer["content_delta"]
                    if delta:
                        full_content += delta
                        buf += delta
                        buffer["content_delta"] = ""

                        while True:
                            m = _SENT_END.search(buf)
                            if not m:
                                break
                            sentence = buf[: m.start() + 1].strip()
                            buf = buf[m.end():]
                            if sentence:
                                yield {"type": "sentence", "text": sentence}

                    if done:
                        if buf.strip():
                            yield {"type": "sentence", "text": buf.strip()}
                        tool_calls = self._build_tool_calls_from_buffer(buffer)
                        yield {
                            "type":       "done",
                            "content":    full_content.strip(),
                            "tool_calls": tool_calls,
                        }
                        return

                if buf.strip():
                    yield {"type": "sentence", "text": buf.strip()}
                tool_calls = self._build_tool_calls_from_buffer(buffer)
                yield {
                    "type":       "done",
                    "content":    full_content.strip(),
                    "tool_calls": tool_calls,
                }

        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"Cannot reach Anthropic API at {self._url}.")
        except requests.exceptions.Timeout:
            raise RuntimeError("Anthropic stream timed out.")
        except requests.exceptions.HTTPError as e:
            body = _error_body(e.response)
            raise RuntimeError(f"Anthropic HTTP error: {e.response.status_code} — {body}")
        except Exception as e:
            raise RuntimeError(f"Anthropic stream failed: {e}")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ProviderRegistry:
    """Maps provider names to implementations and supports fallback chains."""

    _PROVIDERS: dict[str, type[LLMProvider]] = {
        "ollama":     OllamaProvider,
        "kaggle":     KaggleProvider,
        "openai":     OpenAIProvider,
        "groq":       GroqProvider,
        "cloudflare": CloudflareProvider,
        "nvidia":     NVIDIAProvider,
        "kilo":       KiloProvider,
        "gemini":     GeminiProvider,
        "anthropic":  AnthropicProvider,
    }

    def __init__(self):
        self._instances: dict[str, LLMProvider] = {}

    @classmethod
    def register_provider(cls, name: str, provider_cls: type[LLMProvider]) -> None:
        cls._PROVIDERS[name] = provider_cls

    @classmethod
    def list_providers(cls) -> list[str]:
        extra = []
        try:
            from core.gemini_live_provider import GeminiLiveProvider  # noqa: F811
            extra.append("gemini_live")
        except ImportError:
            pass
        return list(cls._PROVIDERS.keys()) + extra

    def _resolve_gemini_live(self) -> type[LLMProvider]:
        try:
            from core.gemini_live_provider import GeminiLiveProvider
            return GeminiLiveProvider
        except ImportError:
            raise ValueError(
                "GeminiLiveProvider requires google-genai. "
                "Install: pip install google-genai"
            )

    def get_provider(self, name: str, **kwargs: Any) -> LLMProvider:
        name = name.strip().lower()
        if name == "gemini_live":
            provider_cls = self._resolve_gemini_live()
        elif name not in self._PROVIDERS:
            raise ValueError(
                f"Unknown provider '{name}'. Available: {', '.join(self._PROVIDERS)}"
            )
        else:
            provider_cls = self._PROVIDERS[name]
        if name not in self._instances:
            self._instances[name] = provider_cls(**kwargs)
        elif name == "gemini_live":
            # Its session is intentionally persistent, so update its own
            # settings instead of silently retaining old Live credentials.
            self._instances[name].reconfigure(**kwargs)
        return self._instances[name]

    def get_provider_with_fallback(
        self,
        primary: str,
        fallbacks: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMProvider:
        """Return a *FallbackProvider* that tries primary then each fallback."""
        if fallbacks is None:
            fallbacks = []
        chain = [primary] + fallbacks
        providers = {name: self.get_provider(name, **kwargs) for name in chain}
        return FallbackProvider(providers, chain)


class FallbackProvider(LLMProvider):
    """Wraps multiple providers and tries each in order on failure."""

    def __init__(self, providers: dict[str, LLMProvider], chain: list[str]):
        self._providers = providers
        self._chain = chain

    def _try_all(self, method: str, *args: Any, **kwargs: Any) -> Any:
        last_error: Exception | None = None
        for name in self._chain:
            provider = self._providers[name]
            try:
                result = getattr(provider, method)(*args, **kwargs)
                return result
            except Exception as e:
                print(f"[Fallback] {name} failed: {e}")
                last_error = e
        raise RuntimeError(
            f"All providers in chain {self._chain} failed. Last error: {last_error}"
        )

    def chat(self, messages: list, tools: list | None = None, timeout: int = 120) -> dict:
        return self._try_all("chat", messages, tools=tools, timeout=timeout)

    def chat_stream(
        self,
        messages: list,
        tools: list | None = None,
        timeout: int = 120,
        cancel_event: threading.Event | None = None,
    ) -> Generator[dict, None, None]:
        return self._try_all("chat_stream", messages, tools=tools, timeout=timeout, cancel_event=cancel_event)

    def warmup(self, system_prompt: str | None = None) -> bool:
        for name in self._chain:
            provider = self._providers[name]
            try:
                if provider.warmup(system_prompt):
                    return True
            except Exception as e:
                print(f"[Fallback] {name} warmup failed: {e}")
        return False

    def ensure_running(self) -> bool:
        for name in self._chain:
            provider = self._providers[name]
            try:
                if provider.ensure_running():
                    return True
            except Exception as e:
                print(f"[Fallback] {name} ensure_running failed: {e}")
        return False

    def get_settings(self) -> tuple[str, str]:
        return self._providers[self._chain[0]].get_settings()


# ---------------------------------------------------------------------------
# Default provider factory
# ---------------------------------------------------------------------------

_registry: ProviderRegistry | None = None


def get_registry() -> ProviderRegistry:
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry


def get_default_provider(**kwargs: Any) -> LLMProvider:
    """Read config and return the appropriate provider instance.

    Config key ``llm_provider`` selects the backend:
      "ollama", "openai", "groq", "cloudflare", "gemini", "anthropic".

    Optional kwargs override config values (url, model, api_key, …).
    """
    cfg = _load_config()
    name = (kwargs.pop("provider", None) or cfg.get("llm_provider", "ollama")).strip().lower()

    # Normalise legacy / api-compatible names
    if name == "openrouter":
        kwargs.setdefault("url", "https://openrouter.ai/api")
        kwargs.setdefault("model", cfg.get("or_model", ""))
        kwargs.setdefault("api_key", cfg.get("or_api_key", ""))
        name = "openai"

    legacy_map = {
        "lmstudio": "openai", "localai": "openai", "jan": "openai", "llamacpp": "openai",
    }
    name = legacy_map.get(name, name)

    fallbacks_str: str = cfg.get("llm_fallbacks", "")
    fallbacks = [f.strip().lower() for f in fallbacks_str.split(",") if f.strip()]

    registry = get_registry()
    if fallbacks:
        return registry.get_provider_with_fallback(name, fallbacks, **kwargs)
    return registry.get_provider(name, **kwargs)
