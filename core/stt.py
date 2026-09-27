"""
Speech-to-Text engines for Orthos.

Whisper  – offline transcription via faster-whisper (VAD-buffered)
Vosk     – offline streaming transcription (lighter)
Gemini   – cloud transcription via Google Gemini API (no local model)
Hashim   – vendored Gemini Live streaming engine (server-side VAD/AEC;
           thin adapter over vendor/hashim_stt — see that README)
"""
import io
import json
import queue

import numpy as np
import soundfile as sf


class GroqSTT:
    """Cloud STT via Groq Whisper API. No local model needed — ideal for low-resource systems.

    Key handling (2026-09 hardening): a 401 here used to be a mystery.
    Root causes found in the field:
      1. An environment variable ``GROQ_API_KEY`` (from a terminal, IDE or
         system settings) silently OVERRIDES the key pasted in Configure —
         ``main._load_config`` prefers env vars. If that env var holds an
         old/revoked key, every fresh key the user pastes still gets 401.
         → ``__init__`` now rejects malformed keys loudly at startup.
      2. Wrong key pasted into the wrong field (Groq LLM key exists as a
         separate ``groq_llm_key``). Both are gsk_… so they look alike.
      3. Key truncated by copy (a real key is ~56 chars: ``gsk_`` + 52).
    ``_resolve_model`` asks the API which audio models the key can see
    (cached), so a deprecated model name never breaks the engine.
    """

    # Newest-first. whisper-large-v3 and whisper-large-v3-turbo are the
    # documented STT models; resolved dynamically at first call anyway.
    _MODEL_PREFS = ["whisper-large-v3-turbo", "whisper-large-v3"]
    _resolved_model: str | None = None

    def __init__(self, api_key: str, language: str = "auto"):
        from groq import Groq
        key = (api_key or "").strip()
        if not key:
            raise RuntimeError(
                "Groq STT needs an API key. Open CONFIGURE → Speech-to-Text → "
                "Groq and paste a key from console.groq.com/keys (starts with gsk_)."
            )
        if not key.startswith("gsk_"):
            raise RuntimeError(
                "This does not look like a Groq key (should start with 'gsk_'). "
                "You may have pasted it into the wrong field — Groq has TWO key "
                "fields: 'Groq API key' (STT, Configure → STT) and 'Groq LLM key' "
                "(LLM section). Get a key from console.groq.com/keys."
            )
        if len(key) < 40:
            raise RuntimeError(
                f"Groq key looks truncated ({len(key)} chars — real keys are ~56). "
                "Copy it again from console.groq.com/keys."
            )
        # Env-var info: _load_config now gives the JSON file priority, so an
        # env var only matters when no key was saved in the UI. Log the source
        # so "which key am I actually using?" is always answerable.
        import os as _os
        env_key = _os.environ.get("GROQ_API_KEY", "").strip()
        if env_key and env_key != key:
            print(
                "[STT] Note: GROQ_API_KEY is also set in the environment with a "
                "different value — the key saved in CONFIGURE takes priority."
            )
        self._client = Groq(api_key=key)
        self._language = language if language and language.strip().lower() != "auto" else None

    def _resolve_model(self) -> str:
        """Pick the best STT model this key can actually see (cached)."""
        if GroqSTT._resolved_model:
            return GroqSTT._resolved_model
        try:
            ids = [m.id for m in self._client.models.list().data]
            audio = [i for i in ids if "whisper" in i.lower()]
            for cand in self._MODEL_PREFS:
                if cand in audio:
                    GroqSTT._resolved_model = cand
                    print(f"[STT] Groq STT model: {cand}")
                    return cand
            if audio:
                GroqSTT._resolved_model = audio[0]
                print(f"[STT] Groq STT model (fallback): {audio[0]}")
                return audio[0]
        except Exception as e:
            print(f"[STT] Groq model list unavailable ({e}) — using preference list")
        GroqSTT._resolved_model = self._MODEL_PREFS[0]
        return GroqSTT._resolved_model

    def transcribe(self, audio: np.ndarray) -> str:
        buf = io.BytesIO()
        sf.write(buf, audio, 16000, format="WAV")
        buf.seek(0)

        kwargs = dict(
            model=self._resolve_model(),
            file=("audio.wav", buf),
        )
        if self._language:
            kwargs["language"] = self._language

        response = self._client.audio.transcriptions.create(**kwargs)
        return response.text.strip()


class GeminiSTT:
    """Cloud STT via Google Gemini API. No local model needed — ideal for low-resource systems.

    Model handling (2026-09): Google rotates Gemini models aggressively —
    the old hardcoded ``gemini-2.0-flash-001`` was SHUT DOWN on 2026-06-01,
    which made this engine fail forever. Instead of pinning one name, the
    engine asks the API which models the key can actually see (cached for
    the process lifetime) and picks the newest audio-capable Flash, with a
    static preference list as fast path. If the chosen model 404s at
    request time (deprecation can land mid-run), the next candidate is
    tried automatically and the cache is invalidated.
    """

    # Newest-first; anything the key can see is fine. Flash is preferred
    # for STT: cheap, fast, and audio-in capable on every release so far.
    _MODEL_PREFS = [
        "gemini-3.5-flash",
        "gemini-3-flash",
        "gemini-3.1-flash-lite",
        "gemini-3-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-flash-latest",          # stable alias if Google offers it
    ]
    _resolved_model: str | None = None          # process-wide cache
    _exhausted: dict = {}                       # model -> date (free-tier per-day quota)

    def __init__(self, api_key: str, language: str = "auto"):
        if not api_key:
            raise RuntimeError(
                "Gemini STT requires 'gemini_voice_api_key' — "
                "set it in Configure → Text-to-Speech → Voice API Key."
            )
        key = api_key.strip()
        # Fail loudly on obviously-wrong keys instead of a mid-conversation
        # 401. Google currently issues TWO valid key families:
        #   AIza…  (μ39 chars, AI Studio)  and  AQ.… (53 chars,
        #   Vertex AI express mode) — both accepted, both verified working.
        # Only a Groq key (gsk_…) is a certain wrong-paste; unknown prefixes
        # only warn so future Google key formats keep working.
        if key.startswith("gsk_"):
            raise RuntimeError(
                "A Groq key (gsk_…) was pasted into the Gemini field. Gemini "
                "keys come from aistudio.google.com/apikey — paste it in "
                "Configure → Speech-to-Text → Gemini."
            )
        if not (key.startswith("AIza") or key.startswith("AQ.")):
            print(f"[STT] Note: Gemini key has unusual prefix "
                  f"{key[:3]!r}… — accepting anyway (Google issues several "
                  "key formats).")
        if len(key) < 30:
            # Warn only — Google key lengths vary across families and CI
            # wiring passes short dummy keys; the API itself 401s on a truly
            # truncated key and transcribe() surfaces a clear message then.
            print(f"[STT] Note: Gemini key is unusually short ({len(key)} "
                  "chars) — if auth fails, re-copy it from "
                  "aistudio.google.com/apikey.")
        from google import genai
        self._client = genai.Client(api_key=key)
        self._language = language

    def _list_audio_models(self) -> set[str]:
        """Model names this key can actually use, best-effort."""
        try:
            names: set[str] = set()
            for m in self._client.models.list():
                name = getattr(m, "name", "") or ""
                name = name.removeprefix("models/")
                actions = getattr(m, "supported_actions", None) or getattr(
                    m, "supported_generation_methods", None) or []
                # audio-capable generateContent models; keep flash variants only
                if ("generateContent" in actions or not actions) and "flash" in name:
                    names.add(name)
            return names
        except Exception as e:
            print(f"[STT] Gemini model list unavailable ({e}) — using preference list")
            return set()

    def _pick_model(self) -> str:
        import datetime as _dt
        today = _dt.date.today().isoformat()
        GeminiSTT._exhausted = {
            m: d for m, d in GeminiSTT._exhausted.items() if d == today
        }
        if (GeminiSTT._resolved_model
                and GeminiSTT._resolved_model not in GeminiSTT._exhausted):
            return GeminiSTT._resolved_model
        available = self._list_audio_models()
        for cand in self._MODEL_PREFS:
            if not available or cand in available:
                GeminiSTT._resolved_model = cand
                print(f"[STT] Gemini STT model: {cand}")
                return cand
        # Fall back to whatever flash model the key exposes, newest-looking first
        for name in sorted(available, reverse=True):
            GeminiSTT._resolved_model = name
            print(f"[STT] Gemini STT model (fallback): {name}")
            return name
        raise RuntimeError(
            "No Gemini Flash model available to this API key — check the key "
            "at aistudio.google.com/apikey or switch STT engine."
        )

    def transcribe(self, audio: np.ndarray) -> str:
        from google.genai import types
        buf = io.BytesIO()
        sf.write(buf, audio, 16000, format="WAV")
        wav_bytes = buf.getvalue()

        tried: list[str] = []
        for attempt in range(3):
            model = self._pick_model()
            try:
                response = self._client.models.generate_content(
                    model=model,
                    contents=[
                        types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
                        "Transcribe this audio exactly. Return only the transcribed text, nothing else.",
                    ]
                )
                GeminiSTT._resolved_model = model   # worked — remember it
                # No-speech audio yields response.text == None; that is an
                # empty transcript, not an error (the old .strip() on None
                # crashed the mic loop's turn handling).
                return (getattr(response, "text", None) or "").strip()
            except Exception as e:
                msg = str(e)
                if ("401" in msg or "403" in msg
                        or "API_KEY_INVALID" in msg or "api key" in msg.lower()):
                    raise RuntimeError(
                        "Gemini rejected the API key (401/403). Get a fresh "
                        "key from aistudio.google.com/apikey and paste it in "
                        "Configure → Speech-to-Text → Gemini."
                    ) from e
                retriable = (
                    "404" in msg or "not found" in msg.lower()
                    or "429" in msg or "quota" in msg.lower()
                    or "503" in msg or "unavailable" in msg.lower()
                )
                if not retriable or attempt == 2:
                    raise
                if "404" in msg or "not found" in msg.lower():
                    GeminiSTT._resolved_model = None    # model died — re-resolve
                    if model in self._MODEL_PREFS:
                        self._MODEL_PREFS.remove(model) # don't pick it again
                    tried.append(model)
                    print(f"[STT] Gemini model '{model}' unavailable — trying next…")
                elif "429" in msg and "perday" in msg.replace("_", "").lower():
                    # Free-tier quota is PER MODEL per day (e.g. 20/day on the
                    # newest flash). Sleeping cannot fix that today — rotate
                    # to the next model, which has its own fresh quota.
                    import datetime as _dt
                    GeminiSTT._exhausted[model] = _dt.date.today().isoformat()
                    GeminiSTT._resolved_model = None
                    GeminiSTT._MODEL_PREFS = [
                        m for m in GeminiSTT._MODEL_PREFS if m != model
                    ] + [model]                     # deprioritise, keep for tomorrow
                    print(f"[STT] Gemini '{model}' daily quota used — rotating model")
                else:
                    import time as _t
                    _t.sleep(1.5 * (attempt + 1))       # quota/503 — brief backoff
        raise RuntimeError(f"Gemini STT failed after retries (tried: {tried})")


class WhisperSTT:
    """Offline transcription using faster-whisper."""

    def __init__(self, model_name: str = "base", language: str | None = None):
        import os
        from faster_whisper import WhisperModel
        print(f"[STT] Loading Whisper '{model_name}'…")
        try:
            import torch
            device  = "cuda" if torch.cuda.is_available() else "cpu"
            compute = "float16" if device == "cuda" else "int8"
        except Exception:
            device, compute = "cpu", "int8"

        try:
            self._model = WhisperModel(model_name, device=device, compute_type=compute)
        except Exception as _first_err:
            # Offline flag set but model not cached yet → download once, then offline forever
            _e = str(_first_err).lower()
            if any(k in _e for k in ("offline", "not found", "cache", "localentry", "does not exist")):
                print(f"[STT] '{model_name}' not cached — downloading (internet required for first run)…")
                os.environ.pop("HF_HUB_OFFLINE",      None)
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
                os.environ.pop("HF_DATASETS_OFFLINE",  None)
                self._model = WhisperModel(model_name, device=device, compute_type=compute)
            else:
                raise

        self._language = None if (not language or language.strip().lower() == "auto") else language.strip().lower()
        print(f"[STT] Whisper '{model_name}' ready ({device})")

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a float32 mono 16 kHz numpy array. Returns transcript string."""
        try:
            segments, _ = self._model.transcribe(
                audio,
                language=self._language,
                beam_size=1,
                best_of=1,
                condition_on_previous_text=True,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 700},
            )
            return " ".join(s.text for s in segments).strip()
        except Exception as e:
            print(f"[STT] Transcription error: {e}")
            raise


class VoskSTT:
    """Streaming transcription using Vosk."""

    def __init__(self, model_path: str | None = None, language: str = "en-us"):
        from vosk import Model, KaldiRecognizer
        print("[STT] Loading Vosk model…")
        if model_path:
            model = Model(model_path)
        else:
            lang  = language.strip().lower() if language and language.strip().lower() != "auto" else "en-us"
            model = Model(lang=lang)
        self._rec = KaldiRecognizer(model, 16000)
        print("[STT] Vosk ready.")

    def process_chunk(self, audio_bytes: bytes) -> tuple[str, bool]:
        """Feed raw int16 LE PCM bytes. Returns (text, is_final)."""
        if self._rec.AcceptWaveform(audio_bytes):
            result = json.loads(self._rec.Result())
            return result.get("text", ""), True
        partial = json.loads(self._rec.PartialResult())
        return partial.get("partial", ""), False


class HashimSTT:
    """Thin adapter around the VENDORED Hashim Malik engine.

    The real implementation lives, unmodified, in
    ``vendor/hashim_stt/stt_engine.py`` (gemini-realtime-speech-to-text-
    translator, https://github.com/hashimmalikdev/…). This adapter does
    NOT reimplement it: ``connect()`` starts the upstream class's own
    thread + mic + Gemini Live session, and sentences arrive through the
    upstream ``sentence_queue``.

    Concurrency note: the upstream ``listen()`` blocks forever on an
    empty queue, so this adapter never calls it on the orchestrating
    thread — it drains ``sentence_queue`` directly, non-blocking.

    The engine's session drops silently on some network failures
    (upstream catches all exceptions in its receive loop), so
    ``is_alive()`` tracks whether the upstream thread is still running
    and lets the caller restart or fall back.
    """

    def __init__(self, api_key: str, silence_threshold: float = 1.2,
                 target_lang: str | None = None):
        import os as _os
        import sys as _sys
        key = (api_key or "").strip()
        if not key:
            raise RuntimeError(
                "Hashim STT needs a Gemini API key — set it in "
                "Configure → Text-to-Speech → Voice API Key."
            )
        if key.startswith("gsk_"):
            raise RuntimeError(
                "A Groq key (gsk_…) was pasted into the Gemini field. "
                "Hashim STT uses Gemini Live — get a key from "
                "aistudio.google.com/apikey and paste it in Configure → "
                "Speech-to-Text → Gemini (Voice API Key)."
            )
        # The upstream engine reads the key from its constructor now; the
        # env export stays for any code path still reading GEMINI_API_KEY.
        _os.environ["GEMINI_API_KEY"] = key
        if "vendor" not in _sys.path:
            _sys.path.insert(0, "vendor")
        try:
            from vendor.hashim_stt.stt_engine import STT as _UpstreamSTT
        except ImportError:
            # Package context may not resolve in some launch layouts.
            from hashim_stt.stt_engine import STT as _UpstreamSTT
        self._engine = _UpstreamSTT(
            silence_threshold=silence_threshold,
            api_key=key,
            target_lang=target_lang or None,
        )
        self._closed = False

    def connect(self) -> None:
        """Start the upstream engine's own thread + mic + live session."""
        self._engine.connect()

    def set_gate(self, fn) -> None:
        """Forward a speaking-gate to the upstream engine: while fn() is
        True its mic frames are dropped (host TTS + room tail), so the
        live session never hears Orthos's own voice."""
        try:
            self._engine.set_gate(fn)
        except Exception:
            pass

    def close(self) -> None:
        self._closed = True
        try:
            stop = getattr(self._engine, "stop", None)
            if callable(stop):
                stop()          # full teardown: mic + live session + loop
            else:
                self._engine.close()
        except Exception:
            pass

    def is_alive(self) -> bool:
        """True while the upstream engine's thread runs and no drain was requested."""
        if self._closed:
            return False
        t = getattr(self._engine, "_thread", None)
        return bool(t and t.is_alive())

    def drain_sentences(self, max_items: int = 10) -> list[str]:
        """Pop every finalized sentence the upstream engine has queued."""
        out: list[str] = []
        q = self._engine.sentence_queue
        while len(out) < max_items:
            try:
                out.append(q.get_nowait())
            except queue.Empty:
                break
        return out

    def last_error(self) -> BaseException | None:
        """The real exception that killed the upstream thread, if any.

        The upstream engine catches everything inside its receive/audio
        tasks, so a bare ``is_alive() == False`` says only 'thread died' —
        this returns the underlying cause (bad config, rejected key,
        network drop, …) for the UI.
        """
        return getattr(self._engine, "last_error", None)
