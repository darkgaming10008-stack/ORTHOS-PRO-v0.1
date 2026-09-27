"""
Text-to-Speech engines for Orthos.

EdgeTTS     – free Microsoft TTS (internet required, no API key)
Kokoro      – fully offline neural TTS (~330 MB model)
ElevenLabs  – cloud API (API key required, best quality)
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import os
import queue as _queue
import re
import threading
import time
from typing import Callable, Optional

import miniaudio
import numpy as np

import requests as _requests
_HTTP = _requests.Session()
import sounddevice as sd



# USE_TF=0 stops transformers from importing TensorFlow (saves 4-8 s startup).
# Do NOT set USE_TORCH or USE_JAX explicitly — forcing those values breaks
# transformers' lazy-loader on certain versions, causing AutoModel and other
# classes to vanish from the public namespace.  Auto-detection is reliable.
os.environ.setdefault("USE_TF",                 "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ---------------------------------------------------------------------------
# Audio playback helpers
# ---------------------------------------------------------------------------

def _to_numpy(samples) -> np.ndarray:
    """Convert samples to float32 numpy array.

    Handles both numpy arrays and PyTorch tensors (Kokoro >= 0.9).

    PyTorch built against numpy 1.x raises RuntimeError('Numpy is not available')
    when numpy 2.x is installed.  The .tolist() fallback always works regardless
    of PyTorch / numpy version pairing.
    """
    if hasattr(samples, "detach"):                  # PyTorch tensor
        t = samples.detach().cpu().float()
        try:
            return t.numpy()                        # fast path (compatible versions)
        except RuntimeError:
            # PyTorch/numpy version mismatch — convert via Python list (always safe)
            return np.asarray(t.tolist(), dtype=np.float32)
    return np.asarray(samples, dtype=np.float32)


def _compress_silence(
    arr: np.ndarray,
    sample_rate: int    = 24_000,
    max_silence_ms: int = 500,    # cap punctuation pauses — keeps natural rhythm
    threshold: float    = 0.003,  # RMS below this = silence; lower = less clipping
) -> np.ndarray:
    """
    Shorten Kokoro's very long punctuation pauses (1-2 s → ≤500 ms).
    Conservative settings preserve natural prosody; only trims extreme pauses.
    """
    max_samp  = int(max_silence_ms * sample_rate / 1000)
    frame_len = 240                   # ~10 ms at 24 kHz
    out: list[np.ndarray] = []
    silent_acc = 0

    for i in range(0, len(arr), frame_len):
        chunk = arr[i : i + frame_len]
        if np.sqrt(np.mean(chunk ** 2) + 1e-12) < threshold:
            silent_acc += len(chunk)
            if silent_acc <= max_samp:
                out.append(chunk)
        else:
            silent_acc = 0
            out.append(chunk)

    return np.concatenate(out) if out else arr


def _play_np(samples, sample_rate: int) -> None:
    """Play float32 mono (or stereo) audio via sounddevice.
    Accepts numpy arrays or PyTorch tensors.
    """
    sd.play(_to_numpy(samples), sample_rate)
    sd.wait()


def _decode_to_np(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    decoded = miniaudio.decode(
        audio_bytes,
        output_format=miniaudio.SampleFormat.FLOAT32,
        nchannels=1,
    )
    return np.array(decoded.samples, dtype=np.float32), decoded.sample_rate


def _play_audio_bytes(audio_bytes: bytes) -> None:
    """Decode MP3/WAV/OGG bytes and play via sounddevice (uses miniaudio)."""
    samples, sr = _decode_to_np(audio_bytes)
    sd.play(samples, sr)
    sd.wait()


_GTT_QUOTA_MARKERS = (
    "quota",
    "RESOURCE_EXHAUSTED",
    "exceeded your current quota",
    "quotaValue",
    "QuotaFailure",
    "1011",
    "429",
)


def _is_gtt_quota_error(msg: str) -> bool:
    low = msg.lower()
    return any(m.lower() in low for m in _GTT_QUOTA_MARKERS)


class GTTSEngine:
    """Google Translate TTS – free, no API key, literal 'Google voice'.
    
    Uses the gTTS library to interface with the undocumented Translate API.
    """

    def __init__(self, voice: str = "hi"):
        self.voice = voice
        self._sample_rate = 24000

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def stop(self) -> None:
        sd.stop()

    def synthesize(self, text: str) -> np.ndarray:
        from gtts import gTTS
        import tempfile
        import pathlib

        temp_path = None
        try:
            # tld='co.in' ensures the natural Indian accent for Hindi/English mixing
            tts = gTTS(text=text, lang=self.voice, tld="co.in")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                temp_path = fp.name
                tts.save(fp.name)
            
            with open(temp_path, "rb") as f:
                audio_bytes = f.read()
            samples, sr = _decode_to_np(audio_bytes)
            self._sample_rate = sr
            return samples
        except Exception as e:
            msg = str(e)
            if _is_gtt_quota_error(msg):
                raise RuntimeError(
                    "gTTS quota exceeded or blocked by Google Translate. "
                    "Falling back to alternative TTS engine."
                ) from e
            raise RuntimeError(f"gTTS synthesis failed: {msg}") from e
        finally:
            if temp_path:
                path = pathlib.Path(temp_path)
                if path.exists():
                    path.unlink()

    def speak(self, text: str) -> None:
        audio = self.synthesize(text)
        if audio.size > 0:
            sd.play(audio, self._sample_rate)
            sd.wait()

class EdgeTTSEngine:
    """Microsoft EdgeTTS – free, requires internet."""

    def __init__(self, voice: str = "en-US-GuyNeural"):
        self.voice = voice
        self._sample_rate = 24000

    def stop(self) -> None:
        sd.stop()

    async def _synth(self, text: str) -> bytes:
        import edge_tts
        comm = edge_tts.Communicate(text, self.voice)
        buf = bytearray()
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                buf.extend(chunk["data"])
        return bytes(buf)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def synthesize(self, text: str) -> np.ndarray:
        loop = asyncio.new_event_loop()
        try:
            audio_bytes = loop.run_until_complete(self._synth(text))
        finally:
            loop.close()
        if not audio_bytes:
            return np.array([], dtype=np.float32)
        samples, sr = _decode_to_np(audio_bytes)
        self._sample_rate = sr
        return samples

    def speak(self, text: str) -> None:
        audio = self.synthesize(text)
        if audio.size > 0:
            sd.play(audio, self._sample_rate)
            sd.wait()


# ---------------------------------------------------------------------------
# Kokoro import helper — auto-upgrades on version-mismatch errors
# ---------------------------------------------------------------------------

# Errors that indicate the installed kokoro uses old transformers classes
# (AlbertModel, AutoModel) that are no longer exported at the top level.
_KOKORO_COMPAT_ERRORS = ("AlbertModel", "AutoModel", "cannot import name")


def _import_kokoro_pipeline():
    """Import KPipeline, auto-upgrading kokoro if a version mismatch is found.

    Old kokoro (<0.9) imports AlbertModel / AutoModel from transformers.
    Newer transformers versions no longer export these at the top level,
    causing an ImportError.  kokoro>=0.9 removed these dependencies.

    When the error is detected we:
      1. Upgrade kokoro to >=0.9 via pip (silent, background)
      2. Flush stale kokoro entries from sys.modules
      3. Re-import — this time it should succeed
    """
    import sys

    def _try_import():
        from kokoro import KPipeline  # noqa: PLC0415
        return KPipeline

    try:
        return _try_import()
    except Exception as first_err:
        err_msg = str(first_err)
        if not any(marker in err_msg for marker in _KOKORO_COMPAT_ERRORS):
            # Unrelated error (kokoro not installed, etc.)
            raise RuntimeError(
                f"Kokoro import failed: {first_err}\n"
                "Run: pip install kokoro>=0.9 soundfile"
            ) from first_err

        # ── Version mismatch: upgrade kokoro silently and retry ──────────
        print("[TTS] Kokoro/transformers version mismatch detected — upgrading kokoro…")
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "kokoro>=0.9",
             "--upgrade", "--quiet", "--disable-pip-version-check"],
            capture_output=True,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace").strip()
            raise RuntimeError(
                f"Kokoro auto-upgrade failed: {stderr[:200]}\n"
                "Run manually: pip install kokoro>=0.9 soundfile"
            ) from first_err

        # Flush any stale kokoro submodules from the import cache
        stale = [k for k in sys.modules if k == "kokoro" or k.startswith("kokoro.")]
        for key in stale:
            del sys.modules[key]

        print("[TTS] Kokoro upgraded — retrying import…")
        try:
            return _try_import()
        except Exception as retry_err:
            raise RuntimeError(
                f"Kokoro still broken after upgrade: {retry_err}\n"
                "Run manually: pip install --upgrade kokoro transformers"
            ) from retry_err


# Kokoro voice prefix → KPipeline lang_code mapping
_KOKORO_LANG_CODES = {
    "a": "a",   # American English  (af_*, am_*)
    "b": "b",   # British English   (bf_*, bm_*)
    "j": "j",   # Japanese          (jf_*, jm_*)
    "z": "z",   # Mandarin Chinese  (zf_*, zm_*)
    "s": "s",   # Spanish           (sf_*, sm_*)
    "f": "f",   # French            (ff_*, fm_*)
    "h": "h",   # Hindi             (hf_*, hm_*)
    "i": "i",   # Italian           (if_*, im_*)
    "p": "p",   # Brazilian Portuguese
    "r": "r",   # Russian           (rf_*, rm_*)
    "e": "e",   # German            (ef_*, em_*)
}


class KokoroTTSEngine:
    """Fully offline Kokoro neural TTS.

    Model (~330 MB) is downloaded from HuggingFace on first use,
    then cached locally — subsequent starts load from disk.

    Warmup strategy: _init() runs synchronously in the background
    _do_tts() thread (not the UI thread).  After the pipeline loads,
    a dummy inference compiles the PyTorch JIT graph immediately so
    the first real speak() call has zero compilation overhead.
    """

    def __init__(self, voice: str = "af_heart", speed: float = 1.0):
        self.voice     = voice
        self.speed     = speed
        self._pipeline = None
        self._lock     = threading.Lock()
        self._init()   # blocking, but called from background thread

    @property
    def _lang_code(self) -> str:
        prefix = self.voice[0].lower() if self.voice else "a"
        return _KOKORO_LANG_CODES.get(prefix, "a")

    def _init(self) -> None:
        if self._pipeline is not None:
            return

        lang = self._lang_code

        # Prefer GPU — Kokoro on CUDA is ~10x faster than CPU.
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                import os as _os
                n_threads = max(1, min(4, (_os.cpu_count() or 4) // 2))
                try:
                    torch.set_num_threads(n_threads)
                    torch.set_num_interop_threads(2)
                except RuntimeError:
                    pass
                print(
                    f"[TTS] Kokoro on CPU — for faster speech install CUDA PyTorch:\n"
                    "      pip install torch --index-url https://download.pytorch.org/whl/cu118"
                )
        except Exception:
            device = "cpu"

        print(f"[TTS] Kokoro — loading (lang='{lang}', device='{device}')…")

        KPipeline = _import_kokoro_pipeline()

        def _create_pipeline():
            try:
                return KPipeline(lang_code=lang, device=device)
            except TypeError:
                return KPipeline(lang_code=lang)   # older build — no device param

        try:
            self._pipeline = _create_pipeline()
        except Exception as _first_err:
            # Offline flag set but model not cached yet → download once
            _e = str(_first_err).lower()
            if any(k in _e for k in ("offline", "not found", "cache", "localentry", "does not exist")):
                print("[TTS] Kokoro model not cached — downloading (internet required for first run)…")
                os.environ.pop("HF_HUB_OFFLINE",      None)
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
                os.environ.pop("HF_DATASETS_OFFLINE",  None)
                self._pipeline = _create_pipeline()
            else:
                raise

        print("[TTS] Kokoro compiling (first-time only)…")
        # Warmup: compiles PyTorch JIT graph so first real speak() call is instant.
        try:
            for _ in self._pipeline("hello", voice=self.voice, speed=self.speed):
                pass
            print("[TTS] Kokoro ready.")
        except Exception as e:
            print(f"[TTS] Kokoro warmup warning: {e}")

    @property
    def sample_rate(self) -> int:
        return 24000

    def synthesize(self, text: str) -> np.ndarray:
        with self._lock:
            if self._pipeline is None:
                self._init()
        chunks: list[np.ndarray] = []
        for _, _, audio in self._pipeline(text, voice=self.voice, speed=self.speed):
            if audio is not None:
                arr = _to_numpy(audio)
                arr = _compress_silence(arr)
                if arr.size > 0:
                    chunks.append(arr)
        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks)

    def speak(self, text: str) -> None:
        audio = self.synthesize(text)
        if audio.size > 0:
            _play_np(audio, 24000)


class ElevenLabsTTSEngine:
    """ElevenLabs cloud TTS – API key required."""

    def __init__(self, api_key: str, voice_id: str = "pNInz6obpgDQGcFmaJgB"):
        self.api_key  = api_key
        self.voice_id = voice_id
        self._sample_rate = 44100

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def synthesize(self, text: str) -> np.ndarray:
        headers = {
            "xi-api-key":   self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "text":     text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }
        resp = _HTTP.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}",
            json=payload, headers=headers, timeout=30,
        )
        resp.raise_for_status()
        samples, sr = _decode_to_np(resp.content)
        self._sample_rate = sr
        return samples

    def speak(self, text: str) -> None:
        audio = self.synthesize(text)
        if audio.size > 0:
            sd.play(audio, self._sample_rate)
            sd.wait()


class GoogleCloudTTSEngine:
    """Google Cloud Text-to-Speech — cloud-based with expressive neural voices.

    Requires the Cloud Text-to-Speech API to be enabled in your Google Cloud project.
    The Gemini API key (AIzaSy...) can be used if the API is enabled on the same project.
    """

    def __init__(self, api_key: str, voice: str = "en-US-Neural2-D", language: str = "en-US", speaking_rate: float = 1.0):
        self.api_key  = api_key
        self.voice    = voice
        self.language = language
        self.speaking_rate = speaking_rate
        self._sample_rate = 24000

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def synthesize(self, text: str) -> np.ndarray:
        import base64
        url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={self.api_key}"
        payload = {
            "input": {"text": text},
            "voice": {
                "languageCode": self.language,
                "name": self.voice,
            },
            "audioConfig": {
                "audioEncoding": "MP3",
                "speakingRate": float(self.speaking_rate),
                "pitch": 0.0,
            },
        }
        resp = _HTTP.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        audio_b64 = resp.json().get("audioContent", "")
        if not audio_b64:
            raise RuntimeError("Google Cloud TTS returned empty audio")
        audio_bytes = base64.b64decode(audio_b64)
        samples, sr = _decode_to_np(audio_bytes)
        self._sample_rate = sr
        return samples

    def speak(self, text: str) -> None:
        audio = self.synthesize(text)
        if audio.size > 0:
            sd.play(audio, self._sample_rate)
            sd.wait()


# ---------------------------------------------------------------------------
# Google Gemini TTS (native Gemini voices)
# ---------------------------------------------------------------------------

# Official prebuilt voices for the Gemini TTS / Live models (Puck is default).
GEMINI_TTS_VOICES: list[str] = [
    "Zephyr", "Kore", "Orus", "Autonoe", "Umbriel", "Erinome", "Laomedeia",
    "Schedar", "Achird", "Sadachbia", "Puck", "Fenrir", "Aoede", "Enceladus",
    "Algieba", "Algenib", "Achernar", "Gacrux", "Zubenelgenubi", "Sadaltager",
    "Charon", "Leda", "Callirrhoe", "Iapetus", "Despina", "Rasalgethi",
    "Alnilam", "Pulcherrima", "Vindemiatrix", "Sulafat",
]

GEMINI_TTS_MODEL_DEFAULT = "gemini-3.1-flash-tts-preview"


class GeminiTTSEngine:
    """Google Gemini TTS — real Gemini voices via gemini-3.1-flash-tts-preview.

    Uses the same Gemini API key as the rest of Orthos (gemini_api_key).
    Output is raw 16-bit little-endian PCM at 24 kHz (audio/l16), decoded here
    to normalized float32 for playback via sounddevice.
    """

    def __init__(
        self,
        api_key: str = "",
        voice: str = "Puck",
        model: str = GEMINI_TTS_MODEL_DEFAULT,
    ):
        self.api_key   = api_key
        self.voice     = voice
        self.model     = model
        self._sample_rate = 24000
        self._client   = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def _get_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _config(self):
        from google.genai import types
        return types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self.voice
                    )
                )
            ),
        )

    def _parse_rate(self, mime_type: str | None) -> int:
        """Read sample rate from mime type, e.g. 'audio/l16; rate=24000'."""
        try:
            return int(re.search(r"rate=(\d+)", mime_type or "").group(1))
        except Exception:
            return 24000

    def synthesize(self, text: str) -> np.ndarray:
        if not self.api_key:
            raise RuntimeError(
                "Gemini TTS requires 'gemini_voice_api_key' — "
                "set it in Configure → Text-to-Speech → Voice API Key."
            )
        if not text.strip():
            return np.zeros(0, dtype=np.float32)
        resp = self._get_client().models.generate_content(
            model=self.model,
            contents=text,
            config=self._config(),
        )
        parts = (resp.candidates or [{}])[0].content.parts or []
        inline = None
        for p in parts:
            if getattr(p, "inline_data", None) and p.inline_data.data:
                inline = p.inline_data
                break
        if inline is None:
            reason = ""
            try:
                reason = resp.prompt_feedback.block_reason
            except Exception:
                pass
            raise RuntimeError(
                f"Gemini TTS returned no audio (block reason: {reason or 'unknown'})"
            )
        self._sample_rate = self._parse_rate(inline.mime_type)
        samples = np.frombuffer(inline.data, dtype="<i2").astype(np.float32) / 32768.0
        return samples

    def synthesize_stream(self, text: str):
        """Yield raw 24 kHz 16-bit PCM bytes as they stream in.

        Handles odd-byte chunk boundaries (a PCM frame must stay aligned).
        """
        if not self.api_key:
            raise RuntimeError(
                "Gemini TTS requires 'gemini_voice_api_key' — "
                "set it in Configure → Text-to-Speech → Voice API Key."
            )
        stream = self._get_client().models.generate_content_stream(
            model=self.model,
            contents=text,
            config=self._config(),
        )
        carry = b""
        for chunk in stream:
            try:
                parts = chunk.candidates[0].content.parts or []
            except (IndexError, AttributeError):
                continue
            for p in parts:
                if getattr(p, "inline_data", None) and p.inline_data.data:
                    data = carry + p.inline_data.data
                    usable = len(data) - (len(data) % 2)
                    carry = data[usable:]
                    if usable:
                        self._sample_rate = self._parse_rate(p.inline_data.mime_type)
                        yield data[:usable]

    def warmup(self) -> None:
        """Pre-create the google.genai client so the first synthesis is fast."""
        self._get_client()

    def speak(self, text: str) -> None:
        audio = self.synthesize(text)
        if audio.size > 0:
            sd.play(audio, self._sample_rate)
            sd.wait()


# ---------------------------------------------------------------------------
# Thread-safe player wrapper
# ---------------------------------------------------------------------------

class NullTTSPlayer:
    """No-op TTS player used when another system (e.g. Gemini Live) handles audio."""

    supports_streaming = False
    sample_rate = 24000

    def synthesize(self, text: str) -> np.ndarray:
        return np.array([], dtype=np.float32)

    def synthesize_stream(self, text: str):
        return iter(())

    def warmup(self) -> None:
        pass

    def speak(
        self,
        text: str,
        on_start: Optional[Callable] = None,
        on_done: Optional[Callable] = None,
    ) -> None:
        if on_done:
            on_done()

    def interrupt(self) -> None:
        pass


class TTSPlayer:
    """
    Wraps any *Engine. Exposes a blocking speak() method
    meant to be called from a dedicated background thread.
    """

    def __init__(self, engine):
        self._engine  = engine
        self._playing = False
        self._lock    = threading.Lock()

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def sample_rate(self) -> int:
        return self._engine.sample_rate

    def synthesize(self, text: str) -> np.ndarray:
        return self._engine.synthesize(text)

    @property
    def supports_streaming(self) -> bool:
        """True when the engine can yield PCM chunks before full synthesis."""
        return hasattr(self._engine, "synthesize_stream")

    def synthesize_stream(self, text: str):
        """Yield raw PCM bytes as they stream in (streaming engines only)."""
        return self._engine.synthesize_stream(text)

    def warmup(self) -> None:
        """Pre-initialise heavy engine resources (client, model, etc.)."""
        warmup = getattr(self._engine, "warmup", None)
        if callable(warmup):
            warmup()

    def speak(
        self,
        text:     str,
        on_start: Optional[Callable] = None,
        on_done:  Optional[Callable] = None,
    ) -> None:
        """Synthesise and play text. BLOCKING – call from a dedicated thread."""
        try:
            with self._lock:
                self._playing = True
            if on_start:
                on_start()
            self._engine.speak(text)
        except Exception as e:
            print(f"[TTS] Error: {e}")
        finally:
            with self._lock:
                self._playing = False
            if on_done:
                on_done()

    def interrupt(self) -> None:
        sd.stop()
        stop = getattr(self._engine, "stop", None)
        if callable(stop):
            try:
                stop()
            except Exception:
                pass
        with self._lock:
            self._playing = False

    def stop(self) -> None:
        sd.stop()
        stop = getattr(self._engine, "stop", None)
        if callable(stop):
            try:
                stop()
            except Exception:
                pass
        with self._lock:
            self._playing = False





# Google Gemini Live as a voice renderer (Live-as-TTS)
# ---------------------------------------------------------------------------

VOICE_RENDERER_PROMPT = (
    "You are ONLY a voice renderer, a text-to-speech engine. "
    "The text you receive is already the final response written by another AI. "
    "Read the supplied text EXACTLY word for word, from the first word to the "
    "last word. Do NOT rephrase, reword, summarize, paraphrase, translate, "
    "correct, or change any word. Do NOT add anything before, inside, or after "
    "the text. Do NOT add filler words such as \"so\", \"well\", \"now\", "
    "\"okay\", or \"sure\". Never acknowledge the text, greet, comment, or ask "
    "questions. Start with the first word and end immediately after the last "
    "word. Speak naturally and expressively, exactly as written, in the "
    "language it is written in."
)

GEMINI_LIVE_TTS_MODEL_DEFAULT = "gemini-2.5-flash-native-audio-preview-12-2025"


class GeminiLiveVoiceEngine:
    """Gemini Live as a voice renderer: text in -> native 24 kHz PCM audio out.

    Uses the Live WebSocket API (same one as GeminiLiveProvider) but in a
    "voice only" configuration: no mic, no tools, no transcription, and an
    aggressive system instruction that forces exact recitation of the
    supplied text. Meant to be driven by Orthos's own LLM (any provider):
    LLM picks the words, Live speaks them.
    """

    def __init__(
        self,
        api_key: str = "",
        voice: str = "Charon",
        model: str = GEMINI_LIVE_TTS_MODEL_DEFAULT,
    ):
        self.api_key = api_key
        self.voice = voice
        self.model = model
        self._sample_rate = 24000

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._loop = None
        self._session = None
        self._connected = threading.Event()
        self._turn_q = None
        self._turn_lock = threading.Lock()
        self._backoff = 3
        self._resumption_handle: str | None = None
        self._go_away = False

        self._debounce = 2.5
        self._max_batch_chars = 4000
        self._pending: list[str] = []
        self._pending_lock = threading.Lock()
        self._new_sentence = threading.Event()
        self._interrupted = threading.Event()
        self._drainer: threading.Thread | None = None

        self._transcribe_output = False
        self._last_transcript = ""

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def warmup(self) -> None:
        """Pre-create the genai client and start the session thread."""
        if not self.api_key:
            return
        from google import genai
        genai.Client(api_key=self.api_key, http_options={"api_version": "v1beta"})
        self.ensure_running()

    def ensure_running(self) -> bool:
        if not self.api_key:
            return False
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._session_main, daemon=True)
            self._thread.start()
        return True

    def interrupt(self) -> None:
        """Drop any in-flight turn audio and pending sentences (barge-in / stop)."""
        self._interrupted.set()
        with self._pending_lock:
            self._pending.clear()
        self._new_sentence.clear()
        with self._turn_lock:
            q = self._turn_q
            self._turn_q = None
        if q is not None:
            while True:
                try:
                    q.get_nowait()
                except Exception:
                    break
        try:
            sd.stop()
        except Exception:
            pass

    def stop(self) -> None:
        self._stop.set()
        self._interrupted.set()
        try:
            sd.stop()
        except Exception:
            pass
        if self._loop is not None and self._session is not None:
            try:
                asyncio.run_coroutine_threadsafe(self._session.close(), self._loop)
            except Exception:
                pass

    # ---------------------------------------------------------------------------
    # Synthesis
    # ---------------------------------------------------------------------------

    def synthesize(self, text: str) -> np.ndarray:
        if not self.api_key:
            raise RuntimeError(
                "Gemini Live voice requires 'gemini_voice_api_key' "
                "set it in Configure -> Text-to-Speech -> Voice API Key."
            )
        audio = self._run_turn(text)
        return audio.astype(np.float32) / 32768.0

    def synthesize_stream(self, text: str):
        """Accept one sentence of the response (voice renderer).

        Sentences accumulate in a pending buffer; the background drainer
        sends the FULL response as a single Live turn (no mid-response
        breaks between sentences). This generator therefore returns no
        audio - playback happens inside the engine.
        """
        if not self.api_key:
            raise RuntimeError(
                "Gemini Live voice requires 'gemini_voice_api_key' "
                "set it in Configure -> Text-to-Speech -> Voice API Key."
            )
        if not text:
            return iter(())
        with self._pending_lock:
            self._pending.append(text)
        self._new_sentence.set()
        self._ensure_drainer()
        return iter(())

    def speak(self, text: str) -> None:
        """Synthesise and play text. BLOCKING - call from a dedicated thread."""
        if not self.api_key:
            return
        audio = self._run_turn(text)
        if audio.size > 0:
            sd.play(audio, self._sample_rate)
            sd.wait()

    # ---------------------------------------------------------------------------
    # Turn execution
    # ---------------------------------------------------------------------------

    def _run_turn(self, text: str) -> np.ndarray:
        max_attempts = 4
        for attempt in range(max_attempts):
            if self._interrupted.is_set():
                return np.zeros(0, dtype="<i2")
            try:
                return self._send_turn(text)
            except RuntimeError as e:
                message = str(e)
                if "quota" in message.lower() or "RESOURCE_EXHAUSTED" in message:
                    raise
                print(
                    f"[GeminiLiveVoice] Send failed "
                    f"(attempt {attempt + 1}/{max_attempts}): {message[:120]}"
                )
                if attempt < max_attempts - 1:
                    print("  -> waiting for reconnect...")
                    self._connected.wait(timeout=5)
                    continue
                raise RuntimeError(f"Gemini Live voice send failed: {message}") from e
        return np.zeros(0, dtype="<i2")

    def _send_turn(self, text: str) -> np.ndarray:
        if not self.ensure_running():
            raise RuntimeError("Gemini Live voice engine is not running.")

        self._interrupted.clear()
        self._last_transcript = ""

        if not self._connected.wait(timeout=30):
            raise RuntimeError("Gemini Live session could not connect - check API key and model.")

        turn_q: "_queue.Queue" = _queue.Queue()
        with self._turn_lock:
            self._turn_q = turn_q

        try:
            fut = asyncio.run_coroutine_threadsafe(
                self._send_text(text), self._loop
            )
            fut.result(timeout=15)
        except Exception as e:
            message = str(e)
            if "quota" in message.lower():
                raise RuntimeError(f"quota exceeded: {message}") from e
            raise RuntimeError(message) from e

        deadline = time.monotonic() + max(120.0, 0.06 * len(text))
        chunks: list[np.ndarray] = []

        while True:
            if self._interrupted.is_set():
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("Gemini Live voice timed out.")
            try:
                ev = turn_q.get(timeout=min(0.5, remaining))
            except _queue.Empty:
                if not self._connected.is_set():
                    raise RuntimeError("Gemini Live session dropped mid-turn.")
                continue
            ev_type = ev.get("type")
            if ev_type == "chunk":
                data = ev.get("data", b"")
                if data:
                    chunks.append(np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0)
            elif ev_type == "done":
                break
            elif ev_type == "error":
                raise RuntimeError(ev.get("message", "unknown error"))

        if not chunks:
            return np.zeros(0, dtype="<i2")
        return np.concatenate(chunks)

    def _get_session(self):
        if self._session is None:
            return None
        return self._session

    # ---------------------------------------------------------------------------
    # Session management
    # ---------------------------------------------------------------------------

    def _session_main(self) -> None:
        while not self._stop.is_set():
            try:
                self._connected.clear()
                asyncio.run(self._session_loop())
            except Exception as e:
                print(f"[GeminiLiveVoice] Session error: {e}")
            finally:
                self._session = None
                self._connected.clear()
            if not self._stop.is_set():
                time.sleep(self._backoff)

    async def _session_loop(self) -> None:
        from google import genai
        backoff = self._backoff
        while not self._stop.is_set():
            try:
                config = self._build_config()
                client = genai.Client(
                    api_key=self.api_key,
                    http_options={"api_version": "v1beta"},
                )
                async with client.aio.live.connect(
                    model=f"models/{self.model}", config=config
                ) as session:
                    self._session = session
                    self._loop = asyncio.get_event_loop()
                    self._connected.set()
                    await session.send_client_content(
                        turns=[{"role": "user", "parts": [{"text": ""}]}],
                        turn_complete=True,
                    )
                    async for response in session.receive():
                        if self._interrupted.is_set():
                            break
                        if response.data:
                            self._emit({"type": "chunk", "data": response.data})
                        if response.server_content and response.server_content.turn_complete:
                            self._emit({"type": "done"})
                            break
            except Exception as e:
                message = str(e)
                self._emit({"type": "error", "message": message, "quota": "quota" in message.lower()})
                if "quota" in message.lower():
                    raise
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    @property
    def _uses_realtime_text(self) -> bool:
        return self.model.startswith("gemini-3")

    @property
    def _supports_compression(self) -> bool:
        return self.model.startswith("gemini-3")

    async def _send_text(self, text: str) -> None:
        if self._session is None:
            return
        if self._uses_realtime_text:
            await self._session.send_realtime_input(text=text)
        else:
            await self._session.send_client_content(
                turns=[{"role": "user", "parts": [{"text": text}]}],
                turn_complete=True,
            )

    def _build_config(self):
        from google.genai import types
        kwargs = {
            "response_modalities": ["AUDIO"],
            "speech_config": types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self.voice
                    )
                )
            ),
            "system_instruction": types.Content(
                parts=[{"text": VOICE_RENDERER_PROMPT}]
            ),
        }
        if self._supports_compression:
            try:
                kwargs["context_window_compression"] = types.ContextWindowCompressionConfig(
                    trigger_tokens=25000,
                )
            except Exception:
                pass
        if self._resumption_handle:
            kwargs["session_resumption"] = types.SessionResumptionConfig(
                handle=self._resumption_handle
            )
        if self.model.startswith("gemini-3"):
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        if self._transcribe_output:
            kwargs["output_audio_transcription"] = types.AudioTranscriptionConfig()
        return types.LiveConnectConfig(**kwargs)

    def _emit(self, ev: dict) -> None:
        with self._turn_lock:
            q = self._turn_q
        if q is not None:
            try:
                q.put_nowait(ev)
            except Exception:
                pass

    def _ensure_drainer(self) -> None:
        if self._drainer is not None and self._drainer.is_alive():
            return
        self._drainer = threading.Thread(target=self._drainer_main, daemon=True)
        self._drainer.start()

    def _drainer_main(self) -> None:
        while not self._stop.is_set():
            self._new_sentence.wait(timeout=self._debounce)
            self._new_sentence.clear()
            text = None
            with self._pending_lock:
                if self._pending:
                    text = "".join(self._pending)
                    self._pending.clear()
            if text:
                self._run_turn(text)


# ---------------------------------------------------------------------------
# HashimMMalik-style standalone Gemini Live TTS (gemini-3.1-flash-live-preview)
# ---------------------------------------------------------------------------

HASHIM_LIVE_TTS_MODEL_DEFAULT = "gemini-3.1-flash-live-preview"


class HashimLiveTTSEngine:
    """Standalone Gemini Live TTS based on hashimmalikdev/gemini-live-tts.

    Uses ``gemini-3.1-flash-live-preview`` and ``send_realtime_input(text=...)``
    — the exact approach from the reference repo.  Completely independent from
    ``GeminiLiveProvider`` (the LLM provider) — it has its own WebSocket session,
    its own thread, and its own connection.  Text in → 24 kHz PCM audio out.
    """

    def __init__(
        self,
        api_key: str = "",
        voice: str = "Kore",
        model: str = HASHIM_LIVE_TTS_MODEL_DEFAULT,
    ):
        self.api_key = api_key
        self.voice = voice
        self.model = model
        self._sample_rate = 24000

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._connected = threading.Event()
        self._turn_done = threading.Event()
        self._turn_q: "_queue.Queue" | None = None
        self._turn_lock = threading.Lock()
        self._loop = None
        self._session = None
        self._backoff = 3

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def _start(self) -> bool:
        if not self.api_key:
            return False
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._bg_main, daemon=True)
            self._thread.start()
        return True

    def _bg_main(self) -> None:
        while not self._stop.is_set():
            try:
                asyncio.run(self._session_loop())
            except Exception as e:
                print(f"[HashimTTS] Session error: {e}")
            finally:
                self._connected.clear()
                self._session = None
            if not self._stop.is_set():
                time.sleep(self._backoff)

    async def _session_loop(self) -> None:
        from google import genai
        from google.genai import types

        client = genai.Client(
            http_options={"api_version": "v1beta"},
            api_key=self.api_key,
        )

        while not self._stop.is_set():
            config = types.LiveConnectConfig(
                response_modalities=["AUDIO"],
                output_audio_transcription=types.AudioTranscriptionConfig(),
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=self.voice
                        )
                    )
                ),
            )
            try:
                async with (
                    client.aio.live.connect(
                        model=f"models/{self.model}", config=config
                    ) as session,
                    asyncio.TaskGroup() as tg,
                ):
                    self._session = session
                    self._loop = asyncio.get_event_loop()
                    self._connected.set()
                    print(f"[HashimTTS] Connected ({self.model}, {self.voice})")

                    tg.create_task(self._receive_audio())
                    # Keep session alive until stop is requested
                    while not self._stop.is_set():
                        await asyncio.sleep(0.5)
            except Exception as e:
                print(f"[HashimTTS] Session error: {e}")
                self._connected.clear()
                await asyncio.sleep(self._backoff)

            if self._connected.is_set():
                # session ended unexpectedly
                self._connected.clear()

    async def _receive_audio(self) -> None:
        chunks: list[bytes] = []
        while True:
            async for response in self._session.receive():
                sc = response.server_content
                if sc:
                    if sc.model_turn:
                        for part in sc.model_turn.parts or []:
                            if part.inline_data and part.inline_data.data:
                                chunks.append(part.inline_data.data)
                    if sc.turn_complete:
                        data = b"".join(chunks)
                        chunks = []
                        with self._turn_lock:
                            q = self._turn_q
                        if q is not None:
                            q.put({"type": "chunk", "data": data})
                            q.put({"type": "done"})

    def synthesize(self, text: str) -> np.ndarray:
        """Send one piece of text and return the full 24 kHz float32 audio."""
        if not self.api_key:
            raise RuntimeError(
                "Hashim Live TTS requires 'gemini_voice_api_key'."
            )
        if self._stop.is_set():
            self._stop.clear()
        if not self._start():
            raise RuntimeError("Could not start TTS thread.")
        if not self._connected.wait(timeout=30):
            raise RuntimeError("Gemini Live TTS could not connect — check API key.")

        turn_q: "_queue.Queue" = _queue.Queue()
        with self._turn_lock:
            self._turn_q = turn_q

        try:
            # Send the text via realtime input
            fut = asyncio.run_coroutine_threadsafe(
                self._session.send_realtime_input(text=text), self._loop
            )
            fut.result(timeout=15)
        except Exception as e:
            with self._turn_lock:
                if self._turn_q is turn_q:
                    self._turn_q = None
            raise RuntimeError(f"HashimTTS send failed: {e}")

        deadline = time.monotonic() + 120
        pcm_chunks: list[bytes] = []
        while True:
            if time.monotonic() >= deadline:
                break
            try:
                ev = turn_q.get(timeout=0.5)
            except _queue.Empty:
                if not self._connected.is_set():
                    break
                continue
            if ev["type"] == "chunk":
                pcm_chunks.append(ev["data"])
            elif ev["type"] == "done":
                break
            elif ev["type"] == "error":
                raise RuntimeError(ev["message"])

        with self._turn_lock:
            if self._turn_q is turn_q:
                self._turn_q = None

        if not pcm_chunks:
            return np.zeros(0, dtype=np.float32)
        raw = b"".join(pcm_chunks)
        if not raw:
            return np.zeros(0, dtype=np.float32)
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        return samples

    def speak(self, text: str) -> None:
        audio = self.synthesize(text)
        if audio.size > 0:
            sd.play(audio, self._sample_rate)
            sd.wait()

    def warmup(self) -> None:
        if self.api_key:
            from google import genai
            genai.Client(api_key=self.api_key, http_options={"api_version": "v1beta"})
            self._start()

    def stop(self) -> None:
        self._stop.set()
        try:
            sd.stop()
        except Exception:
            pass


def _is_reconnectable(message: str) -> bool:
    if not message:
        return False
    lower = message.lower()
    if "quota" in lower:
        return False
    if "1000 (ok)" in lower:
        return True
    if "connection closedok" in lower or "connectionclosedok" in lower:
        return True
    if "connection reset" in lower:
        return True
    return False




# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class FallbackTTSPlayer:
    """Tries multiple TTS engines in order, falling back on failure."""

    def __init__(self, players: list[TTSPlayer]):
        self._players = players
        self._active  = players[0] if players else None
        self._lock    = threading.Lock()

    @property
    def is_playing(self) -> bool:
        return self._active.is_playing if self._active else False

    @property
    def sample_rate(self) -> int:
        return self._active.sample_rate if self._active else 24000

    def synthesize(self, text: str) -> np.ndarray:
        last_err = None
        for p in self._players:
            try:
                return p.synthesize(text)
            except Exception as e:
                last_err = e
        raise last_err or RuntimeError("No TTS engines available")

    def synthesize_stream(self, text: str):
        player = self._active or self._players[0]
        if player and getattr(player, "supports_streaming", False):
            return player.synthesize_stream(text)
        return iter(())

    @property
    def supports_streaming(self) -> bool:
        return any(getattr(p, "supports_streaming", False) for p in self._players)

    def warmup(self) -> None:
        for p in self._players:
            try:
                p.warmup()
            except Exception:
                pass

    def speak(self, text: str, on_start=None, on_done=None) -> None:
        last_err = None
        for p in self._players:
            try:
                with self._lock:
                    self._active = p
                if on_start:
                    on_start()
                p.speak(text)
                return
            except Exception as e:
                last_err = e
        print(f"[TTS] All engines failed: {last_err}")

    def interrupt(self) -> None:
        for p in self._players:
            try:
                p.interrupt()
            except Exception:
                pass

    def stop(self) -> None:
        for p in self._players:
            try:
                p.stop()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_ENGINE_ALIASES: dict[str, str] = {
    "edgetts":    "edgetts",
    "edge":       "edgetts",
    "gtts":       "gtts",
    "google":     "googlecloud",
    "googlecloud":"googlecloud",
    "gcloud":     "googlecloud",
    "gemini":     "gemini",
      "gemini_live":"gemini_live",
    "hashim_live":"hashim_live",
    "hashimlive": "hashim_live",
    "kokoro":     "kokoro",
    "elevenlabs": "elevenlabs",
}


def _normalize_engine(name: str) -> str:
    return _ENGINE_ALIASES.get(name.lower().strip(), name.lower().strip())


def _build_engine(name: str, config: dict) -> TTSPlayer | None:
    name = _normalize_engine(name)
    if name == "kokoro":
        voice  = config.get("tts_voice", "af_heart")
        speed  = float(config.get("tts_speed", 1.0))
        return TTSPlayer(KokoroTTSEngine(voice=voice, speed=speed))
    elif name == "elevenlabs":
        api_key  = config.get("elevenlabs_api_key", "")
        voice_id = config.get("tts_voice", "pNInz6obpgDQGcFmaJgB")
        return TTSPlayer(ElevenLabsTTSEngine(api_key=api_key, voice_id=voice_id))
    elif name == "googlecloud":
        api_key  = config.get("gemini_voice_api_key", "").strip()
        if not api_key:
            return None
        voice    = config.get("tts_voice", "hi-IN-Neural2-A")
        language = config.get("tts_language") or "-".join(voice.split("-")[:2]) or "en-US"
        speed    = float(config.get("tts_speed", 1.0))
        return TTSPlayer(GoogleCloudTTSEngine(api_key=api_key, voice=voice, language=language, speaking_rate=speed))
    elif name == "gtts":
        voice    = config.get("tts_voice", "hi")
        return TTSPlayer(GTTSEngine(voice=voice))
    elif name == "gemini_live":
        if config.get("llm_provider", "").lower() == "gemini_live":
            return None
        api_key  = config.get("gemini_voice_api_key", "").strip()
        if not api_key:
            return None
        voice    = config.get("gemini_live_voice", "Charon")
        model    = config.get("gemini_live_model", GEMINI_LIVE_TTS_MODEL_DEFAULT)
        return TTSPlayer(GeminiLiveVoiceEngine(api_key=api_key, voice=voice, model=model))
    elif name == "hashim_live":
        # Standalone Gemini Live TTS using the EXACT hashimmalikdev/gemini-live-tts
        # tool (cloned at tools/gemini-live-tts). Always available — even when
        # LLM provider is gemini_live.
        api_key  = config.get("gemini_voice_api_key", "").strip()
        if not api_key:
            return None
        voice    = config.get("gemini_live_voice", "Kore")
        try:
            from core.hashim_live_tts_adapter import HashimExactTTSEngine
            return TTSPlayer(HashimExactTTSEngine(api_key=api_key, voice=voice))
        except Exception:
            # Fall back to the in-repo engine if adapter import fails
            return TTSPlayer(HashimLiveTTSEngine(api_key=api_key, voice=voice))
    elif name == "gemini":
        api_key  = config.get("gemini_voice_api_key", "").strip()
        if not api_key:
            return None
        voice    = config.get("gemini_tts_voice", "Puck")
        model    = config.get("gemini_tts_model", GEMINI_TTS_MODEL_DEFAULT)
        return TTSPlayer(GeminiTTSEngine(api_key=api_key, voice=voice, model=model))
    else:   # edgetts (default)
        voice  = config.get("tts_voice", "en-US-GuyNeural")
        return TTSPlayer(EdgeTTSEngine(voice=voice))


def create_tts_player(config: dict) -> TTSPlayer | FallbackTTSPlayer | NullTTSPlayer:
    primary = _normalize_engine(config.get("tts_engine", "edgetts"))
    fallback_raw = config.get("tts_fallback", "")
    fallback_list = [
        _normalize_engine(x) for x in fallback_raw.replace(";", ",").split(",") if x.strip()
    ]

    players: list[TTSPlayer] = []
    seen: set[str] = set()

    # Primary first
    if primary not in seen:
        seen.add(primary)
        p = _build_engine(primary, config)
        if p is not None:
            players.append(p)
        elif not fallback_list:
            if primary in ("gemini", "gemini_live"):
                if config.get("llm_provider", "").lower() == "gemini_live":
                    pass
                else:
                    raise RuntimeError(
                        "Gemini voice requires 'gemini_voice_api_key' — "
                        "set it in Configure → Text-to-Speech → Voice API Key."
                    )
            elif primary == "googlecloud":
                raise RuntimeError(
                    "Google Cloud voice requires 'gemini_voice_api_key' — "
                    "set it in Configure → Text-to-Speech → Voice API Key."
                )
            else:
                raise RuntimeError(f"Failed to initialize primary TTS engine: {primary}")

    # Then fallbacks (skip duplicates)
    for name in fallback_list:
        if name not in seen:
            seen.add(name)
            p = _build_engine(name, config)
            if p is not None:
                players.append(p)

    if not players:
        return NullTTSPlayer()

    if len(players) == 1:
        return players[0]

    return FallbackTTSPlayer(players)
