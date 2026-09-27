"""
Silero VAD v5 (ONNX) — neural voice-activity detection for Orthos.

Drop-in replacement for main._VADBuffer: same ``process(chunk) ->
utterance|None`` and ``reset()`` interface, so the Whisper/Vosk mic loops,
echo gates, mic-hold and barge-in keep working unchanged — only the
"when did the utterance end" decision becomes neural instead of
energy-threshold based.

Why Silero over the energy VAD:
  - classifies SPEECH vs noise/music/TV (not just loud vs quiet)
  - turn-end detected in ~200-400 ms instead of a fixed 3 s wait
  - no threshold tuning per room/mic; TRRS bleed rarely reads as speech

Runtime: onnxruntime CPU (~1 ms per 512-sample frame). The 2.3 MB model is
auto-downloaded once into models/silero_vad.onnx. If either dependency is
missing, ``silero_available()`` returns False and callers fall back to the
energy VAD.
"""
from __future__ import annotations

import collections
import time
import urllib.request
from pathlib import Path

import numpy as np

# NOTE: the v5.1.2 tag, NOT master — master's onnx export has produced a
# file that scores speech at ~0.16 forever (verified 2026-09-21). v5.1.2
# scores the same audio at 1.0. If the file is ever re-fetched, keep the tag.
_MODEL_URL = ("https://github.com/snakers4/silero-vad/raw/v5.1.2/"
              "src/silero_vad/data/silero_vad.onnx")
_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "silero_vad.onnx"

# Valid frame sizes for Silero v5 at the given sample rate.
_FRAME_BY_SR = {8000: 256, 16000: 512, 24000: 576, 48000: 768}

_session = None          # cached ORT InferenceSession (thread-safe Run)
_session_failed = False  # sticky: don't retry a broken load every chunk


def silero_available() -> bool:
    """True when the model + onnxruntime can actually be used."""
    try:
        _get_session()
        return True
    except Exception:
        return False


def _ensure_model() -> Path:
    if _MODEL_PATH.exists() and _MODEL_PATH.stat().st_size > 1_000_000:
        return _MODEL_PATH
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _MODEL_PATH.with_suffix(".onnx.part")
    with urllib.request.urlopen(_MODEL_URL, timeout=30) as resp, \
            open(tmp, "wb") as fh:
        fh.write(resp.read())
    tmp.replace(_MODEL_PATH)
    return _MODEL_PATH


def _get_session():
    global _session, _session_failed
    if _session is not None:
        return _session
    if _session_failed:
        raise RuntimeError("Silero VAD load failed earlier (sticky)")
    import onnxruntime as ort
    path = _ensure_model()
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    opts.log_severity_level = 3
    _session = ort.InferenceSession(
        str(path), sess_options=opts, providers=["CPUExecutionProvider"])
    return _session


class SileroVAD:
    """Buffering neural VAD with main._VADBuffer's external contract.

    process(float32 mono chunk @16 kHz) returns the COMPLETE utterance
    (onset recovered via a rolling prebuffer) when speech ends, else None.
    Hysteresis: speech starts at ``start_prob``, ends after ``end_prob``
    holds for ``silence_ms`` — that gap is what stops mid-word cuts.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        threshold: float = 0.5,        # centre of the hysteresis band
        silence_ms: float = 450,       # below-end-prob silence → turn end
        min_speech_sec: float = 0.28,  # shorter than this = noise → dropped
        max_speech_sec: float = 30.0,
        prebuffer_sec: float = 0.30,   # onset recovery
    ):
        self._sr = int(sample_rate)
        self._frame_n = _FRAME_BY_SR.get(self._sr, 512)
        hi = min(0.95, threshold + 0.08)
        lo = max(0.15, threshold - 0.12)
        self._start_prob = hi
        self._end_prob = lo
        self._sil_frames_needed = max(1, int(silence_ms / 1000 * self._sr / self._frame_n))
        self._min_n = int(min_speech_sec * self._sr)
        self._max_n = int(max_speech_sec * self._sr)
        self._pre_n = int(prebuffer_sec * self._sr)

        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._residual = np.zeros(0, dtype=np.float32)
        self._prebuf: collections.deque[np.ndarray] = collections.deque()
        self._prebuf_n = 0
        self._speech: list[np.ndarray] = []
        self._speech_n = 0
        self._in_speech = False
        self._sil_frames = 0
        self._last = 0.0                # last voice prob (for trailing trim)
        self._last_dbg = 0.0            # exported for logging

    # ── internals ──────────────────────────────────────────────────────
    def _infer_frame(self, frame: np.ndarray) -> float:
        sess = _get_session()
        x = frame.astype(np.float32).reshape(1, -1)
        out, state = sess.run(
            None,
            {"input": x, "state": self._state,
             "sr": np.asarray(self._sr, dtype=np.int64)},
        )
        self._state = np.asarray(state, dtype=np.float32).reshape(2, 1, 128)
        return float(out[0, 0])

    def _push_pre(self, chunk: np.ndarray) -> None:
        self._prebuf.append(chunk.copy())
        self._prebuf_n += len(chunk)
        while self._prebuf_n > self._pre_n:
            dropped = self._prebuf.popleft()
            self._prebuf_n -= len(dropped)

    def _flush(self) -> np.ndarray | None:
        audio = np.concatenate(self._speech) if self._speech else np.zeros(0, np.float32)
        self._speech = []
        self._speech_n = 0
        self._in_speech = False
        self._sil_frames = 0
        if len(audio) < self._min_n:
            return None
        return audio

    # ── public contract (matches main._VADBuffer) ──────────────────────
    def reset(self) -> None:
        """Drop buffered speech + prebuffer; clear the recurrent state.

        Called when the assistant stops talking — whatever sat in the
        buffer came from the speakers, not the user. (Same semantics as
        the energy VAD's reset; the room concept doesn't exist here.)
        """
        self._state[:] = 0.0
        self._residual = np.zeros(0, dtype=np.float32)
        self._prebuf.clear()
        self._prebuf_n = 0
        self._speech = []
        self._speech_n = 0
        self._in_speech = False
        self._sil_frames = 0
        self._last = 0.0

    def process(self, chunk: np.ndarray) -> np.ndarray | None:
        """Feed float32 mono samples; return the utterance when it ends."""
        x = np.asarray(chunk, dtype=np.float32).ravel()
        if x.size == 0:
            return None
        self._push_pre(x)

        stream = np.concatenate([self._residual, x]) if self._residual.size else x
        n_frames = stream.size // self._frame_n
        if n_frames == 0:
            self._residual = stream
            return None
        cut = n_frames * self._frame_n
        self._residual = stream[cut:].copy()
        frames = np.reshape(stream[:cut], (n_frames, self._frame_n))

        for frame in frames:
            prob = self._infer_frame(frame)
            self._last = prob
            self._last_dbg = prob

            if not self._in_speech:
                if prob >= self._start_prob:
                    # Onset: seed the utterance with the prebuffer so the
                    # first word is never clipped.
                    self._speech = list(self._prebuf)
                    self._speech_n = self._prebuf_n
                    self._prebuf.clear()
                    self._prebuf_n = 0
                    self._in_speech = True
                    self._sil_frames = 0
                    self._speech.append(frame.copy())
                    self._speech_n += len(frame)
            else:
                self._speech.append(frame.copy())
                self._speech_n += len(frame)
                if prob < self._end_prob:
                    self._sil_frames += 1
                    if self._sil_frames >= self._sil_frames_needed:
                        # Trim the trailing silence back to the last voiced
                        # frame so the STT doesn't get dead air.
                        tail_n = self._sil_frames * self._frame_n
                        if 0 < tail_n < self._speech_n:
                            flat = np.concatenate(self._speech)
                            self._speech = [flat[:self._speech_n - tail_n]]
                        elif tail_n >= self._speech_n:
                            self._speech = []
                        return self._flush()
                else:
                    self._sil_frames = 0

            if self._in_speech and self._speech_n >= self._max_n:
                return self._flush()
        return None


class SoftSileroVAD(SileroVAD):
    """Same model, but reset() does NOT clear the recurrent state.

    Mic-loop restarts / engine switches reuse reset() between turns of the
    SAME user session; wiping the LSTM then costs ~300 ms of cold start on
    the next utterance. Preserving state across a reset is strictly better
    when only the buffers must go. (The energy VAD keeps its noise floor on
    reset — this is the neural equivalent.)
    """

    def reset(self) -> None:
        self._residual = np.zeros(0, dtype=np.float32)
        self._prebuf.clear()
        self._prebuf_n = 0
        self._speech = []
        self._speech_n = 0
        self._in_speech = False
        self._sil_frames = 0
        self._last = 0.0


def make_vad(sample_rate: int = 16_000, prefer_silero: bool = True,
             soft_reset: bool = True, threshold: float = 0.5,
             silence_ms: float = 450):
    """Factory: SileroVAD when available, else None (caller falls back).

    ``threshold`` is the hysteresis centre (default 0.5): lower = more
    sensitive — softer/shorter speech opens a turn earlier; higher =
    stricter. The cloud-STT loops pass a lowered value by design.
    """
    if not prefer_silero:
        return None
    try:
        _get_session()
    except Exception:
        return None
    cls = SoftSileroVAD if soft_reset else SileroVAD
    return cls(sample_rate=sample_rate, threshold=threshold,
               silence_ms=silence_ms)
