"""
Adapter that uses the EXACT hashimmalikdev/gemini-live-tts tool.

The repo is cloned at  tools/gemini-live-tts  and its
Google_Gemini_Live_TTS.py module exposes a global TTS(query, voice) function
(plus its own background PyAudio playback worker).  This adapter wraps that
function so the rest of Orthos can treat it as a standard TTS engine.
"""
from __future__ import annotations

import os
import sys
import queue
import threading
import time
from pathlib import Path

import numpy as np

# ── Inject the repo path so `import Google_Gemini_Live_TTS` works ──────────
_REPO_DIR = Path(__file__).resolve().parent.parent / "tools" / "gemini-live-tts"
if str(_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(_REPO_DIR))

import Google_Gemini_Live_TTS as _gtts  # noqa: E402


class _Tap:
    """Latest PCM sink callback shared with the tapping playback worker."""
    fn = None


def _tapping_playback_worker(playback_queue=None):
    """Drop-in replacement for the tool's _global_playback_worker.

    Identical queue->PyAudio behaviour, but every batch of raw PCM is handed
    to ``_Tap.fn`` first so Orthos can drive real lip-sync from the actual
    audio instead of estimating a syllable schedule from text.
    """
    pq = playback_queue if playback_queue is not None else _gtts._playback_queue
    gen = _gtts._session_gen
    pya = _gtts.pyaudio.PyAudio()
    try:
        stream = pya.open(
            format=_gtts.FORMAT,
            channels=_gtts.CHANNELS,
            rate=_gtts.RECEIVE_SAMPLE_RATE,
            output=True,
            frames_per_buffer=1024,
        )
        while _gtts._session_gen == gen:
            chunks = []
            while not pq.empty():
                try:
                    chunks.append(pq.get_nowait())
                except queue.Empty:
                    break
            if chunks:
                joined = b"".join(chunks)
                fn = _Tap.fn
                if fn is not None:
                    try:
                        fn(joined)
                    except Exception:
                        pass
                stream.write(joined)
                for _ in range(len(chunks)):
                    pq.task_done()
            else:
                time.sleep(0.001)
    except Exception:
        pass


# Replace the module-global worker BEFORE any background thread is started:
# warmup() runs after this import, and start_background_thread resolves the
# worker from module globals at spawn time.
_gtts._global_playback_worker = _tapping_playback_worker


class HashimExactTTSEngine:
    """Wraps the exact GitHub tool (Google_Gemini_Live_TTS.py).

    The original tool plays audio through its own PyAudio worker thread and
    waits for the full turn.  `synthesize` asks the tool to speak and returns
    an empty numpy array (audio is already played by the tool's worker).
    """

    def __init__(self, api_key: str = "", voice: str = "Kore"):
        self.api_key = api_key
        self.voice = voice
        self._sample_rate = 24000
        self._lock = threading.Lock()
        self._applied_voice = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def set_pcm_sink(self, fn) -> None:
        """Register a callback(raw_pcm_bytes) fired for every playback batch.

        Lets the app drive real lip-sync from the exact audio being played.
        """
        _Tap.fn = fn

    def synthesize(self, text: str) -> np.ndarray:
        if not text.strip():
            return np.zeros(0, dtype=np.float32)
        if self.voice != self._applied_voice:
            _gtts.restart(voice=self.voice)
            self._applied_voice = self.voice
        with self._lock:
            # The exact tool's TTS() blocks until the spoken turn finishes.
            _gtts.TTS(text, voice=self.voice)
        return np.zeros(0, dtype=np.float32)

    def speak(self, text: str) -> None:
        self.synthesize(text)

    def stop(self) -> None:
        try:
            _gtts.restart()
        except Exception:
            pass

    def warmup(self) -> None:
        # Ensure the background thread + session are initialised.
        if not self.api_key:
            return
        os.environ["GEMINI_API_KEY"] = self.api_key
        if self.voice != self._applied_voice:
            _gtts.restart(voice=self.voice)
            self._applied_voice = self.voice
        if _gtts._bg_thread is None:
            # Force init of the background loop
            _gtts.TTS(".", voice=self.voice)