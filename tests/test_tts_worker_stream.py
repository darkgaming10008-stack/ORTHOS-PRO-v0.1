"""Tests for Level 1+ streaming TTS voice system.

Covers: _tts_worker streaming playback (bytes chunks -> OutputStream),
legacy blocking regression (Edge-style engines), barge-in detection,
clause-level sentence splitting, and TTSPlayer warmup/streaming passthrough.
"""

import queue
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ── Import main.py with heavy/stub-incompatible modules faked ──────────────
_STUBS = [
    "ui",
    "memory.memory_manager",
    "memory.chroma_memory",
    "actions.file_processor", "actions.flight_finder", "actions.open_app",
    "actions.weather_report", "actions.send_message", "actions.reminder",
    "actions.computer_settings", "actions.screen_processor",
    "actions.youtube_video", "actions.desktop", "actions.file_controller",
    "actions.code_helper", "actions.dev_agent", "actions.web_search",
    "actions.computer_control", "actions.game_updater", "actions.terminal",
    "core.logging_setup", "core.security", "core.tool_registry",
    "core.llm_provider", "core.observability", "core.mcp_manager",
    "core.tool_index",
]

_SAVED_MODS = {}
for _name in _STUBS:
    _SAVED_MODS[_name] = sys.modules.get(_name)
    sys.modules[_name] = MagicMock()

_SAVED_ARGV = sys.argv
sys.argv = ["main.py"]          # main.py runs argparse.parse_args() at import
import main  # noqa: E402
sys.argv = _SAVED_ARGV

for _name in _STUBS:
    if _SAVED_MODS[_name] is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _SAVED_MODS[_name]


# ── Fakes ──────────────────────────────────────────────────────────────────

class FakeOutStream:
    def __init__(self, recorder, **kw):
        self.recorder = recorder
        self.written = b""
        self.started = self.stopped = self.closed = False
        self.samplerate = kw.get("samplerate")
        self.dtype = kw.get("dtype")

    def start(self):
        self.started = True

    def write(self, arr):
        data = arr.tobytes()
        self.written += data
        self.recorder.append(("write", len(data)))

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class FakeSD:
    def __init__(self):
        self.recorder = []
        self.played = []
        self.streams = []

    def OutputStream(self, **kw):
        s = FakeOutStream(self.recorder, **kw)
        self.streams.append(s)
        return s

    def play(self, audio, rate):
        self.played.append((audio, rate))

    def stop(self):
        pass


class FakeStreamEngine:
    sample_rate = 24_000

    def __init__(self):
        self.texts = []

    def synthesize_stream(self, text):
        self.texts.append(text)
        yield b"\x01\x00\x02\x00" * 100
        yield b"\x03\x00" * 200


class FakeLegacyEngine:
    sample_rate = 22_050

    def __init__(self):
        self.texts = []

    def synthesize(self, text):
        self.texts.append(text)
        return np.ones(22_050, dtype=np.float32) * 0.5


def _make_app(player):
    """Bare Orthos instance with only the TTS worker state wired."""
    app = main.Orthos.__new__(main.Orthos)
    app.ui = MagicMock()
    app.ui.muted = False
    app._tts_ready = threading.Event()
    app._tts_queue = queue.Queue()
    app._audio_queue = queue.Queue(maxsize=2)
    app._should_stop = threading.Event()
    app._speaking = False
    app._speaking_lock = threading.Lock()
    app._tts = player
    app._out_stream = None
    # Speaker-echo rejection state (stop_speaking/speak touch these)
    app._echo_until = 0.0
    app._spoken_tail = []
    app._last_echo_text = None
    app._vad_ref = None
    app._echo_risk = True
    app._measuring_bleed = False
    app._mic_bleed = []
    app._bleed_lock = threading.Lock()
    app._speak_pending = 0
    return app


def _wait_until(cond, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def sd_fake():
    fake = FakeSD()
    with patch.object(main, "sd", fake):
        yield fake


# ── Streaming worker: chunk playback ───────────────────────────────────────

def test_streaming_worker_plays_chunks_via_output_stream(sd_fake):
    from core.tts import TTSPlayer

    engine = FakeStreamEngine()
    app = _make_app(TTSPlayer(engine))
    app._tts_ready.set()
    threading.Thread(target=app._tts_worker, daemon=True).start()

    app.speak("Hello there.")
    ok = _wait_until(lambda: sd_fake.streams and
                     sd_fake.streams[0].written == b"\x01\x00\x02\x00" * 100 + b"\x03\x00" * 200)
    assert ok, f"streamed bytes mismatch, got {sd_fake.streams[0].written if sd_fake.streams else None!r}"

    stream = sd_fake.streams[0]
    assert stream.samplerate == 24_000
    assert stream.dtype == "int16"
    assert stream.started and not stream.stopped
    assert engine.texts == ["Hello there."]

    assert _wait_until(lambda: not app._speaking)
    assert app._tts_queue.empty()
    assert "SPEAKING" in [c.args[0] for c in app.ui.set_state.call_args_list]
    assert "LISTENING" in [c.args[0] for c in app.ui.set_state.call_args_list]


def test_streaming_worker_multiple_utterances(sd_fake):
    from core.tts import TTSPlayer

    engine = FakeStreamEngine()
    app = _make_app(TTSPlayer(engine))
    app._tts_ready.set()
    threading.Thread(target=app._tts_worker, daemon=True).start()

    app.speak("First.")
    app.speak("Second.")
    ok = _wait_until(lambda: engine.texts == ["First.", "Second."] and
                     sd_fake.streams and
                     len(sd_fake.streams[0].written) == 2 * (400 + 400))
    assert ok
    assert _wait_until(lambda: not app._speaking)
    assert app._tts_queue.empty()


def test_stop_speaking_closes_output_stream(sd_fake):
    from core.tts import TTSPlayer

    app = _make_app(TTSPlayer(FakeStreamEngine()))
    app._tts_ready.set()
    threading.Thread(target=app._tts_worker, daemon=True).start()

    app.speak("Interrupt me.")
    assert _wait_until(lambda: sd_fake.streams and sd_fake.streams[0].written)
    app.stop_speaking()

    stream = sd_fake.streams[0]
    assert stream.stopped and stream.closed
    assert not app._speaking
    assert app._should_stop.is_set()
    assert app._audio_queue.empty()


# ── Legacy regression: Edge-style engines keep the blocking path ───────────

def test_legacy_worker_keeps_blocking_playback(sd_fake):
    from core.tts import TTSPlayer

    engine = FakeLegacyEngine()
    app = _make_app(TTSPlayer(engine))
    app._tts_ready.set()
    threading.Thread(target=app._tts_worker, daemon=True).start()

    app.speak("Legacy engine.")
    ok = _wait_until(lambda: sd_fake.played and len(sd_fake.played[0][0]) == 22_050)
    assert ok, "blocking sd.play was never called"
    audio, rate = sd_fake.played[0]
    assert rate == 22_050
    assert np.allclose(audio, np.ones(22_050, dtype=np.float32) * 0.5)
    assert not sd_fake.streams, "legacy path must not open an OutputStream"
    assert engine.texts == ["Legacy engine."]
    assert _wait_until(lambda: not app._speaking)
    assert app._tts_queue.empty()


def test_legacy_engine_reports_no_streaming_support():
    from core.tts import TTSPlayer

    assert TTSPlayer(FakeLegacyEngine()).supports_streaming is False
    assert TTSPlayer(FakeStreamEngine()).supports_streaming is True


# ── Barge-in detector ──────────────────────────────────────────────────────

def test_barge_detector_quiet_audio_never_triggers():
    det = main._BargeDetector(floor_rms=0.02)
    for _ in range(40):
        assert det.speech(0.005) is False


def test_barge_detector_loud_audio_triggers():
    det = main._BargeDetector(floor_rms=0.02)
    for _ in range(10):
        det.speech(0.005)
    assert det.speech(0.1) is True


def test_barge_detector_suppresses_speaker_echo():
    det = main._BargeDetector(floor_rms=0.02)
    for _ in range(30):
        det.speech(0.05)          # Orthos's voice echoes back into the mic
    assert det.speech(0.08) is False   # echo-level audio must NOT self-trigger
    assert det.speech(0.3) is True     # real user speech above echo still triggers


# ── Clause-level sentence splitting (voice mode) ───────────────────────────

@pytest.fixture(autouse=True)
def _voice_mode_off():
    from core import llm_client
    yield
    llm_client.set_voice_mode(False)


def test_sentence_split_always_works():
    from core.llm_client import _next_utterance

    assert _next_utterance("Hi there. Next") == ("Hi there.", "Next")


def test_blank_line_split_always_works():
    from core.llm_client import _next_utterance

    assert _next_utterance("Line one\n\nLine two") == ("Line one", "Line two")


def test_no_clause_split_when_voice_mode_off():
    from core.llm_client import _next_utterance

    long_run = ("alpha, " * 50).strip()   # 100+ chars, commas, no sentence end
    assert _next_utterance(long_run) is None


def test_clause_split_when_voice_mode_on():
    from core.llm_client import _next_utterance, set_voice_mode

    set_voice_mode(True)
    long_run = ("alpha, " * 50).strip()
    split = _next_utterance(long_run)
    assert split is not None
    utterance, rest = split
    assert utterance.endswith(",")
    assert rest and "alpha" in rest


def test_short_run_with_comma_not_split_even_in_voice_mode():
    from core.llm_client import _next_utterance, set_voice_mode

    set_voice_mode(True)
    assert _next_utterance("short, text") is None
    assert _next_utterance("a, b, c, d, e, f, g, h, i, j, k") is None


def test_set_voice_mode_toggle():
    from core.llm_client import _next_utterance, set_voice_mode

    long_run = ("beta, " * 50).strip()
    assert _next_utterance(long_run) is None
    set_voice_mode(True)
    assert _next_utterance(long_run) is not None
    set_voice_mode(False)
    assert _next_utterance(long_run) is None


# ── Warmup + streaming passthrough on TTSPlayer ────────────────────────────

def test_gemini_engine_warmup_preloads_client():
    from core.tts import GeminiTTSEngine

    engine = GeminiTTSEngine(api_key="AIza-test")
    engine._client = MagicMock()
    engine.warmup()
    assert engine._get_client() is engine._client   # client cached, no new build


def test_ttsplayer_warmup_delegates():
    from core.tts import GeminiTTSEngine, TTSPlayer

    engine = GeminiTTSEngine(api_key="AIza-test")
    engine._client = MagicMock()
    player = TTSPlayer(engine)
    player.warmup()
    assert engine._client.models is not None


def test_ttsplayer_warmup_noop_without_engine_support():
    from core.tts import TTSPlayer

    TTSPlayer(FakeLegacyEngine()).warmup()   # must not raise


def test_stream_rate_parsed_from_first_chunk_mime():
    from core.tts import GeminiTTSEngine

    engine = GeminiTTSEngine(api_key="AIza-test")

    def _chunk(mime, data):
        return SimpleNamespace(candidates=[
            SimpleNamespace(content=SimpleNamespace(parts=[
                SimpleNamespace(inline_data=SimpleNamespace(data=data, mime_type=mime)),
            ])),
        ])

    chunks = [
        _chunk("audio/l16; rate=16000; channels=1", b"\x00"),       # odd byte -> carried
        _chunk("audio/l16; rate=16000; channels=1", b"\x01\x02\x03"),
    ]
    client = MagicMock()
    client.models.generate_content_stream.return_value = iter(chunks)
    engine._client = client

    out = list(engine.synthesize_stream("hi"))
    assert out == [b"\x00\x01\x02\x03"]                              # aligned across chunks
    assert engine.sample_rate == 16_000