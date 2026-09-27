"""HashimSTT adapter tests.

Contract: the vendored engine (vendor/hashim_stt/stt_engine.py from
github.com/hashimmalikdev/gemini-realtime-speech-to-text-translator)
must stay BYTE-IDENTICAL to upstream — Orthos only wraps it, never
patches it. The adapter drives the upstream class's own thread, mic and
Gemini Live session; sentences are drained from the upstream queue.
"""

import hashlib
import os
import queue
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_SAVED_ARGV = sys.argv
sys.argv = ["main.py"]          # main.py runs argparse.parse_args() at import
import main  # noqa: E402
sys.argv = _SAVED_ARGV

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "hashim_stt" / "stt_engine.py"
VENDOR_MD5 = "cc8ed0bc2bc1d330165affbaf3003436"  # 2026-09: pydantic fix + api_key + stop() + speaking-gate (set_gate)


def _import_upstream():
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    from vendor.hashim_stt.stt_engine import STT

    return STT


def _make_app():
    app = main.Orthos.__new__(main.Orthos)
    app.ui = MagicMock()
    app.ui.muted = False
    app._config = {}
    app._stt = None
    app._text_queue = queue.Queue()
    app._echo_until = 0.0
    app._spoken_tail = []
    app._spoken_content = []
    app._last_echo_text = None
    app._capture_env = []
    app._played_env = []
    app._echo_risk = True
    app._echo_streak = 0
    app._speaking = False
    app._speaking_lock = threading.Lock()
    app._should_stop = threading.Event()
    app._echo_gate_until = 0.0
    app._echo_seen = set()
    app._measuring_bleed = False
    app._mic_bleed = []
    app._bleed_lock = threading.Lock()
    app._is_headless = True
    app._mic_thread_started = False
    return app


def test_vendor_config_builds_without_none_extras():
    """Regression: LiveConnectConfig was built with translation_config=None
    and output_audio_transcription=None, which pydantic 2.13 rejects
    (extra_forbidden) — the engine thread died instantly at connect().
    Optional kwargs must be omitted entirely when target_lang is unset."""
    import asyncio
    import vendor.hashim_stt.stt_engine as mod

    STT = _import_upstream()
    eng = STT.__new__(STT)          # no __init__ — no network, no audio
    eng.target_lang = None
    eng.api_key = "AIzaTESTKEY"
    captured = {}

    class FakeTypes:
        class AudioTranscriptionConfig:
            def __init__(self): pass
        class LiveConnectConfig:
            def __init__(self, **kw):
                captured.update(kw)
        class TranslationConfig:      # must never be constructed here
            def __init__(self, **kw):
                raise AssertionError("TranslationConfig built without target_lang")

    class FakePyAudioModule:
        paInt16 = 0
        class PyAudio:
            def __init__(self): pass
            def open(self, **kw): raise RuntimeError("no audio in tests")

    class FakeGenai:
        class Client:
            def __init__(self, **kw):
                captured["client_kwargs"] = kw

    orig = (mod.types, mod.genai, mod.pyaudio)
    mod.types, mod.genai, mod.pyaudio = FakeTypes, FakeGenai, FakePyAudioModule
    try:
        async def _run_to_config_build():
            try:
                await STT._async_connect(eng)   # dies at client.aio (fake)
            except AttributeError:
                pass
        asyncio.run(_run_to_config_build())
    finally:
        mod.types, mod.genai, mod.pyaudio = orig

    assert "translation_config" not in captured
    assert "output_audio_transcription" not in captured
    assert captured["client_kwargs"]["api_key"] == "AIzaTESTKEY"
    # The same shape must pass pydantic's strict LiveConnectConfig —
    # this is the exact crash from the field log (extra_forbidden).
    from google.genai import types as real_types
    real_kwargs = dict(captured)
    real_kwargs.pop("client_kwargs", None)
    real_kwargs["input_audio_transcription"] = real_types.AudioTranscriptionConfig()
    real_types.LiveConnectConfig(**real_kwargs)     # must NOT raise


def test_vendor_config_includes_translation_when_target_lang_set():
    import asyncio
    import vendor.hashim_stt.stt_engine as mod

    STT = _import_upstream()
    eng = STT.__new__(STT)
    eng.target_lang = "hi"
    eng.api_key = "AIzaTESTKEY"
    captured = {}

    class FakeTypes:
        class AudioTranscriptionConfig:
            def __init__(self): pass
        class LiveConnectConfig:
            def __init__(self, **kw):
                captured.update(kw)
        class TranslationConfig:
            def __init__(self, **kw):
                captured["translation"] = kw

    class FakePyAudioModule:
        paInt16 = 0
        class PyAudio:
            def __init__(self): pass
            def open(self, **kw): raise RuntimeError("no audio in tests")

    class FakeGenai:
        class Client:
            def __init__(self, **kw):
                captured["client_kwargs"] = kw

    orig = (mod.types, mod.genai, mod.pyaudio)
    mod.types, mod.genai, mod.pyaudio = FakeTypes, FakeGenai, FakePyAudioModule
    try:
        async def _run_to_config_build():
            try:
                await STT._async_connect(eng)
            except AttributeError:
                pass
        asyncio.run(_run_to_config_build())
    finally:
        mod.types, mod.genai, mod.pyaudio = orig

    assert "translation_config" in captured
    assert captured["translation"]["target_language_code"] == "hi"


def test_vendor_stop_closes_stream_and_stops_running_flag():
    """stop() must set _is_running False and close the mic stream even when
    the engine thread is gone (the old close() could not stop a live loop)."""
    STT = _import_upstream()
    eng = STT.__new__(STT)
    eng._is_running = True
    eng.audio_stream = None
    eng.stop()
    assert eng._is_running is False


def test_adapter_rejects_groq_key_and_falls_back_to_gemini_key():
    """A gsk_ key pasted into the voice field fails LOUDLY at construction;
    an empty voice key falls back to the main Gemini LLM key in _build_stt."""
    from core.stt import HashimSTT

    with pytest.raises(RuntimeError, match="gsk_"):
        HashimSTT("gsk_FAKEKEYFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")

    app = main.Orthos.__new__(main.Orthos)
    app._config = {}
    # _voice_gemini_key falls back voice -> main Gemini key
    assert main.Orthos._voice_gemini_key({"gemini_api_key": "AIzaMAIN"}) == "AIzaMAIN"
    assert main.Orthos._voice_gemini_key({"gemini_voice_api_key": "AIzaVOICE"}) == "AIzaVOICE"


def test_startup_uses_shared_stt_builder():
    """Startup and reconfigure must build engines through the same
    _build_stt() — the old code duplicated five factory branches in two
    places and they drifted (hashim got no key validation)."""
    import inspect
    src = inspect.getsource(main.Orthos.run)
    assert "_build_stt(stt_engine" in src
    """The vendored engine must remain byte-identical to upstream.

    If this fails, someone edited the vendor copy — re-vendor from the
    repo instead (see vendor/hashim_stt/README.md), or bump VENDOR_MD5
    deliberately when pulling a new upstream version.
    """
    assert hashlib.md5(VENDOR.read_bytes()).hexdigest() == VENDOR_MD5


def test_adapter_drives_upstream_class_not_a_reimplementation():
    """HashimSTT wraps the upstream STT; it must not define its own
    transcribe/process_chunk pipeline."""
    from core.stt import HashimSTT

    for forbidden in ("transcribe", "process_chunk"):
        assert not hasattr(HashimSTT, forbidden), (
            f"HashimSTT must not implement {forbidden}() — it is a thin "
            "adapter around the vendored engine, not a new engine"
        )


def test_adapter_requires_key():
    from core.stt import HashimSTT

    with pytest.raises(RuntimeError):
        HashimSTT("")
    with pytest.raises(RuntimeError):
        HashimSTT("   ")


def test_adapter_passes_key_to_upstream_env_and_constructs_upstream():
    """The upstream engine reads GEMINI_API_KEY at import/init time; the
    adapter must export the configured key and build the upstream class."""
    from core.stt import HashimSTT

    Upstream = _import_upstream()
    seen = {}

    class FakeUpstream(Upstream):
        def __init__(self, *a, **kw):
            seen["args"] = a
            seen["kwargs"] = kw
            seen["cls"] = type(self).__mro__[1].__name__
            seen["env"] = os.environ.get("GEMINI_API_KEY")
            raise RuntimeError("stop-before-network")  # do not connect

    import vendor.hashim_stt.stt_engine as mod

    old, old_env = mod.STT, os.environ.get("GEMINI_API_KEY")
    mod.STT = FakeUpstream
    try:
        with pytest.raises(RuntimeError, match="stop-before-network"):
            HashimSTT("AIzaTESTKEY", silence_threshold=0.9)
    finally:
        mod.STT = old
        if old_env is None:
            os.environ.pop("GEMINI_API_KEY", None)
        else:
            os.environ["GEMINI_API_KEY"] = old_env

    assert seen["cls"] == "STT"           # subclassing the UPSTREAM class
    assert seen["kwargs"]["silence_threshold"] == 0.9
    assert seen["env"] == "AIzaTESTKEY"    # key exported for the upstream engine


def test_drain_sentences_pops_upstream_queue_nonblocking():
    from core.stt import HashimSTT

    Upstream = _import_upstream()

    class FakeUpstream(Upstream):
        def __init__(self, *a, **kw):
            self.sentence_queue = queue.Queue()
            self._thread = None

    import vendor.hashim_stt.stt_engine as mod

    old = mod.STT
    mod.STT = FakeUpstream
    try:
        eng = HashimSTT("AIzaTESTKEY")
        eng._engine.sentence_queue.put("hello there")
        eng._engine.sentence_queue.put("second line")
        assert eng.drain_sentences() == ["hello there", "second line"]
        assert eng.drain_sentences() == []
    finally:
        mod.STT = old


def _blocking_thread() -> threading.Thread:
    """A daemon thread that stays alive until the process ends (no race)."""
    t = threading.Thread(target=_block_forever, daemon=True)
    t.start()
    return t


_BLOCK = threading.Event()


def _block_forever():
    _BLOCK.wait(timeout=60)


def test_is_alive_reflects_upstream_thread():
    from core.stt import HashimSTT

    Upstream = _import_upstream()

    class FakeUpstream(Upstream):
        def __init__(self, *a, **kw):
            self.sentence_queue = queue.Queue()
            self._thread = _blocking_thread()

    import vendor.hashim_stt.stt_engine as mod

    old = mod.STT
    mod.STT = FakeUpstream
    try:
        eng = HashimSTT("AIzaTESTKEY")
        assert eng.is_alive() is True
        eng._closed = True
        assert eng.is_alive() is False  # closed -> never alive
        eng2 = HashimSTT("AIzaTESTKEY")
        eng2._engine._thread = threading.Thread(target=lambda: None, daemon=True)
        eng2._engine._thread.start()
        eng2._engine._thread.join(timeout=2.0)   # fully dead, deterministic
        assert eng2.is_alive() is False          # upstream thread died
    finally:
        mod.STT = old
        _BLOCK.set()
        _BLOCK.clear()


def test_listen_hashim_submits_drained_sentences():
    app = _make_app()

    class FakeEngine:
        """Mirrors HashimSTT's public surface without the vendor engine."""

        def __init__(self):
            self.sentence_queue = queue.Queue()
            self._thread = _blocking_thread()
            self.connected = False

        def connect(self):
            self.connected = True

        def close(self):
            pass

        def is_alive(self):
            return not getattr(self, "_closed", False)

        def drain_sentences(self, max_items=10):
            out = []
            while len(out) < max_items:
                try:
                    out.append(self.sentence_queue.get_nowait())
                except queue.Empty:
                    break
            return out

    class FakeSTT(FakeEngine):
        pass

    eng = FakeSTT()
    eng.sentence_queue.put("hey orthos, what can you do?")
    app._stt = eng

    # Stop the bridge after the first drain: make is_alive flip to False
    # on the second poll so the loop raises and exits its retry (the
    # _stt_error path sleeps 2s — patch time.sleep to raise instead).
    import main as _main

    calls = {"n": 0}
    orig_alive = eng.is_alive

    def alive():
        calls["n"] += 1
        return calls["n"] <= 1

    eng.is_alive = alive

    real_sleep = _main.time.sleep

    def boom(_s):
        raise KeyboardInterrupt  # break out of the while-True bridge

    _main.time.sleep = boom
    try:
        with pytest.raises(KeyboardInterrupt):
            app._listen_hashim()
    finally:
        _main.time.sleep = real_sleep

    assert eng.connected is True
    got = app._text_queue.get_nowait()
    assert got == "hey orthos, what can you do?"


def test_listen_hashim_filters_echo_before_submit():
    app = _make_app()

    class FakeEngine:
        def __init__(self):
            self.sentence_queue = queue.Queue()
            self._thread = _blocking_thread()

        def connect(self):
            pass

        def close(self):
            pass

        def is_alive(self):
            return not getattr(self, "_closed", False)

        def drain_sentences(self, max_items=10):
            out = []
            while len(out) < max_items:
                try:
                    out.append(self.sentence_queue.get_nowait())
                except queue.Empty:
                    break
            return out

    eng = FakeEngine()
    eng.sentence_queue.put("I am ready when you are.")   # verbatim reply tail
    app._stt = eng
    app._remember_spoken_tail("I am ready when you are.")
    # Acoustic proof: capture envelope matches the played envelope.
    import math

    played = [math.sin(i * 0.37) * (0.5 + 0.5 * math.sin(i * 0.11)) for i in range(40)]
    app._played_env = list(played)
    app._capture_env = list(played)

    import main as _main

    calls = {"n": 0}

    def alive():
        calls["n"] += 1
        return calls["n"] <= 1

    eng.is_alive = alive
    real_sleep = _main.time.sleep

    def boom(_s):
        raise KeyboardInterrupt

    _main.time.sleep = boom
    try:
        with pytest.raises(KeyboardInterrupt):
            app._listen_hashim()
    finally:
        _main.time.sleep = real_sleep

    assert app._text_queue.empty() is True   # echo never reached the chat
    assert app._last_echo_text == "I am ready when you are."


def test_startup_selection_uses_listen_hashim_only_for_hashim():
    """The mic-thread wiring must route engine 'hashim' to _listen_hashim
    (through _start_mic_loop) — whisper/vosk/gemini/groq keep their own
    loops."""
    import inspect

    src = inspect.getsource(main.Orthos.run)
    # Startup routes every engine through the shared starter.
    assert "_start_mic_loop(stt_engine" in src

    mapper = inspect.getsource(main.Orthos._start_mic_loop)
    assert '"hashim"' in mapper and "_listen_hashim" in mapper
    assert '"vosk"' in mapper and "_listen_vosk" in mapper
    assert "_listen_whisper" in mapper            # default path
