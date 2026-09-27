"""Tests for GeminiLiveVoiceEngine (Gemini Live as voice renderer) and factory wiring."""

import numpy as np
import pytest
from unittest.mock import patch

from core.tts import (
    GEMINI_LIVE_TTS_MODEL_DEFAULT,
    VOICE_RENDERER_PROMPT,
    GeminiLiveVoiceEngine,
    TTSPlayer,
    create_tts_player,
)


class _FakeFut:
    def result(self, timeout=None):
        return None


def _patch_threadsafe(fake):
    def _run(coro, loop):
        fake(coro, loop)
        return _FakeFut()

    return patch("core.tts.asyncio.run_coroutine_threadsafe", side_effect=_run)


# ── Factory wiring ─────────────────────────────────────────────────────────

def test_factory_wires_gemini_live_engine():
    player = create_tts_player({
        "tts_engine": "gemini_live",
        "gemini_voice_api_key": "AIza-test",
        "gemini_live_voice": "Kore",
    })
    assert isinstance(player, TTSPlayer)
    eng = player._engine
    assert isinstance(eng, GeminiLiveVoiceEngine)
    assert eng.api_key == "AIza-test"
    assert eng.voice == "Kore"
    assert eng.model == GEMINI_LIVE_TTS_MODEL_DEFAULT
    assert player.sample_rate == 24000


def test_factory_gemini_live_default_voice_is_charon():
    eng = create_tts_player({
        "tts_engine": "gemini_live",
        "gemini_voice_api_key": "AIza-test",
    })._engine
    assert eng.voice == "Charon"


def test_factory_gemini_live_requires_voice_api_key():
    with pytest.raises(RuntimeError, match="gemini_voice_api_key"):
        create_tts_player({"tts_engine": "gemini_live", "gemini_api_key": "AIza-shared"})
    with pytest.raises(RuntimeError, match="gemini_voice_api_key"):
        create_tts_player({"tts_engine": "gemini_live"})


def test_factory_gemini_live_custom_model():
    eng = create_tts_player({
        "tts_engine": "gemini_live",
        "gemini_voice_api_key": "AIza-test",
        "gemini_live_model": "gemini-3.1-flash-live-preview",
    })._engine
    assert eng.model == "gemini-3.1-flash-live-preview"


def test_factory_other_engines_unaffected():
    player = create_tts_player({"tts_engine": "edgetts"})
    assert not isinstance(player._engine, GeminiLiveVoiceEngine)


# ── Hashim Live (gemini-live-tts tool adapter) ──────────────────────────────

def test_hashim_factory_uses_configured_voice():
    player = create_tts_player({
        "tts_engine": "hashim_live",
        "gemini_voice_api_key": "AIza-test",
        "gemini_live_voice": "Fenrir",
    })
    assert isinstance(player, TTSPlayer)
    assert player._engine.voice == "Fenrir"


def test_hashim_factory_default_voice_is_kore():
    eng = create_tts_player({
        "tts_engine": "hashim_live",
        "gemini_voice_api_key": "AIza-test",
    })._engine
    assert eng.voice == "Kore"


try:
    from core.hashim_live_tts_adapter import HashimExactTTSEngine as _HashimEngine
    _HAS_ADAPTER = True
except Exception:
    _HAS_ADAPTER = False


@pytest.mark.skipif(not _HAS_ADAPTER, reason="gemini-live-tts tool deps not installed")
def test_adapter_voice_change_triggers_restart():
    """Changing the engine voice must restart the tool so the new voice
    is guaranteed to apply (the tool's in-place reconnect is unreliable)."""
    import core.hashim_live_tts_adapter as mod

    class _FakeGtts:
        _bg_thread = None
        _main_loop = None
        _input_queue = None

        def __init__(self):
            self.calls = {"restart": [], "tts": []}

        def restart(self, voice=None):
            self.calls["restart"].append(voice)

        def TTS(self, query, voice="Zephyr"):
            self.calls["tts"].append((query, voice))

    fake = _FakeGtts()
    old_gtts = mod._gtts
    mod._gtts = fake
    try:
        eng = _HashimEngine(api_key="AIza-test", voice="Kore")
        eng.synthesize("Hello")
        assert eng._applied_voice == "Kore"
        assert fake.calls["restart"] == ["Kore"]
        assert fake.calls["tts"] == [("Hello", "Kore")]
        fake.calls.clear()
        # Same voice again → no restart
        eng.synthesize("Again")
        assert fake.calls["restart"] == []
        # New voice → restart with the new voice, then speak with it
        eng.voice = "Charon"
        eng.synthesize("Now")
        assert fake.calls["restart"] == ["Charon"]
        assert fake.calls["tts"] == [("Now", "Charon")]
    finally:
        mod._gtts = old_gtts


# ── Engine behaviour ───────────────────────────────────────────────────────

def test_voice_renderer_prompt_forces_exact_recitation():
    low = VOICE_RENDERER_PROMPT.lower()
    assert "only a voice renderer" in low
    assert "do not" in low
    assert "paraphrase" in low
    assert "word for word" in low


def test_engine_synthesize_stream_requires_api_key():
    eng = GeminiLiveVoiceEngine(api_key="")
    with pytest.raises(RuntimeError, match="gemini_voice_api_key"):
        list(eng.synthesize_stream("Hello"))


def test_engine_synthesize_requires_api_key():
    eng = GeminiLiveVoiceEngine(api_key="")
    with pytest.raises(RuntimeError, match="gemini_voice_api_key"):
        eng.synthesize("Hello")


def test_engine_ensure_running_false_without_key():
    eng = GeminiLiveVoiceEngine(api_key="")
    assert eng.ensure_running() is False


def test_engine_defaults():
    eng = GeminiLiveVoiceEngine()
    assert eng.voice == "Charon"
    assert eng.model == GEMINI_LIVE_TTS_MODEL_DEFAULT
    assert eng.sample_rate == 24000


def test_interrupt_clears_pending_turn_and_sentences():
    import queue as _q
    eng = GeminiLiveVoiceEngine(api_key="AIza-test")
    q = _q.Queue()
    eng._turn_q = q
    q.put({"type": "chunk", "data": b"\x00" * 2400})
    with eng._pending_lock:
        eng._pending.append("Hello world.")
    eng.interrupt()
    assert eng._turn_q is None
    assert q.empty()
    assert eng._pending == []
    assert eng._interrupted.is_set()


# ── Full-response batching (drainer) ──────────────────────────────────────

def _spy_drainer(eng, debounce=0.1, max_chars=4000):
    eng._debounce = debounce
    eng._max_batch_chars = max_chars
    captured = {"texts": []}

    def fake_run_turn(text):
        captured["texts"].append(text)
        return np.zeros(100, dtype="<i2")

    eng._run_turn = fake_run_turn
    return captured


def _wait_for(predicate, timeout=3.0):
    import time as _t
    deadline = _t.monotonic() + timeout
    while _t.monotonic() < deadline:
        if predicate():
            return True
        _t.sleep(0.02)
    return False


def test_synthesize_stream_buffers_and_returns_empty():
    eng = GeminiLiveVoiceEngine(api_key="k")
    eng._debounce = 0.1
    with patch("core.tts.sd.play"), patch("core.tts.sd.wait"):
        assert list(eng.synthesize_stream("First.")) == []
        assert list(eng.synthesize_stream("Second.")) == []
    with eng._pending_lock:
        assert eng._pending == ["First.", "Second."]
    eng.stop()


def test_drainer_batches_sentences_into_one_turn():
    import time as _t
    eng = GeminiLiveVoiceEngine(api_key="k")
    captured = _spy_drainer(eng, debounce=0.1)
    with patch("core.tts.sd.play"), patch("core.tts.sd.wait"):
        eng.synthesize_stream("First sentence.")
        _t.sleep(0.03)
        eng.synthesize_stream("Second sentence.")
        assert _wait_for(lambda: len(captured["texts"]) == 1)
    assert captured["texts"] == ["First sentence. Second sentence."]
    eng.stop()


def test_drainer_splits_on_long_gap():
    import time as _t
    eng = GeminiLiveVoiceEngine(api_key="k")
    captured = _spy_drainer(eng, debounce=0.1)
    with patch("core.tts.sd.play"), patch("core.tts.sd.wait"):
        eng.synthesize_stream("First paragraph.")
        assert _wait_for(lambda: len(captured["texts"]) == 1)
        _t.sleep(0.25)
        eng.synthesize_stream("Second paragraph.")
        assert _wait_for(lambda: len(captured["texts"]) == 2)
    assert captured["texts"][0] == "First paragraph."
    assert captured["texts"][1] == "Second paragraph."
    eng.stop()


def test_drainer_flushes_when_batch_cap_reached():
    import time as _t
    eng = GeminiLiveVoiceEngine(api_key="k")
    captured = _spy_drainer(eng, debounce=1.0, max_chars=10)
    with patch("core.tts.sd.play"), patch("core.tts.sd.wait"):
        eng.synthesize_stream("AAAAAAAAAAA")
        assert _wait_for(lambda: len(captured["texts"]) == 1)
        eng.synthesize_stream("BBBB")
        assert _wait_for(lambda: len(captured["texts"]) == 2)
    assert captured["texts"][0] == "AAAAAAAAAAA"
    assert captured["texts"][1] == "BBBB"
    eng.stop()


def test_drainer_skips_playback_after_interrupt():
    import time as _t
    eng = GeminiLiveVoiceEngine(api_key="k")
    captured = _spy_drainer(eng, debounce=0.1)
    played = []
    with patch("core.tts.sd.play", side_effect=lambda a, sr: played.append(a)), \
         patch("core.tts.sd.wait"):
        eng.synthesize_stream("Interrupted turn.")
        _t.sleep(0.05)
        eng.interrupt()
        assert _wait_for(lambda: eng._pending == [] and not eng._new_sentence.is_set())
        _t.sleep(0.15)
    assert played == []
    eng.stop()


# ── Model-aware send routing (fix: no error-based fallback) ───────────────

def test_send_routing_2x_uses_client_content():
    eng = GeminiLiveVoiceEngine(api_key="k", model="gemini-2.5-flash-native-audio-preview-12-2025")
    assert eng._uses_realtime_text is False
    assert eng._supports_compression is False


def test_send_routing_3x_uses_realtime_input():
    eng = GeminiLiveVoiceEngine(api_key="k", model="gemini-3.1-flash-live-preview")
    assert eng._uses_realtime_text is True
    assert eng._supports_compression is True


def test_build_config_2x_omits_compression_and_resumption():
    eng = GeminiLiveVoiceEngine(api_key="k", model="gemini-2.5-flash-native-audio-preview-12-2025")
    cfg = eng._build_config()
    assert getattr(cfg, "context_window_compression", None) is None
    assert getattr(cfg, "session_resumption", None) is None


def test_build_config_3x_includes_compression():
    eng = GeminiLiveVoiceEngine(api_key="k", model="gemini-3.1-flash-live-preview")
    cfg = eng._build_config()
    assert cfg.context_window_compression is not None
    assert getattr(cfg, "session_resumption", None) is None


def test_build_config_uses_resumption_handle_when_present():
    eng = GeminiLiveVoiceEngine(api_key="k", model="gemini-3.1-flash-live-preview")
    eng._resumption_handle = "abc123"
    cfg = eng._build_config()
    assert cfg.session_resumption is not None
    assert cfg.session_resumption.handle == "abc123"


def test_build_config_thinking_disabled_for_3x():
    eng = GeminiLiveVoiceEngine(api_key="k", model="gemini-3.1-flash-live-preview")
    cfg = eng._build_config()
    assert cfg.thinking_config is not None
    assert cfg.thinking_config.thinking_budget == 0

def test_build_config_transcription_opt_in():
    eng = GeminiLiveVoiceEngine(api_key="k")
    eng._transcribe_output = False
    cfg = eng._build_config()
    assert cfg.output_audio_transcription is None
    eng._transcribe_output = True
    cfg = eng._build_config()
    assert cfg.output_audio_transcription is not None


# ── Retry-on-reconnect (fix: no more visible 1000-close errors) ───────────

def test_is_reconnectable_detects_server_close():
    from core.tts import _is_reconnectable
    assert _is_reconnectable("sent 1000 (OK); then received 1000 (OK)")
    assert _is_reconnectable("ConnectionClosedOK: sent 1000")
    assert _is_reconnectable("connection reset by peer")
    assert not _is_reconnectable("quota exceeded")


def test_send_quota_error_raises_without_retry():
    eng = GeminiLiveVoiceEngine(api_key="k")
    eng._connected.set()
    eng._loop = object()

    def fake_send(coro, loop):
        raise Exception("1011 RESOURCE_EXHAUSTED quota exceeded")

    with _patch_threadsafe(fake_send):
        with pytest.raises(RuntimeError, match="quota"):
            eng._run_turn("hi")


def test_send_reconnectable_error_retries_then_succeeds():
    eng = GeminiLiveVoiceEngine(api_key="k")
    eng._connected.set()
    eng._loop = object()
    calls = {"n": 0}

    def fake_send(coro, loop):
        calls["n"] += 1
        if calls["n"] == 1:
            raise Exception("sent 1000 (OK); then received 1000 (OK)")
        eng._emit({"type": "done"})

    with _patch_threadsafe(fake_send):
        out = eng._run_turn("hi")
    assert calls["n"] == 2
    assert out.size == 0


def test_send_hard_error_raises_immediately():
    eng = GeminiLiveVoiceEngine(api_key="k")
    eng._connected.set()
    eng._loop = object()

    def fake_send(coro, loop):
        raise Exception("boom")

    with _patch_threadsafe(fake_send):
        with pytest.raises(RuntimeError, match="send failed"):
            eng._run_turn("hi")


def test_turn_emits_error_event_as_runtime_error():
    eng = GeminiLiveVoiceEngine(api_key="k")
    eng._connected.set()
    eng._loop = object()

    def fake_send(coro, loop):
        eng._emit({"type": "error", "message": "server blew up", "quota": False})

    with _patch_threadsafe(fake_send):
        with pytest.raises(RuntimeError, match="server blew up"):
            eng._run_turn("hi")


def test_run_turn_returns_concatenated_pcm():
    eng = GeminiLiveVoiceEngine(api_key="k")
    eng._connected.set()
    eng._loop = object()

    def fake_send(coro, loop):
        eng._emit({"type": "chunk", "data": b"\x01\x00" * 4})
        eng._emit({"type": "chunk", "data": b"\x02\x00" * 4})
        eng._emit({"type": "done"})

    with _patch_threadsafe(fake_send):
        out = eng._run_turn("hi")
    assert out.dtype == np.dtype("<i2")
    assert len(out) == 8
    assert out.tolist() == [1, 1, 1, 1, 2, 2, 2, 2]


def test_run_turn_interrupt_returns_early():
    eng = GeminiLiveVoiceEngine(api_key="k")
    eng._connected.set()
    eng._loop = object()
    eng._interrupted.set()

    def fake_send(coro, loop):
        eng._emit({"type": "done"})

    with _patch_threadsafe(fake_send):
        out = eng._run_turn("hi")
    assert out.size == 0
