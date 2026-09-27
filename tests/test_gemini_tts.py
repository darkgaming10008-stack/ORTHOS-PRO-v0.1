"""Tests for GeminiTTSEngine (gemini-3.1-flash-tts-preview) and factory wiring."""

import base64
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from core import tts
from core.tts import (
    GEMINI_TTS_MODEL_DEFAULT,
    GEMINI_TTS_VOICES,
    GeminiTTSEngine,
    GoogleCloudTTSEngine,
    GTTSEngine,
    FallbackTTSPlayer,
    TTSPlayer,
    create_tts_player,
)


# ── Fakes ──────────────────────────────────────────────────────────────────

def _inline(data: bytes, mime: str = "audio/l16; rate=24000; channels=1"):
    return SimpleNamespace(inline_data=SimpleNamespace(data=data, mime_type=mime))


def _resp(*parts):
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=list(parts)))],
        prompt_feedback=SimpleNamespace(block_reason=None),
    )


class FakeModels:
    def __init__(self, responses=None, stream_chunks=None):
        self.responses = list(responses or [])
        self.stream_chunks = list(stream_chunks or [])
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)

    def generate_content_stream(self, **kwargs):
        self.calls.append(kwargs)

        def _gen():
            for c in self.stream_chunks:
                yield c

        return _gen()


class FakeClient:
    def __init__(self, **kwargs):
        self.models = FakeModels(**kwargs)


def _engine(**kwargs):
    eng = GeminiTTSEngine(api_key="AIza-test", **kwargs)
    eng._client = FakeClient()
    return eng


# ── Factory wiring ─────────────────────────────────────────────────────────

def test_factory_wires_gemini_engine():
    player = create_tts_player({
        "tts_engine": "gemini",
        "gemini_voice_api_key": "AIza-test",
        "gemini_tts_voice": "Kore",
    })
    assert isinstance(player, TTSPlayer)
    eng = player._engine
    assert isinstance(eng, GeminiTTSEngine)
    assert eng.api_key == "AIza-test"
    assert eng.voice == "Kore"
    assert eng.model == GEMINI_TTS_MODEL_DEFAULT
    assert player.sample_rate == 24000


def test_factory_gemini_default_voice_is_puck():
    eng = create_tts_player({
        "tts_engine": "gemini",
        "gemini_voice_api_key": "AIza-test",
    })._engine
    assert eng.voice == "Puck"


def test_factory_gemini_requires_voice_api_key():
    with pytest.raises(RuntimeError, match="gemini_voice_api_key"):
        create_tts_player({"tts_engine": "gemini", "gemini_api_key": "AIza-shared"})
    with pytest.raises(RuntimeError, match="gemini_voice_api_key"):
        create_tts_player({"tts_engine": "gemini"})


def test_factory_googlecloud_requires_voice_api_key():
    with pytest.raises(RuntimeError, match="gemini_voice_api_key"):
        create_tts_player({"tts_engine": "googlecloud", "gemini_api_key": "AIza-shared"})
    with pytest.raises(RuntimeError, match="gemini_voice_api_key"):
        create_tts_player({"tts_engine": "googlecloud"})


def test_factory_other_engines_unaffected():
    player = create_tts_player({"tts_engine": "edgetts"})
    assert not isinstance(player._engine, GeminiTTSEngine)


# ── synthesize ─────────────────────────────────────────────────────────────

def test_synthesize_decodes_raw_pcm():
    eng = _engine()
    pcm = np.array([0, 1000, -1000, 32767], dtype="<i2").tobytes()
    eng._client.models.responses = [_resp(_inline(pcm))]

    out = eng.synthesize("Hello there.")

    assert eng._client.models.calls[0]["model"] == GEMINI_TTS_MODEL_DEFAULT
    assert out.dtype == np.float32
    assert out.shape == (4,)
    np.testing.assert_allclose(
        out,
        np.array([0, 1000, -1000, 32767], dtype=np.float32) / 32768.0,
    )
    assert eng.sample_rate == 24000


def test_synthesize_sends_audio_modality_and_voice():
    eng = _engine(voice="Fenrir")
    eng._client.models.responses = [_resp(_inline(b"\x00" * 4))]

    eng.synthesize("Hi.")

    cfg = eng._client.models.calls[0]["config"]
    assert cfg.response_modalities == ["AUDIO"]
    voice = cfg.speech_config.voice_config.prebuilt_voice_config.voice_name
    assert voice == "Fenrir"


def test_synthesize_reads_rate_from_mime():
    eng = _engine()
    eng._client.models.responses = [_resp(_inline(b"\x00" * 4, "audio/l16; rate=48000; channels=1"))]

    eng.synthesize("Hi.")

    assert eng.sample_rate == 48000


def test_synthesize_no_audio_raises():
    eng = _engine()
    eng._client.models.responses = [_resp()]  # no inline parts

    with pytest.raises(RuntimeError, match="no audio"):
        eng.synthesize("Hi.")


def test_synthesize_no_api_key_raises():
    eng = GeminiTTSEngine(api_key="")
    with pytest.raises(RuntimeError, match="gemini_voice_api_key"):
        eng.synthesize("Hi.")


def test_synthesize_empty_text_skips_call():
    eng = _engine()
    out = eng.synthesize("   ")
    assert out.size == 0
    assert eng._client.models.calls == []


# ── synthesize_stream ──────────────────────────────────────────────────────

def test_stream_aligns_odd_byte_boundaries():
    eng = _engine()
    pcm = np.array([111, 222, 333, 444, 555, 666], dtype="<i2").tobytes()
    eng._client.models.stream_chunks = [
        _resp(_inline(pcm[:5])),   # odd byte boundary (5 bytes)
        _resp(_inline(pcm[5:9])),  # another odd split (4 bytes)
        _resp(_inline(pcm[9:])),   # remainder
    ]

    chunks = list(eng.synthesize_stream("Hello."))

    joined = b"".join(chunks)
    assert joined == pcm
    assert all(len(c) % 2 == 0 for c in chunks)


def test_stream_skips_non_audio_chunks():
    eng = _engine()
    pcm = np.array([1, 2], dtype="<i2").tobytes()
    eng._client.models.stream_chunks = [
        SimpleNamespace(candidates=[]),                     # empty chunk
        SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text="partial")]))]),
        _resp(_inline(pcm)),
    ]

    chunks = list(eng.synthesize_stream("Hello."))

    assert b"".join(chunks) == pcm


def test_stream_no_api_key_raises():
    eng = GeminiTTSEngine(api_key="")
    with pytest.raises(RuntimeError, match="gemini_voice_api_key"):
        list(eng.synthesize_stream("Hi."))


# ── speak ──────────────────────────────────────────────────────────────────

def test_speak_plays_decoded_audio(monkeypatch):
    eng = _engine()
    pcm = np.array([0, 1000], dtype="<i2").tobytes()
    eng._client.models.responses = [_resp(_inline(pcm))]
    played = {}

    def fake_play(audio, rate):
        played["audio"] = audio
        played["rate"] = rate

    monkeypatch.setattr(tts.sd, "play", fake_play)
    monkeypatch.setattr(tts.sd, "wait", lambda: None)

    eng.speak("Hello.")

    assert played["rate"] == 24000
    assert played["audio"].shape == (2,)


# ── Voice catalogue ────────────────────────────────────────────────────────

def test_voice_catalogue_has_all_official_voices():
    assert len(GEMINI_TTS_VOICES) == 30
    assert "Puck" in GEMINI_TTS_VOICES
    assert "Charon" in GEMINI_TTS_VOICES
    assert "Fenrir" in GEMINI_TTS_VOICES
    assert len(set(GEMINI_TTS_VOICES)) == 30


# ── Gemini STT shares the voice API key ────────────────────────────────────

def test_gemini_stt_requires_voice_api_key():
    from core.stt import GeminiSTT

    with pytest.raises(RuntimeError, match="gemini_voice_api_key"):
        GeminiSTT("")


def test_gemini_stt_accepts_voice_api_key():
    from core.stt import GeminiSTT

    stt = GeminiSTT("AIza-test")
    assert stt._client is not None


# ── GoogleCloudTTSEngine fixes ───────────────────────────────────────────────

def test_googlecloud_speaking_rate_defaults_to_one():
    eng = GoogleCloudTTSEngine(api_key="AIza-test")
    assert eng.speaking_rate == 1.0

def test_googlecloud_speaking_rate_custom():
    eng = GoogleCloudTTSEngine(api_key="AIza-test", speaking_rate=1.2)
    assert eng.speaking_rate == 1.2


# ── gTTS error handling ─────────────────────────────────────────────────────

def test_gtts_synthesize_raises_on_quota(monkeypatch):
    from gtts.tts import gTTSError
    eng = GTTSEngine(voice="hi")
    monkeypatch.setattr(tts.sd, "play", lambda *a, **k: None)
    monkeypatch.setattr(tts.sd, "wait", lambda: None)

    def fake_save(self, path):
        raise gTTSError("You exceeded your current quota, quotaValue: '10'")

    monkeypatch.setattr("gtts.tts.gTTS.save", fake_save)

    with pytest.raises(RuntimeError, match="gTTS quota exceeded"):
        eng.synthesize("Hello world")


def test_gtts_synthesize_raises_on_generic_error(monkeypatch):
    eng = GTTSEngine(voice="hi")
    monkeypatch.setattr("gtts.tts.gTTS.save", lambda self, path: (_ for _ in ()).throw(RuntimeError("Network error")))

    with pytest.raises(RuntimeError, match="gTTS synthesis failed"):
        eng.synthesize("Hello world")


# ── Fallback chain ───────────────────────────────────────────────────────────

def test_fallback_returns_fallback_player_when_multiple_engines():
    cfg = {
        "tts_engine": "edgetts",
        "tts_fallback": "gtts",
    }
    player = create_tts_player(cfg)
    assert isinstance(player, FallbackTTSPlayer)
    assert len(player._players) >= 2

def test_fallback_single_engine_returns_plain_player():
    cfg = {
        "tts_engine": "edgetts",
        "tts_fallback": "",
    }
    player = create_tts_player(cfg)
    assert isinstance(player, TTSPlayer)
    assert not isinstance(player, FallbackTTSPlayer)

def test_fallback_skips_duplicates():
    cfg = {
        "tts_engine": "edgetts",
        "tts_fallback": "edgetts,gtts",
    }
    player = create_tts_player(cfg)
    names = [type(p._engine).__name__ for p in player._players]
    assert names.count("EdgeTTSEngine") == 1

def test_fallback_googlecloud_skipped_when_no_api_key():
    cfg = {
        "tts_engine": "googlecloud",
        "tts_fallback": "gtts",
        "gemini_voice_api_key": "",
    }
    player = create_tts_player(cfg)
    assert isinstance(player, TTSPlayer)
    assert isinstance(player._engine, GTTSEngine)

def test_fallback_falls_back_on_synthesize_failure():
    class FailingEngine:
        def synthesize(self, text):
            raise RuntimeError("primary failed")
        def speak(self, text):
            raise RuntimeError("primary failed")
        def stop(self):
            pass
        @property
        def sample_rate(self):
            return 24000

    class GoodEngine:
        def __init__(self):
            self.called = False
        def synthesize(self, text):
            self.called = True
            return np.zeros(10, dtype=np.float32)
        def speak(self, text):
            pass
        def stop(self):
            pass
        @property
        def sample_rate(self):
            return 24000

    good = GoodEngine()
    players = [TTSPlayer(FailingEngine()), TTSPlayer(good)]
    fb = FallbackTTSPlayer(players)
    out = fb.synthesize("hi")
    assert good.called
    assert out.shape == (10,)

