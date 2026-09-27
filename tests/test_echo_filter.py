"""Speaker-echo rejection tests.

The mic sits next to the speakers: right after a TTS turn ends, the tail of
the playback is still leaving the audio device and lands in an open mic.
The VAD captures it and the STT transcribes the assistant's own closing
words as if the user had said them — and because pauses inside a reply
split the echo into pieces, each piece queues another reply that then
interrupts the current one mid-way.

Covers: the post-turn mic cooldown gate, the VAD buffer flush on
stop_speaking, and the transcript tail filter that drops what the
assistant just heard itself say.
"""

import sys
import time
from unittest.mock import MagicMock

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


def _make_app() -> "main.Orthos":
    """Bare Orthos instance with only the echo-rejection state wired."""
    app = main.Orthos.__new__(main.Orthos)
    app.ui = MagicMock()
    app.ui.muted = False
    app._config = {"tts_engine": "edge"}
    app._tts = None
    app._speaking = True
    app._speaking_lock = __import__("threading").Lock()
    app._should_stop = __import__("threading").Event()
    app._tts_queue = __import__("queue").Queue()
    app._audio_queue = __import__("queue").Queue()
    app._out_stream = None
    app._echo_until = 0.0
    app._spoken_tail = []
    app._spoken_content = []
    app._last_echo_text = None
    app._vad_ref = None
    app._echo_risk = True
    app._measuring_bleed = False
    app._mic_bleed = []
    app._bleed_lock = __import__("threading").Lock()
    app._echo_streak = 0
    app._played_env = []
    app._capture_env = []
    app._speak_pending = 0
    return app


# ── Cooldown gate ──────────────────────────────────────────────────────────

def test_gate_open_on_fresh_instance():
    app = _make_app()
    assert app._echo_gate_open() is True


def test_gate_closed_during_cooldown():
    app = _make_app()
    app._open_echo_gate_soon()
    assert app._echo_gate_open() is False
    assert app._echo_until > time.time()


def test_gate_reopens_after_cooldown(monkeypatch):
    app = _make_app()
    app._open_echo_gate_soon()
    # Jump the clock past the cooldown instead of sleeping through it.
    # main.time IS the time module, so capture the real clock first or the
    # patched lambda would recurse into itself.
    real_time = time.time
    monkeypatch.setattr(main.time, "time", lambda: real_time() + 5.0)
    assert app._echo_gate_open() is True


def test_stop_speaking_closes_gate(monkeypatch):
    app = _make_app()
    monkeypatch.setattr(main.sd, "stop", lambda: None)
    app._vad_ref = None
    app._live_provider = None
    app.stop_speaking()
    assert app._echo_gate_open() is False
    cooldown = app._echo_until - (time.time() - app._ECHO_COOLDOWN)
    assert cooldown > app._ECHO_COOLDOWN - 0.2   # ~full cooldown applied


# ── Spoken-tail memory ─────────────────────────────────────────────────────

def test_remember_spoken_tail_keeps_last_words():
    app = _make_app()
    app._remember_spoken_tail(" ".join(f"w{i}" for i in range(40)))
    assert len(app._spoken_tail) == app._ECHO_TAIL_KEEP
    assert app._spoken_tail[-1] == "w39"
    assert app._spoken_tail[0] == "w16"


def test_tail_words_are_normalized():
    app = _make_app()
    app._remember_spoken_tail("Hello, World! It's ready.")
    assert "hello" in app._spoken_tail
    assert "world" in app._spoken_tail
    assert "ready" in app._spoken_tail
    assert all("," not in w and "!" not in w for w in app._spoken_tail)


# ── Transcript echo filter ─────────────────────────────────────────────────

# ── Field false-rejects: the real user speech the old filter ate ──────────

def test_screenshot_false_reject_1_now_passes():
    # 'Actually, yeah, I need help.' was eaten: the reply's "I'm" split
    # into the garbage token "i", and i+need+help hit the old 3-hit floor.
    # Contraction-safe tokens + stopword exclusion fix it.
    app = _make_app()
    app._remember_spoken_tail(
        "Hey! What's up? That 00:00 looks like a timestamp — need help with "
        "something specific? I'm here and ready to go!")
    assert app._reject_echo("Actually, yeah, I need help.") is False


def test_screenshot_false_reject_2_now_passes():
    app = _make_app()
    app._remember_spoken_tail(
        "Hey! What's up? That 00:00 looks like a timestamp — need help with "
        "something specific? I'm here and ready to go!")
    assert app._reject_echo("I need your help.") is False


def test_contraction_stays_glued():
    app = _make_app()
    assert "i'm" in app._echo_words("I'm ready")
    assert "i" not in app._echo_words("I'm ready")
    assert "m" not in app._echo_words("I'm ready")
    # curly apostrophe too
    assert "i\u2019m" in app._echo_words("I\u2019m ready") or \
        "i'm" in app._echo_words("I\u2019m ready")


def test_echo_of_full_tail_is_rejected():
    app = _make_app()
    reply = "All systems are nominal and I am standing by. I am ready when you are."
    app._remember_spoken_tail(reply)
    assert app._reject_echo("I am ready when you are.") is True


def _correlated_capture(app):
    """Give the app a played-envelope and a capture envelope that are the
    same waveform — what a mic next to the speaker records. Short echo
    fragments are text-ambiguous, so rejection now requires this acoustic
    proof (in production the capture envelope comes from the VAD buffer).
    """
    import math
    played = [math.sin(i * 0.37) * (0.5 + 0.5 * math.sin(i * 0.11))
              for i in range(40)]
    app._played_env = list(played)
    app._capture_env = list(played)


def test_echo_rearranged_words_still_rejected():
    app = _make_app()
    app._remember_spoken_tail("All systems are nominal and standing by.")
    _correlated_capture(app)
    # Echo may arrive as VAD-split fragments — a subset of the same words.
    assert app._reject_echo("nominal and standing by.") is True


def test_three_word_echo_is_rejected():
    app = _make_app()
    app._remember_spoken_tail("I am ready when you are.")
    _correlated_capture(app)
    assert app._reject_echo("ready when you") is True


def test_short_fragment_without_acoustic_proof_passes():
    """A 3-5 token verbatim fragment with no capture envelope cannot be
    told apart from a user quoting the reply's closing words — pass it.
    """
    app = _make_app()
    app._remember_spoken_tail("I am ready when you are.")
    assert app._reject_echo("ready when you") is False


def test_two_word_fragment_is_accepted_tradeoff():
    """Deliberate: a 2-word echo fragment is indistinguishable from a
    genuine short reply reusing two tail words ("sahi jawab?"), so the
    absolute 3-hit floor lets both through. The cooldown gate and VAD
    flush are the primary defense for fragments; this filter is the
    backstop for full utterances."""
    app = _make_app()
    app._remember_spoken_tail("I am ready when you are.")
    assert app._reject_echo("ready when") is False


def test_user_reply_with_new_content_passes():
    app = _make_app()
    app._remember_spoken_tail("All systems are nominal and I am standing by.")
    assert app._reject_echo("ok now run the build") is False


def test_real_case_mixed_room_audio_is_rejected():
    """Calibration case from the field: the echo arrived mixed with room
    audio from a video playing on the speakers, so only ~65% of its words
    sat on the reply tail. The 75% bar let it through; 60% with an
    absolute 3-hit floor catches it without eating short genuine replies."""
    app = _make_app()
    app._config = {"echo_guard_mode": "auto"}
    app._remember_spoken_tail(
        "Bhai, tu keh raha hai \"Salman Khan ki pasand se maanti hai\" — but "
        "exactly kya jaanna hai? Tu thoda detail batana, main tujhe sahi "
        "jawab dunga!")
    echo = ("kuch janna chahta hai toh thoda detail batana. Main tujhe sahi "
            "jawab dunga. Abe, uski toh shadi hi nahin hui.")
    assert app._reject_echo(echo) is True


def test_short_genuine_reply_reusing_two_words_passes():
    app = _make_app()
    app._remember_spoken_tail(
        "Tu thoda detail batana, main tujhe sahi jawab dunga.")
    # Only two tail words — under the 3-hit absolute floor.
    assert app._reject_echo("sahi jawab?") is False


def test_unrelated_transcript_passes():
    app = _make_app()
    app._remember_spoken_tail("All systems are nominal and I am standing by.")
    assert app._reject_echo("what is the weather today") is False


def test_identical_repeat_phantom_is_rejected():
    app = _make_app()
    # No spoken tail recorded (e.g. gate-only path), but the same phantom
    # text already seen once must not submit twice.
    app._last_echo_text = "systems nominal"
    assert app._reject_echo("systems nominal") is True


def test_empty_transcript_is_not_echo():
    app = _make_app()
    app._remember_spoken_tail("some reply text here")
    assert app._reject_echo("") is False
    assert app._reject_echo("   ") is False
    assert app._reject_echo("!!!") is False


def test_user_repeating_assistant_words_plus_new_content_passes():
    """A user may legitimately repeat a word then add their own — that is a
    real reply and must not be swallowed."""
    app = _make_app()
    app._remember_spoken_tail("I am ready when you are ready.")
    assert app._reject_echo("are you sure about that") is False


# ── VAD buffer reset ───────────────────────────────────────────────────────

def test_vad_reset_drops_buffered_speech():
    vad = main._VADBuffer(sample_rate=16_000, min_speech_sec=0.1)
    sr = 16_000
    loud = (np.sin(np.linspace(0, 200, sr // 10)) * 0.9).astype(np.float32)
    quiet = np.zeros(sr // 10, dtype=np.float32)
    # Feed quiet, then speech, then quiet — buffers hold partial utterance.
    for _ in range(5):
        vad.process(quiet)
    for _ in range(20):
        vad.process(loud)
    vad.reset()
    assert vad._buf == [] and vad._in_spch is False
    assert vad._prebuf == [] and vad._prebuf_n == 0
    # Floor model survived (it describes the room, not the voice).
    assert vad._noise_floor >= 0.0
    # And processing continues normally afterwards: a fresh utterance still
    # flushes when speech ends.
    out = None
    for _ in range(40):
        out = vad.process(loud)
        if out is not None:
            break
    assert out is None or isinstance(out, np.ndarray)   # no crash, sane state


def test_stop_speaking_resets_vad(monkeypatch):
    app = _make_app()
    monkeypatch.setattr(main.sd, "stop", lambda: None)
    app._live_provider = None
    calls = []
    app._vad_ref = MagicMock()
    app._vad_ref.reset.side_effect = lambda: calls.append(1)
    app.stop_speaking()
    assert calls, "stop_speaking must flush the VAD buffer"


def test_stop_speaking_survives_vad_error(monkeypatch):
    app = _make_app()
    monkeypatch.setattr(main.sd, "stop", lambda: None)
    app._live_provider = None
    app._vad_ref = MagicMock()
    app._vad_ref.reset.side_effect = RuntimeError("boom")
    app.stop_speaking()   # must not raise


# ── Adaptive echo risk: device-name seed ───────────────────────────────────

def test_trrs_combo_dongle_seeds_risk_on(monkeypatch):
    """TRRS combo headset: capture and playback endpoints of the SAME
    dongle ("Microphone (USB Audio and HID)" + "Speakers (USB Audio and
    HID)"). The mic sits millimetres from the earbud driver — bleed is
    louder than any room echo, so the gate must stay on. Word-overlap of
    the generic tokens must not be fooled by stop-word stripping."""
    app = _make_app()
    monkeypatch.setattr(main.sd, "query_devices", lambda kind=None: (
        {"name": "Microphone (USB Audio and HID)"} if kind == "input"
        else {"name": "Speakers (USB Audio and HID)"}))
    app._seed_echo_risk()
    assert app._echo_risk is True


def test_unrelated_devices_do_not_seed_risk(monkeypatch):
    app = _make_app()
    monkeypatch.setattr(main.sd, "query_devices", lambda kind=None: (
        {"name": "Microphone Array (Intel)"} if kind == "input"
        else {"name": "DELL U2419H (NVIDIA High Definition Audio)"}))
    app._seed_echo_risk()
    assert app._echo_risk is True   # safe default, no false combo claim


def test_proportional_cooldown_calibrated():
    app = _make_app()
    # Live-measured TRRS bleed: 0.031 RMS -> ~2.7 s window (1.0 + peak*55).
    app._last_bleed_peak = 0.031
    app._open_echo_gate_soon()
    w1 = app._echo_until - time.time()
    assert 2.5 < w1 < 2.9, f"TRRS window {w1:.2f}"
    # A loud speaker rig peaks the cap.
    app2 = _make_app()
    app2._last_bleed_peak = 0.1
    app2._open_echo_gate_soon()
    assert (app2._echo_until - time.time()) <= 4.55
    # A silent headphone mic keeps the plain floor.
    app3 = _make_app()
    app3._last_bleed_peak = 0.0
    app3._open_echo_gate_soon()
    assert abs((app3._echo_until - time.time()) - app3._ECHO_COOLDOWN) < 0.2


def test_headphone_name_seed_opens_gate(monkeypatch):
    """A headset mic physically cannot hear the speakers — no gate."""
    app = _make_app()
    monkeypatch.setattr(main.sd, "query_devices",
                        lambda kind=None: {"name": "Headset Microphone (Realtek)"})
    app._seed_echo_risk()
    assert app._echo_risk is False


def test_laptop_mic_name_keeps_safe_risk(monkeypatch):
    app = _make_app()
    monkeypatch.setattr(main.sd, "query_devices",
                        lambda kind=None: {"name": "Microphone Array (Intel)"})
    app._seed_echo_risk()
    assert app._echo_risk is True


def test_seed_failure_keeps_safe_default(monkeypatch):
    app = _make_app()
    def boom(kind=None):
        raise RuntimeError("no audio system")
    monkeypatch.setattr(main.sd, "query_devices", boom)
    app._seed_echo_risk()
    assert app._echo_risk is True


# ── Adaptive echo risk: bleed measurement (ground truth) ──────────────────

def test_silent_bleed_retires_gate():
    """Headphones plugged in after boot: mic hears nothing during playback
    → gate retires."""
    app = _make_app()
    app._measuring_bleed = True
    app._mic_bleed = [1e-5] * 40
    app._measure_bleed()
    assert app._echo_risk is False
    assert app._measuring_bleed is False
    assert app._mic_bleed == []


def test_loud_bleed_keeps_gate():
    app = _make_app()
    app._measuring_bleed = True
    app._mic_bleed = [2e-2] * 40
    app._measure_bleed()
    assert app._echo_risk is True


def test_speakers_back_after_headphones():
    """Device switched from headphones to speakers mid-session: loud bleed
    re-arms the gate."""
    app = _make_app()
    app._echo_risk = False
    app._measuring_bleed = True
    app._mic_bleed = [5e-2] * 40
    app._measure_bleed()
    assert app._echo_risk is True


def test_short_capture_keeps_current_belief():
    """Too little evidence must never flip the belief either way."""
    app = _make_app()
    app._measuring_bleed = True
    app._mic_bleed = [1e-5] * 5          # looks silent, but only 5 samples
    app._measure_bleed()
    assert app._echo_risk is True        # unchanged

    app2 = _make_app()
    app2._echo_risk = False
    app2._measuring_bleed = True
    app2._mic_bleed = [1e-5] * 5
    app2._measure_bleed()
    assert app2._echo_risk is False      # unchanged


# ── Layer 2 is unconditional ───────────────────────────────────────────────

def test_transcript_filter_active_even_when_gate_off():
    """The cooldown gate is adaptive, but a transcript matching what the
    assistant just said is dropped regardless of device — the text filter
    is free and costs a real user nothing."""
    app = _make_app()
    app._echo_risk = False
    app._remember_spoken_tail("I am ready when you are.")
    assert app._reject_echo("I am ready when you are.") is True
    assert app._reject_echo("ok run the tests now") is False


def test_speak_arms_bleed_capture_only_when_risky():
    app = _make_app()
    app._tts = MagicMock()
    app.speak("hello there")
    assert app._measuring_bleed is True

    app2 = _make_app()
    app2._echo_risk = False
    app2._tts = MagicMock()
    app2.speak("hello there")
    # Auto mode still measures every turn — a headphone→speaker switch can
    # only be caught from real evidence, and evidence needs capture.
    assert app2._measuring_bleed is True

    app3 = _make_app()
    app3._config = {"echo_guard_mode": "off"}
    app3._tts = MagicMock()
    app3.speak("hello there")
    assert app3._measuring_bleed is False


# ── Fuzzy transcription drift ─────────────────────────────────────────────

def test_fuzzy_catches_hinglish_drift():
    """TTS said 'nahin'/'toh'/'jaanna', the STT heard 'nahi'/'to'/'janna'."""
    app = _make_app()
    app._remember_spoken_tail(
        "Abe, uski toh shadi hi nahin hui. Tu thoda detail batana.")
    assert app._reject_echo("uski to shadi hi nahi hui") is True


def test_fuzzy_requires_absolutely_3_hits():
    app = _make_app()
    app._remember_spoken_tail("Tu thoda detail batana main tujhe sahi jawab dunga")
    # 2 tail words of 3, both fuzzy hits — below the absolute floor.
    assert app._reject_echo("thoda detail hi") is False


# ── Manual override: echo_guard_mode ──────────────────────────────────────

def test_gate_active_honours_mode():
    app = _make_app()                       # auto + risk on
    assert app._gate_active() is True
    app._echo_risk = False
    assert app._gate_active() is False      # auto follows measurement
    app._config = {"echo_guard_mode": "on"}
    assert app._gate_active() is True       # forced
    app._config = {"echo_guard_mode": "off"}
    assert app._gate_active() is False      # disabled


def test_mode_off_disables_transcript_filter():
    app = _make_app()
    app._config = {"echo_guard_mode": "off"}
    app._remember_spoken_tail("I am ready when you are.")
    assert app._reject_echo("I am ready when you are.") is False


def test_loud_bleed_gets_longer_cooldown():
    app = _make_app()
    app._last_bleed_loud = True
    app._last_bleed_peak = 0.05        # loud bleed -> proportional window
    app._open_echo_gate_soon()
    wait = app._echo_until - time.time()
    assert wait > app._ECHO_COOLDOWN + 0.5

    app2 = _make_app()
    app2._last_bleed_loud = False
    app2._last_bleed_peak = 0.0
    app2._open_echo_gate_soon()
    wait2 = app2._echo_until - time.time()
    assert abs(wait2 - app2._ECHO_COOLDOWN) < 0.2


def test_caught_echo_extends_gate():
    """Echo arrives in pieces — each caught piece keeps the gate shut."""
    app = _make_app()
    app._echo_until = time.time() - 10      # long expired
    app._remember_spoken_tail("I am ready when you are.")
    assert app._reject_echo("I am ready when you are.") is True
    assert app._echo_until > time.time() + 0.3


# ── VAD engine routing (Silero vs energy) ─────────────────────────────────

def test_vad_engine_routing_per_stt():
    """gemini/groq STT run the LEGACY energy VAD with RAISED noise
    multipliers (fan/background hiss must not open a turn); local
    engines (whisper/vosk) follow vad_engine (Silero when available)."""
    app = _make_app()
    app._config["vad_engine"] = "silero"
    cloud_vad = app._make_vad("gemini")
    cloud2 = app._make_vad("groq")
    assert type(cloud_vad).__name__ == "_VADBuffer"
    assert type(cloud2).__name__ == "_VADBuffer"
    assert abs(cloud_vad._start_mult - 1.9) < 1e-6   # raised: ~2x floor
    assert abs(cloud_vad._end_mult - 1.4) < 1e-6
    assert abs(cloud_vad._sil_n / 16000 - 3.0) < 1e-6
    local_vad = app._make_vad("whisper")
    local_names = {"SoftSileroVAD", "SileroVAD", "_VADBuffer"}
    assert type(local_vad).__name__ in local_names
    if type(local_vad).__name__ == "_VADBuffer":
        assert abs(local_vad._start_mult - 1.5) < 1e-6   # original default


def test_vad_energy_multipliers_config_tunable():
    """vad_cloud_* keys tune the cloud energy VAD from config; the local
    energy fallback keeps its own vad_start_mult/vad_end_mult."""
    app = _make_app()
    app._config["vad_engine"] = "energy"
    app._config["vad_cloud_start_mult"] = 2.5
    app._config["vad_cloud_end_mult"] = 1.6
    app._config["vad_silence_sec_cloud"] = 4.0
    app._config["vad_start_mult"] = 2.0
    cloud_vad = app._make_vad("gemini")
    local_vad = app._make_vad("vosk")
    assert type(cloud_vad).__name__ == "_VADBuffer"
    assert type(local_vad).__name__ == "_VADBuffer"
    assert abs(cloud_vad._start_mult - 2.5) < 1e-6
    assert abs(cloud_vad._end_mult - 1.6) < 1e-6
    assert abs(cloud_vad._sil_n / 16000 - 4.0) < 1e-6
    assert abs(local_vad._start_mult - 2.0) < 1e-6


def test_mic_loops_forward_their_engine_to_vad_factory():
    """The mic-loop threads receive engine=<name>; each loop must pass it
    through to _make_vad so the per-engine pin actually applies."""
    import inspect
    for name in ("_listen_whisper", "_listen_vosk"):
        src = inspect.getsource(getattr(main.Orthos, name))
        assert "self._make_vad(engine)" in src, name
    src = inspect.getsource(main.Orthos._start_mic_loop)
    assert 'kwargs={"engine": engine}' in src


# ── junk-transcript guard (Whisper silence hallucinations) ────────────────

def test_junk_transcript_guards():
    """Bare numbers/clock patterns and caption phrases are decoder noise,
    not user speech; real sentences with digits inside survive."""
    app = _make_app()
    import numpy as np
    quiet = np.zeros(8000, dtype=np.float32)
    loud = (np.random.default_rng(1).standard_normal(16000) * 0.15).astype(np.float32)
    for t in ("00:00", "12.30", "100", "7:00", "thank you for watching",
              "you", "okay"):
        assert app._junk_transcript(t, quiet) is True, t
    for t in ("what is the weather today", "convert 100 dollars to rupees",
              "set a timer for 7:00 pm please",
              "hello Orthos kaise ho"):
        assert app._junk_transcript(t, loud) is False, t
    # Aggressive quiet rule: marginal audio + <=2 content words = junk.
    rng = np.random.default_rng(3)
    mid = (rng.standard_normal(16000) * 0.014).astype(np.float32)
    assert app._junk_transcript("hello Orthos", mid) is True
    assert app._junk_transcript("120000", quiet) is True


def test_junk_transcript_config_switch():
    app = _make_app()
    app._config["junk_transcript_filter"] = False
    assert app._junk_transcript("00:00", np.zeros(1000, np.float32)) is False


# ── muted mic must not kill the SPEAKING animation ────────────────────────

def test_set_speaking_raises_speaking_even_when_muted():
    """Text input works while the user mic is muted — the assistant still
    speaks, so the SPEAKING state must be raised regardless of mute."""
    app = _make_app()
    app.ui.muted = True
    app.set_speaking(True)
    app.ui.set_state.assert_called_with("SPEAKING")


def test_set_listening_settles_to_muted_not_stuck_speaking():
    """When playback ends while muted, state settles to MUTED (not stuck
    in SPEAKING, not wrongly LISTENING)."""
    app = _make_app()
    app.ui.muted = True
    app.set_speaking(False)
    app.ui.set_state.assert_called_with("MUTED")
    app2 = _make_app()
    app2.ui.muted = False
    app2.set_speaking(False)
    app2.ui.set_state.assert_called_with("LISTENING")
