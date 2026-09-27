"""
Orthos — Local LLM Edition
STT (Whisper / Vosk)  +  Ollama LLM  +  TTS (EdgeTTS / Kokoro / ElevenLabs)
All Gemini / Google-AI dependencies removed.
"""
# ── Silence verbose logs + block heavy unused backends ─────────────────────
import os as _os
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL",  "3")   # TensorFlow C++ noise
_os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")   # oneDNN banner
_os.environ.setdefault("GRPC_VERBOSITY",         "ERROR")
# USE_TF=0 prevents transformers from importing TensorFlow (saves 4-8 s).
# We intentionally do NOT set USE_TORCH or USE_JAX — forcing those values
# breaks transformers' lazy-loader on some versions (AutoModel disappears
# from the namespace).  Let transformers auto-detect the available backends.
_os.environ.setdefault("USE_TF",                 "0")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Offline mode — use cached models, no HuggingFace network calls on startup.
# On first run the model isn't cached yet; tts.py / stt.py detect this and
# temporarily clear these flags to allow the one-time download, then they
# stay in effect for every subsequent launch (fully offline).
# ── Migrate from deprecated TRANSFORMERS_CACHE to HF_HOME ──────────────────
_hf_cache = _os.environ.pop("TRANSFORMERS_CACHE", None)
if _hf_cache:
    _os.environ.setdefault("HF_HOME", _hf_cache)
_os.environ.setdefault("HF_HOME", _os.path.join(_os.environ.get("USERPROFILE", "C:\\Users\\default"), ".cache\\huggingface"))
# ───────────────────────────────────────────────────────────────────────────
_os.environ.setdefault("HF_HUB_OFFLINE",      "1")
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_os.environ.setdefault("HF_DATASETS_OFFLINE",  "1")
import warnings as _warnings
_warnings.filterwarnings("ignore", category=UserWarning)
_warnings.filterwarnings("ignore", category=DeprecationWarning)
_warnings.filterwarnings("ignore", category=FutureWarning)
# ───────────────────────────────────────────────────────────────────────────

# ── Windows console safety ─────────────────────────────────────────────────
# Printing unicode (→, ⚠️, emoji) raises 'charmap' codec errors when
# stdout/stderr is piped or redirected (which would crash the Gemini Live
# session mid-tool-round).  Keep the detected encoding, never raise.
import sys as _sys
for _stream in (_sys.stdout, _sys.stderr):
    try:
        if _stream is not None and hasattr(_stream, "reconfigure"):
            _stream.reconfigure(errors="replace")
    except Exception:
        pass

# ── Global exception handler (catch ALL unhandled exceptions) ───────────────
import traceback as _traceback
import threading as _threading

def _global_excepthook(exc_type, exc_value, exc_tb):
    """Print FULL traceback for any unhandled exception."""
    print("=" * 70, file=sys.stderr)
    print("UNHANDLED EXCEPTION (main thread):", file=sys.stderr)
    _traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.stderr)
    print("=" * 70, file=sys.stderr)

def _thread_excepthook(args):
    """Print FULL traceback for unhandled exceptions in threads."""
    print("=" * 70, file=sys.stderr)
    print(f"UNHANDLED EXCEPTION in thread '{args.thread.name}':", file=sys.stderr)
    _traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback, file=sys.stderr)
    print("=" * 70, file=sys.stderr)

import sys
import random
sys.excepthook = _global_excepthook
_threading.excepthook = _thread_excepthook
# ───────────────────────────────────────────────────────────────────────────

# ── CLI arguments ─────────────────────────────────────────────────────────
import argparse as _argparse
from pathlib import Path as _Path

_CAPTURES_DIR = _Path(__file__).resolve().parent / "memory" / "captures"
_CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

_ARGS = _argparse.ArgumentParser(description="Orthos — AI Assistant")
_ARGS.add_argument("--headless", "--no-gui", action="store_true",
                    help="Run without PyQt6 GUI (Telegram-only terminal mode)")
_ARGS = _ARGS.parse_args()
# ───────────────────────────────────────────────────────────────────────────

# ── Bootstrap: auto-install base UI packages before anything else ──────────
# Uses only stdlib so it works even on a completely fresh Python install.
import importlib.util as _ilu
import subprocess      as _sp
import sys             as _sys

_BASE_PKGS = [
    ("psutil",      "psutil"),
    ("numpy",       "numpy"),
    ("PIL",         "pillow"),
    ("requests",    "requests"),
]
if not _ARGS.headless:
    _BASE_PKGS.extend([
        ("PyQt6",       "PyQt6"),
        ("sounddevice", "sounddevice"),
    ])

def _bootstrap() -> None:
    need = [pkg for mod, pkg in _BASE_PKGS if _ilu.find_spec(mod) is None]
    if not need:
        return
    print(f"\n[Orthos] First-run setup — installing: {', '.join(need)}")
    print("[Orthos] This happens only once.\n")
    _sp.run([_sys.executable, "-m", "pip", "install", *need], check=True)
    print("\n[Orthos] Base packages ready — restarting…\n")
    # Replace current process with a fresh one (picks up newly installed packages)
    _os.execv(_sys.executable, [_sys.executable] + _sys.argv)

_bootstrap()
# ───────────────────────────────────────────────────────────────────────────

import base64
import json
import io
import queue
import re
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd

from ui import OrthosUI
from memory.memory_manager import (
    load_memory, update_memory, format_memory_for_prompt,
    format_memory_items_from_sqlite,
    save_conversation, load_recent_conversations,
    create_session, list_sessions, get_session, rename_session,
    close_and_summarize, load_session_turns, load_all_summaries,
    auto_title_session, delete_session,
    set_session_memory, get_session_memory, format_session_memory,
    get_session_recent_turns,
    log_timeline_event, format_timeline, auto_detect_timeline_events,
    search_timeline,
    # ── Memory items CRUD (used by cleanup worker) ──
    get_memory_items,
    update_memory_item,
    # ── New retrieval-first ──
    hybrid_search,
    invalidate_search_cache,
    format_memory_items_for_prompt,
    add_memory_item,
    fork_session,
    compress_oldest_turns,
    estimate_session_tokens,
    # ── Project ──
    _ensure_default_hierarchy,
    list_projects,
    create_project,
    get_project,
    set_session_project,
    # ── Memory versioning ──
    save_memory_version,
    get_memory_versions,
    # ── Tool memory ──
    set_tool_memory,
    get_tool_memory,
    # ── Entity / Graph ──
    get_entity_graph,
    # ── Logging ──
    log_retrieval,
    log_prompt,
)
from core.llm_client import call_llm, call_llm_stream, get_llm_settings, _get_provider_config

try:
    from core.gemini_live_provider import QuotaExceededError as _QuotaExceededError
except Exception:
    _QuotaExceededError = None

from actions.file_processor    import file_processor
from actions.flight_finder     import flight_finder
from actions.open_app          import open_app
from actions.weather_report    import weather_action
from actions.send_message      import send_message
from actions.reminder          import reminder
from actions.computer_settings import computer_settings
from actions.screen_processor  import screen_process, screen_locate, _capture_screen_annotated, _get_screenshot_quality
from actions.youtube_video     import youtube_video
from actions.desktop           import desktop_control
from actions.file_controller   import file_controller
from actions.code_helper       import code_helper
from actions.dev_agent         import dev_agent
from actions.web_search        import web_search as web_search_action, webfetch as webfetch_action
from actions.computer_control  import computer_control
from actions.game_updater      import game_updater
from actions.terminal          import run_terminal

# ── Improved modules ───────────────────────────────────────────────────
from core.logging_setup import setup_logging, get_logger, patch_print, catch_exceptions
from core.security import SecretsManager, Sanitizer
from core.tool_registry import discover as discover_tools
from core.llm_provider import get_default_provider
from core.observability import MetricsCollector, HealthCheck, start_metrics_server, Timer, init_tracing
from memory.chroma_memory import ChromaMemory, MemoryBridge, CHROMA_VECTOR_SEARCH_ENABLED
from core.mcp_manager import MCPManager

# ── Global logging ─────────────────────────────────────────────────────
logger = get_logger("mark")
setup_logging()
patch_print()  # replaces all print() calls with loguru

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent


BASE_DIR        = _get_base_dir()
API_CONFIG_PATH = BASE_DIR / "config" / "api_keys.json"
PROMPT_PATH     = BASE_DIR / "core" / "prompt.txt"

SAMPLE_RATE_IN = 16_000
BLOCK_SIZE     = 1_024
CHANNELS       = 1

# ---------------------------------------------------------------------------
# Secrets manager
# ---------------------------------------------------------------------------

_secrets = SecretsManager()


def _load_config() -> dict:
    """Load config/api_keys.json with env-var fallback.

    Priority (2026-09 fix): the JSON file WINS over the environment when it
    holds a non-empty value. The old order (env first) meant a stale
    GROQ_API_KEY exported in a terminal/system settings silently replaced
    every freshly-pasted key — the user kept getting 401s no matter how many
    new keys they saved, because the request always went out with the old
    env var. Env vars now only fill keys the user never set in the file.
    """
    try:
        cfg = json.loads(API_CONFIG_PATH.read_text(encoding="utf-8"))
        for key in list(cfg.keys()):
            if isinstance(cfg.get(key), str) and cfg[key].strip():
                continue                      # user-set value wins
            env_val = _secrets.get(key.upper())
            if env_val:
                cfg[key] = env_val
        return cfg
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Dynamic tool registry
# ---------------------------------------------------------------------------

_tool_registry = discover_tools()
# If no dynamically discovered tools (no @tool decorators on action files),
# fall back to the static legacy TOOL_DECLARATIONS from main.py.original
if not _tool_registry.list_tools():
    from core.tool_declarations import TOOL_DECLARATIONS as _LEGACY_DECLARATIONS
    _tool_registry.register_legacy(_LEGACY_DECLARATIONS)
TOOL_DECLARATIONS = _tool_registry.get_declarations()
OLLAMA_TOOLS = TOOL_DECLARATIONS

# ── Dynamic Tool Index (semantic search + on-demand schema loading) ──────
from core.tool_index import ToolIndex, TOOL_SUMMARIES, META_TOOL_DEFINITIONS, \
    make_search_tools_handler, make_get_tool_definition_handler

_tool_index = ToolIndex(_tool_registry)
_tool_active_full_schemas: set[str] = set()

# Register meta-tools so they can be executed
_search_tools_fn = make_search_tools_handler(_tool_index)
_get_tool_def_fn = make_get_tool_definition_handler(_tool_index, _tool_active_full_schemas)
_tool_registry.register_tool("search_tools", _search_tools_fn,
    description="Search for available tools by capability. Returns matching tool names, descriptions, and full parameter schemas.",
    parameters={"type": "OBJECT", "properties": {"query": {"type": "STRING", "description": "What capability you need"}, "k": {"type": "INTEGER", "description": "Number of results (default 5, max 10)"}}, "required": ["query"]})
_tool_registry.register_tool("get_tool_definition", _get_tool_def_fn,
    description="Load the full parameter schema for a specific tool. Call this when search_tools shows a matching tool and you need its complete parameters.",
    parameters={"type": "OBJECT", "properties": {"tool_name": {"type": "STRING", "description": "Exact tool name e.g. 'web_search'"}}, "required": ["tool_name"]})


def _build_dynamic_tools() -> list[dict]:
    """Build the tools payload: meta-tools + internal tools (full schemas) + loaded MCP tools.
    
    Internal tools: always sent with full parameter schemas (not empty {}).
    MCP tools: NEVER sent in payload. Only available via search_tools, loaded on demand.
    """
    tools = list(META_TOOL_DEFINITIONS)

    # Internal tools — always send full schemas
    for name in _tool_index._tool_names:
        full = _tool_index.get_tool_definition(name)
        if full:
            tools.append(full)
        else:
            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": TOOL_SUMMARIES.get(name, name),
                    "parameters": {"type": "object", "properties": {}},
                }
            })

    # MCP tools — only if LLM explicitly loaded via get_tool_definition
    for name in _tool_index._mcp_tool_names:
        if name in _tool_active_full_schemas:
            full = _tool_index.get_tool_definition(name)
            if full:
                tools.append(full)

    return tools


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_system_prompt() -> str:
    try:
        return PROMPT_PATH.read_text(encoding="utf-8")
    except Exception:
        return (
            "You are Orthos, an AI assistant. "
            "Be concise, direct, and always use the provided tools to complete tasks. "
            "Never simulate or guess results — always call the appropriate tool."
        )


# ---------------------------------------------------------------------------
# Voice Activity Detection (used for Whisper listen loop)
# ---------------------------------------------------------------------------

class _VADBuffer:
    """Adaptive energy-based VAD: buffers audio until end of utterance.

    Continuously tracks the noise floor so it works with any mic
    (including WO Mic which has a high constant noise floor ~0.24 RMS).
    Speech is detected when RMS significantly exceeds the noise floor.

    2026-09 fix — three changes that together stop the one-word-per-turn
    truncation:

      1. The noise-floor history is updated ONLY with chunks classified as
         non-speech. (Previously every chunk — speech included — fed the
         median, so after ~2 s of talking the floor rose to the speaker's
         own level, the speech threshold climbed above the voice, the next
         word read as "silence" and the utterance flushed early.)
      2. Hangover/fade: after an utterance the floor holds for hangover_ms
         and only re-learns if the room stays louder for fade_ms, so the
         floor cannot swing mid-utterance.
      3. A prebuffer records continuously, so the onset of an utterance —
         the ~250 ms before the classifier fires — is recovered on flush
         instead of being clipped off the first word.
    """

    # Chunks at or below this fraction of the speech threshold are treated
    # as quiet and may update the floor history. Above it, chunks never
    # touch the history (see docstring point 1).
    _LEARN_MAX_FRAC = 0.60

    def __init__(
        self,
        sample_rate:    int   = 16_000,
        silence_sec:    float = 3.0,     # silence after last word → send to STT
        min_speech_sec: float = 0.3,
        max_speech_sec: float = 30.0,
        hangover_ms:    int   = 400,     # floor holds this long after speech
        fade_ms:        int   = 800,     # …then refreezes over this long
        prebuffer_sec:  float = 0.35,    # onset recovery
        start_mult:     float = 1.5,     # speech onset: RMS > floor × start_mult
        end_mult:       float = 1.15,    # speech end:   RMS < floor × end_mult
    ):
        self._sr        = sample_rate
        self._start_mult = float(start_mult)
        self._end_mult   = float(end_mult)
        self._sil_n     = int(silence_sec * sample_rate)
        self._min_n     = int(min_speech_sec * sample_rate)
        self._max_n     = int(max_speech_sec * sample_rate)
        self._hang_n    = int(hangover_ms / 1000 * sample_rate)
        self._fade_n    = max(1, int(fade_ms / 1000 * sample_rate))
        self._pre_n     = int(prebuffer_sec * sample_rate)
        self._buf: list[np.ndarray] = []
        self._in_spch   = False
        self._sil_cnt   = 0
        self._history: list[float] = []
        self._noise_floor = 0.0
        self._since_spch = self._hang_n   # samples since speech was last seen
        self._since_learn = 0             # samples since the floor last moved
        self._prebuf: list[np.ndarray] = []
        self._prebuf_n  = 0

    def _learn(self, rms: float) -> None:
        """Feed one quiet-chunk RMS to the floor history."""
        self._history.append(rms)
        if len(self._history) > 30:
            self._history.pop(0)
        self._noise_floor = float(np.median(self._history))

    def _flush(self) -> np.ndarray | None:
        audio = np.concatenate(self._buf)
        self._buf = []
        self._in_spch = False
        self._sil_cnt = 0
        if len(audio) < self._min_n:
            return None
        return audio

    def reset(self) -> None:
        """Drop buffered speech and the prebuffer; keep the room model.

        Called when the assistant stops talking: whatever sits in the
        buffer then was captured from the speakers, not the user, and must
        never reach the STT. The learned noise floor is deliberately kept —
        it describes the room, not the voice.
        """
        self._buf = []
        self._in_spch = False
        self._sil_cnt = 0
        self._prebuf = []
        self._prebuf_n = 0
        self._since_spch = self._hang_n

    def process(self, chunk: np.ndarray) -> np.ndarray | None:
        """
        Feed one audio chunk (float32 mono).
        Returns complete utterance when speech ends, otherwise None.

        Thresholds adapt to the noise floor of SILENCE only:
          - speech starts when RMS > noise_floor × start_mult (default 1.5;
            cloud STT raises it so fan/background hiss never opens a turn)
          - speech ends when RMS < noise_floor × end_mult
        The gap between the two prevents mid-sentence cuts.
        """
        rms = float(np.sqrt(np.mean(chunk ** 2)))

        # Establish an initial floor from the first quiet chunks after boot.
        if len(self._history) < 8:
            self._learn(rms)

        speech_thresh = self._noise_floor * self._start_mult
        silence_thresh = self._noise_floor * self._end_mult

        # ── Prebuffer: always rolling; recovered on flush (onset fix) ──
        self._prebuf.append(chunk.copy())
        self._prebuf_n += len(chunk)
        while self._prebuf_n > self._pre_n:
            dropped = self._prebuf.pop(0)
            self._prebuf_n -= len(dropped)

        learn_max = speech_thresh * self._LEARN_MAX_FRAC

        if rms > speech_thresh:
            if not self._in_spch:
                # Speech onset: recover the pre-speech ring so the first
                # word is complete (it was recorded before the classifier
                # could fire). The onset chunk is already inside _prebuf.
                self._in_spch = True
                self._sil_cnt = 0
                self._since_spch = 0
                self._buf = list(self._prebuf)
                self._prebuf = []
                self._prebuf_n = 0
            else:
                self._buf.append(chunk.copy())
            self._sil_cnt = 0
            if sum(len(c) for c in self._buf) >= self._max_n:
                return self._flush()
        elif self._in_spch:
            self._buf.append(chunk.copy())
            if rms < silence_thresh:
                self._sil_cnt += len(chunk)
            else:
                self._sil_cnt = 0        # between-word gap, not end of turn
            if self._sil_cnt >= self._sil_n or sum(len(c) for c in self._buf) >= self._max_n:
                return self._flush()
        else:
            # Idle: only genuinely quiet chunks update the floor, so the
            # thresholds always describe the room — never the user's voice.
            if rms <= learn_max:
                self._learn(rms)
                self._since_spch += len(chunk)
                self._since_learn = 0
            else:
                self._since_learn += len(chunk)
                if self._since_spch < self._hang_n:
                    # Hangover right after an utterance: hold the floor —
                    # room decay must not pull it down yet.
                    self._since_spch += len(chunk)
                elif self._since_learn >= self._fade_n:
                    # Ambient got louder than the learned floor (fan, TV)
                    # and stayed there: re-learn instead of going deaf.
                    self._history = [rms] * 8
                    self._noise_floor = rms
                    self._since_learn = 0
        return None


class _BargeDetector:
    """Detects real speech while Orthos is speaking (barge-in).

    Only audio well above the ambient noise floor triggers an interrupt,
    so speaker echo of Orthos's own voice does not self-trigger. The noise
    floor adapts via a rolling median over the last ~3 seconds.
    """

    def __init__(self, floor_rms: float = 0.02, history: int = 30):
        self._floor_rms = float(floor_rms)
        self._hist: list[float] = []

    def speech(self, rms: float) -> bool:
        self._hist.append(rms)
        if len(self._hist) > 30:
            self._hist.pop(0)
        noise = max(self._floor_rms, float(np.median(self._hist)) * 2.5)
        return rms > noise


# ---------------------------------------------------------------------------
# Conversation trim policy (2026-09-06, user-decided numbers)
# ---------------------------------------------------------------------------
# Raw conversation hits 50k tokens → summarize ONLY the oldest ~15k into the
# rolling summary; the newest ~35k stay raw in the prompt so recent exchanges
# keep full fidelity (the old behaviour summarized the ENTIRE conversation,
# which caused "context amnesia" after every trim).
from core.conversation_trim import (
    RAW_CONV_TRIGGER_TOKENS as _RAW_CONV_TRIGGER_TOKENS,
    TRIM_CHUNK_TOKENS as _TRIM_CHUNK_TOKENS,
    oldest_trim_indices as _oldest_trim_split,
    compact_slice_tokens as _compact_slice_tokens,
)


# ---------------------------------------------------------------------------
# Orthos
# ---------------------------------------------------------------------------

class _STTSwitched(Exception):
    """Raised in a mic loop when the STT engine was replaced live."""


class Orthos:
    """
    Main assistant class.
    Replaces JarvisLive (Gemini Live API) with:
      STT (Whisper/Vosk) → Ollama LLM (tool calling) → TTS (Edge/Kokoro/ElevenLabs)
    """

    def __init__(self, ui: OrthosUI):
        self.ui               = ui
        self._config          = _load_config()
        self._stt             = None
        self._tts             = None
        self._tts_ready       = threading.Event()
        self._speaking        = False
        self._speaking_lock   = threading.Lock()
        self._should_stop     = threading.Event()
        self._cancel_event    = threading.Event()
        self._cancel_seq      = 0
        self._cancel_lock     = threading.Lock()
        # Gemini Live user-transcript accumulation (fragments → one message)
        self._live_user_acc   = ""
        self._live_user_timer: threading.Timer | None = None
        self._live_user_lock  = threading.Lock()
        self._text_queue:     queue.Queue = queue.Queue()
        self._tts_queue:      queue.Queue = queue.Queue()
        self._audio_queue:    queue.Queue = queue.Queue(maxsize=2)
        self._streaming_tts   = False       # True when TTS engine yields PCM chunks (Gemini)
        self._out_stream      = None        # sd.OutputStream for streaming playback
        self._play_at         = time.time()
        self._conversation:   list[dict]  = []
        self._conv_saved_idx  = 0
        self._rolling_summary: str = ""
        self._conv_budget: int = 5000
        self._summary_rolling_budget: int = 50000
        self._summary_past_budget: int = 100000
        self._cur_session_id: str | None = None
        self._cur_project_id: str | None = None
        self._session_list: list[dict] = []
        self._response_callbacks: list = []
        # Speaker-echo rejection state (see _reject_echo below): the mic
        # hears the tail of the TTS playback right after a turn ends, and
        # without these it transcribes that echo as the user's reply.
        self._echo_until = 0.0              # mic chunks dropped until this time
        self._spoken_tail: list[str] = []   # last words the TTS is saying
        self._last_echo_text: str | None = None
        # Echo protection is ADAPTIVE: speakers leak into a laptop mic, but
        # a headset or USB mic physically cannot hear the speakers. Risk is
        # assumed on first (safe), then refined from the device name and —
        # decisively — from mic samples captured while the TTS plays.
        self._echo_risk = True
        self._measuring_bleed = False
        self._mic_bleed: list[float] = []
        self._bleed_lock = threading.Lock()
        self._last_bleed_loud = False   # did the mic hear the last turn?
        self._last_bleed_peak = 0.0     # how loud, for a proportional gate
        # Acoustic echo fingerprint: what the TTS ACTUALLY played (PCM tap),
        # downsampled to a ~10 Hz loudness envelope. A captured utterance
        # whose envelope correlates with this IS the playback at the mic;
        # anything else deserves to be heard as a real voice.
        self._played_env: list[float] = []
        self._capture_env: list[float] = []
        self._gateway = None
        self._is_headless = _ARGS.headless
        self._barge: _BargeDetector | None = None   # lazy — see _get_barge()
        self._mic_thread_started = False
        # Bumped whenever a new STT engine is built (startup or live
        # reconfigure). Mic loops capture the value at start and exit
        # when it changes, so an engine switch needs no app restart.
        self._stt_generation = 0
        # Number of speak() sentences queued but not yet fully played.
        # The TTS worker ends the turn (mic-hold, LISTENING) only when this
        # hits 0 — so sentences synthesized after a tool call play as ONE
        # turn instead of flickering SPEAKING↔PROCESSING between them.
        self._speak_pending = 0

        # ── 5 Background worker queues ──
        self._embedding_queue: queue.Queue = queue.Queue()
        self._summary_queue: queue.Queue = queue.Queue()
        self._reflection_queue: queue.Queue = queue.Queue()
        self._extraction_queue: queue.Queue = queue.Queue()
        self._cleanup_queue: queue.Queue = queue.Queue()
        self._start_background_workers()

        # ── Ensure default project hierarchy exists ──
        try:
            _ensure_default_hierarchy()
        except Exception:
            pass

        # ── Improved module integrations ──────────────────────────────────
        self._metrics = MetricsCollector()
        self._health = HealthCheck()
        self._chroma = ChromaMemory() if CHROMA_VECTOR_SEARCH_ENABLED else None
        self._memory_bridge = MemoryBridge()
        self._provider = get_default_provider()
        self._sanitizer = Sanitizer()
        self._secrets = _secrets

        self.ui.on_text_command = self._on_text_command
        self.ui.on_text_command_with_files = self._on_text_command_with_files
        self.ui.on_query_typing = self._prefetch_query   # E: typing prefetch
        # Connect session UI callbacks
        try:
            self.ui._win._session_cb_new = self._new_session
            self.ui._win._session_cb_switch = self._switch_session
            self.ui._win._session_cb_delete = self._delete_session
            self.ui._win._session_cb_rename = self._rename_session_ui
            self.ui._win._session_cb_fork = self._fork_session_ui
            # ── Project & memory version callbacks ──
            self.ui._win._project_cb_list = self._list_projects_ui
            self.ui._win._project_cb_switch = self._switch_project_ui
            self.ui._win._project_cb_create = self._create_project_ui
            self.ui._win._version_cb_browse = self._browse_memory_versions_ui
            # ── Inline chat actions (F7): Regenerate / Continue ──
            if hasattr(self.ui._win, "_log"):
                self.ui._win._log._on_action_cb = self._handle_chat_action
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Response delivery (UI + Telegram gateway fan-out)
    # ------------------------------------------------------------------

    def register_response_callback(self, callback) -> None:
        self._response_callbacks.append(callback)

    def _deliver_response(self, text: str, source: str = "ui", quiet: bool = False) -> None:
        if not text:
            return
        if not quiet:
            self.ui.write_log(f"Orthos: {text}")
        for cb in self._response_callbacks:
            try:
                cb(text, source)
            except Exception as e:
                print(f"[ResponseRouter] callback error: {e}")

    # ------------------------------------------------------------------
    # Async reflection queue
    # ------------------------------------------------------------------

    def _start_background_workers(self) -> None:
        """Start 5 background worker threads: embedding, summary, reflection, extraction, cleanup."""

        def _embedding_worker():
            while True:
                try:
                    task = self._embedding_queue.get(timeout=60)
                except queue.Empty:
                    continue
                if task is None:
                    break
                try:
                    item_id = task.get("item_id", 0)
                    content = task.get("content", "")
                    if item_id and content:
                        from memory.chroma_memory import ChromaMemory, CHROMA_VECTOR_SEARCH_ENABLED
                        if CHROMA_VECTOR_SEARCH_ENABLED:
                            chroma = ChromaMemory()
                            chroma.add_memory("memory_items", str(item_id), content)
                except Exception as e:
                    print(f"[Worker] Embedding error: {e}")

        def _summary_worker():
            while True:
                try:
                    task = self._summary_queue.get(timeout=60)
                except queue.Empty:
                    continue
                if task is None:
                    break
                try:
                    session_id = task.get("session_id", self._cur_session_id)
                    compress_oldest_turns(session_id)
                except Exception as e:
                    print(f"[Worker] Summary error: {e}")

        def _reflection_worker():
            while True:
                try:
                    task = self._reflection_queue.get(timeout=60)
                except queue.Empty:
                    continue
                if task is None:
                    break
                try:
                    session_id = task.get("session_id", self._cur_session_id)
                    text = task.get("text", "")
                    if session_id and text:
                        log_timeline_event(
                            session_id, "reflection",
                            f"Async reflection: {text[:200]}",
                            auto=True,
                        )
                except Exception as e:
                    print(f"[Worker] Reflection error: {e}")

        def _extraction_worker():
            while True:
                try:
                    task = self._extraction_queue.get(timeout=60)
                except queue.Empty:
                    continue
                if task is None:
                    break
                try:
                    text = task.get("text", "")
                    if text:
                        self._extract_session_memory(text)
                except Exception as e:
                    print(f"[Worker] Extraction error: {e}")

        def _cleanup_worker():
            """Aging: decay scores for memory_items not recently accessed."""
            while True:
                try:
                    task = self._cleanup_queue.get(timeout=3600)  # run hourly
                except queue.Empty:
                    pass
                if task is None:
                    break
                try:
                    items = get_memory_items(limit=200)
                    now = datetime.now()
                    for item in items:
                        last_acc = item.get("last_access", "")
                        if last_acc:
                            days = (now - datetime.fromisoformat(last_acc)).days
                        else:
                            days = 365
                        if days > 90:
                            new_decay = max(0.1, item.get("decay_score", 1.0) - 0.1 * (days / 30))
                            update_memory_item(item["id"], importance=new_decay * item.get("importance", 1.0))
                except Exception as e:
                    print(f"[Worker] Cleanup error: {e}")

        threads = [
            ("embedding", _embedding_worker),
            ("summary", _summary_worker),
            ("reflection", _reflection_worker),
            ("extraction", _extraction_worker),
            ("cleanup", _cleanup_worker),
        ]
        for name, target in threads:
            t = threading.Thread(target=target, daemon=True)
            t.start()

    def _enqueue_reflection(self, text: str, session_id: str | None = None) -> None:
        """Queue a reflection task for async processing."""
        try:
            self._reflection_queue.put_nowait({
                "text": text,
                "session_id": session_id or self._cur_session_id,
            })
        except Exception:
            pass

    def _enqueue_embedding(self, item_id: int, content: str) -> None:
        try:
            self._embedding_queue.put_nowait({"item_id": item_id, "content": content})
        except Exception:
            pass

    def _enqueue_extraction(self, text: str) -> None:
        try:
            self._extraction_queue.put_nowait({"text": text})
        except Exception:
            pass

    def _enqueue_summary(self, session_id: str | None = None) -> None:
        try:
            self._summary_queue.put_nowait({"session_id": session_id or self._cur_session_id})
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _init_session_system(self) -> None:
        """Called on startup. Creates a fresh session, loads past summaries."""
        self._cur_project_id = _ensure_default_hierarchy()[-1]  # get default project id
        # Migrate old sessions (project_id IS NULL) to default project
        try:
            import sqlite3
            from memory.conversation_db import DB_PATH
            with sqlite3.connect(str(DB_PATH)) as _conn:
                _conn.execute(
                    "UPDATE sessions SET project_id = ? WHERE project_id IS NULL",
                    (self._cur_project_id,),
                )
                _conn.commit()
        except Exception:
            pass
        self._session_list = list_sessions(project_id=self._cur_project_id)
        self._cur_session_id = create_session("New Chat", project_id=self._cur_project_id)
        self._conversation = []
        self._conv_saved_idx = 0
        self._rolling_summary = ""
        try:
            self.ui._win._cur_session_id = self._cur_session_id
            self.ui._win._cur_project_id = self._cur_project_id
            self.ui._win._session_refresh_sig.emit()
        except Exception:
            pass
        print(f"[Session] Started: {self._cur_session_id}")

    def _new_session(self) -> None:
        """Close current session and start a new one."""
        if self._cur_session_id:
            self._close_current_session()
        self._cur_session_id = create_session("New Chat", project_id=self._cur_project_id)
        self._conversation = []
        self._conv_saved_idx = 0
        self._rolling_summary = ""
        self._session_list = list_sessions(project_id=self._cur_project_id)
        try:
            self.ui._win._cur_session_id = self._cur_session_id
            self.ui._win._refresh_session_list()
            self.ui._win._clear_log()
        except Exception:
            pass
        print(f"[Session] New session started: {self._cur_session_id}")

    def _switch_session(self, session_id: str) -> None:
        """Switch to an existing session. Close current, load target."""
        if session_id == self._cur_session_id:
            return
        if self._cur_session_id:
            self._close_current_session()
        self._cur_session_id = session_id
        self._conversation = load_session_turns(session_id)
        self._conv_saved_idx = len(self._conversation)
        try:
            sm = get_session_memory(session_id, "_rolling_summary")
            self._rolling_summary = sm.get("_rolling_summary", "") if sm else ""
        except Exception:
            self._rolling_summary = ""
        # ── Trim conversation when rolling summary exists (avoid 2x payload) ──
        if self._rolling_summary and len(self._conversation) > 3:
            try:
                from memory.conversation_db import _estimate_tokens
                total = 0
                trimmed = []
                for t in reversed(self._conversation):
                    t_tok = _estimate_tokens(t.get("content", "") or "")
                    if total + t_tok > 10000:
                        break
                    trimmed.insert(0, t)
                    total += t_tok
                if trimmed:
                    self._conversation = trimmed
                    self._conv_saved_idx = len(trimmed)
            except Exception:
                pass
        self._session_list = list_sessions(project_id=self._cur_project_id)
        try:
            self.ui._win._cur_session_id = self._cur_session_id
            self.ui._win._refresh_session_list()
            self.ui._win._populate_log(self._conversation)
        except Exception as e:
            print(f"[Session] Failed to restore chat log: {e}")
            traceback.print_exc()
        session = get_session(session_id)
        title = session["title"] if session else session_id
        print(f"[Session] Switched to: {title} ({session_id})")

    def _close_current_session(self) -> None:
        """Save remaining turns, close and summarize the current session."""
        if not self._cur_session_id:
            return
        try:
            new_turns = self._conversation[self._conv_saved_idx:]
            if new_turns:
                save_conversation(new_turns, self._cur_session_id)
            summary = close_and_summarize(self._cur_session_id)
            if summary:
                print(f"[Session] Summary: {summary[:80]}...")
            # Auto-title from first user message if still "New Chat"
            session = get_session(self._cur_session_id)
            if session and session.get("title") == "New Chat":
                auto_title_session(self._cur_session_id)
        except Exception as e:
            print(f"[Session] Close error: {e}")
        self._cur_session_id = None

    def _delete_session(self, session_id: str) -> None:
        """Delete a session and all its data."""
        if session_id == self._cur_session_id:
            self._close_current_session()
            delete_session(session_id)
            self._cur_session_id = create_session("New Chat", project_id=self._cur_project_id)
            self._conversation = []
            self._conv_saved_idx = 0
        else:
            delete_session(session_id)
        self._session_list = list_sessions(project_id=self._cur_project_id)
        try:
            self.ui._win._refresh_session_list()
        except Exception:
            pass

    def _rename_session(self, session_id: str, title: str) -> None:
        rename_session(session_id, title)
        self._session_list = list_sessions(project_id=self._cur_project_id)
        try:
            self.ui._win._refresh_session_list()
        except Exception:
            pass

    def _rename_session_ui(self, session_id: str) -> None:
        """Show rename dialog and apply."""
        try:
            from PyQt6.QtWidgets import QInputDialog
            session = get_session(session_id)
            old_title = session["title"] if session else "Chat"
            new_title, ok = QInputDialog.getText(
                None, "Rename Session", "New title:",
                text=old_title,
            )
            if ok and new_title and new_title.strip():
                self._rename_session(session_id, new_title.strip())
        except Exception as e:
            print(f"[Session] Rename dialog error: {e}")

    def _fork_session_ui(self, session_id: str) -> None:
        """Fork a session: create new session from current state of the fork source."""
        try:
            from memory.conversation_db import fork_session
            new_id = fork_session(session_id, from_turn_id=0, title=f"Fork of {session_id[:8]}")
            # Switch to the new fork
            self._switch_session(new_id)
            print(f"[Session] Forked {session_id} → {new_id}")
        except Exception as e:
            print(f"[Session] Fork error: {e}")

    # ── Project management ─────────────────────────────────────────────

    def _list_projects_ui(self) -> list[dict]:
        return list_projects()

    def _switch_project_ui(self, project_id: str) -> None:
        """Switch current project view — reload session list for the new project."""
        self._cur_project_id = project_id
        self._session_list = list_sessions(project_id=project_id)
        # If current session belongs to old project, switch to most recent in new project
        if self._session_list:
            self._switch_session(self._session_list[0]["id"])
        else:
            if self._cur_session_id:
                self._close_current_session()
            self._cur_session_id = create_session("New Chat", project_id=project_id)
            self._conversation = []
            self._conv_saved_idx = 0
        try:
            self.ui._win._cur_project_id = project_id
            self.ui._win._refresh_session_list()
            if not self._session_list:
                self.ui._win._clear_log()
        except Exception:
            pass
        proj = get_project(project_id)
        pname = proj["name"] if proj else project_id
        print(f"[Project] Switched to: {pname} ({project_id})")

    def _create_project_ui(self, name: str) -> str:
        pid = create_project(name)
        self._session_list = list_sessions(project_id=self._cur_project_id)
        try:
            self.ui._win._refresh_session_list()
        except Exception:
            pass
        return pid

    def _browse_memory_versions_ui(self) -> list[dict]:
        return get_memory_versions()

    # ── Session memory extraction ─────────────────────────────────────────

    def _extract_session_memory(self, text: str) -> None:
        """Parse AI output for goals, issues, code, files, todos and store in session_memory."""
        if not self._cur_session_id:
            return
        import re
        patterns = {
            "goal":     r"(?:current\s+)?(?:goal|objective|aim)[:\s]+(.+?)(?=\n\n|\n(?:\w+\s*:)|\Z)",
            "issue":    r"(?:issue|blocker|problem|challenge)[:\s]+(.+?)(?=\n\n|\n(?:\w+\s*:)|\Z)",
            "code":     r"(?:code|function|class|script)[:\s]+(.+?)(?=\n\n|\n(?:\w+\s*:)|\Z)",
            "files":    r"(?:files?|file\s+created|edited|modified)[:\s]+(.+?)(?=\n\n|\n(?:\w+\s*:)|\Z)",
            "todo":     r"(?:todo|next\s+step|remaining)[:\s]+(.+?)(?=\n\n|\n(?:\w+\s*:)|\Z)",
        }
        for key, pat in patterns.items():
            m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
            if m:
                val = m.group(1).strip()[:500]
                if val:
                    set_session_memory(self._cur_session_id, key, val)

    def _extract_and_log(self, text: str) -> None:
        """Call after each assistant response to update session memory + timeline + entity graph."""
        self._extract_session_memory(text)
        if self._cur_session_id:
            try:
                _store_count = getattr(self, "_memory_extract_count", 0)
                if _store_count % 5 == 0:  # every 5 turns, detect timeline events
                    from memory.conversation_db import get_session_recent_turns as _gsrt
                    recent = _gsrt(self._cur_session_id)
                    auto_detect_timeline_events(self._cur_session_id, recent,
                                                project_id=self._cur_project_id or "")
                setattr(self, "_memory_extract_count", _store_count + 1)
            except Exception:
                pass
        # Extract entities for knowledge graph (async via queue).
        # Dedup by content hash — the same response used to be linked
        # dozens of times, flooding the graph with boilerplate edges.
        try:
            import hashlib as _hashlib
            _h = _hashlib.md5((text or "").encode("utf-8")).hexdigest()[:12]
            _recent = getattr(self, "_kg_resp_hashes", [])
            if _h not in _recent and (text or "").strip():
                from memory.entity_store import link_fact
                link_fact("conversation", "recent_response", text[:500])
                _recent = (_recent + [_h])[-3:]
                setattr(self, "_kg_resp_hashes", _recent)
        except Exception:
            pass

    # ── Reflection & Decay ───────────────────────────────────────────────

    def _build_reflection(self) -> str:
        """Build a short reflection summary from recent conversation turns."""
        try:
            recent = self._conversation[-6:]
            lines = []
            for msg in recent:
                role = msg.get("role", "?")
                content = (msg.get("content") or "")[:120]
                lines.append(f"[{role}] {content}")
            return "; ".join(lines[-3:]) or "No recent content"
        except Exception:
            return "Reflection unavailable"

    def _decay_old_memories(self) -> None:
        """Decay importance of session memory items not recently updated."""
        try:
            from memory.conversation_db import get_session_recent_turns as _gsrt
            recent = _gsrt(self._cur_session_id, limit=3)
            recent_text = " ".join(t.get("content", "") for t in recent).lower()
            if self._cur_session_id:
                from memory.conversation_db import get_session_memory as _gsm
                mem = _gsm(self._cur_session_id)
                for key, value in mem.items():
                    # NEVER decay internal framework keys — `_rolling_summary`
                    # and `_summary_*` carry the persisted older-context and
                    # deleting them silent-wipes the model's history memory.
                    if key.startswith("_"):
                        continue
                    if key not in recent_text and len(self._conversation) > 20:
                        log_timeline_event(
                            self._cur_session_id, "decay",
                            f"Session memory '{key}' faded (not recently referenced)",
                            auto=True,
                        )
                        from memory.conversation_db import delete_session_memory as _dsm
                        _dsm(self._cur_session_id, key)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    # ── Prompt assembly (2026-09 prefetch split) ─────────────────────────
    # The prompt is assembled from SEGMENTS in the original order:
    #   - static segments  → cached (key: session/rolling/project/memory-rev),
    #                         pre-built in background after each response (D)
    #   - query segments   → depend on the user's text; pre-built while the
    #                         user types (E), consumed if query matches
    #   - fresh segments   → time/mode, computed every call (cheap)
    _PROMPT_ORDER = (
        "sys", "summaries", "rolling", "session_mem",
        "retrieved", "chroma",
        "core", "ltm",
        "timeline",
        "project", "tool",
        "kg",
        "time", "mode",
    )
    _STATIC_SEG_KEYS = frozenset(
        ("sys", "summaries", "rolling", "session_mem", "core", "project", "tool")
    )
    _STATIC_PROMPT_TTL = 300.0

    def _extract_query(self) -> str:
        """Query = last user message (first 200 chars) — same as always."""
        for msg in reversed(self._conversation):
            if msg.get("role") == "user":
                return (msg.get("content") or "")[:200]
        return ""

    def _build_system_prompt(self, query: str | None = None) -> str:
        if query is None:
            query = self._extract_query()
        static = self._static_prompt_segments()
        qsegs = self._query_prompt_segments_cached(query)
        fresh = self._fresh_prompt_segments()
        parts: list[str] = []
        for key in self._PROMPT_ORDER:
            if key in fresh:
                val = fresh[key]
            elif key in self._STATIC_SEG_KEYS:
                val = static.get(key, "")
            else:
                val = qsegs.get(key, "")
            if val:
                parts.append(val)
        return "\n\n".join(parts)

    def _static_prompt_segments(self) -> dict[str, str]:
        """Cached static segments; invalidated by session/rolling/project
        changes or a memory-revision bump (set_session_memory etc.)."""
        try:
            from memory.conversation_db import get_memory_rev
            rev = get_memory_rev()
        except Exception:
            rev = -1
        key = (self._cur_session_id, self._rolling_summary,
               self._cur_project_id, rev)
        cached = getattr(self, "_static_prompt_cache", None)
        now = time.time()
        if (cached and cached.get("key") == key
                and (now - cached.get("built_at", 0.0)) < self._STATIC_PROMPT_TTL):
            return cached.get("seg") or {}
        seg = self._build_static_prompt_segments()
        self._static_prompt_cache = {"key": key, "seg": seg, "built_at": now}
        return seg

    def _build_static_prompt_segments(self) -> dict[str, str]:
        """Prompt blocks that do NOT depend on the user's text."""
        seg: dict[str, str] = {}
        seg["sys"] = _load_system_prompt()

        # 1. Past session summaries (raw turns are in messages array — no duplicate)
        if self._cur_session_id:
            try:
                from memory.conversation_db import get_session_context
                conv_ctx = get_session_context(self._cur_session_id, include_summaries=True, include_raw_turns=False, max_tokens=self._summary_past_budget)
                if conv_ctx:
                    seg["summaries"] = conv_ctx
            except Exception:
                pass

        # 1b. Current session rolling summary (oldest turns summarized to stay within budget)
        if self._rolling_summary:
            seg["rolling"] = f"[CURRENT SESSION SUMMARY — older context summarized]\n{self._rolling_summary}"

        # 2. Session memory (current goals, issues, code, files)
        if self._cur_session_id:
            try:
                sm = format_session_memory(self._cur_session_id)
                if sm:
                    seg["session_mem"] = sm
            except Exception as e:
                print(f"[Prompt] Session memory error: {e}")

        # 5. Core memory (ChromaDB facts — identity, preferences, projects, notes)
        try:
            from memory.memory_manager import format_core_memory
            core = format_core_memory()
            if core:
                seg["core"] = core
        except Exception as e:
            print(f"[Prompt] Core memory error: {e}")

        # 8. Active project context
        try:
            if self._cur_project_id and self._cur_project_id != "default_project":
                proj = get_project(self._cur_project_id)
                if proj:
                    proj_lines = [f"[ACTIVE PROJECT — {proj.get('name', 'Unknown')}]"]
                    from memory.conversation_db import search_memory_items
                    proj_mems = search_memory_items(proj.get('name', ''), limit=5)
                    for m in proj_mems:
                        content = (m.get('content') or '')[:200]
                        if content:
                            proj_lines.append(f"  {content}")
                    seg["project"] = "\n".join(proj_lines)
        except Exception as e:
            print(f"[Prompt] Project context error: {e}")

        # 9. Tool memory (current session tool states)
        try:
            if self._cur_session_id:
                from memory.conversation_db import get_tool_memory
                known_tools = ["browser", "python", "vision", "git", "terminal", "file_controller", "code_helper"]
                ts_lines = []
                for tname in known_tools:
                    states = get_tool_memory(tname, self._cur_session_id)
                    if states:
                        ts_lines.append(f"  [{tname}]")
                        for k, v in states.items():
                            ts_lines.append(f"    {k}: {v[:100]}")
                if ts_lines:
                    ts_lines.insert(0, "[TOOL STATES — remembered tool context]")
                    seg["tool"] = "\n".join(ts_lines)
        except Exception as e:
            print(f"[Prompt] Tool state error: {e}")

        return seg

    def _fresh_prompt_segments(self) -> dict[str, str]:
        """Time + mode blocks — trivial to rebuild every call."""
        now = datetime.now()
        time_ctx = (
            f"[CURRENT DATE & TIME]\n"
            f"Right now it is: {now.strftime('%A, %B %d, %Y — %I:%M %p')}\n"
            f"Use this to calculate exact times for reminders."
        )
        mode_ctx = (
            "[CONTINUOUS TOOL EXECUTION MODE]\n"
            "When doing multi-step tasks, keep calling tools "
            "in a continuous loop using tool results to decide the next step.\n"
            "RULE: Do NOT stop mid-task to ask the user 'what to do next'. "
            "Each tool result should tell you what to do next. "
            "Keep the loop going until the task is fully complete or user says stop."
        )
        return {"time": time_ctx, "mode": mode_ctx}

    def _query_prompt_segments_cached(self, query: str) -> dict[str, str]:
        """Consume the typing-prefetched query blocks (E) if still valid."""
        q = (query or "").strip()
        cached = getattr(self, "_prefetched_query", None)
        if (cached and isinstance(cached.get("seg"), dict)
                and cached.get("query") == q
                and (time.time() - cached.get("built_at", 0.0)) < 60.0):
            return cached["seg"]
        return self._build_query_prompt_segments(query)

    def _build_query_prompt_segments(self, query: str) -> dict[str, str]:
        """Retrieval blocks that depend on the user's text.

        Shared dedup budget across retrieved/chroma/ltm — same behaviour
        as the original monolithic build.
        """
        seg: dict[str, str] = {}
        MEM_BUDGET_TOKENS = 1500
        mem_seen: set = set()
        mem_used = 0

        def _mem_block(key: str, header: str, lines: list) -> None:
            nonlocal mem_used
            kept = []
            for ln in lines:
                if not ln or not ln.strip():
                    continue
                sig = re.sub(r"^\s*\[[^\]]*\]\s*", "", ln)[:150]
                if sig in mem_seen:
                    continue
                toks = max(1, len(ln) // 4)
                if mem_used + toks > MEM_BUDGET_TOKENS:
                    break
                mem_seen.add(sig)
                mem_used += toks
                kept.append(ln)
            if kept:
                seg[key] = "\n".join([header] + kept)

        _trivial = self._is_trivial_query(query)

        # 3. Retrieved memories (hybrid search across memory_items)
        try:
            if query and not _trivial:
                retrieved = hybrid_search(query, limit=5)
                lines = []
                for r in retrieved:
                    ctype = r.get("type", "memory")
                    content = r.get("content", "")
                    if content:
                        lines.append(f"  [{ctype}] {content[:300]}")
                _mem_block("retrieved", "[RETRIEVED MEMORIES — relevant past context]", lines)
        except Exception as e:
            print(f"[Prompt] Retrieval error: {e}")

        # 4. ChromaDB auto-retrieval (vector search across long-term facts + session summaries)
        try:
            if not hasattr(self, '_chroma') or not self._chroma:
                from memory.chroma_memory import ChromaMemory, CHROMA_VECTOR_SEARCH_ENABLED
                if CHROMA_VECTOR_SEARCH_ENABLED:
                    self._chroma = ChromaMemory()
            if self._chroma and query and not _trivial:
                chroma_results = self._chroma.search_memory(query, n_results=5)
                lines = []
                for r in chroma_results:
                    val = (r.get("value") or "")[:200]
                    cat = r.get("category", "")
                    key = r.get("key", "")
                    if val:
                        lines.append(f"  [{cat}/{key}] {val}")
                _mem_block("chroma", "[LONG-TERM FACTS — ChromaDB vector search]", lines)
        except Exception as e:
            print(f"[Prompt] ChromaDB retrieval error: {e}")

        # 6. Long-term memory items (from SQLite memory_items table)
        try:
            ltm = format_memory_items_from_sqlite()
            if ltm:
                _mem_block("ltm", "[LONG-TERM MEMORY ITEMS]", ltm.split("\n"))
        except Exception as e:
            print(f"[Prompt] Long-term memory error: {e}")

        # 7. Timeline (relevant milestones — semantic search based on current context)
        try:
            if query:
                tl_events = search_timeline(query, limit=10)
                if tl_events:
                    tl_lines = ["[TIMELINE — relevant historical events]"]
                    for ev in tl_events:
                        ts = ev.get('timestamp', '')[:10]
                        desc = ev.get('description', '')
                        imp = ev.get('importance', 0.5)
                        marker = " ★" if imp >= 0.8 else ""
                        tl_lines.append(f"  {ts} — {desc}{marker}")
                    seg["timeline"] = "\n".join(tl_lines)
            else:
                tl = format_timeline(limit=15)
                if tl:
                    seg["timeline"] = tl
        except Exception as e:
            print(f"[Prompt] Timeline error: {e}")

        # 10. Knowledge graph (entities related to current query)
        try:
            if query and not _trivial:
                from memory.entity_store import get_entity_graph
                kg = get_entity_graph(query)
                if kg:
                    seg["kg"] = kg
        except Exception as e:
            print(f"[Prompt] Knowledge graph error: {e}")

        return seg

    # ── Background prefetch (D: response-time, E: typing-time) ──────────

    def _prefetch_next_payload(self) -> None:
        """D — runs in a daemon thread AFTER a response lands: pre-build
        static prompt segments + warm Chroma so the next send is cheap."""
        try:
            self._static_prompt_segments()   # populate/refresh the cache
        except Exception as e:
            print(f"[Prefetch] static build failed: {e}")
        try:
            if not hasattr(self, "_chroma") or not self._chroma:
                from memory.chroma_memory import ChromaMemory, CHROMA_VECTOR_SEARCH_ENABLED
                if CHROMA_VECTOR_SEARCH_ENABLED:
                    self._chroma = ChromaMemory()
            if getattr(self, "_chroma", None):
                self._chroma.search_memory("warmup", n_results=1)  # loads embedding model
        except Exception:
            pass
        try:
            self._soft_precompact()
        except Exception as e:
            print(f"[Prefetch] soft pre-compact failed: {e}")

    def _soft_precompact(self) -> None:
        """Pre-summarize the oldest slice at ~90% budget so the actual
        trigger turn consumes it without an LLM wait. Read-only — the
        snapshot is applied (or discarded) in _auto_compact_if_needed."""
        if not self._cur_session_id:
            return
        conv = list(self._conversation)
        if len(conv) <= 2:
            return
        from memory.conversation_db import estimate_messages_tokens, summarize_turns, _estimate_tokens
        raw_tokens = estimate_messages_tokens(conv)
        sys_est = getattr(self, "_last_system_prompt_tokens", 7443)
        total_est = int(sys_est) + raw_tokens + int(getattr(self, "_tool_cost_est", 2500))
        budget = int(getattr(self, "_input_budget", 50000))
        if total_est < int(budget * 0.90):
            return
        slice_tokens = _compact_slice_tokens(raw_tokens)
        if raw_tokens < slice_tokens * 2:
            return
        candidates = conv[:-2]
        if not candidates:
            return
        trim_idx = _oldest_trim_split(candidates, slice_tokens)
        if not trim_idx:
            return
        oldest = [candidates[i] for i in trim_idx]
        summary = summarize_turns(oldest, target_tokens=max(500, slice_tokens // 3))
        if not summary:
            return
        self._precompact = {
            "summary": summary,
            "refs": conv,               # identity check: only +1 user msg allowed
            "conv_len": len(conv),
            "trim_idx": trim_idx,
            "built_at": time.time(),
        }
        print(f"[Prefetch] pre-compact ready: ~{slice_tokens} tok -> "
              f"{_estimate_tokens(summary):,} tok (applied at next send)")

    def _prefetch_query(self, query: str) -> None:
        """E — runs in a daemon thread while the user types: build the
        query-dependent prompt blocks for the text so far."""
        try:
            q = (query or "").strip()
            if len(q) < 4 or self._is_trivial_query(q):
                return
            seg = self._build_query_prompt_segments(q)
            self._prefetched_query = {"query": q, "seg": seg, "built_at": time.time()}
        except Exception as e:
            print(f"[Prefetch] query prefetch failed: {e}")

    # ------------------------------------------------------------------
    # Speaking state & TTS
    # ------------------------------------------------------------------

    @staticmethod
    def _is_trivial_query(query: str) -> bool:
        """Retrieval gating (self-RAG style): skip memory/context retrieval
        for greetings and 1-2 word queries — saves tokens and avoids
        context distraction."""
        q = (query or "").strip().lower()
        if len(q) < 6:
            return True
        if re.fullmatch(r"(hi+|hey+|hello+|yo+|ok+|okay+|thanks|thank you|nice|great|good)\W*", q):
            return True
        return False

    def _tts_worker(self) -> None:
        self._tts_ready.wait(timeout=120)

        def _synth_loop():
            while True:
                text = self._tts_queue.get()
                if self._should_stop.is_set():
                    self._should_stop.clear()
                    self._tts_queue.task_done()
                    continue
                try:
                    streaming = bool(getattr(self._tts, "supports_streaming", False))
                    if streaming and text and self._tts:
                        for chunk in self._tts.synthesize_stream(text):
                            if self._should_stop.is_set():
                                break
                            self._audio_queue.put(chunk)
                    else:
                        # Self-playing engines (hashim_live) BLOCK inside
                        # synthesize() while their own worker plays the audio,
                        # so SPEAKING must be set BEFORE the call — setting it
                        # after was visible only for a split second once the
                        # reply had already finished playing.
                        if text.strip() and not self._should_stop.is_set():
                            self._self_playing_begin(text)
                        audio = self._tts.synthesize(text) if text and self._tts else np.array([], dtype=np.float32)
                        if self._should_stop.is_set():
                            self._should_stop.clear()
                            self._tts_queue.task_done()
                            continue
                        self._audio_queue.put(audio)
                except Exception as e:
                    print(f"[TTS] synth error: {e}")
                    # Surface in the activity log too — a print-only error is
                    # invisible to the user, who just sees the face never
                    # enter SPEAKING (e.g. TTS quota exhausted).
                    try:
                        self.ui.write_log(f"ERR: TTS — {str(e)[:180]}")
                    except Exception:
                        pass
                    try:
                        self.ui.glance(0.8, 0.25, 1.5)
                    except Exception:
                        pass
                self._audio_queue.put(None)

        threading.Thread(target=_synth_loop, daemon=True).start()

        def _turn_playback_done() -> None:
            # Turn boundary: the play-cursor belongs to ONE continuous
            # playback. Retire it so the next turn's first batch anchors
            # fresh instead of inheriting a stale (past/future) position.
            self._reset_play_cursor()

        while True:
            try:
                item = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if self._should_stop.is_set():
                self._should_stop.clear()
                continue
            if item is None:
                # One sentence finished. End the turn ONLY when every queued
                # sentence has played — otherwise a sentence synthesized
                # after a tool call drops _speaking mid-reply (state flaps
                # SPEAKING→PROCESSING→SPEAKING, mic-hold re-arms repeatedly
                # and the mouth freezes while audio continues).
                self._speak_pending = max(0, getattr(self, "_speak_pending", 1) - 1)
                self._tts_queue.task_done()
                if self._speak_pending == 0 and self._tts_queue.empty():
                    self._end_speaking()   # whole batch played → mic through tail
                    _turn_playback_done()
                    if not self.ui.muted:
                        self.ui.set_state("LISTENING")
                continue
            streaming = bool(getattr(self._tts, "supports_streaming", False))
            if streaming and isinstance(item, bytes):
                self._play_stream_chunk(item)
            else:
                self._play_blocking(item)

    def _self_playing_begin(self, text: str) -> None:
        """Announce SPEAKING for engines that play their own audio.

        hashim_live blocks inside synthesize() until its own PyAudio worker
        has played the whole turn, so turn-start is the only hook BEFORE
        playback. When a PCM tap is attached the mouth is driven by the REAL
        audio batches during the blocking call; without a tap a syllable-
        timed fallback schedule keeps the mouth roughly in sync.
        """
        if not text.strip():
            return
        self._play_at = time.time() + 0.25   # typical synthesis latency
        self._tap_seen = False
        with self._speaking_lock:
            self._speaking = True
        self.ui.set_state("SPEAKING")
        # No PCM tap: keep the mouth's ground-truth window alive for the
        # estimated speech duration so PROCESSING can't freeze it mid-reply.
        try:
            self.ui.notify_audio_activity(min(20.0, max(1.5, len(text) / 13.0)))
        except Exception:
            pass
        if not getattr(self, "_pcm_sink_attached", False):
            try:
                self.ui.push_visemes(self._pseudo_viseme_schedule(text), 0.048,
                                     self._play_at)
            except Exception:
                pass

    def _reset_play_cursor(self) -> None:
        """Kill the viseme play-cursor + any pending schedule.

        The cursor is only valid INSIDE one continuous playback turn. A
        new turn must never inherit it: the old cursor points into the
        past (mouth freezes while audio plays) or the future (lip-sync
        delayed by the whole previous turn). Called on barge-in AND on
        normal turn end.
        """
        self._play_cursor = None
        self._play_at = time.time()
        self._tap_seen = False
        try:
            self.ui.clear_visemes()
        except Exception:
            pass

    def _feed_face_pcm_bytes(self, data: bytes) -> None:
        """PCM tap from a self-playing TTS engine -> real lip-sync.

        Runs on the engine's playback thread; _feed_face_audio computes the
        level and formant visemes from the exact samples being played, so
        the mouth matches the voice instead of an estimated schedule.

        Timing uses Mark-LIV's play-cursor: each batch sounds one batch-
        duration after the previous one, so the cursor only needs an
        anchor. A small LEAD is deliberate — a mouth slightly ahead of the
        voice reads as sync, one behind reads as dubbing. The cursor is
        re-anchored only when it leaves the physically possible window
        (behind now = device drained; too far ahead = drift).
        """
        try:
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            if not samples.size:
                return
            if self._measuring_bleed:
                with self._bleed_lock:
                    b = self._mic_bleed
                    if b is not None and len(b) < 2000:
                        b.append(float(np.sqrt(np.mean(samples ** 2))))
            # Fingerprint of what the TTS actually played (10 Hz envelope).
            self._envelope_push(getattr(self, "_played_env", []), samples,
                                24000, self._ENV_RATE)
            now = time.time()
            cursor = getattr(self, "_play_cursor", None)
            horizon = 0.6 + 0.15          # PyAudio buffer capacity + slack
            stall = now - 0.08            # device underrun / stalled worker
            if cursor is None or not (stall <= cursor <= now + horizon):
                # NOT in the physically-possible window:
                #  - first sound of a turn  → anchor with a small lead
                #  - cursor in the past     → device drained mid-turn
                #    (Gemini synthesis stall after a tool call) — re-anchor
                #    NOW so visemes land in the future, not in a graveyard
                #    where the HUD purges them instantly (frozen mouth).
                #  - cursor far ahead       → drift after a pause, resync.
                cursor = now + 0.043      # first-sound lead
            self._play_cursor = cursor
            self._play_at = cursor
            self._tap_seen = True
            self._feed_face_audio(samples, 24000)
            # _feed_face_audio advanced _play_at by this batch's duration;
            # keep the cursor in step for the next batch. If the push was
            # clamped (past-schedule graveyard below), _play_at may lag the
            # cursor — never move the cursor BACKWARDS, that would
            # re-anchor every batch and jitter the lip-sync.
            advanced = getattr(self, "_play_at", cursor)
            self._play_cursor = cursor if advanced <= cursor else advanced
        except Exception as e:
            print(f"[HUD] face pcm tap error: {e}")

    def _attach_pcm_sink(self) -> None:
        """Hook real-time PCM from any self-playing engine, if supported."""
        engines: list = []
        t = self._tts
        if hasattr(t, "_engine"):
            engines.append(t._engine)
        elif hasattr(t, "_players"):
            engines.extend(getattr(pl, "_engine", None) for pl in t._players)
        for eng in engines:
            if eng is not None and hasattr(eng, "set_pcm_sink"):
                try:
                    eng.set_pcm_sink(self._feed_face_pcm_bytes)
                    self._pcm_sink_attached = True
                    self.ui.write_log("SYS: Real lip-sync attached (PCM tap).")
                except Exception as e:
                    print(f"[TTS] pcm sink attach failed: {e}")

    @staticmethod
    def _pseudo_viseme_schedule(text: str) -> list[tuple[float, float, float]]:
        """Approximate [level, open, wide] frames from syllable timing.

        Real visemes need the audio samples, which self-playing engines keep
        to themselves; syllable-count timing still gives a believable talk
        cadence (pause on punctuation, ~140 ms per syllable).
        """
        frames: list[tuple[float, float, float]] = []
        import re as _re
        words = _re.findall(r"[\w\u0900-\u097F\u0600-\u06FF]+", text)
        for wi, word in enumerate(words):
            syl = max(1, len(_re.findall(r"[aeiouyAEIOUY\u0900-\u097F\u0600-\u06FF]", word)))
            # 2 frames (~96 ms) per syllable: older 4-5 frames read as a
            # mouth flapping at half the actual speech rate. Real TTS runs
            # ~4-5 syllables/sec — keep the fallback near that cadence.
            for _ in range(min(2, syl)):
                frames.append((round(random.uniform(0.55, 0.95), 2),
                               round(random.uniform(0.35, 0.8), 2),
                               round(random.uniform(0.3, 0.7), 2)))
            # small gap between words, longer at punctuation
            if wi < len(words) - 1:
                ch = text[max(0, text.find(word) + len(word))]
                gap = 3 if ch in ",;:" else (5 if ch in ".!?\u0964\u0965" else 1)
                frames.extend([(0.0, 0.0, 0.0)] * gap)
        return frames or [(0.6, 0.4, 0.4), (0.0, 0.0, 0.0)]

    def _end_speaking(self, tail_hold: float | None = None) -> None:
        """Playback finished: drop _speaking and HARD-MUTE the mic for the
        room-tail window before auto-reopening.

        The mic is the echo's only door — no frames, no echo, no filter
        needed. While the assistant talks the callbacks already drop
        everything; after the turn ends the device buffer is still
        sounding out, so frames stay DROPPED for a FIXED hold (config
        'mic_hold_after_speech', default 3.0 s — the user asked for a
        hard 3-second mute after every spoken turn), then the mic
        reopens by itself. User barge-in passes a short tail_hold — the
        audio was hard-stopped and the user is mid-sentence.
        """
        if tail_hold is None:
            cfg = getattr(self, "_config", None) or {}
            tail_hold = float(cfg.get("mic_hold_after_speech", 3.0))
        self._mic_hold_until = time.time() + tail_hold
        with self._speaking_lock:
            self._speaking = False

    def _mic_held(self) -> bool:
        return time.time() < getattr(self, "_mic_hold_until", 0.0)

    def _set_listening_if_idle(self) -> None:
        """Set LISTENING only when nothing is being spoken.

        End-of-turn handlers used to call set_state("LISTENING") directly;
        the TTS worker thread then set SPEAKING a moment later and one of
        the two signals arrived last at random — the face could freeze in
        LISTENING while audio was still playing. Guarded by _speaking.
        """
        with self._speaking_lock:
            speaking = self._speaking
        if speaking:
            return
        # When muted, still settle to MUTED (not LISTENING) — but never
        # leave the face stuck in SPEAKING after playback really ended.
        self.ui.set_state("MUTED" if self.ui.muted else "LISTENING")

    def _feed_face_audio(self, samples, sample_rate: int) -> None:
        # Audio ground truth: PCM is being played RIGHT NOW. The HUD keeps
        # the mouth live for a short grace window after the last batch —
        # independent of the state string, which may legitimately read
        # PROCESSING/THINKING while a queued reply is still sounding.
        try:
            self.ui.notify_audio_activity()
        except Exception:
            pass
        try:
            from core.face_renderer import pcm_level, pcm_visemes
            level = pcm_level(samples)
            self.ui.set_audio_level(level)
            frames = pcm_visemes(samples, sample_rate)
            if frames:
                play_at = getattr(self, "_play_at", time.time())
                # A batch must never be scheduled in the PAST: the HUD's
                # push_visemes purges frames older than 'now', so past
                # frames = instant delete = frozen mouth while audio runs.
                play_at = max(play_at, time.time())
                self.ui.push_visemes(frames, 480 / max(1, sample_rate), play_at)
                self._play_at = play_at + len(samples) / max(1, sample_rate)
        except Exception as e:
            print(f"[HUD] face audio feed error: {e}")

    def _play_stream_chunk(self, data: bytes) -> None:
        """Write one raw PCM chunk to the streaming output device."""
        if self._out_stream is None:
            if not self._open_out_stream():
                return
            self._play_at = time.time()
            with self._speaking_lock:
                was_speaking = self._speaking
                self._speaking = True
            if not was_speaking:
                self._play_at = time.time()
            self.ui.set_state("SPEAKING")
        try:
            samples = np.frombuffer(data, dtype=np.int16)
            self._out_stream.write(samples)
            self._feed_face_audio(samples, self._tts.sample_rate)
        except Exception as e:
            print(f"[TTS] playback error: {e}")
            self._close_out_stream()

    def _open_out_stream(self) -> bool:
        try:
            self._out_stream = sd.OutputStream(
                samplerate=self._tts.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=2400,
            )
            self._out_stream.start()
            return True
        except Exception as e:
            print(f"[TTS] output stream error: {e}")
            self._out_stream = None
            return False

    def _close_out_stream(self) -> None:
        if self._out_stream is not None:
            try:
                self._out_stream.stop()
                self._out_stream.close()
            except Exception:
                pass
            self._out_stream = None

    def _play_blocking(self, audio: np.ndarray) -> None:
        """Legacy playback path (non-streaming engines: Edge, Kokoro, ...)."""
        if audio.size > 0:
            self._play_at = time.time()
            with self._speaking_lock:
                self._speaking = True
            self.ui.set_state("SPEAKING")
            self._feed_face_audio(audio, self._tts.sample_rate)
            sd.play(audio, self._tts.sample_rate)
            duration = len(audio) / self._tts.sample_rate
            # Blocking path feeds the whole array ONCE — extend the mouth's
            # ground-truth window to cover the whole sentence, not just 0.9s.
            try:
                self.ui.notify_audio_activity(duration + 0.5)
            except Exception:
                pass
            time.sleep(duration)
        # Turn end is handled by the None-sentinel in the TTS worker —
        # per-sentence _end_speaking here would cut multi-sentence replies
        # into separate turns (state flicker + repeated mic-hold).

    def set_speaking(self, value: bool) -> None:
        with self._speaking_lock:
            self._speaking = value
        if value:
            # Mute = the USER's mic is off; the assistant's mouth is not
            # bound to it. Text input while muted still speaks, and the
            # face must animate — so SPEAKING is always raised.
            self.ui.set_state("SPEAKING")
        else:
            self.ui.set_state("MUTED" if self.ui.muted else "LISTENING")

    def speak(self, text: str) -> None:
        if not text:
            return
        # Gemini Live mode speaks natively through the persistent session
        _live = getattr(self, "_live_provider", None)
        if _live is not None and _live.session_active:
            _live.speak(text)
            return
        if not self._tts:
            return
        # Remember the tail of what is about to be said: whatever the
        # speakers play back, the microphone hears again a moment later,
        # and the transcript filter matches against this tail.
        try:
            self._remember_spoken_tail(text)
        except Exception:
            pass
        cfg = getattr(self, "_config", None) or {}
        if str(cfg.get("echo_guard_mode", "auto") or "auto").lower() != "off":
            # Sample the mic while this turn plays — even when the gate is
            # currently off, so a headphone→speaker switch re-arms it from
            # real evidence instead of never being noticed.
            self._mic_bleed = []
            self._measuring_bleed = True
        with self._speaking_lock:
            self._speaking = True
        self._speak_pending = getattr(self, "_speak_pending", 0) + 1
        self._tts_queue.put(text)

    def speak_error(self, tool_name: str, error) -> None:
        short = str(error)[:2000]
        print(f"ERR: {tool_name} — {short}")
        self.ui.write_log(f"ERR: {tool_name} — {short}")
        self.speak(f"{tool_name} encountered an error.")

    def _on_live_output(self, text: str) -> None:
        """Model output transcription from a voice-initiated Live turn."""
        try:
            self._extract_and_log(text)
        except Exception:
            pass
        try:
            self._deliver_response(text, source="live")
        except Exception:
            pass

    def _on_live_user(self, text: str) -> None:
        """User speech transcription from the Live session (logging only).

        The Live API streams input transcription as MANY small fragments
        (often one word each). Logging each fragment as it arrives produced
        one "You: <word>" bubble per word in the activity log. Fragments are
        therefore accumulated and the completed sentence is emitted once,
        after a short debounce window with no new fragments (handles final
        fragments that arrive just after turn_complete too). Cumulative
        transcripts (each event repeating the full text so far) collapse to
        their latest form instead of being joined twice.
        """
        frag = (text or "").strip()
        if not frag:
            return
        with self._live_user_lock:
            acc = self._live_user_acc
            if acc and frag.startswith(acc):
                self._live_user_acc = frag        # cumulative update
            else:
                self._live_user_acc = (acc + " " + frag).strip()
            timer = self._live_user_timer
        if timer is not None:
            timer.cancel()
        timer = threading.Timer(1.2, self._flush_live_user)
        timer.daemon = True
        with self._live_user_lock:
            self._live_user_timer = timer
        timer.start()

    def _flush_live_user(self) -> None:
        with self._live_user_lock:
            full = self._live_user_acc
            self._live_user_acc = ""
            self._live_user_timer = None
        if not full:
            return
        self.ui.write_log(f"You: {full}")
        try:
            from memory.entity_store import link_fact
            link_fact("user_message", "input", full[:500])
        except Exception:
            pass

    def _on_live_speaking(self, value: bool) -> None:
        if value:
            self.ui.set_state("SPEAKING")
        elif not self.ui.muted:
            self.ui.set_state("LISTENING")

    def stop_speaking(self) -> None:
        sd.stop()
        self._close_out_stream()
        self._should_stop.set()
        # The speakers are still sounding the end of this turn while the mic
        # reopens: gate mic input until the tail has decayed, and drop any
        # half-captured utterance — it is the assistant's own voice, not the
        # user's. The stale spoken-tail match is retired with the turn; the
        # next speak() records a fresh one.
        self._measure_bleed()
        # Keep only the last ~4 s of each envelope: enough for correlation
        # against the next capture, never a memory leak over hours.
        for attr in ("_played_env", "_capture_env"):
            env = getattr(self, attr, None)
            if env is not None and len(env) > 40:
                del env[:-40]
        if self._gate_active():
            # The speakers are still sounding the end of this turn while the
            # mic reopens: gate mic input until the tail has decayed, and
            # drop any half-captured utterance — it is the assistant's own
            # voice, not the user's.
            self._open_echo_gate_soon()
        self._last_echo_text = None
        _vad = getattr(self, "_vad_ref", None)
        if _vad is not None and self._echo_risk:
            try:
                _vad.reset()
            except Exception:
                pass
        # User barge-in: audio is hard-stopped, so only a short tail —
        # open the mic FAST, the user is mid-sentence.
        self._end_speaking(0.25)
        # Gemini Live mode: drain the persistent session's audio queue too
        _live = getattr(self, "_live_provider", None)
        if _live is not None:
            try:
                _live.interrupt()
            except Exception:
                pass
        # Regular TTS engines: interrupt in-flight synthesis as well
        if (_live is None or not getattr(_live, "session_active", False)) and self._tts is not None:
            try:
                self._tts.interrupt()
            except Exception:
                pass
        while not self._tts_queue.empty():
            try:
                self._tts_queue.get_nowait()
                self._tts_queue.task_done()
            except queue.Empty:
                break
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break
        self._speak_pending = 0   # barge-in voided the whole batch
        # Barge-in killed playback: the play-cursor is now garbage. Reset
        # BEFORE the next turn or its first visemes land in the past
        # (mouth stays frozen through the whole new reply).
        self._reset_play_cursor()
        # Hashim Live TTS is a blocking engine — clear the stop flag so the
        # next response is not dropped (the interrupted turn already stopped)
        if (getattr(self, "_config", None) or {}).get("tts_engine", "").lower() == "hashim_live":
            self._should_stop.clear()
        self.ui.set_state("LISTENING")

    def _signal_cancel(self) -> None:
        with self._cancel_lock:
            self._cancel_seq += 1
            self._cancel_event.set()

    # ── speaker-echo rejection ──────────────────────────────────────────────
    # The laptop mic sits next to the speakers. While the assistant talks,
    # the mic callback drops chunks — but the moment the turn ends the last
    # few hundred milliseconds of speech are still in the audio device's
    # buffer and play out into an open mic. The VAD hears real speech (the
    # assistant's own final words) and the STT returns it as if the user
    # had said it. Pauses inside the reply split that echo into several
    # phantom utterances, each of which queues a reply, and the first of
    # those interrupts the current reply mid-way. Two layers reject it: a
    # time gate right after the turn, and a text match against the tail of
    # what was actually said.

    _ECHO_COOLDOWN = 1.0    # seconds of mic quiet enforced right after a turn
    _ECHO_TAIL_KEEP = 24    # words of spoken text retained for matching
    # Mic-bleed classification (float32 RMS). Below the floor the mic cannot
    # hear the playback at all (headphones, far device); above the speech
    # mark the playback is arriving loud and clear (speakers next to the mic).
    _BLEED_FLOOR = 3e-4
    _BLEED_SPEECH = 3e-3
    # Device names that cannot reproduce the classic speaker-echo path.
    _HEADPHONE_MARKERS = ("headphone", "headset", "earphone", "airpod",
                          "earbuds", "buds", "virtual")
    # A mic that demonstrably hears the playback (an earphone boom sits
    # centimetres from the driver, and MME/WDM-KS buffers hold seconds of
    # audio) keeps receiving the tail well past the second mark, so its
    # quiet window is longer than the clean desktop path needs.
    _ECHO_COOLDOWN_LOUD = 2.2
    _ENV_RATE = 10            # envelope samples per second
    _ECHO_CORR = 0.60         # envelope correlation that proves echo
    # Only content words are evidence. Contraction fragments stay glued
    # ('i'm', 'you're') and grammatical glue is excluded, so an utterance
    # sharing nothing but glue and two common verbs with a reply —
    # "Actually, yeah, I need help." — can never again read as an echo.
    _STOPWORDS = frozenset((
        " a an and are be being been do does did doing for from had has "
        "have having he her hers him his how i if in into is it its me my "
        "of on or our ours she should so some than that the their them "
        "then there these they this those to too up was we were what when "
        "where which who whose why will with would you your yours am as "
        "at but by just okay ok oh um ha hai na nahi bhai ya to tho aur "
        "yeh vo kya kab kahan ").split())

    def _echo_words(self, text: str) -> list[str]:
        """Contraction-safe lowercase tokens.

        The naive split tore "I'm" into "i" + "m", and the stray "i" went
        on to match a real user utterance containing "I" — three such
        straws later the filter was eating genuine speech. Apostrophes
        stay inside words here.
        """
        import re as _re
        return [w.lower() for w in
                _re.findall(r"[\w']+", (text or "").replace("\u2019", "'"))]

    def _remember_spoken_tail(self, text: str) -> None:
        words = self._echo_words(text)
        self._spoken_tail = words[-self._ECHO_TAIL_KEEP:]
        # Evidence index: content words only (see _STOPWORDS).
        self._spoken_content = [w for w in words
                                if w not in self._STOPWORDS][-self._ECHO_TAIL_KEEP:]
        # Fresh reply: its acoustic fingerprint is rebuilt from the PCM tap
        # as the TTS actually plays it.
        self._played_env = []
        self._capture_env = []
        self._echo_streak = 0

    def _echo_gate_open(self) -> bool:
        """False while the previous turn's audio tail is still ringing out."""
        return time.time() >= getattr(self, "_echo_until", 0.0)

    def _gate_active(self) -> bool:
        """Whether echo gating is in force, honouring the manual override.

        echo_guard_mode: "auto" (default) trusts the bleed measurement,
        "on" forces the gate for a mic that leaks, "off" disables it
        for a headset user who wants zero-latency barge-in. The transcript
        filter honours the same switch.
        """
        cfg = getattr(self, "_config", None) or {}
        mode = str(cfg.get("echo_guard_mode", "auto") or "auto").lower()
        if mode == "on":
            return True
        if mode == "off":
            return False
        return bool(getattr(self, "_echo_risk", True))

    def _open_echo_gate_soon(self) -> None:
        # The quiet window scales with how loudly THIS mic heard the last
        # turn. Calibrated live on a TRRS combo headset (earbud driver and
        # mic in the same shell, one USB dongle): its mic picks the
        # playback up at ~0.031 RMS — ten times the speech mark — and that
        # bleed outlives the fixed window a clean desktop mic needs.
        # peak 0.031 -> ~2.7 s, a 0.1-RMS speaker rig hits the 4.5 s cap,
        # and a silent (headphone) mic keeps the plain 1.0 s floor.
        peak = float(getattr(self, "_last_bleed_peak", 0.0) or 0.0)
        window = min(4.5, max(self._ECHO_COOLDOWN, 1.0 + peak * 55.0))
        self._echo_until = time.time() + window

    _JUNK_PHRASES = frozenset((
        "thank you", "thank you for watching", "thanks for watching",
        "thank you bye", "thank you very much", "please subscribe",
        "subscribe", "amara org", "subtitles by the amara org community",
        "you", "you you", "bye", "bye bye", "ok", "okay", "mm hmm", "hmm",
        "uh", "um", "uhh", "umm", "music", "applause", "silence",
        "the", "and", "so", "mhm", "uh huh",
    ))

    def _junk_transcript(self, text: str, audio) -> bool:
        """True = STT emitted a phantom, drop it before it becomes a turn.

        Whisper-family decoders hallucinate on marginal audio: a beep, a
        chair creak or trailing silence comes back as "00:00", a bare
        number, or one of the classic sub-caption phrases ("thank you for
        watching"). These are NOT echo (that filter stays independent and
        off); they are decoder noise. A real command contains words — an
        utterance that is ONLY a clock pattern / digits / a caption
        phrase is junk, and quiet audio with almost no content words is
        junk too (aggressive defaults by user request). Config:
        'junk_transcript_filter' (on), 'junk_phrases' (extra phrases),
        'junk_quiet_rms' (0.018), 'junk_max_content_words' (2).
        """
        cfg = getattr(self, "_config", None) or {}
        if not cfg.get("junk_transcript_filter", True):
            return False
        t = (text or "").strip().lower()
        if not t:
            return False
        import re as _re
        bare = _re.sub(r"[^\w\s:.]", "", _re.sub(r"\s+", " ", t)).strip()
        # Whole utterance is just a time or number: 00:00, 12.30, 100, 7:00
        if _re.fullmatch(r"\d{1,2}[:.]\d{2}", bare):
            return True
        if _re.fullmatch(r"\d{1,6}", bare.replace(":", "").replace(".", "")):
            return True
        phrases = set(self._JUNK_PHRASES)
        extra = cfg.get("junk_phrases")
        if isinstance(extra, (list, tuple)):
            phrases.update(str(x).strip().lower() for x in extra if str(x).strip())
        if bare in phrases:
            return True
        # Marginal audio + barely any content: decoder noise, junk.
        # Tunables: 'junk_quiet_rms' (default 0.018 — anything quieter is
        # treated as marginal), 'junk_max_content_words' (default 2).
        try:
            import numpy as _np
            a = _np.asarray(audio, dtype=_np.float32)
            rms = float(_np.sqrt(_np.mean(a ** 2))) if a.size else 0.0
        except Exception:
            rms = 0.0
        content = [w for w in bare.split() if w not in self._STOPWORDS]
        quiet_rms = float(cfg.get("junk_quiet_rms", 0.018))
        max_content = int(cfg.get("junk_max_content_words", 2))
        if rms < quiet_rms and len(content) <= max_content:
            return True
        try:
            self.ui.write_log(f"SYS: Junk transcript dropped — “{t[:40]}”")
        except Exception:
            pass
        return False

    def _reject_echo(self, text: str) -> bool:
        """True = drop this transcript; the assistant is hearing itself.

        Order-insensitive match against the spoken tail, tolerant of the
        spelling drift between what the TTS *said* and what the STT
        *heard* (nahin/nahi, toh/to, jaanna/janna — Hindi-English
        transcriptions wobble constantly). A genuine user reply adds new
        content and survives; a near-verbatim repeat of the reply does
        not. A duplicate check also catches the same phantom arriving
        twice through different VAD splits.
        """
        cfg = getattr(self, "_config", None) or {}
        if str(cfg.get("echo_guard_mode", "auto") or "auto").lower() == "off":
            return False   # manual override: the user wants no filtering
        got = self._echo_words(text)
        if not got:
            return False
        tail = getattr(self, "_spoken_tail", []) or []
        content = [w for w in got if w not in self._STOPWORDS]

        # Tier 0 — pure acoustic proof. If the utterance's own envelope
        # correlates strongly with what the TTS just played, the mic
        # replayed the reply; text cannot argue with physics. The stricter
        # bar (0.75) keeps a coincidentally similar real reply safe.
        if len(getattr(self, "_capture_env", []) or []) >= 15                 and self._capture_is_echo(min_corr=0.75):
            self._last_echo_text = text
            self._extend_echo_gate()
            self._log_echo_reject(text)
            return True

        import difflib

        def _sim(a: str, b: str) -> float:
            return difflib.SequenceMatcher(None, a, b).ratio()

        def _reject() -> bool:
            self._last_echo_text = text
            self._extend_echo_gate()
            self._log_echo_reject(text)
            return True

        # Path B — verbatim echo regardless of part of speech. Six-plus
        # tokens: below that the utterance is too short for text alone
        # (a user quoting the closing words looks identical to a
        # VAD-chopped echo fragment) and falls to the acoustic branch.
        if tail and len(got) >= 6:
            hits_all = sum(1 for w in got
                           if w in tail or any(_sim(w, t) >= 0.8 for t in tail))
            if hits_all >= 0.75 * len(got):
                return _reject()

        # Short verbatim fragments are ambiguous, so text never decides
        # alone here: full match on <=5 tokens rejects only when the
        # capture's own envelope also correlates with what played. A
        # missing envelope passes — the user is heard.
        if tail and 3 <= len(got) < 6:
            hits_all = sum(1 for w in got
                           if w in tail or any(_sim(w, t) >= 0.8 for t in tail))
            if hits_all == len(got) and self._capture_is_echo(min_corr=0.5):
                return _reject()

        # Tier 1 — near-verbatim echo of the reply's content words. An
        # echo repeats them wholesale; a human reply adds its own. Five
        # content words minimum and at most three strays keep diluted
        # real speech ("Actually, yeah, I need help.") safely outside.
        if tail and len(content) >= 5:
            hits = sum(1 for w in content
                       if w in tail or any(_sim(w, t) >= 0.8 for t in tail))
            if hits >= 0.75 * len(content) and (len(content) - hits) <= 3:
                return _reject()

        # Tier 3 — contiguous verbatim run. Echo mixed with room audio
        # carries a long run of the reply's words in order even when its
        # overall overlap ratio is low; a genuine reply does not.
        if tail and len(got) >= 6:
            run = max((b.size for b in difflib.SequenceMatcher(
                None, got, tail).get_matching_blocks()), default=0)
            if run >= 6:
                return _reject()

        # Tier 2 — near-total content set overlap, order-insensitive.
        ref = self._spoken_content or tail
        if tail and len(content) >= 5 and difflib.SequenceMatcher(
                None, " ".join(sorted(content)),
                " ".join(sorted(ref))).ratio() >= 0.70:
            return _reject()

        # Tier 2 — partial overlap needs SUPPORT, never a bare count:
        # acoustic proof that the mic heard the playback, or a streak of
        # already-caught phantoms this turn. "Actually, yeah, I need
        # help." lands here and passes — as it must.
        if getattr(self, "_last_echo_text", None) == text:
            return True   # same phantom again — already logged once
        if tail and len(content) >= 3:
            hits = sum(1 for w in content if w in tail)
            if hits >= max(3, int(0.5 * len(content))):
                if self._capture_is_echo():
                    self._last_echo_text = text
                    self._extend_echo_gate()
                    self._log_echo_reject(text)
                    return True
                if getattr(self, "_echo_streak", 0) >= 1:
                    self._last_echo_text = text
                    self._extend_echo_gate()
                    self._log_echo_reject(text)
                    return True
        return False

    def _extend_echo_gate(self) -> None:
        """An echo arrives in pieces; each caught piece keeps the gate shut
        a little longer so its siblings cannot follow it through. Three or
        more caught pieces in one turn means an echo STORM — the tail is
        bouncing around the room — so the hold escalates hard."""
        self._echo_until = max(self._echo_until, time.time() + 0.9)
        if getattr(self, "_echo_streak", 0) >= 3:
            self._echo_until = max(self._echo_until, time.time() + 2.0)

    def _seed_echo_risk(self) -> None:
        """First guess before any playback has been measured.

        Headset/headphone mics sit next to the ears, not next to the
        speakers — the classic echo path does not exist there, so the gate
        starts open for instant barge-in. Any lookup failure keeps the safe
        default (risk on) for this session.
        """
        try:
            d = sd.query_devices(kind="input") or {}
            name = str(d.get("name", "")).lower()
        except Exception:
            return
        if any(m in name for m in self._HEADPHONE_MARKERS):
            self._echo_risk = False
            try:
                self.ui.write_log(
                    "SYS: Headphones detected — echo gate off (instant barge-in).")
            except Exception:
                pass
            return
        # Combo (TRRS) headset: capture and playback endpoints exposed by
        # the SAME dongle — "Microphone (USB Audio and HID)" alongside
        # "Speakers (USB Audio and HID)". The mic sits millimetres from the
        # earbud driver, so it hears the playback far louder than any room
        # echo could be, and the gate must stay on from the first turn.
        import re as _re
        try:
            out_d = sd.query_devices(kind="output") or {}
            out_name = str(out_d.get("name", "")).lower()
        except Exception:
            return
        if not out_name:
            return
        stops = {"and", "audio", "input", "output", "microphone", "speaker",
                 "speakers", "digital", "high", "definition", "device",
                 "sound", "r", "mapper", "primary", "driver", "capture",
                 "line", "lines", "default"}
        tin = {t for t in _re.findall(r"[a-z0-9]+", name) if t not in stops}
        tout = {t for t in _re.findall(r"[a-z0-9]+", out_name) if t not in stops}
        if len(tin & tout) >= 2:
            self._echo_risk = True
            try:
                self.ui.write_log(
                    "SYS: Combo headset detected (mic + speakers on the same "
                    "dongle) — echo gate on.")
            except Exception:
                pass

    def _measure_bleed(self) -> None:
        """Settle echo risk from mic samples captured during playback.

        Whatever the microphone actually heard while the TTS was sounding
        is the ground truth: silent mic = headphones (or an isolated
        device), loud mic = speakers in the acoustic path. A too-short
        capture changes nothing — belief is only ever revised on evidence.
        """
        self._measuring_bleed = False
        with self._bleed_lock:
            samples, self._mic_bleed = self._mic_bleed, []
        if len(samples) < 20:
            return   # not enough evidence; keep current belief
        peak = max(samples)
        self._last_bleed_peak = peak
        self._last_bleed_loud = peak > self._BLEED_SPEECH
        if peak < self._BLEED_FLOOR:
            if self._echo_risk:
                self._echo_risk = False
                try:
                    self.ui.write_log(
                        "SYS: Mic heard nothing during playback — echo gate off "
                        "(instant barge-in).")
                except Exception:
                    pass
        elif peak > self._BLEED_SPEECH and not self._echo_risk:
            self._echo_risk = True   # device switched back to speakers
            try:
                self.ui.write_log(
                    "SYS: Speakers audible to mic again — echo gate on.")
            except Exception:
                pass

    # ── acoustic echo fingerprint ─────────────────────────────────────────
    # Mark-LIV never hits this bug because Gemini Live cancels echo on the
    # SERVER (WebRTC AEC) before transcription. A local VAD+STT pipeline
    # has no such protection, so it builds its own evidence: the PCM tap
    # already carries what the TTS actually played; both sides are reduced
    # to a ~10 Hz loudness envelope and correlated. Strong correlation is
    # physical proof the mic heard the playback — a real voice in the room
    # does not correlate with a random utterance's envelope.

    @staticmethod
    def _envelope_push(env: list, samples, rate: int, per_sec: int) -> None:
        """Append this batch's RMS to a downsampled loudness envelope."""
        import numpy as _np
        per = max(1, int(rate / per_sec))
        total = len(samples)
        i = 0
        while i < total:
            seg = samples[i:i + per]
            if seg.size:
                env.append(float(_np.sqrt(_np.mean(seg ** 2))))
            i += per

    def _capture_is_echo(self, min_corr: float | None = None) -> bool:
        """Compare the capture's envelope against what actually played.

        Needs >= ~1.2 s of overlap on both sides (12 samples at 10 Hz);
        anything shorter cannot support a rejection and stays silent —
        the user is heard. Translation invariance: the capture may begin
        mid-reply, so the best alignment offset wins, not offset zero.
        `min_corr` overrides the class threshold — Tier 0 rejects on its
        own only above a stricter bar than Tier 2's support needs.
        """
        import numpy as _np
        played = list(getattr(self, "_played_env", []) or [])
        cap = list(getattr(self, "_capture_env", []) or [])
        if len(played) < 12 or len(cap) < 12:
            return False
        a = _np.asarray(played, dtype=_np.float64)
        b = _np.asarray(cap, dtype=_np.float64)
        best = 0.0
        for off in range(0, max(1, len(a) - 1)):
            bb = b[:len(a) - off]
            aa = a[off:off + len(bb)]
            if len(aa) < 12:
                break
            sa, sb = aa.std(), bb.std()
            if sa < 1e-9 or sb < 1e-9:
                continue
            r = float(((aa - aa.mean()) * (bb - bb.mean())).mean()
                      / (sa * sb))
            best = max(best, r)
        return best >= (self._ECHO_CORR if min_corr is None else min_corr)

    def _log_echo_reject(self, text: str) -> None:
        self._echo_streak = getattr(self, "_echo_streak", 0) + 1
        """One quiet line in the activity log per rejected echo.

        Deliberately throttled to the first rejection of each phantom text:
        an echo storm would otherwise flood the log and hide real events.
        """
        try:
            self.ui.write_log(f"SYS: Mic echo rejected — “{text[:60]}”")
        except Exception:
            pass

    def _is_cancelled(self, my_seq: int) -> bool:
        with self._cancel_lock:
            return self._cancel_seq != my_seq

    def _stop_and_listen(self) -> None:
        self.stop_speaking()
        if not self.ui.muted:
            self.ui.set_state("LISTENING")

    # ------------------------------------------------------------------
    # Live reconfigure (called when user clicks Apply in Configure panel)
    # ------------------------------------------------------------------

    def reconfigure(self, new_config: dict) -> None:
        threading.Thread(
            target=self._do_reconfigure, args=(new_config,), daemon=True
        ).start()

    def _do_reconfigure(self, new_config: dict) -> None:
        old_stt_engine = self._config.get("stt_engine", "whisper").lower()
        old_llm_model  = self._config.get("llm_model", "")
        new_stt_engine = new_config.get("stt_engine", "whisper").lower()
        self._config = new_config

        try:
            from core.installer import install_for_config
            install_for_config(new_config, log=self.ui.write_log)
        except Exception as e:
            self.ui.write_log(f"ERR: Dependency install — {e}")

        try:
            from core.tts import create_tts_player
            self._tts = create_tts_player(new_config)
            self._pcm_sink_attached = False
            self._attach_pcm_sink()
            self._tts_ready.set()
            self.ui.write_log("SYS: TTS reconfigured.")
        except Exception as e:
            self.ui.write_log(f"ERR: TTS reconfigure — {e}")

        if old_stt_engine == new_stt_engine:
            if new_stt_engine == "hashim":
                # Same engine, but a live Gemini session must be rebuilt to
                # pick up a new key/model. Swap atomically: build first, bump
                # the generation so the old bridge loop stands down, connect
                # the new session, then close the old one.
                try:
                    _new = self._build_stt("hashim", new_config)
                except Exception as e:
                    self.ui.write_log(f"ERR: STT reconfigure — {e}")
                    _new = None
                if _new is not None:
                    _old = self._stt
                    self._stt = _new
                    self._stt_generation += 1
                    try:
                        _new.connect()
                    except Exception as e:
                        self.ui.write_log(f"ERR: STT reconfigure — {e}")
                    if _old is not None and hasattr(_old, "close"):
                        try:
                            _old.close()
                        except Exception:
                            pass
                    self.ui.write_log("SYS: STT reconfigured (Hashim live session rebuilt).")
            else:
                try:
                    self._stt = self._build_stt(new_stt_engine, new_config)
                    self.ui.write_log("SYS: STT reconfigured.")
                except Exception as e:
                    self.ui.write_log(f"ERR: STT reconfigure — {e}")
        else:
            # Engine CHANGED: full live switch, no restart. Build the new
            # engine first (failure keeps the old one working), bump the
            # generation so the old mic loop exits, close the old engine,
            # then start the new mic loop.
            try:
                _new = self._build_stt(new_stt_engine, new_config)
            except Exception as e:
                self.ui.write_log(f"ERR: STT reconfigure — {e}")
                _new = None
            if _new is not None:
                _old = self._stt
                self._stt = _new
                self._stt_generation += 1
                if _old is not None and hasattr(_old, "close"):
                    try:
                        _old.close()
                    except Exception:
                        pass
                self.ui.write_log(f"SYS: STT engine switched to {new_stt_engine} — live, no restart.")
                self._mic_thread_started = False
                self._start_mic_loop(new_stt_engine)

        # Save config to disk
        try:
            API_CONFIG_PATH.write_text(json.dumps(new_config, indent=4), encoding="utf-8")
        except Exception as e:
            self.ui.write_log(f"ERR: Config save — {e}")

        if new_config.get("llm_model", "") != old_llm_model:
            self.ui.write_log("SYS: Warming up new LLM model…")
            from core.llm_client import warmup_model
            warmup_model()
            self.ui.write_log("SYS: New LLM model ready.")

                # Re-wire Gemini Live callbacks if provider mode changed
        try:
            if new_config.get("llm_provider", "") == "gemini_live":
                from core.llm_provider import get_registry
                _live = get_registry().get_provider(
                    "gemini_live",
                    api_key=new_config.get("gemini_api_key", ""),
                    model=new_config.get("gemini_live_model", ""),
                )
                _live.register_tool_handler(self._execute_tool)
                _live.register_output_cb(self._on_live_output)
                _live.register_user_cb(self._on_live_user)
                _live.register_muted_getter(lambda: self.ui.muted)
                _live.register_speaking_cb(self._on_live_speaking)
                _live.set_system_prompt(_load_system_prompt())
                self._live_provider = _live
            else:
                if self._live_provider is not None:
                    try:
                        self._live_provider.stop()
                    except Exception:
                        pass
                self._live_provider = None
        except Exception as e:
            print(f"[Live] Callback re-wire failed: {e}")

        if old_stt_engine == new_stt_engine:
            self.speak("Configuration applied.")
        else:
            self.speak(f"Speech engine switched to {new_stt_engine}.")

    # ------------------------------------------------------------------
    # Text command (from UI input box)
    # ------------------------------------------------------------------

    def _on_text_command(self, text: str) -> None:
        self.stop_speaking()
        self._signal_cancel()
        self._text_queue.put(text)

    # ------------------------------------------------------------------
    # Inline chat actions (F7: Regenerate / Continue)
    # ------------------------------------------------------------------

    def _handle_chat_action(self, action: str) -> None:
        """Handle action:// chips from the chat log.

        Runs on the GUI thread (anchor click) — so it only *queues* the work;
        the text-pipeline worker performs all conversation mutation safely.
        """
        try:
            self._signal_cancel()
            self._text_queue.put(("action_msg", action))
        except Exception as e:
            print(f"[ChatAction] {action} queue error: {e}")

    def _run_chat_action(self, action: str) -> None:
        """Worker-side execution of a chat action (regenerate/continue)."""
        try:
            if action == "regenerate":
                # Drop the trailing assistant/tool turns and re-send the
                # newest user message for a fresh answer.
                while self._conversation and self._conversation[-1].get("role") in (
                        "assistant", "tool"):
                    self._conversation.pop()
                last_user = next(
                    (m["content"] for m in reversed(self._conversation)
                     if m.get("role") == "user"), None)
                if not last_user:
                    self.ui.write_log("SYS: Nothing to regenerate yet.")
                    return
                self.ui.write_log("SYS: ↻ Regenerating last answer…")
                self._process_message_inner(last_user, source="ui", extra_meta=None)
            elif action == "continue":
                last_ai = next(
                    (m["content"] for m in reversed(self._conversation)
                     if m.get("role") == "assistant"), None)
                if not last_ai:
                    self.ui.write_log("SYS: Nothing to continue yet.")
                    return
                self.ui.write_log("SYS: ▸ Continuing last answer…")
                self._process_message_inner(
                    "Continue your previous answer from where it stopped. "
                    "Do not repeat what you already wrote.",
                    source="ui", extra_meta=None)
            else:
                self.ui.write_log(f"SYS: Unknown chat action '{action}'.")
        except Exception as e:
            print(f"[ChatAction] {action} error: {e}")
            import traceback
            traceback.print_exc()

    def _on_text_command_with_files(self, text: str, extra: dict = None) -> None:
        """Handle text command with attached files (ChatGPT style).
        
        NOTE: stop_speaking() is NOT called here — this runs in a daemon thread
        and sounddevice access from non-main thread is unsafe.
        stop_speaking() is called later in _process_message_inner.
        """
        try:
            self._signal_cancel()
            self._text_queue.put(("file_msg", text, extra or {}))
        except Exception as e:
            print(f"[Orthos] ERROR in _on_text_command_with_files: {e}")
            import traceback
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Tool execution (routing unchanged from original)
    # ------------------------------------------------------------------

    def _recall_conversation(self, args: dict) -> str:
        """recall_conversation tool: verbatim search over raw past chat.

        Fuses FTS keyword matches (turns_fts) with semantic matches
        (Chroma conversation_turns), dedups by turn id, and renders
        date + session + speaker + snippet. Optional full-session expand.
        """
        import sqlite3 as _sq
        from memory.conversation_db import (
            search_turns_fts, get_session, get_turns_by_session, format_turn_for_prompt, DB_PATH,
        )
        query = (args.get("query") or "").strip()
        if not query:
            return "Error: query required"
        try:
            limit = int(args.get("limit", 8) or 8)
        except (TypeError, ValueError):
            limit = 8
        limit = max(1, min(limit, 25))
        full_session = bool(args.get("full_session", False))

        try:
            fts_hits = search_turns_fts(query, limit=limit)
        except Exception as e:
            fts_hits = []
            print(f"[Recall] FTS error: {e}")

        vec_hits: list = []
        try:
            from memory.chroma_memory import search_turn_embeddings
            vec_hits = search_turn_embeddings(query, n_results=limit)
        except Exception as e:
            print(f"[Recall] Vector error: {e}")

        # Merge with FTS priority, dedup by turn id; top up with recent
        # turns when no signal hits so the tool is always useful.
        merged: dict[int, dict] = {}
        order: list[int] = []
        for t in list(fts_hits) + list(vec_hits):
            tid = int(t.get("id", 0) or 0)
            if tid in merged:
                continue
            merged[tid] = t
            order.append(tid)
        if not merged:
            try:
                conn = _sq.connect(f"file:{DB_PATH}?mode=ro", uri=True)
                rows = conn.execute(
                    "SELECT t.id, t.role, t.content, t.timestamp, t.session_id, s.title "
                    "FROM turns t LEFT JOIN sessions s ON s.id = t.session_id "
                    "WHERE t.archived = 0 ORDER BY t.id DESC LIMIT ?",
                    (min(limit, 5),),
                ).fetchall()
                conn.close()
                for tid, role, content, ts, sid, title in rows:
                    merged[int(tid)] = {
                        "type": "turn", "id": int(tid), "role": role, "content": content,
                        "timestamp": ts or "", "session_id": sid or "",
                        "session_title": title or "(untitled)",
                        "snippet": (content or "")[:200], "score": 0.0,
                    }
                    order.append(int(tid))
            except Exception:
                pass

        if not merged:
            return f"No past conversation found for: {query}"

        lines = [f"[PAST CONVERSATIONS — matched '{query}' verbatim]"]
        best_sid = ""
        for tid in order[:limit]:
            t = merged[tid]
            ts = str(t.get("timestamp", ""))[:10] or "(no date)"
            title = t.get("session_title") or "(untitled)"
            role = t.get("role", "?")
            body = (t.get("snippet") or t.get("content") or "").strip()
            body = body.replace("\n", " ")[:300]
            lines.append(f"• {ts} — session \"{title}\" — {role}: {body}")
            if not best_sid:
                best_sid = t.get("session_id", "")

        if full_session and best_sid:
            sess = get_session(best_sid)
            s_title = (sess or {}).get("title", best_sid) if sess else best_sid
            lines.append("")
            lines.append(f"[FULL SESSION — \"{s_title}\"]")
            try:
                turns = get_turns_by_session(best_sid)
                budget = 6000
                for t in turns:
                    formatted = format_turn_for_prompt(t)
                    if not formatted:
                        continue
                    toks = max(1, len(formatted) // 4)
                    if budget - toks < 0:
                        lines.append("  …(session truncated — too long)")
                        break
                    budget -= toks
                    lines.append(f"  {formatted}")
            except Exception as e:
                lines.append(f"  (session expand failed: {e})")

        return "\n".join(lines)

    def _execute_tool(self, name: str, args: dict) -> str:
        with Timer() as _timer:
            result = self._exec_tool_inner(name, args)
        elapsed = _timer.elapsed
        success = "failed" not in result.lower() and "error" not in result.lower()
        self._metrics.inc_tool_call(name, success=success)
        self._metrics.observe_request_duration("tool", name, elapsed)
        return result

    def _exec_tool_inner(self, name: str, args: dict) -> str:
        print(f"[Orthos] 🔧 {name}  {args}")
        self.ui.set_state("THINKING")

        if name == "save_memory":
            category = args.get("category", "notes")
            key      = args.get("key", "")
            value    = args.get("value", "")
            if key and value:
                update_memory({category: {key: {"value": value}}},
                              session_id=self._cur_session_id,
                              project_id=self._cur_project_id)
                if hasattr(self, '_chroma') and self._chroma:
                    self._chroma.add_memory(category, key, value[:2000])
                print(f"[Memory] 💾 {category}/{key} = {value}")
            self._set_listening_if_idle()
            return "__SILENT__"

        if name == "forget_memory":
            from memory.memory_manager import forget
            result = forget(args.get("key", ""), args.get("category", "notes"))
            print(f"[Memory] 🗑️ {result}")
            return result

        if name == "search_memory":
            from memory.chroma_memory import search_all_memories
            query = args.get("query", "")
            limit_val = args.get("limit", 10)
            mode = args.get("mode", "summary")
            if isinstance(limit_val, str):
                try:
                    limit_val = int(limit_val)
                except ValueError:
                    limit_val = 10
            # Multi-signal fusion: BM25 + FTS5 + ChromaDB vector + entity graph
            chroma_result = search_all_memories(query, limit_val, mode=mode,
                                                project_id=self._cur_project_id or "")
            return chroma_result

        if name == "list_memories":
            from memory.memory_manager import format_memory_summary, clear_expired_memories, enforce_memory_cap
            _ = clear_expired_memories()
            _ = enforce_memory_cap()
            result = format_memory_summary()
            return result

        if name == "save_procedure":
            from memory.memory_manager import remember
            key = args.get("key", "")
            value = args.get("value", "")
            if key and value:
                result = remember(key, value, category="procedures",
                                  session_id=self._cur_session_id,
                                  project_id=self._cur_project_id)
                print(f"[Memory] 📋 Procedure: {key} = {value[:80]}")
            else:
                result = "Error: both key and value required"
            return result

        if name == "list_procedures":
            from memory.memory_manager import load_memory
            memory = load_memory()
            procs = memory.get("procedures", {})
            if not procs:
                return "No procedures stored yet."
            lines = ["[PROCEDURES — learned workflows and rules]"]
            for key, entry in procs.items():
                val = entry.get("value") if isinstance(entry, dict) else entry
                if val:
                    lines.append(f"  {key}: {val}")
            return "\n".join(lines)

        if name == "core_memory_append":
            from memory.memory_manager import core_memory_append as _cma
            return _cma(args.get("key", ""), args.get("value", ""),
                        session_id=self._cur_session_id,
                        project_id=self._cur_project_id)

        if name == "core_memory_replace":
            from memory.memory_manager import core_memory_replace as _cmr
            return _cmr(args.get("key", ""), args.get("value", ""),
                        session_id=self._cur_session_id,
                        project_id=self._cur_project_id)

        if name == "archival_memory_search":
            from memory.memory_manager import archival_memory_search as _ams
            query = args.get("query", "")
            limit_val = args.get("limit", 10)
            if isinstance(limit_val, str):
                try:
                    limit_val = int(limit_val)
                except ValueError:
                    limit_val = 10
            return _ams(query, limit_val)

        if name == "context_status":
            from memory.memory_manager import context_status as _cs
            return _cs()

        # ── Raw conversation recall (exact past-chat search) ──
        if name == "recall_conversation":
            return self._recall_conversation(args)

        # ── Timeline tools ──
        if name == "search_timeline":
            query = args.get("query", "")
            limit_val = args.get("limit", 10)
            event_type = args.get("event_type", "")
            days_back = args.get("days_back", 0)
            if isinstance(limit_val, str):
                try: limit_val = int(limit_val)
                except ValueError: limit_val = 10
            if isinstance(days_back, str):
                try: days_back = int(days_back)
                except ValueError: days_back = 0
            results = search_timeline(query, limit=limit_val, event_type=event_type,
                                      project_id=self._cur_project_id or "", days_back=days_back)
            if not results:
                return f"No timeline events found for: {query}"
            lines = ["[TIMELINE SEARCH RESULTS]"]
            for ev in results:
                ts = ev.get('timestamp', '')[:10]
                desc = ev.get('description', '')
                etype = ev.get('event_type', 'note')
                imp = ev.get('importance', 0.5)
                marker = " ★" if imp >= 0.8 else ""
                lines.append(f"  [{ts}] ({etype}){marker} {desc}")
            return "\n".join(lines)

        if name == "save_timeline_event":
            event_type = args.get("event_type", "note")
            description = args.get("description", "")
            importance = args.get("importance", 0.5)
            if isinstance(importance, str):
                try: importance = float(importance)
                except ValueError: importance = 0.5
            if not description:
                return "Error: description is required"
            log_timeline_event(
                self._cur_session_id or "", event_type, description,
                auto=False, importance=importance,
                project_id=self._cur_project_id or "",
            )
            print(f"[Timeline] Event logged: {event_type} — {description[:80]}")
            return f"Timeline event saved: {event_type} — {description[:100]}"

        # ── Project tools ──
        if name == "list_projects":
            projects = list_projects()
            if not projects:
                return "No projects found. Use create_project to make one."
            lines = ["[PROJECTS]"]
            for p in projects:
                active_marker = " ← active" if p.get("id") == self._cur_project_id else ""
                lines.append(f"  {p['id']}: {p['name']}{active_marker}")
            return "\n".join(lines)

        if name == "create_project":
            pname = args.get("name", "")
            if not pname:
                return "Error: name is required"
            pid = create_project(pname)
            print(f"[Project] Created: {pid} — {pname}")
            return f"Project created: {pname} (id: {pid})"

        if name == "set_active_project":
            pid_or_name = args.get("project_id", "")
            if not pid_or_name:
                return "Error: project_id is required"
            # Resolve name → id if needed
            projects = list_projects()
            target_id = None
            for p in projects:
                if p["id"] == pid_or_name or p["name"].lower() == pid_or_name.lower():
                    target_id = p["id"]
                    break
            if not target_id:
                return f"Project not found: {pid_or_name}. Use list_projects to see available projects."
            self._cur_project_id = target_id
            if self._cur_session_id:
                set_session_project(self._cur_session_id, target_id)
            print(f"[Project] Switched to: {target_id}")
            return f"Switched to project: {target_id}"

        if name == "get_project_memories":
            pid_or_name = args.get("project_id", "")
            if not pid_or_name:
                return "Error: project_id is required"
            projects = list_projects()
            target_id = None
            target_name = ""
            for p in projects:
                if p["id"] == pid_or_name or p["name"].lower() == pid_or_name.lower():
                    target_id = p["id"]
                    target_name = p["name"]
                    break
            if not target_id:
                return f"Project not found: {pid_or_name}"
            # Get project-scoped memory items
            from memory.conversation_db import search_memory_items
            mems = search_memory_items(target_name, limit=10)
            lines = [f"[PROJECT MEMORIES — {target_name}]"]
            for m in mems:
                content = (m.get("content") or "")[:300]
                lines.append(f"  • {content}")
            # Get project timeline
            tl = format_timeline(limit=10, project_id=target_id)
            if tl:
                lines.append(tl)
            return "\n".join(lines)

        # ── Tool memory ──
        if name == "save_tool_state":
            tool_name = args.get("tool_name", "")
            key = args.get("key", "")
            value = args.get("value", "")
            if not all([tool_name, key, value]):
                return "Error: tool_name, key, and value are all required"
            set_tool_memory(tool_name, self._cur_session_id or "", key, value)
            print(f"[ToolMemory] {tool_name}/{key} = {value[:80]}")
            return f"Tool state saved: {tool_name}/{key}"

        if name == "get_tool_state":
            tool_name = args.get("tool_name", "")
            if not tool_name:
                return "Error: tool_name is required"
            from memory.conversation_db import get_tool_memory
            states = get_tool_memory(tool_name, self._cur_session_id or "")
            if not states:
                return f"No saved state for tool: {tool_name}"
            lines = [f"[TOOL STATE — {tool_name}]"]
            for k, v in states.items():
                lines.append(f"  {k}: {v}")
            return "\n".join(lines)

        # ── GraphRAG ──
        if name == "search_knowledge_graph":
            query = args.get("query", "")
            max_hops = args.get("max_hops", 2)
            if isinstance(max_hops, str):
                try: max_hops = int(max_hops)
                except ValueError: max_hops = 2
            if not query:
                return "Error: query is required"
            from memory.entity_store import get_entity_graph, graph_retrieve
            # First try graph_retrieve (multi-hop with full context)
            graph_results = graph_retrieve(query, max_hops=max_hops)
            if graph_results:
                lines = ["[KNOWLEDGE GRAPH — entity relationships]"]
                seen = set()
                for r in graph_results:
                    sig = (r.get("entity", ""), r.get("relation", ""), r.get("connected_entity", ""))
                    if sig not in seen:
                        seen.add(sig)
                        lines.append(f"  [{r.get('entity_type', '?')}] {r.get('entity', '')} "
                                     f"--[{r.get('relation', '?')}]--> "
                                     f"[{r.get('connected_type', '?')}] {r.get('connected_entity', '')}")
                        obs = r.get("observations", "")
                        if obs:
                            lines.append(f"    {obs[:200]}")
                return "\n".join(lines)
            # Fallback: format entity graph
            kg = get_entity_graph(query)
            return kg or f"No entities found for: {query}"

        result = "Done."
        try:
            if name == "open_app":
                r = open_app(parameters=args, response=None, player=self.ui)
                result = r or f"Opened {args.get('app_name')}."

            elif name == "weather_report":
                r = weather_action(parameters=args, player=self.ui)
                result = r or "Weather delivered."

            elif name == "file_controller":
                r = file_controller(parameters=args, player=self.ui)
                result = r or "Done."

            elif name == "send_message":
                r = send_message(parameters=args, response=None, player=self.ui, session_memory=None)
                result = r or f"Message sent to {args.get('receiver')}."

            elif name == "reminder":
                r = reminder(parameters=args, response=None, player=self.ui)
                result = r or "Reminder set."

            elif name == "youtube_video":
                r = youtube_video(parameters=args, response=None, player=self.ui)
                result = r or "Done."

            elif name == "screen_process":
                r = screen_process(parameters=args, response=None, player=self.ui, session_memory=self._conversation[-10:])
                result = r if isinstance(r, str) and r else "Screen analyzed."

            elif name == "screen_locate":
                r = screen_locate(parameters=args, player=self.ui, session_memory=self._conversation[-10:])
                result = r if isinstance(r, str) and r else "Element not found."

            elif name == "computer_settings":
                r = computer_settings(parameters=args, response=None, player=self.ui)
                result = r or "Done."

            elif name == "desktop_control":
                r = desktop_control(parameters=args, player=self.ui)
                result = r or "Done."

            elif name == "code_helper":
                r = code_helper(parameters=args, player=self.ui, speak=self.speak)
                result = r or "Done."

            elif name == "dev_agent":
                r = dev_agent(parameters=args, player=self.ui, speak=self.speak)
                result = r or "Done."

            elif name == "agent_task":
                from agent.task_queue import get_queue, TaskPriority
                priority_map = {
                    "low": TaskPriority.LOW,
                    "normal": TaskPriority.NORMAL,
                    "high": TaskPriority.HIGH,
                }
                priority = priority_map.get(
                    args.get("priority", "normal").lower(), TaskPriority.NORMAL
                )
                task_id = get_queue().submit(
                    goal=args.get("goal", ""), priority=priority, speak=self.speak
                )
                result = f"Task started (ID: {task_id})."

            elif name == "web_search":
                self.ui.set_state("PROCESSING")
                r = web_search_action(parameters=args, player=self.ui)
                result = r or "Done."

            elif name == "webfetch":
                self.ui.set_state("PROCESSING")
                r = webfetch_action(parameters=args, player=self.ui)
                result = r or "Could not fetch URL."

            elif name == "file_processor":
                if not args.get("file_path") and self.ui.current_file:
                    args["file_path"] = self.ui.current_file
                r = file_processor(parameters=args, player=self.ui, speak=self.speak)
                result = r or "Done."

            elif name == "computer_control":
                r = computer_control(parameters=args, player=self.ui)
                result = r or "Done."

            elif name == "run_terminal":
                r = run_terminal(parameters=args, player=self.ui)
                result = r or "Done."

            elif name == "game_updater":
                r = game_updater(parameters=args, player=self.ui, speak=self.speak)
                result = r or "Done."

            elif name == "flight_finder":
                r = flight_finder(parameters=args, player=self.ui)
                result = r or "Done."

            elif name == "list_mcp_servers":
                if not self._mcp_manager:
                    return "MCP manager not initialized."
                config = self._mcp_manager.get_all_servers_config()
                lines = ["## MCP Servers\n"]
                for srv_name, cfg in config.items():
                    en = cfg.get("enabled", False)
                    status = self._mcp_manager.get_server_status(srv_name)
                    cat = cfg.get("category", "Uncategorized")
                    desc = cfg.get("description", "")
                    tools = [t for t in _tool_registry.list_tools()
                             if t["name"].startswith(f"mcp_{srv_name}_")]
                    icon = "🟢" if en and status == "running" else "🔴" if en else "⚪"
                    lines.append(
                        f"{icon} **{srv_name}** ({cat}) — {status} | {len(tools)} tools\n"
                        f"   {desc}\n"
                        f"   Load all tools: `get_tool_definition(tool_name=\"{srv_name}\")`\n"
                    )
                lines.append(
                    "\nTo use a tool: call `search_tools(query)` to find it, "
                    "then `get_tool_definition(tool_name='TOOL_NAME')` to load its full schema.\n"
                    "To load ALL tools from a server at once: "
                    "`get_tool_definition(tool_name='SERVER_NAME')`."
                )
                return "\n".join(lines)

            elif name == "manage_mcp_server":
                if not self._mcp_manager:
                    return "MCP manager not initialized."
                action = str(args.get("action", "")).strip().lower()
                server = str(args.get("server", "")).strip()
                config = self._mcp_manager.get_all_servers_config()
                if action == "install":
                    package = str(args.get("package", "")).strip()
                    if not package:
                        return "Tell me the MCP package to install, for example @modelcontextprotocol/server-github."
                    generated_name = server or package.split("/")[-1].replace("server-", "").replace("-mcp", "")
                    command = str(args.get("command") or "npx").strip().lower()
                    if command not in ("npx", "uvx"):
                        return "For safety, MCP installation supports npx or uvx only."
                    install_config = {
                        "enabled": True,
                        "category": str(args.get("category") or "Development"),
                        "command": command,
                        "args": (["-y", package] if command == "npx" else [package]),
                        "description": f"Installed by Orthos: {package}",
                    }
                    return self._mcp_manager.install_server(generated_name, install_config)
                if not server or server not in config:
                    available = ", ".join(sorted(config)) or "none"
                    return f"I couldn't find '{server}'. Available MCP servers: {available}."
                if action == "start":
                    return self._mcp_manager.start_server(server)
                if action == "stop":
                    return self._mcp_manager.stop_server(server)
                if action == "restart":
                    return self._mcp_manager.restart_server(server)
                if action == "enable":
                    if config[server].get("enabled", False):
                        return self._mcp_manager.start_server(server)
                    return self._mcp_manager.toggle_server(server)
                if action == "disable":
                    if not config[server].get("enabled", False):
                        return self._mcp_manager.stop_server(server)
                    return self._mcp_manager.toggle_server(server)
                return "Supported MCP actions are: start, stop, restart, enable, disable, and install."

            elif name == "shutdown_orthos":
                self.ui.write_log("SYS: Shutdown requested.")

                def _shutdown():
                    self.speak("Goodbye.")
                    time.sleep(2.5)
                    _os._exit(0)

                threading.Thread(target=_shutdown, daemon=True).start()
                return "Shutting down."

            elif name.startswith("mcp_"):
                if self._mcp_manager is not None:
                    try:
                        if self._mcp_manager.is_write_tool(name) and self._mcp_manager.confirm_write():
                            self.ui.write_log(f"SYS: ⚠ MCP write action: {name}")
                    except Exception:
                        pass
                r = _tool_registry.execute(name, args)
                result = r or "Done."

            elif name in ("search_tools", "get_tool_definition"):
                fn = _tool_registry.get(name)
                if fn:
                    r = fn(parameters=args, response=None, player=self.ui)
                    result = r or "Done."
                else:
                    result = f"Meta-tool '{name}' not found in registry"

            else:
                # Try registry as fallback (handles dynamically registered tools)
                fn = _tool_registry.get(name)
                if fn:
                    r = fn(parameters=args, response=None, player=self.ui)
                    result = r or "Done."
                else:
                    result = f"Unknown tool: {name}"

        except Exception as e:
            result = f"Tool '{name}' failed: {e}"
            traceback.print_exc()
            self.speak_error(name, e)

        if not self.ui.muted:
            self._set_listening_if_idle()

        print(f"[Orthos] 📤 {name} → {str(result)[:80]}")
        return result

    # ------------------------------------------------------------------
    # LLM processing loop
    # ------------------------------------------------------------------

    def _process_message(self, user_text: str, source: str = "ui", extra_meta: dict | None = None) -> None:
        try:
            self._process_message_inner(user_text, source, extra_meta)
        except Exception as e:
            print(f"[Orthos] FATAL ERROR in _process_message: {e}")
            import traceback
            traceback.print_exc()

    # ── Auto-compact (2026-09-25 spec) ─────────────────────────────────
    # Budget gate: when the ESTIMATED total input (system + tools + raw
    # conversation) reaches input_budget (50k / kaggle 25k), summarize the
    # OLDEST slice only — min(15k, 45% of raw conversation) tokens — into
    # <= slice/3 tokens via ONE LLM call. Everything else (system prompt,
    # tools, recent turns, current message) stays raw.
    #
    #     50k − 15k + 5k = 40k  → next turn below trigger → NO LLM call
    #
    # Deliberately NOT implemented: force-trim (summarize-everything) and
    # word-drop trimming. LLM failure → this turn simply sends raw, and
    # after 3 consecutive failures we cool down instead of retrying.
    _COMPACT_FAIL_LIMIT = 3
    _COMPACT_COOLDOWN_S = 300

    def _auto_compact_if_needed(self, tool_cost_est: int) -> bool:
        """Summarize the oldest slice when total input would hit the budget.

        Runs BEFORE the prompt build so the refreshed rolling summary is
        part of this turn's system prompt. Returns True if compacted.
        """
        if not self._cur_session_id:
            return False
        # Circuit breaker (Claude Code pattern): repeated LLM failures →
        # stop attempting for a cooldown instead of failing every message.
        fails = getattr(self, "_compact_fail_count", 0)
        if fails >= self._COMPACT_FAIL_LIMIT:
            if time.time() < getattr(self, "_compact_blocked_until", 0.0):
                return False
            fails = 0
            self._compact_fail_count = 0
        conv = list(self._conversation)
        if len(conv) <= 2:
            return False
        try:
            from memory.conversation_db import (
                estimate_messages_tokens, summarize_turns, _estimate_tokens,
                _summarize_with_llm,
            )
            raw_tokens = estimate_messages_tokens(conv)
            sys_est = getattr(self, "_last_system_prompt_tokens", 7443)
            total_est = int(sys_est) + raw_tokens + int(tool_cost_est)
            input_budget = getattr(self, "_input_budget", 50000)
            if total_est < input_budget:
                return False
            # Slice rule (conversation_trim.compact_slice_tokens): oldest 45%
            # of the raw conversation, cap 15k (TRIM_CHUNK), summary = /3.
            slice_tokens = _compact_slice_tokens(raw_tokens)
            if raw_tokens < slice_tokens * 2:
                # System+tools dominate (or chat too small): an LLM call
                # cannot free enough to matter — send raw, don't waste a call.
                print(f"[Compact] budget hit (est {total_est:,} >= {input_budget:,}) "
                      f"but raw conv only {raw_tokens:,} tok — no useful cut, sending raw")
                return False
            # Current message + the previous one always stay raw.
            candidates = conv[:-2]
            if not candidates:
                return False
            trim_idx = _oldest_trim_split(candidates, slice_tokens)
            if not trim_idx:
                return False
            trim_set = set(trim_idx)
            oldest = [candidates[i] for i in trim_idx]
            rest = [m for i, m in enumerate(candidates) if i not in trim_set]
            # A4: persist ALL unsaved turns first — the DB keeps the full
            # history; compact only rewrites the in-memory prompt copy.
            unsaved = self._conversation[self._conv_saved_idx:]
            if unsaved:
                save_conversation(unsaved, self._cur_session_id)
                try:
                    self.ui._win._session_refresh_sig.emit()
                except Exception:
                    pass
            # Use the background pre-compact (D) if it matches this exact
            # cut — otherwise summarize synchronously right here.
            summary = None
            pc = getattr(self, "_precompact", None)
            if pc:
                self._precompact = None
                try:
                    if (time.time() - float(pc.get("built_at", 0.0)) < 120.0
                            and pc.get("conv_len") == len(conv) - 1
                            and pc.get("trim_idx") == trim_idx
                            and pc.get("refs")
                            and len(pc["refs"]) == len(conv) - 1
                            and all(a is b for a, b in zip(conv, pc["refs"]))):
                        summary = pc.get("summary")
                        if summary:
                            print("[Compact] using pre-fetched summary (no LLM wait)")
                except Exception:
                    summary = None
            if not summary:
                summary = summarize_turns(oldest, target_tokens=max(500, slice_tokens // 3))
            if not summary:
                self._compact_fail_count = fails + 1
                if self._compact_fail_count >= self._COMPACT_FAIL_LIMIT:
                    self._compact_blocked_until = time.time() + self._COMPACT_COOLDOWN_S
                    print(f"[Compact] LLM summarize failed "
                          f"{self._compact_fail_count}x — cooling down "
                          f"{self._COMPACT_COOLDOWN_S}s, sending raw until then")
                else:
                    print("[Compact] LLM summarize failed — sending raw this turn")
                return False
            # Merge into the rolling summary (lives in the system prompt).
            previous = self._rolling_summary
            if previous:
                self._rolling_summary = (f"[Earlier context]\n{previous}\n\n"
                                         f"[Recent summarized context]\n{summary}")
            else:
                self._rolling_summary = summary
            # Rolling-summary cap: LLM-condense when over budget. NO
            # word-drop — an oversized summary beats data loss.
            if _estimate_tokens(self._rolling_summary) > self._summary_rolling_budget:
                condensed = _summarize_with_llm(
                    self._rolling_summary, timeout_s=30,
                    target_note=f"Hard limit: under {self._summary_rolling_budget} tokens.",
                )
                if condensed:
                    self._rolling_summary = condensed
                else:
                    print("[Compact] WARN: rolling summary over budget, "
                          "condense failed — keeping as-is")
            set_session_memory(self._cur_session_id, "_rolling_summary", self._rolling_summary)
            # NOTE: no local 'import time' here — module-level import at
            # main.py top is authoritative (a local import would shadow it
            # and crash the tool-timing path with UnboundLocalError).
            set_session_memory(self._cur_session_id, f"_summary_{int(time.time())}", summary)
            # Rebuild: oldest replaced by summary, recent + current stay raw.
            self._conversation = rest + conv[-2:]
            self._conv_saved_idx = len(self._conversation)
            self._compact_fail_count = 0
            print(f"[Compact] budget hit (est {total_est:,} >= {input_budget:,}): "
                  f"summarized oldest {len(oldest)} msgs (~{slice_tokens} tok) -> "
                  f"{_estimate_tokens(summary):,} tok; kept {len(rest) + 2} recent msgs raw")
            return True
        except Exception as e:
            print(f"[Compact] auto-compact failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _process_message_inner(self, user_text: str, source: str, extra_meta: dict | None) -> None:
        self.stop_speaking()
        self._should_stop.clear()
        with self._cancel_lock:
            self._cancel_seq += 1
            _my_seq = self._cancel_seq
            self._cancel_event.clear()
        self.ui.set_state("THINKING")
        self.ui.write_log(f"You: {user_text}")

        # ── Per-stage latency instrumentation (temp, 2026-09) ──
        import time as _pt
        _latency = {"last": _pt.perf_counter(), "t0": _pt.perf_counter()}

        def _pt_mark(stage: str) -> None:
            now = _pt.perf_counter()
            print(f"[Timing] {stage}: {(now - _latency['last']) * 1000:.0f} ms  (total {(now - _latency['t0']) * 1000:.0f} ms)")
            _latency["last"] = now

        self._conversation.append({"role": "user", "content": user_text})

        # Extract entities from user message for knowledge graph.
        # Background: link_fact makes an HTTP NER call + graph JSON I/O —
        # inline it delayed every message send on this CPU.
        try:
            from memory.entity_store import link_fact as _link_fact
            threading.Thread(
                target=_link_fact,
                args=("user_message", "input", user_text[:500]),
                daemon=True,
                name="kg-link-fact",
            ).start()
        except Exception:
            pass

        for msg in self._conversation:
            msg.pop("images", None)

        # ── Budget: 75% input, 25% output. 2-way split (backup-compatible) ──
        self._conv_budget = 5000
        self._summary_rolling_budget = 15000
        self._summary_past_budget = 15000
        _tool_cost_est = 2500
        input_budget = 8192  # default (overridden inside try block below)
        try:
            cfg = _get_provider_config()
            from memory.conversation_db import get_model_context_window, _estimate_tokens
            ctx_window = get_model_context_window(cfg["model"], cfg["provider"])
            # Pinned input budget (Sept 2026): kaggle 25k, everything else
            # 50k. Deterministic — no more ctx*0.50 drifting with model pins.
            input_budget = 25_000 if cfg.get("provider", "").lower().strip() == "kaggle" else 50_000
            mcp_reserve = int(ctx_window * 0.25)
            _current_tools = _build_dynamic_tools()
            if _current_tools:
                _internal_tools = [t for t in _current_tools
                                   if not (t.get("function", {}).get("name", "")).startswith("mcp_")]
                _mcp_tools = [t for t in _current_tools
                              if (t.get("function", {}).get("name", "")).startswith("mcp_")]
                _tool_cost_est = _estimate_tokens(json.dumps(_internal_tools, ensure_ascii=False))
                _mcp_cost_est = _estimate_tokens(json.dumps(_mcp_tools, ensure_ascii=False))
            else:
                _tool_cost_est = 2500
                _mcp_cost_est = 0
            _base_core = _estimate_tokens(_load_system_prompt())
            _now = datetime.now()
            _base_core += _estimate_tokens(
                f"[CURRENT DATE & TIME]\nRight now it is: {_now.strftime('%A, %B %d, %Y — %I:%M %p')}\n"
                f"Use this to calculate exact times for reminders."
                f"\n\n[CONTINUOUS TOOL EXECUTION MODE]\nWhen doing multi-step tasks, keep calling tools "
                f"in a continuous loop using tool results to decide the next step.\n"
                f"RULE: Do NOT stop mid-task to ask the user 'what to do next'. "
                f"Each tool result should tell you what to do next. "
                f"Keep the loop going until the task is fully complete or user says stop."
            ) + 5000
            _remaining = input_budget - _tool_cost_est - _base_core
            if _remaining > 6000:
                self._summary_rolling_budget = int(_remaining * 0.20)
                self._summary_past_budget = int(_remaining * 0.20)
                self._conv_budget = int(_remaining * 0.55)
            if _mcp_cost_est > mcp_reserve:
                print(f"[Budget] WARN: MCP tools ({_mcp_cost_est:,} tok) exceed reserve ({mcp_reserve:,} tok)")
        except Exception:
            pass
        self._input_budget = input_budget
        self._tool_cost_est = _tool_cost_est

        # ── Auto-compact BEFORE the prompt build: the refreshed rolling
        #    summary must land in THIS turn's system prompt (old code built
        #    first and compacted after → summary missing for one turn). ──
        try:
            self._auto_compact_if_needed(_tool_cost_est)
        except Exception as _ce:
            print(f"[Compact] pre-build compact error: {_ce}")

        messages = [
            {"role": "system", "content": self._build_system_prompt()}
        ] + list(self._conversation)
        try:
            from memory.conversation_db import _estimate_tokens as _tok
            self._last_system_prompt_tokens = _tok(messages[0]["content"] or "")
        except Exception:
            pass
        _pt_mark("build system prompt + context")

        # ── Inbound media (e.g. Telegram images) → attach to latest user turn ──
        _attachment_content_ready = False
        if extra_meta:
            try:
                _img_files = extra_meta.get("image_files") or []
                if _img_files:
                    _inb64s = []
                    import PIL.Image as _PILImg
                    for _p in _img_files:
                        try:
                            if Path(_p).exists():
                                # Compress: resize to 720px max + JPEG 60% quality
                                _img = _PILImg.open(Path(_p))
                                if _img.mode in ("RGBA", "P"):
                                    _img = _img.convert("RGB")
                                _MAX_DIM = 720
                                _img.thumbnail((_MAX_DIM, _MAX_DIM), _PILImg.LANCZOS)
                                _buf = io.BytesIO()
                                _img.save(_buf, format="JPEG", quality=60, optimize=True)
                                _raw_bytes = _buf.getvalue()
                                _b64 = base64.b64encode(_raw_bytes).decode("ascii")
                                _inb64s.append(_b64)
                                _orig_kb = Path(_p).stat().st_size // 1024
                                _comp_kb = len(_raw_bytes) // 1024
                                print(f"[Orthos] Image compressed: {_p} ({_orig_kb}KB -> {_comp_kb}KB, {len(_b64)} chars)")
                            else:
                                print(f"[Orthos] WARNING: Image file not found: {_p}")
                        except Exception as e:
                            print(f"[Orthos] ERROR encoding image {_p}: {e}")
                    if _inb64s:
                        for _i in range(len(messages) - 1, -1, -1):
                            if messages[_i].get("role") == "user":
                                messages[_i] = {**messages[_i], "images": _inb64s, "image_files": list(_img_files)}
                                print(f"[Orthos] Images attached to user message")
                                break
                    else:
                        print(f"[Orthos] WARNING: No images could be encoded from {_img_files}")
                # Scanned PDFs have no text layer. The UI renders their pages and
                # supplies them to the vision model directly - never via a viewer or
                # repeated screenshots.
                _document_images = extra_meta.get("document_images") or []
                if _document_images:
                    for _i in range(len(messages) - 1, -1, -1):
                        if messages[_i].get("role") == "user":
                            _existing_images = list(messages[_i].get("images") or [])
                            messages[_i] = {
                                **messages[_i],
                                "images": _existing_images + list(_document_images),
                                "image_files": list(messages[_i].get("image_files") or ["rendered-pdf-pages"]),
                            }
                            _attachment_content_ready = True
                            print(f"[Orthos] Attached {len(_document_images)} rendered PDF page(s) to vision context")
                            break

                _attached = extra_meta.get("files") or []
                _non_img = [p for p in _attached if p not in _img_files]
                if _non_img:
                    _note = (chr(10) + chr(10) + "[User attached file(s): " + "; ".join(_non_img)
                             + " — use the file_processor or file_controller tools "
                               "to read/analyze them if needed.]")
                    for _i in range(len(messages) - 1, -1, -1):
                        if messages[_i].get("role") == "user":
                            messages[_i]["content"] = messages[_i].get("content", "") + _note
                            print(f"[Orthos] Non-image files attached to user message")
                            break

                # The UI pre-reads a bounded local preview for documents/data files.
                # Put that context on the user turn so analysis does not depend solely
                # on the model deciding to call a file tool first.
                _contexts = extra_meta.get("attachment_context") or []
                _context_parts = []
                _context_budget = 24000
                for _ctx in _contexts:
                    if not isinstance(_ctx, dict):
                        continue
                    _name = str(_ctx.get("name") or "attachment")
                    _kind = str(_ctx.get("type") or "file")
                    _detail = str(_ctx.get("detail") or "Ready to analyze")
                    _preview = str(_ctx.get("preview") or "").strip()
                    if _preview and _context_budget > 0:
                        _preview = _preview[:_context_budget]
                        _context_budget -= len(_preview)
                        _context_parts.append(
                            f"[Pre-read {_kind} attachment: {_name} — {_detail}]\n{_preview}"
                        )
                    else:
                        _context_parts.append(f"[Attachment metadata: {_kind} {_name} — {_detail}]")
                if _context_parts:
                    _attachment_context_note = "\n\n" + "\n\n".join(_context_parts)
                    for _i in range(len(messages) - 1, -1, -1):
                        if messages[_i].get("role") == "user":
                            messages[_i]["content"] = messages[_i].get("content", "") + _attachment_context_note
                            print(f"[Orthos] Added pre-read context for {len(_context_parts)} attachment(s)")
                            _attachment_content_ready = _attachment_content_ready or any(
                                bool(str(_ctx.get("preview") or "").strip())
                                for _ctx in _contexts if isinstance(_ctx, dict)
                            )
                            break
                if _attachment_content_ready:
                    messages[0]["content"] += (
                        "\n\n[ATTACHMENT HANDLING]\n"
                        "The attached document has already been extracted or rendered into this conversation. "
                        "Answer from that context. Do not open the file, capture the screen, or repeatedly invoke "
                        "file tools for this attachment."
                    )
            except Exception as e:
                print(f"[Orthos] ERROR processing extra_meta: {e}")
                import traceback
                traceback.print_exc()

        # ── Post-build budget check (informational only) ────────────────
        # Auto-compact already ran BEFORE the prompt build. If we are still
        # over budget it means the LLM summarize failed / was cooled down —
        # per spec we send raw (no force-trim, no word-drop). Provider hard
        # limits (100k / kaggle 60k) still have plenty of headroom.
        try:
            _total_input = _estimate_tokens(json.dumps(messages, ensure_ascii=False))
            _total_input += _tool_cost_est
            if _total_input > input_budget:
                print(f"[Budget] WARN: total_input ({_total_input:,}) > input_budget "
                      f"({input_budget:,}) — sending raw this turn (compact unavailable)")
        except Exception:
            pass

        _pt_mark("token estimate + trim/budget")
        try:
            _sys_len = sum(len(str(m.get("content") or "")) for m in messages if m.get("role") == "system")
            print(f"[Timing] prompt size: system {_sys_len:,} chars, messages {len(messages)}, "
                  f"total ~{sum(len(str(m.get('content') or '')) for m in messages):,} chars")
        except Exception:
            pass

        _NEEDS_LLM_ROUND = {"web_search", "webfetch", "screen_process", "screen_locate", "agent_task"}

        # Gemini Live mode speaks natively — never echo text through TTS
        _current_live_native = bool(getattr(self, "_live_provider", None))

        _round = 0
        _consecutive_no_tool = 0
        while True:
            _round += 1
            if self._is_cancelled(_my_seq):
                self._stop_and_listen()
                return

            # Skip auto screen capture if the user attached an image (e.g. via Telegram)
            _user_has_image = False
            for _m in reversed(messages):
                if _m.get("role") == "user":
                    _user_has_image = bool(_m.get("image_files"))
                    break

            if not _user_has_image and getattr(self.ui, "screen_vision", True):
                try:
                    _fresh, _ = _capture_screen_annotated(0, 0)
                    import PIL.Image as _PIL_Image
                    _img = _PIL_Image.open(io.BytesIO(_fresh))
                    # Resize to 720p (max width 1280) to reduce payload size
                    _MAX_W = 1280
                    if _img.width > _MAX_W:
                        _ratio = _MAX_W / _img.width
                        _new_h = int(_img.height * _ratio)
                        _img = _img.resize((_MAX_W, _new_h), _PIL_Image.LANCZOS)
                    _buf = io.BytesIO()
                    _img.save(_buf, format="JPEG", quality=_get_screenshot_quality("llm"))
                    _b64 = base64.b64encode(_buf.getvalue()).decode("ascii")
                    _cap_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    _cap_path = str(_CAPTURES_DIR / f"{_cap_ts}.jpg")
                    _img.save(_cap_path, format="JPEG", quality=_get_screenshot_quality("llm"))
                    for i in range(len(messages) - 1, -1, -1):
                        if messages[i].get("role") == "user":
                            messages[i] = {**messages[i], "images": [_b64], "image_files": [_cap_path]}
                            break
                except Exception as _e:
                    print(f"[Vision] Auto-fresh capture skipped: {_e}")

            final_content    = ""
            final_tool_calls: list = []
            _streamed: list[str] = []
            _spoke_sentences = False
            # Hashim Live TTS: buffer all sentences and speak the FULL output
            # at once for a more natural voice. Other TTS engines keep the
            # old sentence-by-sentence streaming flow.
            _is_hashim_tts = self._config.get("tts_engine", "").lower() == "hashim_live"
            _hashim_buffer: list[str] = []

            _pt_mark("screen capture (vision)")
            try:
                _current_tools = _build_dynamic_tools()
                _pt_mark("build tools payload")
                if _attachment_content_ready:
                    _blocked_attachment_tools = {
                        "file_processor", "file_controller", "open_app",
                        "screen_process", "screen_locate",
                    }
                    _current_tools = [tool for tool in _current_tools if tool.get("function", {}).get("name") not in _blocked_attachment_tools]
                # ── Live streaming render (F1): tokens appear in the chat as
                # they arrive instead of one final card after the full reply.
                _stream_ui = getattr(self.ui, "_win", None)
                _stream_log = getattr(_stream_ui, "_log", None) if _stream_ui else None
                _stream_active = False
                # Reasoning/thinking card state (ChatGPT-style collapsible).
                # thinking events arrive before the first sentence; the card
                # finalizes (auto-collapses) when the reply starts streaming.
                _think_ui = _stream_log
                _think_active = False
                # Stays True after streaming_end: the final card is already
                # in the log, so the deliver path must not log it again.
                _stream_rendered = False
                for event in call_llm_stream(messages, _current_tools, cancel_event=self._cancel_event):
                    if event["type"] == "thinking":
                        # Show live reasoning in the chat (never spoken).
                        if _think_ui is not None and not event.get("native_audio"):
                            if not _think_active:
                                _think_ui.thinking_start_threadsafe()
                                _think_active = True
                            _think_ui.thinking_append_threadsafe(event["text"])
                        continue
                    if event["type"] == "sentence":
                        # First visible token: collapse the thinking card.
                        if _think_active and _think_ui is not None:
                            _think_ui.thinking_end_threadsafe(None)
                            _think_active = False
                        _streamed.append(event["text"])
                        # Live card update (skip for Gemini Live: its sentences
                        # are already written below and the card would dupe).
                        # Threadsafe: the LLM loop runs off the GUI thread.
                        if _stream_log is not None and not event.get("native_audio"):
                            if not _stream_active:
                                _stream_log.streaming_start_threadsafe()
                                _stream_active = True
                            _stream_log.streaming_append_threadsafe(event["text"])
                        # Gemini Live plays native audio — never re-speak via TTS/session
                        if not event.get("native_audio"):
                            if _is_hashim_tts:
                                # Buffer for full-output TTS
                                _hashim_buffer.append(event["text"])
                            else:
                                self.speak(event["text"])
                        else:
                            # Show streaming text for Gemini Live even though audio plays natively
                            self.ui.write_log(f"Orthos: {event['text']}")
                        _spoke_sentences = True
                    elif event["type"] == "done":
                        final_content    = event["content"]
                        final_tool_calls = event["tool_calls"]
                        # Finalize the live card with the authoritative text;
                        # the assembled card replaces the streaming draft.
                        if _stream_log is not None and _stream_active:
                            _stream_log.streaming_end_threadsafe(
                                final_content or "".join(_streamed))
                            _stream_active = False
                            _stream_rendered = True
                        # Hashim Live TTS: send the FULL response at once
                        if _is_hashim_tts and _hashim_buffer:
                            full_text = " ".join(_hashim_buffer).strip()
                            if full_text:
                                self.speak(full_text)
                            _hashim_buffer = []
                    elif event["type"] == "cancelled":
                        if _stream_log is not None and _stream_active:
                            _stream_log.streaming_abort_threadsafe()
                        if _think_active and _think_ui is not None:
                            _think_ui.thinking_abort_threadsafe()
                        self._stop_and_listen()
                        return
            except RuntimeError as e:
                if _stream_log is not None and _stream_active:
                    _stream_log.streaming_abort_threadsafe()
                    _stream_active = False
                if _think_active and _think_ui is not None:
                    _think_ui.thinking_abort_threadsafe()
                    _think_active = False
                if _QuotaExceededError is not None and isinstance(e, _QuotaExceededError):
                    print(f"ERR: {e}")
                    self.ui.write_log(f"ERR: {e}")
                    self.speak(str(e))
                elif _current_live_native:
                    # Gemini Live (Mark-L style): session errors are
                    # console-only — the provider reconnects on its own and
                    # the next turn works normally.
                    print(f"[GeminiLive] LLM round aborted: {e}")
                else:
                    self.speak_error("LLM", e)
                return

            if not final_tool_calls:
                if final_content:
                    assistant_msg = {"role": "assistant", "content": final_content}
                    messages.append(assistant_msg)
                    self._conversation.append(assistant_msg)
                    # Already rendered live (F1) or Gemini Live streamed it —
                    # logging again would duplicate the reply card.
                    _already_shown = _stream_rendered or (
                        _current_live_native and _spoke_sentences)
                    if not _already_shown:
                        self._deliver_response(final_content, source)
                    else:
                        self._deliver_response(final_content, source, quiet=True)
                    self._extract_and_log(final_content)
                    if not _spoke_sentences and not _current_live_native:
                        self.speak(final_content)
                break

            assistant_msg = {
                "role":       "assistant",
                "content":    final_content or "",
                "tool_calls": final_tool_calls,
            }
            messages.append(assistant_msg)
            self._conversation.append(assistant_msg)

            _only_memory = all(
                tc.get("function", {}).get("name") == "save_memory"
                for tc in final_tool_calls
            )
            if _only_memory and final_content:
                for tc in final_tool_calls:
                    fn    = tc.get("function", {})
                    targs = fn.get("arguments", {})
                    if isinstance(targs, str):
                        try:
                            targs = json.loads(targs)
                        except Exception:
                            targs = {}
                    self._execute_tool("save_memory", targs)
                assistant_msg2 = {"role": "assistant", "content": final_content}
                messages.append(assistant_msg2)
                self._conversation.append(assistant_msg2)
                if _stream_rendered or (_current_live_native and _spoke_sentences):
                    self._deliver_response(final_content, source, quiet=True)
                else:
                    self._deliver_response(final_content, source)
                self._extract_and_log(final_content)
                if not _spoke_sentences and not _current_live_native:
                    self.speak(final_content)
                break

            all_silent    = True
            _tool_results: list[tuple[str, str]] = []
            _has_image = False

            for tc in final_tool_calls:
                if self._is_cancelled(_my_seq):
                    self._stop_and_listen()
                    return

                fn    = tc.get("function", {})
                tname = fn.get("name", "")
                targs = fn.get("arguments", {})
                if isinstance(targs, str):
                    try:
                        targs = json.loads(targs)
                    except Exception:
                        targs = {}

                tc_id = tc.get("id", "")
                # Live tool progress: name + a compact one-line args preview
                # so the user sees WHAT runs, not just that something did.
                _args_preview = ""
                try:
                    _pv = ", ".join(f"{k}={str(v)[:40]!r}" for k, v in
                                    list(targs.items())[:2])
                    if _pv:
                        _args_preview = f"({_pv})"
                except Exception:
                    _args_preview = ""
                self.ui.write_log(f"SYS: ▶ {tname}{_args_preview}")
                _tool_t0 = time.time()
                if _attachment_content_ready and tname in {
                    "file_processor", "file_controller", "open_app",
                    "screen_process", "screen_locate",
                }:
                    result = (
                        "The attachment has already been extracted or rendered into the conversation. "
                        "Do not open it or capture the screen; answer the user's question using that context now."
                    )
                    self.ui.write_log("SYS: Attachment already available - skipped redundant tool")
                else:
                    result = self._execute_tool(tname, targs)
                # Completion status with duration: the missing "real-time"
                # feedback for tool calls.
                _dt = time.time() - _tool_t0
                _ok = "failed" not in (result or "").lower() and "error" not in (result or "").lower()
                _mark = "✓" if _ok else "✗"
                self.ui.write_log(f"SYS: {_mark} {tname} ({_dt:.1f}s)")

                # ── Auto-capture tool state for continuity ──
                try:
                    if self._cur_session_id:
                        if tname == "computer_control":
                            action = targs.get("action", "")
                            if action in ("type", "smart_type", "click", "hotkey", "press"):
                                set_tool_memory("browser", self._cur_session_id, "last_action", action)
                        elif tname == "run_terminal":
                            cmd = targs.get("command", "")
                            cwd = targs.get("cwd", "")
                            set_tool_memory("terminal", self._cur_session_id, "last_command", cmd[:200])
                            if cwd:
                                set_tool_memory("terminal", self._cur_session_id, "last_cwd", cwd)
                            if result and "error" not in result.lower():
                                set_tool_memory("terminal", self._cur_session_id, "last_success", "true")
                        elif tname == "screen_process":
                            text = targs.get("text", "")
                            set_tool_memory("vision", self._cur_session_id, "last_question", text[:200])
                        elif tname == "file_controller":
                            fpath = targs.get("path", "") or targs.get("destination", "") or ""
                            if fpath:
                                set_tool_memory("file_controller", self._cur_session_id, "last_path", fpath)
                        elif tname == "code_helper":
                            fp = targs.get("file_path", "") or targs.get("output_path", "") or ""
                            if fp:
                                set_tool_memory("code_helper", self._cur_session_id, "last_file", fp)
                except Exception:
                    pass

                if result != "__SILENT__":
                    all_silent = False
                    _tool_results.append((tname, result))

                _img_match = re.match(r'^__IMG__:(.+?):__IMG__(?:\|(.*))?$', str(result), re.DOTALL) if result != "__SILENT__" else None
                if _img_match:
                    _has_image = True
                    raw_b64 = _img_match.group(1)
                    img_text = (_img_match.group(2) or "").strip()
                    images_list = raw_b64.split("|")
                    for i in range(len(messages) - 1, -1, -1):
                        if messages[i].get("role") == "user":
                            messages[i] = {**messages[i], "images": images_list}
                            break
                    tool_msg = {
                        "role": "tool",
                        "content": img_text or "Image captured.",
                    }
                else:
                    tool_msg: dict = {
                        "role":    "tool",
                        "content": "Done." if result == "__SILENT__" else str(result),
                    }
                if tc_id is not None:
                    tool_msg["tool_call_id"] = tc_id

                messages.append(tool_msg)
                self._conversation.append(tool_msg)

            if all_silent:
                _saved_name: str | None = None
                for _tc in final_tool_calls:
                    _fn = _tc.get("function", {})
                    if _fn.get("name") == "save_memory":
                        _a = _fn.get("arguments", {})
                        if isinstance(_a, str):
                            try:
                                _a = json.loads(_a)
                            except Exception:
                                _a = {}
                        if isinstance(_a, dict) and _a.get("key") == "name" and _a.get("value"):
                            _saved_name = str(_a["value"])
                            break
                _ack = f"Got it, {_saved_name}." if _saved_name else "Noted."
                _amsg = {"role": "assistant", "content": _ack}
                messages.append(_amsg)
                self._conversation.append(_amsg)
                self._deliver_response(_ack, source)
                self._extract_and_log(_ack)
                self.speak(_ack)
                break

            if _round > 200:
                _amsg = {"role": "assistant", "content": "Task completed after maximum rounds."}
                messages.append(_amsg)
                self._conversation.append(_amsg)
                self._deliver_response(_amsg["content"], source)
                self._extract_and_log(_amsg["content"])
                self.speak(_amsg["content"])
                break

        if not self.ui.muted:
            self._set_listening_if_idle()

        for msg in self._conversation:
            msg.pop("images", None)

        try:
            new_turns = self._conversation[self._conv_saved_idx:]
            if new_turns:
                save_conversation(new_turns, self._cur_session_id)
                self._conv_saved_idx = len(self._conversation)
                # Auto-title on first user message
                if len(self._conversation) <= 2:
                    session = get_session(self._cur_session_id)
                    if session and session.get("title") == "New Chat":
                        auto_title_session(self._cur_session_id)
                # Real-time session info (turns/tokens/updated_at) after save
                try:
                    self.ui._win._session_refresh_sig.emit()
                except Exception:
                    pass
        except Exception as e:
            print(f"[Memory] ⚠️ Conversation save error: {e}")

        # ── Turn-reflection (async via queue) ────────────────────────────
        try:
            turn_count = len(self._conversation) // 2
            if turn_count > 0 and turn_count % 10 == 0 and self._cur_session_id:
                recent_text = " ".join(
                    (m.get("content") or "")[:120] for m in self._conversation[-4:]
                )
                self._enqueue_reflection(recent_text, self._cur_session_id)
        except Exception as e:
            print(f"[Timeline] Reflection queue error: {e}")

        # ── Importance decay (every 20 turns) ────────────────────────────
        try:
            if turn_count > 0 and turn_count % 20 == 0:
                self._decay_old_memories()
        except Exception:
            pass

        # ── Response delivered → pre-build next payload in background (D) ──
        try:
            import threading as _pf_thread
            _pf_thread.Thread(
                target=self._prefetch_next_payload, daemon=True
            ).start()
        except Exception as _pfe:
            print(f"[Prefetch] spawn failed: {_pfe}")

    # ------------------------------------------------------------------
    # STT listening loops
    # ------------------------------------------------------------------

    def _submit_voice_text(self, text: str) -> None:
        """Hand a finished utterance to the chat worker and return immediately.

        The old loops called ``_process_message`` on the MIC thread, so the
        whole LLM/tool turn ran inside the audio callback path: mic frames
        piled up and overflowed the 200-slot queue while the turn ran (the
        tail of a long utterance was lost), and no VAD ran during the turn
        (everything the user said next was dropped). Posting to
        ``_text_queue`` — the same path the UI text box uses — keeps the
        mic loop responsive and unifies ordering with typed input.
        """
        text = (text or "").strip()
        if text:
            self._text_queue.put(text)

    # ------------------------------------------------------------------
    # STT engine lifecycle (shared build + live-switch plumbing)
    # ------------------------------------------------------------------

    def _close_stt_quietly(self) -> None:
        """Stop the current STT engine (mic, session, thread) — never raises."""
        old = getattr(self, "_stt", None)
        if old is None:
            return
        try:
            old.close()
        except Exception:
            pass
        self._stt = None

    def _build_stt(self, engine: str, config: dict):
        """Build an STT engine from config. Single source of truth used by
        startup and live reconfigure alike.

        Hashim/Gemini STT use the Voice API key; if it is missing, fall
        back to the main Gemini LLM key so a single pasted key can run
        the whole voice pipeline (the UI accepts it in either field).
        """
        stt_language = (config.get("stt_language", "auto") or "auto").strip()
        if engine == "vosk":
            from core.stt import VoskSTT
            return VoskSTT(config.get("vosk_model_path"), language=stt_language)
        if engine == "gemini":
            from core.stt import GeminiSTT
            return GeminiSTT(self._voice_gemini_key(config), language=stt_language)
        if engine == "groq":
            from core.stt import GroqSTT
            return GroqSTT((config.get("groq_api_key", "") or "").strip(), language=stt_language)
        if engine == "hashim":
            from core.stt import HashimSTT
            return HashimSTT(self._voice_gemini_key(config), silence_threshold=1.2)
        from core.stt import WhisperSTT
        return WhisperSTT(config.get("stt_model", "base"), language=stt_language)

    @staticmethod
    def _voice_gemini_key(config: dict) -> str:
        """Voice API key with fallback to the main Gemini LLM key."""
        return (config.get("gemini_voice_api_key", "")
                or config.get("gemini_api_key", "") or "").strip()

    def _start_mic_loop(self, engine: str) -> None:
        """Start the right mic-loop thread for ``engine`` (idempotent per engine)."""
        if self._mic_thread_started:
            return
        self._mic_thread_started = True
        target = {
            "vosk":   self._listen_vosk,
            "hashim": self._listen_hashim,
        }.get(engine, self._listen_whisper)
        threading.Thread(target=target, daemon=True,
                         kwargs={"engine": engine}).start()

    def _stt_error(self, where: str, e: Exception) -> None:
        """Report an STT failure without killing the mic loop.

        A cloud-STT error (Groq 401, Gemini quota, network drop) used to
        escape the transcribe() call and end the whole mic thread — after
        that the assistant never heard anything again until restart. Now
        every failure is logged to the UI and the loop keeps running.
        """
        msg = str(e).strip().splitlines()[0][:200] if str(e).strip() else type(e).__name__
        if "401" in msg or "invalid_api_key" in msg.lower() or "api key" in msg.lower():
            engine = (self._config.get("stt_engine", "") or "").lower()
            if engine == "groq":
                msg = ("Groq STT key rejected (401). Open CONFIGURE → Speech-to-Text → "
                       "Groq and paste a valid key from console.groq.com/keys "
                       "(must start with gsk_).")
            elif engine == "gemini":
                msg = ("Gemini STT key rejected (401/400). Open CONFIGURE → Speech-to-Text → "
                       "Gemini and paste a valid key from aistudio.google.com/apikey "
                       "(AIza… or AQ.… format — both valid).")
            elif engine == "hashim":
                msg = ("Hashim STT: Gemini rejected the key (401/403). Open CONFIGURE → "
                       "Speech-to-Text → Gemini (Voice API Key) and paste a valid key "
                       "from aistudio.google.com/apikey — it applies live, no restart.")
            else:
                msg = f"STT key rejected (401) — check the {engine or 'STT'} API key in CONFIGURE."
        elif "quota" in msg.lower() or "429" in msg:
            msg = "STT quota/rate limit hit — waiting before the next attempt."
        elif "connect" in msg.lower() or "timeout" in msg.lower():
            msg = "STT network problem — will retry on the next utterance."
        print(f"[STT] {where} error: {msg}")
        try:
            self.ui.write_log(f"ERR: STT — {msg}")
        except Exception:
            pass
        # The face looks at the activity log when something goes wrong —
        # a silent visual cue that the panel now holds the reason.
        try:
            self.ui.glance(0.8, 0.25, 1.5)
        except Exception:
            pass

    def _get_barge(self):
        """Barge-in detector, created lazily.

        TTS finishes loading AFTER the mic loop starts (it loads in the
        background), so deciding barge support from ``_streaming_tts`` at
        mic-start time is always too early. Re-evaluate on every call: the
        detector is built the first time a streaming TTS is actually up.
        """
        if not self._config.get("tts_barge_in", True):
            return None
        if self._barge is None and self._streaming_tts:
            self._barge = _BargeDetector()
        return self._barge

    def _make_vad(self, engine: str = "whisper"):
        """Pick the VAD for a mic loop — engine-aware.

        Gemini STT and Groq STT: the LEGACY energy VAD (user choice —
        Silero kept mis-splitting their turns), with RAISED noise
        multipliers so fans/background hiss cannot open a turn:
            vad_cloud_start_mult  (default 1.9 — speech must be ~2x the floor)
            vad_cloud_end_mult    (default 1.4)
            vad_silence_sec_cloud (default 3.0 — silence that ends a turn)
        Local engines (whisper/vosk) run Silero ('vad_engine': 'silero',
        threshold 'vad_threshold' default 0.5). 'energy' config — or a
        missing runtime/model — falls back to the energy VAD everywhere,
        with its own tunables vad_start_mult / vad_end_mult /
        vad_silence_sec (original defaults 1.5 / 1.15 / 3.0).
        """
        if engine in ("gemini", "groq"):
            start_mult = float(self._config.get("vad_cloud_start_mult", 1.9))
            end_mult = float(self._config.get("vad_cloud_end_mult", 1.4))
            sil_sec = float(self._config.get("vad_silence_sec_cloud", 3.0))
            try:
                self.ui.write_log(
                    f"SYS: VAD — energy (cloud STT, noise×{start_mult:g}, "
                    f"pause={sil_sec:g}s).")
            except Exception:
                pass
            return _VADBuffer(start_mult=start_mult, end_mult=end_mult,
                              silence_sec=sil_sec)
        mode = (self._config.get("vad_engine", "silero") or "silero").lower()
        if mode in ("silero", "neural", "auto"):
            try:
                from core.vad_silero import make_vad
                th = float(self._config.get("vad_threshold", 0.5))
                v = make_vad(threshold=th)
                if v is not None:
                    self.ui.write_log(
                        f"SYS: VAD — Silero neural (fast turn-end, th={th:g}).")
                    return v
            except Exception:
                pass
            self.ui.write_log("SYS: VAD — Silero unavailable, energy fallback.")
        return _VADBuffer(
            start_mult=float(self._config.get("vad_start_mult", 1.5)),
            end_mult=float(self._config.get("vad_end_mult", 1.15)),
            silence_sec=float(self._config.get("vad_silence_sec", 3.0)),
        )
    def _listen_whisper(self, engine: str = "whisper") -> None:
        vad = self._make_vad(engine)
        self._vad_ref = vad
        q: queue.Queue = queue.Queue(maxsize=200)

        def callback(indata, frames, time_info, status):
            # Echo-risk measurement: record how loud the room (and the
            # speakers) are in this mic while the TTS is playing.
            if self._measuring_bleed:
                try:
                    r = float(np.sqrt(np.mean(indata.astype(np.float32) ** 2))) / 32768.0
                    with self._bleed_lock:
                        b = self._mic_bleed
                        if b is not None and len(b) < 2000:
                            b.append(r)
                except Exception:
                    pass
            with self._speaking_lock:
                is_speaking = self._speaking
            if is_speaking or self._mic_held() or self.ui.muted:
                return
            if self._gate_active() and not self._echo_gate_open():
                return   # the previous turn's tail is still leaving the speakers
            try:
                q.put_nowait(indata.tobytes())
            except queue.Full:
                pass

        while True:
            gen = getattr(self, "_stt_generation", 0)
            try:
                with sd.InputStream(
                    samplerate=SAMPLE_RATE_IN,
                    channels=CHANNELS,
                    dtype="int16",
                    blocksize=BLOCK_SIZE,
                    callback=callback,
                ):
                    self.ui.write_log("SYS: Mic active (Whisper STT).")
                    self.ui.set_state("LISTENING")   # honest: mic is really open now
                    while True:
                        if getattr(self, "_stt_generation", 0) != gen:
                            raise _STTSwitched()
                        try:
                            data = q.get(timeout=0.1)
                            chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                            # User's live mic level → face reacts while listening
                            # (same normalized loudness mapping the mouth uses)
                            try:
                                from core.face_renderer import pcm_level
                                self.ui.set_audio_level(pcm_level(chunk))
                            except Exception:
                                pass
                            barge = self._get_barge()
                            if barge is not None:
                                with self._speaking_lock:
                                    is_speaking = self._speaking
                                if is_speaking:
                                    rms = float(np.sqrt(np.mean(chunk ** 2)))
                                    if barge.speech(rms):
                                        # THINKING is set only when a transcript
                                        # will actually be submitted — setting it
                                        # earlier left the typing badge stuck for
                                        # the whole idle period after a failed or
                                        # empty transcription.
                                        # TRANSCRIBE FIRST, STOP LATER: a
                                        # marginal/echo frame must not kill the
                                        # playing reply — only a transcript that
                                        # survives junk/echo checks (and is
                                        # submitted) hard-stops speech. Previously
                                        # any loud blip cut Orthos off mid-sentence
                                        # even when its transcript was junk.
                                        audio = vad.process(chunk)
                                        submitted = False
                                        if audio is not None:
                                            self._capture_env = []
                                            try:
                                                self._envelope_push(
                                                    self._capture_env, audio,
                                                    SAMPLE_RATE_IN, self._ENV_RATE)
                                            except Exception:
                                                self._capture_env = []
                                            try:
                                                text = self._stt.transcribe(audio)
                                            except Exception as e:
                                                self._stt_error("transcribe", e)
                                                text = ""
                                            if text.strip():
                                                if not (self._junk_transcript(text, audio) or self._reject_echo(text)):
                                                    self.ui.set_state("THINKING")
                                                    self._submit_voice_text(text)
                                                    submitted = True
                                        if submitted:
                                            self.stop_speaking()
                                            self._signal_cancel()
                                            vad.reset()   # buffer was echo/mixed
                                    continue
                            audio = vad.process(chunk)
                            if audio is not None:
                                # The utterance's own loudness envelope is
                                # the capture side of the echo fingerprint.
                                self._capture_env = []
                                try:
                                    self._envelope_push(
                                        self._capture_env, audio,
                                        SAMPLE_RATE_IN, self._ENV_RATE)
                                except Exception:
                                    self._capture_env = []
                                # TRANSCRIBE FIRST, STOP LATER (same as the
                                # barge path): junk/echo utterances must not
                                # hard-stop a playing reply — only a validated
                                # user utterance does.
                                try:
                                    text = self._stt.transcribe(audio)
                                except Exception as e:
                                    self._stt_error("transcribe", e)
                                    continue
                                if text.strip():
                                    if self._junk_transcript(text, audio) or self._reject_echo(text):
                                        continue
                                    self.ui.set_state("THINKING")
                                    self._submit_voice_text(text)
                                    self.stop_speaking()
                                    self._signal_cancel()
                                    vad.reset()   # buffer was echo/mixed
                        except queue.Empty:
                            pass
            except _STTSwitched:
                return                     # engine swapped — new loop takes over
            except Exception as e:
                # Mic device failure (unplugged, driver reset): back off and
                # reopen instead of dying.
                self._stt_error("mic", e)
                time.sleep(2.0)

    def _listen_vosk(self, engine: str = "vosk") -> None:
        vad = self._make_vad(engine)
        self._vad_ref = vad
        q: queue.Queue = queue.Queue(maxsize=200)

        def callback(indata, frames, time_info, status):
            # Echo-risk measurement: record how loud the room (and the
            # speakers) are in this mic while the TTS is playing.
            if self._measuring_bleed:
                try:
                    r = float(np.sqrt(np.mean(indata.astype(np.float32) ** 2))) / 32768.0
                    with self._bleed_lock:
                        b = self._mic_bleed
                        if b is not None and len(b) < 2000:
                            b.append(r)
                except Exception:
                    pass
            with self._speaking_lock:
                is_speaking = self._speaking
            if is_speaking or self._mic_held() or self.ui.muted:
                return
            if self._gate_active() and not self._echo_gate_open():
                return   # the previous turn's tail is still leaving the speakers
            try:
                q.put_nowait(indata.tobytes())
            except queue.Full:
                pass

        while True:
            gen = getattr(self, "_stt_generation", 0)
            try:
                with sd.InputStream(
                    samplerate=SAMPLE_RATE_IN,
                    channels=CHANNELS,
                    dtype="int16",
                    blocksize=4096,
                    callback=callback,
                ):
                    self.ui.write_log("SYS: Mic active (Vosk STT).")
                    self.ui.set_state("LISTENING")   # honest: mic is really open now
                    final_texts: list[str] = []
                    while True:
                        if getattr(self, "_stt_generation", 0) != gen:
                            raise _STTSwitched()
                        try:
                            data = q.get(timeout=0.1)
                            chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                            # User's live mic level → face reacts while listening
                            # (same normalized loudness mapping the mouth uses)
                            try:
                                from core.face_renderer import pcm_level
                                self.ui.set_audio_level(pcm_level(chunk))
                            except Exception:
                                pass
                            barge = self._get_barge()
                            if barge is not None:
                                with self._speaking_lock:
                                    is_speaking = self._speaking
                                if is_speaking:
                                    rms = float(np.sqrt(np.mean(chunk ** 2)))
                                    if barge.speech(rms):
                                        # TRANSCRIBE FIRST, STOP LATER — a
                                        # junk/echo frame must not kill the
                                        # playing reply (see whisper loop).
                                        vad_audio = vad.process(chunk)
                                        submitted = False
                                        if vad_audio is not None:
                                            self._capture_env = []
                                            try:
                                                self._envelope_push(
                                                    self._capture_env, vad_audio,
                                                    SAMPLE_RATE_IN, self._ENV_RATE)
                                            except Exception:
                                                self._capture_env = []
                                            try:
                                                text, _ = self._stt.process_chunk(vad_audio.tobytes())
                                            except Exception as e:
                                                self._stt_error("vosk", e)
                                                text = ""
                                            if text.strip():
                                                if not (self._junk_transcript(text, vad_audio) or self._reject_echo(text)):
                                                    final_texts.append(text)
                                                    self.ui.set_state("THINKING")
                                                    self._submit_voice_text(" ".join(final_texts))
                                                    submitted = True
                                            final_texts.clear()
                                        if submitted:
                                            self.stop_speaking()
                                            self._signal_cancel()
                                            vad.reset()   # buffer was echo/mixed
                                    continue
                            vad_audio = vad.process(chunk)
                            if vad_audio is not None:
                                try:
                                    self._envelope_push(
                                        self._capture_env, vad_audio,
                                        SAMPLE_RATE_IN, self._ENV_RATE)
                                except Exception:
                                    self._capture_env = []
                                text, _ = self._stt.process_chunk(vad_audio.tobytes())
                                if text.strip():
                                    final_texts.append(text)
                            text, is_final = self._stt.process_chunk(data)
                            if is_final and text.strip():
                                final_texts.append(text)
                            if vad_audio is None and final_texts:
                                joined = " ".join(final_texts)
                                final_texts.clear()
                                # Validate BEFORE stopping speech: a silent
                                # stretch that produced only echo/junk must not
                                # cut the playing reply (whisper-loop fix).
                                if self._reject_echo(joined):
                                    continue
                                self._submit_voice_text(joined)
                                self.stop_speaking()
                                self._signal_cancel()
                        except queue.Empty:
                            pass
            except _STTSwitched:
                return                     # engine swapped — new loop takes over
            except Exception as e:
                self._stt_error("mic", e)
                time.sleep(2.0)

    def _listen_hashim(self, engine: str = "hashim") -> None:
        """Bridge the vendored Hashim engine's sentences into the chat.

        The upstream engine owns mic capture, VAD and (server-side) echo
        suppression — this loop never opens an audio device. It drains
        ``sentence_queue`` and submits through the same gate every other
        voice path uses, so the transcript filter still guards against
        any echo that survives the server-side AEC.

        Liveness is polled via ``_stt_generation``: a live reconfigure
        builds a NEW engine and bumps the generation, so this loop exits
        cleanly and the mic is re-opened for the new engine — no app
        restart needed when switching STT engines in Configure.
        """
        while True:
            gen = getattr(self, "_stt_generation", 0)
            try:
                self.ui.write_log("SYS: Mic active (Hashim STT — Gemini Live).")
                self.ui.set_state("LISTENING")
                self._stt.connect()
                if hasattr(self._stt, "set_gate"):
                    self._stt.set_gate(lambda: self._speaking or self._mic_held())
                while True:
                    if not self._stt.is_alive():
                        # Report the REAL failure (bad key, config error,
                        # network drop) instead of a bare "thread died".
                        last = None
                        try:
                            last = self._stt.last_error()
                        except Exception:
                            pass
                        raise RuntimeError(
                            str(last) if last else "Hashim engine thread died"
                        )
                    if getattr(self, "_stt_generation", 0) != gen:
                        raise _STTSwitched()          # engine replaced — stand down
                    sents = self._stt.drain_sentences(max_items=10)
                    if not sents:
                        time.sleep(0.1)
                        continue
                    for sent in sents:
                        sent = (sent or "").strip()
                        if not sent:
                            continue
                        self.ui.set_state("THINKING")
                        if self._reject_echo(sent):
                            continue
                        self._submit_voice_text(sent)
            except _STTSwitched:
                # The reconfigure thread owns the swap AND the old engine's
                # teardown — this loop just stands down.
                self.ui.write_log("SYS: STT engine switched — Hashim session closed.")
                return
            except Exception as e:
                if getattr(self, "_stt_generation", 0) != gen:
                    return                             # superseded while failing
                self._stt_error("hashim", e)
                time.sleep(2.0)                        # loop-top reconnects

    # ------------------------------------------------------------------
    # Text command loop (UI input box)
    # ------------------------------------------------------------------

    def _text_command_loop(self) -> None:
        while True:
            try:
                item = self._text_queue.get(timeout=0.5)
                if isinstance(item, tuple) and item[0] == "file_msg":
                    _, text, extra = item
                    print(f"[Orthos] file_msg received: text={text[:50]}... extra_keys={list(extra.keys())}")
                    try:
                        self._process_message(text, source="ui", extra_meta=extra)
                    except Exception as e:
                        print(f"[Orthos] ERROR in _process_message (file_msg): {e}")
                        import traceback
                        traceback.print_exc()
                elif isinstance(item, tuple) and item[0] == "action_msg":
                    # F7: Regenerate/Continue — handled on this worker thread
                    # so conversation mutation stays race-free.
                    self._run_chat_action(item[1])
                elif isinstance(item, str) and item.strip():
                    self._process_message(item)
            except queue.Empty:
                pass
            except Exception as e:
                print(f"[Orthos] ERROR in _text_command_loop: {e}")
                import traceback
                traceback.print_exc()

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        try:
            self.ui.on_reconfigure = self.reconfigure

            # ── Metrics server (non-blocking) ──────────────────────────────
            try:
                start_metrics_server(9090)
            except Exception:
                pass

            # ── OpenTelemetry tracing ──────────────────────────────────────
            init_tracing()

            # ── ChromaDB health check ──────────────────────────────────────
            if self._chroma:
                try:
                    health = self._chroma.health()
                    self.ui.write_log(f"SYS: ChromaDB {health.get('status', 'unknown')}"
                                      f" — {health.get('count', 0)} entries")
                except Exception as e:
                    self.ui.write_log(f"SYS: ChromaDB init skipped — {e}")

            # ── LLM health ────────────────────────────────────────────────
            from core.llm_client import ensure_ollama_running, warmup_model, get_llm_provider
            _llm_prov = get_llm_provider()
            self.ui.write_log(f"SYS: Checking {_llm_prov}…")
            if ensure_ollama_running():
                self.ui.write_log(f"SYS: {_llm_prov} OK.")
            else:
                self.ui.write_log(f"ERR: {_llm_prov} unavailable.")

            # ── Config ────────────────────────────────────────────────────
            stt_engine   = self._config.get("stt_engine",   "whisper").lower()
            stt_language = self._config.get("stt_language", "auto")
            stt_model    = self._config.get("stt_model",    "base")
            tts_engine   = self._config.get("tts_engine",   "edgetts").lower()

            # ── Startup progress panel ────────────────────────────────────
            self.ui.show_startup_panel()

            _warmup_done = threading.Event()
            _stt_done    = threading.Event()

            # ── LLM warmup thread ─────────────────────────────────────────
            def _do_warmup():
                try:
                    static_prompt = _load_system_prompt()
                    warmup_model(system_prompt=static_prompt)
                    self.ui.write_log("SYS: LLM ready.")
                    self.ui.mark_startup_ready("llm")
                except Exception as e:
                    self.ui.write_log(f"ERR: LLM warmup — {e}")
                    self.ui.mark_startup_ready("llm", error=True)
                finally:
                    _warmup_done.set()

            # ── STT load thread ───────────────────────────────────────────
            def _do_stt():
                self._seed_echo_risk()
                try:
                    self.ui.write_log(f"SYS: Loading {stt_engine.upper()} STT…")
                    self._stt = self._build_stt(stt_engine, self._config)
                    self.ui.write_log("SYS: STT ready.")
                    self.ui.mark_startup_ready("stt")
                    # Mic opens THE MOMENT the STT engine is ready — not after
                    # MCP servers, MiniLM, NER and index loads finish. Those can
                    # take minutes on a loaded machine; the old ordering left
                    # the assistant deaf the whole time (mic loop was the last
                    # statement of run()).
                    if not self._is_headless:
                        self._start_mic_loop(stt_engine)
                except Exception as e:
                    self.ui.write_log(f"ERR: STT — {e}")
                    self.ui.mark_startup_ready("stt", error=True)
                finally:
                    _stt_done.set()

            # ── TTS load thread — does NOT block going online ─────────────
            def _do_tts():
                try:
                    self.ui.write_log(f"SYS: Loading {tts_engine.upper()} TTS…")
                    if tts_engine == "kokoro":
                        self.ui.write_log("SYS: Kokoro — loading model + compiling JIT…")
                        _os.environ.pop("HF_HUB_OFFLINE", None)
                        _os.environ.pop("TRANSFORMERS_OFFLINE", None)
                        _os.environ.pop("HF_DATASETS_OFFLINE", None)
                    from core.tts import create_tts_player
                    self._tts = create_tts_player(self._config)
                    self._attach_pcm_sink()
                    self._streaming_tts = bool(getattr(self._tts, "supports_streaming", False))
                    if self._streaming_tts:
                        from core.llm_client import set_voice_mode
                        set_voice_mode(True)
                        try:
                            self._tts.warmup()
                        except Exception as e:
                            print(f"[TTS] warmup error: {e}")
                    self._tts_ready.set()
                    self.ui.write_log("SYS: TTS ready.")
                    self.ui.mark_startup_ready("tts")
                    self.ui.set_startup_status("● All systems ready.")
                    self.ui.hide_startup_panel()
                    self.speak("Orthos fully online.")
                except Exception as e:
                    import traceback as _tb; _tb.print_exc()
                    self.ui.write_log(f"ERR: TTS — {e}")
                    self.ui.mark_startup_ready("tts", error=True)
                    self._tts_ready.set()

            # ── Gemini Live persistent mode: wire session callbacks ──────────
            self._live_provider = None
            try:
                if _get_provider_config()["provider"] == "gemini_live":
                    from core.llm_provider import get_registry
                    _live_cfg = _get_provider_config()
                    _live = get_registry().get_provider(
                        "gemini_live",
                        api_key=_live_cfg.get("api_key", ""),
                        model=_live_cfg.get("model", ""),
                    )
                    _live.register_tool_handler(self._execute_tool)
                    _live.register_output_cb(self._on_live_output)
                    _live.register_user_cb(self._on_live_user)
                    _live.register_muted_getter(lambda: self.ui.muted)
                    _live.register_speaking_cb(self._on_live_speaking)
                    self._live_provider = _live
                    self.ui.write_log("SYS: Gemini Live persistent session mode.")
            except Exception as e:
                print(f"[Live] Callback wiring failed: {e}")

            # Launch all startup threads simultaneously
            self.ui.write_log("SYS: Loading systems in parallel…")
            threading.Thread(target=_do_warmup, daemon=True).start()
            if self._live_provider is not None:
                self._stt = None
                _stt_done.set()
                self.ui.write_log("SYS: STT skipped (Gemini Live handles mic input natively).")
            else:
                threading.Thread(target=_do_stt,    daemon=True).start()
            threading.Thread(target=_do_tts,    daemon=True).start()

            # ── Wait ONLY for STT + LLM (fast) ────────────────────────────
            _warmup_done.wait(timeout=60)
            _stt_done.wait(timeout=60)

            # ── Initialise session system ──────────────────────────────────
            self._init_session_system()
            self.ui.write_log(f"SYS: Session {self._cur_session_id[:13]}... ready.")

            # ── Prompt-cache warmup (perf) ──────────────────────────────
            # Pre-build static prompt segments + warm the vector stack in the
            # background so the FIRST real message doesn't pay the cold cost
            # (Chroma init, summary JOIN build, condense-cache disk load).
            def _do_prompt_warmup():
                try:
                    self._prefetch_next_payload()
                    self.ui.write_log("SYS: Memory caches warm.")
                except Exception as e:
                    print(f"[Warmup] prompt prefetch failed: {e}")

            threading.Thread(target=_do_prompt_warmup, daemon=True,
                             name="prompt-warmup").start()

            # ── Go online immediately ──────────────────────────────────────
            self.ui.write_log("SYS: Orthos online.")
            self.ui.set_state("LISTENING")
            self.ui.set_startup_status("● Orthos online · Voice loading in background…")

            if not self._is_headless:
                threading.Thread(target=self._tts_worker,        daemon=True).start()
            threading.Thread(target=self._text_command_loop,  daemon=True).start()

            # ── Gateway (Telegram, etc.) ─────────────────────────────────────
            try:
                from gateway import Gateway
                self._gateway = Gateway(
                    process_callback=self._process_message,
                    log_callback=self.ui.write_log,
                    cancel_callback=self._signal_cancel,
                )
                self._gateway.start()
                self.register_response_callback(self._gateway.route_response)
                from gateway.log_capture import LogCapture
                self._log_capture = LogCapture(self._gateway.send_log)
                self._log_capture.start()
                from core.logging_setup import rehook_console_sink
                rehook_console_sink()
                self.ui.write_log("SYS: Terminal log capture started.")
            except Exception as e:
                self.ui.write_log(f"ERR: Gateway init — {e}")

            # ── MiniLM inline load (use boot-time memory service if available) ──
            self._memory_svc_ok = False
            try:
                from memory.memory_client import health as check_memory_svc
                svc = check_memory_svc()
                if svc and svc.get("status") == "ok":
                    self._memory_svc_ok = True
                    self.ui.write_log("SYS: Using preloaded memory service (boot-time).")
                else:
                    raise ConnectionError("Memory service not reachable")
            except Exception:
                try:
                    self.ui.write_log("SYS: Loading MiniLM L6 v2…")
                    from memory.vector_index import preload_model
                    preload_model()
                    self.ui.write_log("SYS: MiniLM ready.")
                    self.ui.write_log("SYS: Loading spaCy NER…")
                    from memory.entity_store import preload_ner
                    preload_ner()
                    self.ui.write_log("SYS: spaCy NER ready.")
                except Exception as e:
                    self.ui.write_log(f"ERR: local model load — {e}")

            # ── Pre-start background index worker (async raw-turn indexing) ──
            try:
                from memory.index_worker import warm_start
                warm_start()
            except Exception:
                pass

            # ── Background memory dreaming (background fact extraction) ──────
            try:
                from memory.dreaming import start_dream_loop
                start_dream_loop(interval_seconds=120)
                self.ui.write_log("SYS: Memory dreaming started.")
            except Exception as e:
                self.ui.write_log(f"ERR: Memory dreaming — {e}")

            # ── Background backfill: embed existing raw turns for semantic recall ──
            try:
                def _turn_vector_backfill():
                    # Wait until the app is quiet — this competes with the
                    # first user message (SQLite + embedding service) and in
                    # cloud mode each batch is a REMOTE call (50+ round-trips
                    # at startup froze prompt building for 20+ seconds).
                    time.sleep(120)
                    try:
                        from memory.chroma_memory import backfill_turn_embeddings
                        n = backfill_turn_embeddings(batch_size=512)
                        if n:
                            self.ui.write_log(f"SYS: Raw-turn vectors backfilled — {n} messages now semantically searchable.")
                        else:
                            self.ui.write_log("SYS: Raw-turn vectors up to date.")
                    except Exception as ex:
                        self.ui.write_log(f"ERR: Turn-vector backfill — {ex}")
                threading.Thread(target=_turn_vector_backfill, name="turn-vector-backfill", daemon=True).start()
            except Exception as e:
                self.ui.write_log(f"ERR: Turn-vector backfill thread — {e}")

            # ── MCP Server connections ────────────────────────────────────
            try:
                from core.mcp_manager import get_global_manager, attach_runtime
                attach_runtime(
                    registry=_tool_registry,
                    tool_index=_tool_index,
                    loaded_set=_tool_active_full_schemas,
                )
                self._mcp_manager = get_global_manager()
                mcp_decls = self._mcp_manager.discover_and_register(_tool_registry)
                if mcp_decls:
                    self.ui.write_log(f"SYS: MCP — {len(mcp_decls)} tool(s) registered (hidden from payload, searchable via search_tools)")
                    print(f"[MCP] {len(mcp_decls)} MCP tool(s) registered")
            except Exception as e:
                self.ui.write_log(f"ERR: MCP init — {e}")
                self._mcp_manager = None

            # ── Warm vector + BM25 indexes (skip if memory service already loaded them) ──
            if not self._memory_svc_ok:
                try:
                    from memory.memory_manager import load_memory, get_recent_turns_for_vector
                    from memory.vector_index import rebuild_index
                    turns = get_recent_turns_for_vector(100)
                    facts = load_memory()
                    rebuild_index(turns, facts)
                    self.ui.write_log("SYS: Vector index built.")
                except Exception as e:
                    self.ui.write_log(f"ERR: Vector index — {e}")
                try:
                    from memory.bm25_search import rebuild_bm25_index
                    rebuild_bm25_index()
                    self.ui.write_log("SYS: BM25 index built.")
                except Exception as e:
                    self.ui.write_log(f"ERR: BM25 index — {e}")
            else:
                self.ui.write_log("SYS: Indexes already loaded via memory service.")

            # ── Block — STT loop or headless sleep ──────────────────────────
            if self._is_headless:
                self.ui.write_log("SYS: Headless mode active — Telegram bot running.")
                self.ui.write_log("SYS: Press Ctrl+C to stop.")
                while True:
                    import time as _time
                    _time.sleep(1)
            elif self._live_provider is not None:
                # Gemini Live: the persistent session streams the mic and
                # plays Gemini's native voice — no STT loop / TTS speaking.
                self.ui.write_log("SYS: Gemini Live — hands-free voice mode.")
                self.ui.set_state("LISTENING")
                while True:
                    import time as _time
                    _time.sleep(1)
            else:
                # The mic loop already runs on its own thread (started the
                # moment STT became ready). Stay here to keep run() alive.
                self.ui.write_log("SYS: Voice loop running on background thread.")
                while True:
                    import time as _time
                    _time.sleep(1)

        except Exception as e:
            self.ui.write_log(f"ERR: Init failed — {e}")
            traceback.print_exc()


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main() -> None:
    print("[Orthos] Starting with improved modules: Loguru · SecretsManager · "
          "ToolRegistry · LLM Provider · Prometheus · ChromaDB · OpenTelemetry · HealthChecks")

    # ── Pre-import torch in background immediately ─────────────────────────
    def _preload_torch():
        try:
            import torch  # noqa: F401
        except Exception:
            pass
    threading.Thread(target=_preload_torch, daemon=True).start()
    # ───────────────────────────────────────────────────────────────────────

    if _ARGS.headless:
        from core.headless_ui import HeadlessUI
        ui = HeadlessUI()
    else:
        ui = OrthosUI("face.png")

    def runner():
        if not _ARGS.headless:
            ui.wait_for_api_key()

        ui.write_log("SYS: Checking dependencies…")
        cfg = _load_config()
        _install_done = threading.Event()

        def _do_install():
            try:
                from core.installer import install_for_config
                install_for_config(cfg, log=ui.write_log)
            except Exception as e:
                ui.write_log(f"ERR: Dependency install — {e}")
            finally:
                _install_done.set()

        threading.Thread(target=_do_install, daemon=True).start()
        _install_done.wait()

        orthos = Orthos(ui)
        try:
            orthos.run()
        except KeyboardInterrupt:
            print("\n[Orthos] Shutting down…")
        finally:
            try:
                orthos._close_current_session()
            except Exception:
                pass

    threading.Thread(target=runner, daemon=True).start()
    if _ARGS.headless:
        try:
            while True:
                import time as _time
                _time.sleep(1)
        except KeyboardInterrupt:
            print("\n[Orthos] Shutting down…")
    else:
        ui.root.mainloop()


if __name__ == "__main__":
    main()
