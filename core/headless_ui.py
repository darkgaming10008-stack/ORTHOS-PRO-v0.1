"""
Orthos — Headless UI

Stand-in for OrthosUI (PyQt6) when running with --headless flag.
All UI methods become print() or no-ops.

This allows the assistant (Gateway/Telegram bot, LLM) to run
on a headless server, WSL, or Windows terminal without PyQt6.
"""

from __future__ import annotations

import sys
from datetime import datetime


def _stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


class HeadlessUI:
    """
    Mock UI that replaces PyQt6 OrthosUI in --headless mode.

    Implements the same interface consumed by Orthos so
    no other code changes are needed.
    """

    is_headless: bool = True  # Used by Orthos.run() to skip STT/TTS
    muted: bool = True   # No TTS in headless mode

    # Callbacks — left as None; Orthos sets these at runtime
    on_text_command = None
    on_reconfigure   = None

    # ── Logging ─────────────────────────────────────────────────────────

    def write_log(self, msg: str) -> None:
        """Print log messages to stderr so they appear in the terminal."""
        print(f"[{_stamp()}] {msg}", file=sys.stderr, flush=True)

    # ── State transitions (no-op) ───────────────────────────────────────

    def set_state(self, _state: str) -> None:
        pass

    def set_audio_level(self, _level: float) -> None:
        pass

    def notify_audio_activity(self, _grace: float = 0.9) -> None:
        pass

    def push_visemes(self, _frames, _hop: float, _at: float) -> None:
        pass

    def clear_visemes(self) -> None:
        pass

    # ── Startup panel (no-op) ───────────────────────────────────────────

    def show_startup_panel(self) -> None:
        pass

    def hide_startup_panel(self) -> None:
        pass

    def mark_startup_ready(self, _component: str, error: bool = False) -> None:
        pass

    def set_startup_status(self, _status: str) -> None:
        pass

    # ── First-run API key setup (no-op in headless) ─────────────────────

    def wait_for_api_key(self) -> None:
        """Config is loaded directly; no PyQt6 overlay needed."""
        pass
