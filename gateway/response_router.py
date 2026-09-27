"""
Orthos — Response Router

Collects final assistant responses and delivers them to all
active output channels (TTS, UI log, Telegram, WhatsApp, …).

Each registered callback receives (response_text, source).
"""

from __future__ import annotations

import threading
from typing import Callable


class ResponseRouter:
    """
    Fan-out dispatcher for assistant responses.

    Register callbacks with register(callback) where callback
    has signature: fn(response_text: str, source: str) -> None.
    """

    def __init__(self):
        self._callbacks: list[Callable[[str, str], None]] = []
        self._lock = threading.Lock()

    def register(self, callback: Callable[[str, str], None]):
        with self._lock:
            self._callbacks.append(callback)

    def unregister(self, callback: Callable[[str, str], None]):
        with self._lock:
            try:
                self._callbacks.remove(callback)
            except ValueError:
                pass

    def dispatch(self, text: str, source: str) -> None:
        """Send *text* to every registered output channel."""
        if not text:
            return
        with self._lock:
            cbs = list(self._callbacks)
        for cb in cbs:
            try:
                cb(text, source)
            except Exception as e:
                print(f"[ResponseRouter] callback error: {e}")
