"""
Orthos — Message Router

Normalises inbound messages from any platform into a common
(full_text, source) pair and dispatches to the central pipeline.

Sources: "ui", "voice", "telegram", "whatsapp", etc.
"""

from __future__ import annotations

from typing import Callable


class MessageRouter:
    """
    Receives raw messages from platform adapters, normalises them,
    and calls the processing callback.

    The callback signature is: fn(text, source, **extra)
    where extra contains platform-specific metadata (chat_id, user_id, etc.)
    """

    def __init__(self, process_callback: Callable):
        self._process = process_callback

    def route(self, text: str, source: str, **extra) -> None:
        """
        Route a user message to the central LLM pipeline.

        Args:
            text:   The user's message text.
            source: Platform identifier ("telegram", "whatsapp", "ui", "voice").
            extra:  Platform-specific metadata (chat_id, user_id, etc.).
        """
        text = text.strip()
        if not text:
            return

        self._process(text, source, **extra)
