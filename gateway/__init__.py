"""
Orthos — Multi-platform Gateway

Architecture (OpenClaw/Hermes inspired):
  Inbound adapters (Telegram, WhatsApp, …) → MessageRouter
  → serialised queue → _process_message()
  → Gateway.route_response() → outbound adapters

All inbound messages are serialised through a single queue to
avoid race conditions in the shared _conversation state.

Usage:
    from gateway import Gateway
    gw = Gateway(process_callback=my_fn, log_callback=print)
    gw.start()
    gw.route_response("Hello!", "telegram")
"""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Callable

from .message_router import MessageRouter

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "telegram_config.json"


class Gateway:
    """
    Single control plane for all messaging platforms.
    Owns adapter connections and serialises inbound messages.
    """

    def __init__(
        self,
        process_callback: Callable,
        log_callback: Callable[[str], None] | None = None,
        cancel_callback: Callable[[], None] | None = None,
    ):
        self._process_callback = process_callback  # fn(text, source, **extra)
        self._log = log_callback or print
        self._cancel_callback = cancel_callback    # fn() — signals current turn to abort
        self._message_router = MessageRouter(self._enqueue_message)
        self._adapters: list = []
        self._running = False
        self._inbound_queue: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None

        self._load_config()

    def _load_config(self):
        self._config = {}
        try:
            if _CONFIG_PATH.exists():
                with open(_CONFIG_PATH, encoding="utf-8") as f:
                    self._config = json.load(f)
        except Exception as e:
            self._log(f"GATEWAY: Config load error — {e}")

    def _enqueue_message(self, text: str, source: str, **extra):
        """
        Called by MessageRouter from any thread.

        Cancels any in-flight processing so the queued message preempts
        the current turn.  The worker loop drains intermediate messages
        and processes only the latest.
        """
        if self._cancel_callback:
            self._cancel_callback()
        self._inbound_queue.put((text, source, extra))

    def _worker_loop(self):
        """
        Process inbound messages with preemption.

        When a new message arrives while one is being processed, the previous
        turn is cancelled and the queue is drained so only the latest message
        is processed — effectively a "typeahead" that skips stale turns.
        """
        while self._running:
            try:
                text, source, extra = self._inbound_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # Cancel any in-flight processing from a previous turn
            if self._cancel_callback:
                self._cancel_callback()

            # Drain queue: keep only the *latest* message (skips stale ones)
            while True:
                try:
                    text, source, extra = self._inbound_queue.get_nowait()
                except queue.Empty:
                    break

            # Process the latest message only
            try:
                self._process_callback(text, source, extra_meta=extra)
            except Exception as e:
                self._log(f"GATEWAY: Processing error — {e}")

    def start(self):
        if self._running:
            return
        self._running = True

        self._worker = threading.Thread(
            target=self._worker_loop,
            name="gateway-worker",
            daemon=True,
        )
        self._worker.start()

        from .async_worker import get_async_worker

        get_async_worker()  # Ensure shared event loop is available

        from .telegram_adapter import TelegramAdapter

        front_token = self._config.get("front_bot_token", "")
        if front_token:
            try:
                adapter = TelegramAdapter(
                    token=front_token,
                    on_message=self._message_router.route,
                    log=self._log,
                )
                self._adapters.append(adapter)
                adapter.start()
                self._log("GATEWAY: Front Telegram adapter started.")
            except Exception as e:
                self._log(f"GATEWAY: Front Telegram init error — {e}")
        else:
            self._log("GATEWAY: No front_bot_token found — skipping front Telegram bot.")

        backend_token = self._config.get("backend_bot_token", "")
        if backend_token:
            try:
                self._backend_adapter = TelegramAdapter(
                    token=backend_token,
                    backend_mode=True,
                    log=self._log,
                )
                self._adapters.append(self._backend_adapter)
                self._backend_adapter.start()
                self._log("GATEWAY: Backend Telegram adapter started.")
            except Exception as e:
                self._log(f"GATEWAY: Backend Telegram init error — {e}")
                self._backend_adapter = None
        else:
            self._backend_adapter = None
            self._log("GATEWAY: No backend_bot_token found — skipping backend log bot.")

    def stop(self):
        self._running = False
        for adapter in self._adapters:
            try:
                adapter.stop()
            except Exception:
                pass
        self._adapters.clear()

    def send_log(self, text: str) -> None:
        """Forward a terminal log line to the backend Telegram bot."""
        if self._backend_adapter:
            try:
                self._backend_adapter.send_log(text)
            except Exception as e:
                self._log(f"GATEWAY: Backend send_log error — {e}")

    def route_response(self, text: str, source: str) -> None:
        """
        Route an assistant response back to the originating platform adapter.
        Called by Orthos._deliver_response for each response turn.
        """
        for adapter in self._adapters:
            if hasattr(adapter, "send_response"):
                try:
                    adapter.send_response(text, source)
                except Exception as e:
                    self._log(f"GATEWAY: Adapter send error — {e}")
