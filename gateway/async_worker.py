"""
Orthos — Async Worker

Manages a single asyncio event loop on a dedicated daemon thread.
Provides a singleton accessor for use across the gateway module.
"""

from __future__ import annotations

import asyncio
import threading


class AsyncWorker:
    """Single event loop on a dedicated daemon thread."""

    def __init__(self):
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self):
        if self._running:
            return
        self._loop = asyncio.new_event_loop()
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            name="async-worker",
            daemon=True,
        )
        self._thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro):
        """Schedule *coro* on the worker loop (fire-and-forget).

        Returns a concurrent.futures.Future.
        """
        if not self._running or self._loop is None:
            raise RuntimeError("AsyncWorker is not running")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def run_sync(self, coro, timeout=30):
        """Schedule *coro* and block the calling thread for the result."""
        future = self.run(coro)
        return future.result(timeout=timeout)

    def stop(self, timeout=5):
        if not self._running:
            return
        self._running = False
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def is_running(self):
        return self._running

    @property
    def loop(self):
        return self._loop


_worker: AsyncWorker | None = None
_lock = threading.Lock()


def get_async_worker() -> AsyncWorker:
    global _worker
    if _worker is None:
        with _lock:
            if _worker is None:
                _worker = AsyncWorker()
    if not _worker.is_running():
        _worker.start()
    return _worker
