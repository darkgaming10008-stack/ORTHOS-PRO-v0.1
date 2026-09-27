"""
Production-ready logging system using loguru.

Provides console logging, rotating file handlers, stdout capture,
print() replacement, and an exception-catching decorator.
"""

import io
import functools
import sys
from pathlib import Path
from typing import Callable, TextIO

from loguru import logger

_console_sink_id: int | None = None


def _log_dir() -> Path:
    """Return the project-level ``logs`` directory, creating it if needed."""
    # This module lives at core/logging_setup.py → project root is two levels up.
    base = Path(__file__).resolve().parent.parent
    path = base / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(
    *,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    """Configure the root loguru logger with three handlers.

    Parameters
    ----------
    console_level
        Minimum severity for the coloured console handler.
    file_level
        Minimum severity for the rotating debug log file.
    rotation
        Maximum size before a new log file is started (e.g. ``"10 MB"``).
    retention
        Maximum age before a log file is cleaned up (e.g. ``"7 days"``).
    """
    # Remove the default ``sys.stderr`` handler that loguru installs implicitly.
    logger.remove()

    log_path = _log_dir()

    # ── Console handler (stdout, coloured) ──────────────────────────────
    global _console_sink_id
    _console_sink_id = logger.add(
        sys.stdout,
        level=console_level,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        enqueue=True,
    )

    # ── Rotating debug file ─────────────────────────────────────────────
    logger.add(
        str(log_path / "debug_{time:YYYY-MM-DD}.log"),
        level=file_level,
        rotation=rotation,
        retention=retention,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{name}:{function}:{line} | {message}"
        ),
        enqueue=True,
        encoding="utf-8",
    )

    # ── Error-only file with detailed format ────────────────────────────
    logger.add(
        str(log_path / "error_{time:YYYY-MM-DD}.log"),
        level="ERROR",
        rotation=rotation,
        retention=retention,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{name}:{function}:{line} | "
            "{message}\n{extra[exception] if 'exception' in extra else ''}"
        ),
        enqueue=True,
        encoding="utf-8",
        backtrace=True,
        diagnose=True,
    )


def get_logger(name: str) -> "loguru.Logger":
    """Return a loguru logger bound with the given *name*.

    Parameters
    ----------
    name
        Usually ``__name__`` from the calling module.

    Returns
    -------
    loguru.Logger
        A child logger whose output includes *name* in every record.
    """
    return logger.bind(name=name)


class LogCapture:
    """Redirect Python's ``sys.stdout`` to the loguru logging system.

    Every line written to the captured stream is logged at the *level*
    passed at construction time (default ``"INFO"``).

    Use as a context manager::

        with LogCapture():
            print("This goes to loguru instead of the terminal.")
    """

    def __init__(self, level: str = "INFO") -> None:
        self._level = level
        self._capture: io.StringIO = io.StringIO()

    def __enter__(self) -> io.StringIO:
        self._stdout: TextIO = sys.stdout
        sys.stdout = self._capture
        return self._capture

    def __exit__(self, *exc: object) -> None:
        sys.stdout = self._stdout
        written = self._capture.getvalue()
        if written:
            logger.opt(depth=1).log(self._level, written.rstrip("\n"))
        self._capture.close()


def patch_print() -> None:
    """Replace the built-in ``print`` with a function that delegates to loguru.

    This is useful for legacy code or third-party libraries that use raw
    ``print()`` calls — their output will appear in the logging pipeline.

    The original ``print`` is preserved as ``builtins._original_print``.
    """
    import builtins

    builtins._original_print = builtins.print

    def _loguru_print(*args: object, **kwargs: object) -> None:
        """Wrapper that sends ``print`` output through loguru."""
        # Flush the message via loguru as a single INFO record.
        # ``kwargs`` such as ``file``, ``flush`` and ``end`` are ignored.
        message = " ".join(str(a) for a in args)
        logger.opt(depth=1).info(message)

    builtins.print = _loguru_print  # type: ignore[assignment]


def catch_exceptions(
    logger: "loguru.Logger",
    *,
    default: object = None,
    reraise: bool = False,
) -> Callable:
    """Decorate an async or sync function to catch and log unhandled exceptions.

    Parameters
    ----------
    logger
        A loguru logger instance (e.g. returned by :func:`get_logger`).
    default
        Value to return when an exception is caught and *reraise* is ``False``.
    reraise
        If ``True`` the exception is logged **and** re-raised.

    Examples
    --------
    ::

        @catch_exceptions(get_logger(__name__))
        def risky() -> int:
            return 1 // 0

    ::

        @catch_exceptions(get_logger(__name__), reraise=True)
        async def failing() -> None:
            await some_coro()  # raises
    """

    def _decorator(func: Callable) -> Callable:
        import asyncio
        import inspect

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def _async_wrapper(*args: object, **kwargs: object) -> object:
                try:
                    return await func(*args, **kwargs)
                except Exception:
                    logger.exception(f"Unhandled exception in {func.__name__}")
                    if reraise:
                        raise
                    return default

            return _async_wrapper

        @functools.wraps(func)
        def _sync_wrapper(*args: object, **kwargs: object) -> object:
            try:
                return func(*args, **kwargs)
            except Exception:
                logger.exception(f"Unhandled exception in {func.__name__}")
                if reraise:
                    raise
                return default

        return _sync_wrapper

    return _decorator


def rehook_console_sink(level: str = "DEBUG") -> None:
    """Re-attach the loguru console sink to the current ``sys.stdout``.

    Call this *after* ``gateway.log_capture.LogCapture`` (or any other
    wrapper) has replaced ``sys.stdout`` so that loguru messages are
    forwarded through the wrapper (e.g. to Telegram).
    """
    global _console_sink_id
    if _console_sink_id is not None:
        logger.remove(_console_sink_id)
    _console_sink_id = logger.add(
        sys.stdout,
        level=level,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        enqueue=True,
    )
