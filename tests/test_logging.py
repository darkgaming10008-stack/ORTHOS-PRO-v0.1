import io
import sys
import builtins
from unittest.mock import patch, MagicMock

import pytest
from loguru import logger

from core.logging_setup import setup_logging, get_logger, LogCapture, patch_print, catch_exceptions


# ── setup_logging ────────────────────────────────────────────────────────────


class TestSetupLogging:
    def test_adds_handlers(self):
        with patch("core.logging_setup.logger.add") as mock_add, \
             patch("core.logging_setup.logger.remove") as mock_remove:
            setup_logging(console_level="DEBUG", file_level="DEBUG")
            assert mock_remove.called
            # console + debug file + error file = 3 handlers
            assert mock_add.call_count == 3

    def test_removes_default_handler(self):
        with patch("core.logging_setup.logger.add"), \
             patch("core.logging_setup.logger.remove") as mock_remove:
            setup_logging()
            mock_remove.assert_called_once_with()


# ── get_logger ───────────────────────────────────────────────────────────────


class TestGetLogger:
    def test_returns_bound_logger(self):
        bound = get_logger("test_module")
        assert bound is not None


# ── LogCapture ───────────────────────────────────────────────────────────────


class TestLogCapture:
    def test_captures_stdout(self):
        cap = io.StringIO()
        with patch("sys.stdout", cap):
            print("hello from capture")
        assert "hello from capture" in cap.getvalue()

    def test_context_manager_restores_stdout(self):
        original = sys.stdout
        with LogCapture():
            sys.stdout.write("test")
        assert sys.stdout is original

    def test_logs_captured_content(self):
        with patch.object(logger, "opt", return_value=logger) as mock_opt:
            with patch.object(logger, "log") as mock_log:
                with LogCapture(level="INFO") as capture:
                    capture.write("captured line")
                mock_log.assert_called_once()


# ── patch_print ──────────────────────────────────────────────────────────────


class TestPatchPrint:
    def test_replaces_builtin_print(self):
        patch_print()
        assert builtins.print is not builtins._original_print
        # restore
        builtins.print = builtins._original_print

    def test_original_preserved(self):
        orig = builtins.print
        patch_print()
        assert builtins._original_print is orig
        builtins.print = builtins._original_print

    def test_patched_print_goes_to_loguru(self):
        patch_print()
        with patch.object(logger, "opt", return_value=logger) as mock_opt:
            with patch.object(logger, "info") as mock_info:
                builtins.print("log message")
                mock_info.assert_called_once_with("log message")
        builtins.print = builtins._original_print


# ── catch_exceptions ─────────────────────────────────────────────────────────


class TestCatchExceptions:
    def test_catches_and_logs(self):
        test_logger = get_logger("test_catch")
        with patch.object(test_logger, "exception") as mock_exc:

            @catch_exceptions(test_logger)
            def failing():
                raise ValueError("boom")

            result = failing()
            assert result is None  # default
            mock_exc.assert_called_once()

    def test_returns_default(self):
        test_logger = get_logger("test_default")

        @catch_exceptions(test_logger, default=42)
        def failing():
            raise ValueError("boom")

        result = failing()
        assert result == 42

    def test_reraise_when_set(self):
        test_logger = get_logger("test_reraise")

        @catch_exceptions(test_logger, reraise=True)
        def failing():
            raise ValueError("should reraise")

        with pytest.raises(ValueError, match="should reraise"):
            failing()

    def test_passes_through_success(self):
        test_logger = get_logger("test_success")

        @catch_exceptions(test_logger)
        def working() -> str:
            return "ok"

        assert working() == "ok"

    def test_async_catches_exception(self):
        test_logger = get_logger("test_async")

        @catch_exceptions(test_logger, default="fallback")
        async def failing_async():
            raise RuntimeError("async fail")

        import asyncio
        result = asyncio.run(failing_async())
        assert result == "fallback"

    def test_async_reraise(self):
        test_logger = get_logger("test_async_reraise")

        @catch_exceptions(test_logger, reraise=True)
        async def failing_async():
            raise RuntimeError("async boom")

        import asyncio
        with pytest.raises(RuntimeError, match="async boom"):
            asyncio.run(failing_async())
