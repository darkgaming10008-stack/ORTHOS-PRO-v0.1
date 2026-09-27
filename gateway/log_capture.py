import sys
import threading


class LogCapture:
    """Redirect sys.stdout (and stderr) to both the terminal and a Telegram backend bot."""

    def __init__(self, send_log_callback):
        self._send_log = send_log_callback
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._buffer = []
        self._lock = threading.RLock()
        self._capturing = False

    def start(self):
        if self._capturing:
            return
        self._capturing = True
        sys.stdout = self
        sys.stderr = self

    def stop(self):
        if not self._capturing:
            return
        self._capturing = False
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        with self._lock:
            remaining = "".join(self._buffer).strip()
            if remaining:
                self._send_log(remaining)
            self._buffer = []

    def write(self, text):
        with self._lock:
            self._original_stdout.write(text)
            self._buffer.append(text)
            if "\n" in text:
                parts = "".join(self._buffer).split("\n")
                for part in parts[:-1]:
                    line = part.strip("\r").rstrip()
                    if line:
                        self._send_log(line)
                self._buffer = [parts[-1]] if parts[-1] else []

    def flush(self):
        with self._lock:
            self._original_stdout.flush()
