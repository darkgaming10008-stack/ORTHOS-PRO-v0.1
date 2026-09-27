import subprocess
import sys
import time
from pathlib import Path

MAX_OUTPUT_CHARS = 5000
MAX_TIMEOUT = 120


def run_terminal(parameters: dict, player=None) -> str:
    command = (parameters.get("command") or "").strip()
    timeout = min(int(parameters.get("timeout", 30)), MAX_TIMEOUT)
    cwd = parameters.get("cwd") or None

    if not command:
        return "No command provided."

    if player:
        player.write_log(f"SYS: ▶ terminal — {command[:80]}")

    if sys.platform == "win32":
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True,
            cwd=cwd,
        )
    else:
        proc = subprocess.Popen(
            ["bash", "-c", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd,
        )

    try:
        output_parts = []
        start = time.time()
        for raw in iter(proc.stdout.readline, b""):
            text = raw.decode("utf-8", errors="replace")
            output_parts.append(text)
            if time.time() - start > timeout:
                proc.kill()
                leftover = "".join(output_parts)
                if len(leftover) > MAX_OUTPUT_CHARS:
                    leftover = leftover[:MAX_OUTPUT_CHARS] + "\n... [truncated]"
                return f"[Timed out after {timeout}s]\n{leftover}" if leftover else f"[Timed out after {timeout}s]"
        proc.wait()
        output = "".join(output_parts)
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + "\n... [truncated]"
        return output.strip() or "(no output)"
    except FileNotFoundError as e:
        return f"Shell not found: {e}"
    except Exception as e:
        return f"Error: {e}"
