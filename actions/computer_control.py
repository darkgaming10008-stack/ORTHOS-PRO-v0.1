#computer_control.py
import base64
import io
import json
import re
import string
import subprocess
import sys
import time
import random
from pathlib import Path

import requests

_HTTP = requests.Session()

# Moondream VLM via Ollama (local vision AI)
from actions.moondream_llama import get_detector as _get_moondream_detector

try:
    import pyautogui
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE    = 0.05
    _PYAUTOGUI = True
except ImportError:
    _PYAUTOGUI = False

try:
    import pyperclip
    _PYPERCLIP = True
except ImportError:
    _PYPERCLIP = False

def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


_BASE         = _base_dir()
_CONFIG_PATH  = _BASE / "config" / "api_keys.json"

def _load_config() -> dict:
    try:
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _get_os() -> str:
    import platform
    s = platform.system().lower()
    if s == "darwin":
        return "mac"
    return s  # "windows" or "linux"


def _get_screenshot_quality(target: str = "llm") -> int:
    cfg = _load_config()
    key = "screenshot_quality_moondream" if target == "moondream" else "screenshot_quality_llm"
    return int(cfg.get(key, 85 if target == "moondream" else 65))


def _get_llm_settings() -> tuple[str, str]:
    cfg = _load_config()
    url = cfg.get("llm_url", "http://localhost:11434").rstrip("/")
    model = cfg.get("llm_model", "gemma4:31b-cloud")
    return url, model


def _call_ollama_vision(image: bytes, prompt: str, system: str | None = None, timeout: int = 60) -> str:
    """Send an image + prompt to Ollama vision model directly."""
    url, model = _get_llm_settings()
    b64 = base64.b64encode(image).decode("ascii")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt, "images": [b64]})
    try:
        resp = _HTTP.post(
            f"{url}/api/chat",
            json={"model": model, "messages": messages, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return (data.get("message", {}).get("content") or "").strip()
    except Exception as e:
        return f"Ollama vision call failed: {e}"


_SAFE_SCREENSHOT_ROOTS = (
    Path.home(),
)

def _safe_screenshot_path(requested: str | None) -> Path:
    fallback = Path.home() / "Desktop" / "orthos_screenshot.png"
    if not requested:
        return fallback
    try:
        p = Path(requested).expanduser().resolve()
        for root in _SAFE_SCREENSHOT_ROOTS:
            if p.is_relative_to(root.resolve()):
                p.parent.mkdir(parents=True, exist_ok=True)
                return p
    except Exception:
        pass
    return fallback

def _require_pyautogui():
    if not _PYAUTOGUI:
        raise RuntimeError("PyAutoGUI not installed. Run: pip install pyautogui")

_FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Drew", "Quinn",
    "Avery", "Blake", "Cameron", "Dakota", "Emerson", "Finley", "Harper",
]
_LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Wilson", "Moore", "Taylor", "Anderson", "Thomas", "Jackson",
]
_DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "proton.me", "mail.com"]


def _random_data(data_type: str) -> str:
    dt = data_type.lower().strip()

    if dt == "first_name":
        return random.choice(_FIRST_NAMES)

    if dt == "last_name":
        return random.choice(_LAST_NAMES)

    if dt == "name":
        return f"{random.choice(_FIRST_NAMES)} {random.choice(_LAST_NAMES)}"

    if dt == "email":
        first = random.choice(_FIRST_NAMES).lower()
        last  = random.choice(_LAST_NAMES).lower()
        num   = random.randint(10, 999)
        return f"{first}.{last}{num}@{random.choice(_DOMAINS)}"

    if dt == "username":
        return f"{random.choice(_FIRST_NAMES).lower()}{random.randint(100, 9999)}"

    if dt == "password":
        chars = string.ascii_letters + string.digits + "!@#$%"
        raw   = (
            random.choice(string.ascii_uppercase)
            + random.choice(string.digits)
            + random.choice("!@#$%")
            + "".join(random.choices(chars, k=9))
        )
        return "".join(random.sample(raw, len(raw)))

    if dt == "phone":
        return f"+1{random.randint(200,999)}{random.randint(1_000_000, 9_999_999)}"

    if dt == "birthday":
        y = random.randint(1980, 2000)
        m = random.randint(1, 12)
        d = random.randint(1, 28)
        return f"{m:02d}/{d:02d}/{y}"

    if dt == "address":
        num    = random.randint(100, 9999)
        street = random.choice(["Main St", "Oak Ave", "Park Blvd", "Elm St", "Cedar Ln"])
        return f"{num} {street}"

    if dt == "zip_code":
        return str(random.randint(10000, 99999))

    if dt == "city":
        return random.choice(["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"])

    return f"random_{data_type}_{random.randint(1000, 9999)}"

def _user_profile() -> dict:
    """Read identity fields from long-term memory (SQLite memory_items)."""
    try:
        from memory.conversation_db import get_memory_items as _gmi
        items = _gmi(type="identity", limit=200)
        return {item.get("key", "?"): item.get("content", "") for item in items}
    except Exception:
        return {}

def _type(text: str, interval: float = 0.03) -> str:
    _require_pyautogui()
    time.sleep(0.3)
    pyautogui.typewrite(text, interval=interval)
    return f"Typed: {text[:60]}{'…' if len(text) > 60 else ''}"


def _smart_type(text: str, clear_first: bool = True) -> str:
    _require_pyautogui()
    if clear_first:
        _clear_field()
        time.sleep(0.1)

    if len(text) > 20 and _PYPERCLIP:
        pyperclip.copy(text)
        time.sleep(0.1)
        pyautogui.hotkey("ctrl", "v")
        return f"Smart-typed (clipboard): {text[:60]}{'…' if len(text) > 60 else ''}"

    pyautogui.typewrite(text, interval=0.04)
    return f"Smart-typed: {text[:60]}{'…' if len(text) > 60 else ''}"


def _click(x=None, y=None, button: str = "left", clicks: int = 1) -> str:
    _require_pyautogui()
    if x is not None and y is not None:
        pyautogui.click(x, y, button=button, clicks=clicks)
        return f"{'Double-c' if clicks == 2 else 'C'}licked ({x}, {y}) [{button}]"
    pyautogui.click(button=button, clicks=clicks)
    return f"Clicked at current position [{button}]"


def _hotkey(*keys) -> str:
    _require_pyautogui()
    pyautogui.hotkey(*keys)
    return f"Hotkey: {'+'.join(keys)}"


def _press(key: str) -> str:
    _require_pyautogui()
    pyautogui.press(key)
    return f"Pressed: {key}"


def _scroll(direction: str = "down", amount: int = 3) -> str:
    _require_pyautogui()
    vertical   = direction in ("up", "down")
    clicks     = amount if direction in ("up", "right") else -amount
    pyautogui.scroll(clicks) if vertical else pyautogui.hscroll(clicks)
    return f"Scrolled {direction} ×{amount}"


def _move(x: int, y: int, duration: float = 0.3) -> str:
    _require_pyautogui()
    pyautogui.moveTo(x, y, duration=duration)
    return f"Mouse → ({x}, {y})"


def _annotate_screenshot(img, cursor_x: int, cursor_y: int) -> bytes:
    try:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        r = 12
        draw.ellipse([cursor_x - r, cursor_y - r, cursor_x + r, cursor_y + r], outline="red", width=3)
        draw.line([cursor_x - r - 6, cursor_y, cursor_x + r + 6, cursor_y], fill="red", width=2)
        draw.line([cursor_x, cursor_y - r - 6, cursor_x, cursor_y + r + 6], fill="red", width=2)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        print(f"[ComputerControl] ⚠️ Annotation failed: {e}")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


def _drag(x1: int, y1: int, x2: int, y2: int, duration: float = 0.5) -> str:
    _require_pyautogui()
    pyautogui.moveTo(x1, y1, duration=0.2)
    pyautogui.dragTo(x2, y2, duration=duration, button="left")
    return f"Dragged ({x1},{y1}) → ({x2},{y2})"


def _clipboard_get() -> str:
    if _PYPERCLIP:
        return pyperclip.paste()
    _hotkey("ctrl", "c")
    time.sleep(0.2)
    return "(copied — pyperclip unavailable for read)"


def _clipboard_paste(text: str) -> str:
    if _PYPERCLIP:
        pyperclip.copy(text)
        time.sleep(0.1)
        _require_pyautogui()
        pyautogui.hotkey("ctrl", "v")
        return f"Pasted: {text[:60]}{'…' if len(text) > 60 else ''}"
    return "pyperclip not available"


def _screenshot(save_path: str | None = None) -> str:
    _require_pyautogui()
    path = _safe_screenshot_path(save_path)
    img  = pyautogui.screenshot()
    cx, cy = pyautogui.position()
    _annotate_screenshot(img, int(cx), int(cy))
    img.save(str(path))
    return f"Screenshot saved: {path}"


def _clear_field() -> str:
    _require_pyautogui()
    pyautogui.hotkey("ctrl", "a")
    time.sleep(0.1)
    pyautogui.press("delete")
    return "Field cleared"

def _focus_window(title: str) -> str:
    os_name = _get_os()

    if os_name == "windows":
        try:
            script = f'(New-Object -ComObject WScript.Shell).AppActivate("{title}")'
            subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                capture_output=True, timeout=5,
            )
            time.sleep(0.3)
            return f"Focused window: {title}"
        except Exception as e:
            return f"focus_window (Windows) failed: {e}"

    if os_name == "mac":
        script = (
            f'tell application "System Events" to '
            f'set frontmost of (first process whose name contains "{title}") to true'
        )
        try:
            subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, timeout=5,
            )
            time.sleep(0.3)
            return f"Focused window: {title}"
        except Exception as e:
            return f"focus_window (macOS) failed: {e}"

    if os_name == "linux":
        try:
            result = subprocess.run(
                ["wmctrl", "-a", title],
                capture_output=True, timeout=5,
            )
            if result.returncode == 0:
                time.sleep(0.3)
                return f"Focused window: {title}"
        except FileNotFoundError:
            pass
        try:
            result = subprocess.run(
                ["xdotool", "search", "--name", title, "windowactivate"],
                capture_output=True, timeout=5,
            )
            time.sleep(0.3)
            return f"Focused window: {title}"
        except FileNotFoundError:
            return "focus_window (Linux) requires wmctrl or xdotool"
        except Exception as e:
            return f"focus_window (Linux) failed: {e}"

    return f"focus_window: unknown OS '{os_name}'"

def _clamp_pixel(x: int, y: int, max_w: int, max_h: int, margin: int = 10) -> tuple[int, int]:
    return max(margin, min(x, max_w - margin)), max(margin, min(y, max_h - margin))


def _screen_find(description: str) -> tuple[int, int] | None:
    try:
        _require_pyautogui()
        img = pyautogui.screenshot()
        cx, cy = pyautogui.position()
        _annotate_screenshot(img, int(cx), int(cy))
        sw, sh = img.size
        _PAD = 10

        # Compress to JPEG at Moondream quality (no lossless PNG)
        from PIL import Image as _PIL_Image
        _md_buf = io.BytesIO()
        img.save(_md_buf, format="JPEG", quality=_get_screenshot_quality("moondream"))
        img = _PIL_Image.open(io.BytesIO(_md_buf.getvalue()))

        moondream = _get_moondream_detector()
        if not moondream.loaded:
            print(f"[ComputerControl] Moondream not available: {moondream.error}")
            return None

        coords = moondream.locate(img, description)
        if coords is None:
            print(f"[ComputerControl] Moondream ✗ '{description}' not found")
            return None

        abs_x, abs_y = coords
        abs_x, abs_y = _clamp_pixel(abs_x, abs_y, sw, sh, _PAD)
        print(f"[ComputerControl] Moondream → '{description}' at ({abs_x}, {abs_y})")
        return (abs_x, abs_y)

    except Exception as e:
        print(f"[ComputerControl] ⚠️ screen_find failed: {e}")

    return None


def _screen_describe(description: str) -> str:
    try:
        _require_pyautogui()
        img = pyautogui.screenshot()

        if description and description.lower() not in ("screen", "whole screen", "full screen", ""):
            try:
                coords = _screen_find(description)
                if coords:
                    cx, cy = coords
                    crop = 64
                    w, h = img.size
                    box = (
                        max(0, cx - crop), max(0, cy - crop),
                        min(w, cx + crop), min(h, cy + crop),
                    )
                    img = img.crop(box)
            except Exception:
                pass

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=_get_screenshot_quality("llm"))
        img_bytes = buf.getvalue()

        result = _call_ollama_vision(img_bytes, "Describe what you see in this screenshot briefly.", timeout=60)
        print(f"[ComputerControl] Ollama vision describe: {result[:200]}...")
        return result
    except Exception as e:
        print(f"[ComputerControl] _screen_describe error: {e}")
        return f"screen_describe failed: {e}"


def _screen_get_text() -> str:
    try:
        _require_pyautogui()
        img = pyautogui.screenshot()

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=_get_screenshot_quality("llm"))
        img_bytes = buf.getvalue()

        result = _call_ollama_vision(img_bytes, "Read all visible text in this screenshot. Return only the text content line by line, nothing else.", timeout=60)
        print(f"[ComputerControl] Ollama vision text: {len(result)} chars")
        return result or "(no visible text)"
    except Exception as e:
        print(f"[ComputerControl] _screen_get_text error: {e}")
        return f"screen_get_text failed: {e}"


def computer_control(
    parameters: dict,
    response=None,
    player=None,
    session_memory=None,
) -> str:
    """
    Dispatch table for all computer control actions.

    parameters keys (all optional unless noted):
      action        : (required) one of the actions listed below
      text          : text to type or paste
      x, y          : screen coordinates
      button        : 'left' | 'right' (default: left)
      keys          : hotkey string, e.g. 'ctrl+c'
      key           : single key name, e.g. 'enter'
      direction     : 'up' | 'down' | 'left' | 'right'
      amount        : scroll amount (default: 3)
      seconds       : wait duration
      title         : window title fragment for focus_window
      description   : natural-language element description for screen_find/click
      type          : data type for random_data
      field         : memory field name for user_data
      clear_first   : bool, clear field before typing (default: true)
      path          : save path for screenshot (must be inside home dir)

    Actions:
      type            — type text at cursor
      smart_type      — clear field + type (clipboard-backed)
      click           — left click
      double_click    — double left click
      right_click     — right click
      move            — move mouse to (x, y)
      drag            — click-drag between two points
      hotkey          — key combination
      press           — single key
      scroll          — scroll the wheel
      copy            — read clipboard
      paste           — write + paste clipboard
      screenshot      — capture screen (safe path only)
      wait            — sleep N seconds
      clear_field     — select-all + delete
      focus_window    — bring window to foreground
      screen_find     — AI element finder (returns x,y)
      screen_click    — AI element finder + click
      screen_describe — AI screen description
      screen_get_text — AI screen text extraction
      random_data     — generate fake form data
      user_data       — pull real data from memory
    """
    params = parameters or {}
    action = params.get("action", "").lower().strip()

    if not action:
        return "No action specified for computer_control."

    if player:
        player.write_log(f"[Computer] {action}")

    print(f"[ComputerControl] ▶ {action}  {params}")

    try:

        if action == "type":
            return _type(params.get("text", ""))

        if action == "smart_type":
            return _smart_type(
                params.get("text", ""),
                clear_first=params.get("clear_first", True),
            )

        if action in ("click", "left_click"):
            return _click(params.get("x"), params.get("y"), "left", 1)

        if action == "double_click":
            return _click(params.get("x"), params.get("y"), "left", 2)

        if action == "right_click":
            return _click(params.get("x"), params.get("y"), "right", 1)

        if action == "move":
            return _move(int(params.get("x", 0)), int(params.get("y", 0)))

        if action == "drag":
            return _drag(
                int(params.get("x1", 0)), int(params.get("y1", 0)),
                int(params.get("x2", 0)), int(params.get("y2", 0)),
            )

        if action == "hotkey":
            raw  = params.get("keys", "")
            keys = [k.strip() for k in raw.split("+")] if isinstance(raw, str) else raw
            return _hotkey(*keys)

        if action == "press":
            return _press(params.get("key", "enter"))

        if action == "scroll":
            return _scroll(
                direction=params.get("direction", "down"),
                amount=int(params.get("amount", 3)),
            )

        if action == "copy":
            return _clipboard_get()

        if action == "paste":
            return _clipboard_paste(params.get("text", ""))

        if action == "screenshot":
            return _screenshot(params.get("path"))

        if action == "screen_find":
            desc   = params.get("description") or params.get("text", "")
            coords = _screen_find(desc)
            return f"{coords[0]},{coords[1]}" if coords else "NOT_FOUND"

        if action == "screen_click":
            desc   = params.get("description") or params.get("text", "")
            coords = _screen_find(desc)
            if coords:
                time.sleep(0.2)
                _click(x=coords[0], y=coords[1])
                return f"Clicked '{desc}' at {coords}"
            return f"Element not found on screen: '{desc}'"

        if action == "screen_describe":
            desc = params.get("description") or params.get("text", "")
            return _screen_describe(desc)

        if action == "screen_get_text":
            return _screen_get_text()

        if action == "wait":
            secs = float(params.get("seconds", 1.0))
            secs = min(secs, 30.0)
            time.sleep(secs)
            return f"Waited {secs}s"

        if action == "clear_field":
            return _clear_field()

        if action == "focus_window":
            return _focus_window(params.get("title", ""))

        if action == "random_data":
            dt     = params.get("type", "name")
            result = _random_data(dt)
            print(f"[ComputerControl] 🎲 random {dt} → {result}")
            return result

        if action == "user_data":
            field   = params.get("field", "name")
            profile = _user_profile()
            value   = profile.get(field, "")
            if not value:
                value = _random_data(field)
                print(f"[ComputerControl] ⚠️ No '{field}' in memory, using random: {value}")
            return value

        return f"Unknown action: '{action}'"

    except Exception as e:
        print(f"[ComputerControl] ❌ {action}: {e}")
        return f"computer_control '{action}' failed: {e}"