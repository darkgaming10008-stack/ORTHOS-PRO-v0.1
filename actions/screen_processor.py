"""
Orthos — Screen / Camera Processor
Replaces Gemini Live vision session with a direct Ollama vision-model call.
The analysis text is returned (and optionally spoken via the `speak` callback).
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Callable

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

try:
    import mss
    import mss.tools
    _MSS = True
except ImportError:
    _MSS = False

try:
    import PIL.Image
    _PIL = True
except ImportError:
    _PIL = False

import platform


def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


_BASE        = _base_dir()
_CONFIG_PATH = _BASE / "config" / "api_keys.json"


import pyautogui as _pyautogui

from actions.moondream_llama import get_detector as _get_moondream_detector


# ── GDI screen capture with real cursor rendering ──────────────────────
try:
    import ctypes
    _GDI_OK = True

    class _POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    class _CURSORINFO(ctypes.Structure):
        _fields_ = [
            ("cbSize", ctypes.c_ulong),
            ("flags", ctypes.c_ulong),
            ("hCursor", ctypes.c_void_p),
            ("ptScreenPos", _POINT),
        ]

    class _BMI_HEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", ctypes.c_ulong),
            ("biWidth", ctypes.c_long),
            ("biHeight", ctypes.c_long),
            ("biPlanes", ctypes.c_ushort),
            ("biBitCount", ctypes.c_ushort),
            ("biCompression", ctypes.c_ulong),
            ("biSizeImage", ctypes.c_ulong),
            ("biXPelsPerMeter", ctypes.c_long),
            ("biYPelsPerMeter", ctypes.c_long),
            ("biClrUsed", ctypes.c_ulong),
            ("biClrImportant", ctypes.c_ulong),
        ]

    class _BITMAPINFO(ctypes.Structure):
        _fields_ = [("bmiHeader", _BMI_HEADER)]

    _user32 = ctypes.windll.user32
    _gdi32  = ctypes.windll.gdi32

    # Set proper 64-bit argtypes/restype to prevent "int too long to convert"
    _user32.GetDC.argtypes = [ctypes.c_void_p]
    _user32.GetDC.restype = ctypes.c_void_p
    _user32.ReleaseDC.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _user32.ReleaseDC.restype = ctypes.c_int
    _user32.GetCursorInfo.argtypes = [ctypes.c_void_p]
    _user32.GetCursorInfo.restype = ctypes.c_bool
    _user32.DrawIconEx.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    _user32.DrawIconEx.restype = ctypes.c_bool

    _gdi32.CreateCompatibleDC.argtypes = [ctypes.c_void_p]
    _gdi32.CreateCompatibleDC.restype = ctypes.c_void_p
    _gdi32.CreateCompatibleBitmap.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    _gdi32.CreateCompatibleBitmap.restype = ctypes.c_void_p
    _gdi32.SelectObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _gdi32.SelectObject.restype = ctypes.c_void_p
    _gdi32.BitBlt.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                               ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                               ctypes.c_int, ctypes.c_int, ctypes.c_int]
    _gdi32.BitBlt.restype = ctypes.c_bool
    _gdi32.GetDIBits.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
                                  ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_uint]
    _gdi32.GetDIBits.restype = ctypes.c_int
    _gdi32.DeleteDC.argtypes = [ctypes.c_void_p]
    _gdi32.DeleteDC.restype = ctypes.c_int
    _gdi32.DeleteObject.argtypes = [ctypes.c_void_p]
    _gdi32.DeleteObject.restype = ctypes.c_int
except Exception:
    _GDI_OK = False




def _load_config() -> dict:
    try:
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _get_screenshot_quality(target: str = "llm") -> int:
    cfg = _load_config()
    key = "screenshot_quality_moondream" if target == "moondream" else "screenshot_quality_llm"
    return int(cfg.get(key, 85 if target == "moondream" else 65))


def _get_screenshot_format() -> str:
    """Configured capture format: 'jpeg' (default, fast) or 'png' (lossless).

    PNG is ~5-10x bigger and slower to upload — the LLM also takes longer to
    process a larger image. Only enable when text/detail fidelity matters.
    """
    cfg = _load_config()
    return str(cfg.get("screenshot_format", "jpeg")).strip().lower()


def _save_screenshot(img, out: io.BytesIO) -> None:
    """Encode the captured PIL image in the configured format/quality."""
    fmt = _get_screenshot_format()
    if fmt == "png":
        img.save(out, format="PNG", optimize=False)   # lossless, full quality
    else:
        img.save(out, format="JPEG", quality=_get_screenshot_quality("llm"))


def _compress_image(image_bytes: bytes, quality: int) -> bytes:
    img = PIL.Image.open(io.BytesIO(image_bytes))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=quality)
    return out.getvalue()


def _save_config_key(key: str, value) -> None:
    try:
        cfg = _load_config()
        cfg[key] = value
        _CONFIG_PATH.write_text(json.dumps(cfg, indent=4), encoding="utf-8")
    except Exception as e:
        print(f"[Vision] ⚠️ Could not save config key '{key}': {e}")


def _get_os() -> str:
    s = platform.system().lower()
    if s == "darwin":  return "mac"
    if s == "windows": return "windows"
    return "linux"


# ---------------------------------------------------------------------------
# Image capture helpers (unchanged from original)
# ---------------------------------------------------------------------------

def _get_cursor() -> tuple[int, int]:
    try:
        x, y = _pyautogui.position()
        return int(x), int(y)
    except Exception:
        return 0, 0


def _pass(img_bytes: bytes, source_format: str = "PNG") -> tuple[bytes, str]:
    return img_bytes, f"image/{source_format.lower()}"


def _capture_gdi() -> bytes:
    """
    Capture screen via GDI BitBlt + overlay actual Windows cursor via DrawIconEx.
    Returns JPEG bytes.
    """
    _t0 = time.time()
    if not _GDI_OK:
        raise RuntimeError("GDI not available")

    w = _user32.GetSystemMetrics(0)
    h = _user32.GetSystemMetrics(1)

    hdc_screen = _user32.GetDC(None)
    hdc_mem    = _gdi32.CreateCompatibleDC(hdc_screen)
    hbitmap    = _gdi32.CreateCompatibleBitmap(hdc_screen, w, h)

    try:
        _gdi32.SelectObject(hdc_mem, hbitmap)
        _gdi32.BitBlt(hdc_mem, 0, 0, w, h, hdc_screen, 0, 0, 0x00CC0020)

        cursor_info = _CURSORINFO()
        cursor_info.cbSize = ctypes.sizeof(cursor_info)
        if _user32.GetCursorInfo(ctypes.byref(cursor_info)):
            if cursor_info.flags == 1:
                _user32.DrawIconEx(
                    hdc_mem,
                    cursor_info.ptScreenPos.x,
                    cursor_info.ptScreenPos.y,
                    cursor_info.hCursor,
                    0, 0, 0, None, 3
                )

        bmi = _BITMAPINFO()
        bmi.bmiHeader.biSize       = ctypes.sizeof(_BMI_HEADER)
        bmi.bmiHeader.biWidth      = w
        bmi.bmiHeader.biHeight     = -h
        bmi.bmiHeader.biPlanes     = 1
        bmi.bmiHeader.biBitCount   = 32
        bmi.bmiHeader.biCompression = 0

        stride  = w * 4
        pixels  = ctypes.create_string_buffer(h * stride)
        ret     = _gdi32.GetDIBits(hdc_mem, hbitmap, 0, h, pixels, ctypes.byref(bmi), 0)
        if not ret:
            raise RuntimeError("GetDIBits failed")

        from PIL import Image as _PIL_Image
        buf = bytes(pixels)
        img = _PIL_Image.frombytes("RGBA", (w, h), buf, "raw", "BGRA", 0, 1).convert("RGB")
        out = io.BytesIO()
        _save_screenshot(img, out)
        result = out.getvalue()
        _t1 = time.time()
        print(f"[Timing] GDI capture ({w}x{h}): {(_t1-_t0)*1000:.0f}ms -> {len(result):,} bytes")
        return result

    finally:
        _gdi32.DeleteObject(hbitmap)
        _gdi32.DeleteDC(hdc_mem)
        _user32.ReleaseDC(None, hdc_screen)


def _capture_screen() -> tuple[bytes, str]:
    _t0 = time.time()
    if not _GDI_OK:
        raise RuntimeError("GDI not available for screen capture")
    result = _pass(_capture_gdi(), "JPEG")
    print(f"[Timing] Screen capture: {(_t0-time.time())*-1000:.0f}ms")
    return result


def _capture_screen_annotated(cursor_x: int, cursor_y: int) -> tuple[bytes, str]:
    """Capture screen — real cursor is already rendered via GDI, no fake marker needed."""
    return _capture_screen()


def _capture_zoomed(cx: int, cy: int, radius: int = 350) -> tuple[bytes, str]:
    """Zoom crop around (cx, cy) — real cursor already visible via GDI, no fake marker."""
    _t0 = time.time()
    raw_bytes, mime = _capture_screen()
    try:
        import PIL.Image as _PIL_Image
        img = _PIL_Image.open(io.BytesIO(raw_bytes))
        sw, sh = img.size
        left   = max(0, cx - radius)
        top    = max(0, cy - radius)
        right  = min(sw, cx + radius)
        bottom = min(sh, cy + radius)
        crop = img.crop((left, top, right, bottom))
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=_get_screenshot_quality("llm"))
        result = buf.getvalue(), mime
        print(f"[Timing] Zoom crop ({radius}x{radius}): {(_t0-time.time())*-1000:.0f}ms → {len(result[0]):,} bytes")
        return result
    except Exception as e:
        print(f"[Vision] ⚠️ Zoom crop failed: {e}")
        return raw_bytes, mime


def _cv2_backend() -> int:
    if not _CV2:
        return 0
    os_name = _get_os()
    if os_name == "windows":
        return cv2.CAP_DSHOW
    if os_name == "mac":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_ANY


def _probe_camera(index: int, backend: int, warmup: int = 5) -> bool:
    if not _CV2:
        return False
    import numpy as np
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release(); return False
    for _ in range(warmup):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return False
    return bool(np.mean(frame) > 8)


def _detect_camera_index() -> int:
    backend = _cv2_backend()
    print("[Vision] 🔍 Auto-detecting camera…")
    for idx in range(6):
        if _probe_camera(idx, backend):
            print(f"[Vision] ✅ Camera found at index {idx}")
            _save_config_key("camera_index", idx)
            return idx
        print(f"[Vision] ⚠️ Camera index {idx}: no usable frame")
    print("[Vision] ⚠️ No camera found — defaulting to index 0")
    _save_config_key("camera_index", 0)
    return 0


def _get_camera_index() -> int:
    cfg = _load_config()
    if "camera_index" in cfg:
        return int(cfg["camera_index"])
    return _detect_camera_index()


def _capture_camera() -> tuple[bytes, str]:
    if not _CV2:
        raise RuntimeError("OpenCV (cv2) is not installed. Run: pip install opencv-python")
    _t0 = time.time()
    import numpy as np
    index   = _get_camera_index()
    backend = _cv2_backend()
    cap     = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        raise RuntimeError(f"Camera index {index} could not be opened.")
    for _ in range(10):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError("Camera returned no frame.")
    if _PIL:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(rgb)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=_get_screenshot_quality("llm"))
        result = buf.getvalue(), "image/jpeg"
    else:
        _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), _get_screenshot_quality("llm")])
        result = buf.tobytes(), "image/jpeg"
    _t1 = time.time()
    print(f"[Timing] Camera capture: {(_t1-_t0)*1000:.0f}ms → {len(result[0]):,} bytes")
    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def screen_process(
    parameters:     dict,
    response=None,
    player=None,
    session_memory=None,
    speak:          Optional[Callable[[str], None]] = None,
) -> str:
    """
    Capture screen or camera, analyse with Ollama vision model.

    Returns the analysis text (str).
    Optionally speaks via `speak` callback and logs to `player`.
    """
    _t_start = time.time()
    params    = parameters or {}
    user_text = (params.get("text") or params.get("user_text") or "").strip()
    angle     = params.get("angle", "screen").lower().strip()
    verify_x  = params.get("verify_x")
    verify_y  = params.get("verify_y")

    if not user_text:
        user_text = "What do you see? Describe briefly."

    if player:
        player.write_log(f"SYS: Vision [{angle}] — {user_text[:60]}")

    # Capture — zoomed crop for verification, full screen otherwise
    try:
        _VERIFY_KW = {"verify", "cursor", "position", "check", "precisely", "exact", "correctly", "confirm"}
        _is_verification = any(kw in user_text.lower() for kw in _VERIFY_KW)
        zoom = int(params.get("zoom", 350))

        if angle == "camera":
            _t_cap = time.time()
            image_bytes, _ = _capture_camera()
            print(f"[Vision] 📷 Camera: {len(image_bytes):,} bytes ({(_t_cap-time.time())*-1000:.0f}ms)")
        elif _is_verification or "zoom" in params:
            _t_cap = time.time()
            cx = verify_x if verify_x is not None else _get_cursor()[0]
            cy = verify_y if verify_y is not None else _get_cursor()[1]
            full_img, _ = _capture_screen_annotated(cx, cy)
            zoom_img, _ = _capture_zoomed(cx, cy, radius=zoom)
            image_bytes = [full_img, zoom_img]
            user_text = f"[Full screen + zoomed crop at ({cx},{cy})] {user_text}"
            print(f"[Vision] 🔍 Full ({len(full_img):,}) + zoomed ({len(zoom_img):,}) at ({cx}, {cy}) — {(_t_cap-time.time())*-1000:.0f}ms")
        else:
            _t_cap = time.time()
            cx = verify_x if verify_x is not None else _get_cursor()[0]
            cy = verify_y if verify_y is not None else _get_cursor()[1]
            image_bytes, _ = _capture_screen_annotated(cx, cy)
            print(f"[Vision] 🖥️  Screen+marker at ({cx}, {cy}): {len(image_bytes):,} bytes ({(_t_cap-time.time())*-1000:.0f}ms)")
    except Exception as e:
        msg = f"Capture error: {e}"
        print(f"[Vision] ❌ {msg}")
        if player: player.write_log(f"ERR: {msg}")
        return msg

    # ── Encode image(s) as base64 and return __IMG__ marker ────────────────
    _t_enc = time.time()
    quality = _get_screenshot_quality("llm")

    if isinstance(image_bytes, list):
        b64_list = []
        for img in image_bytes:
            compressed = _compress_image(img, quality)
            b64_list.append(base64.b64encode(compressed).decode("ascii"))
        b64_payload = "|".join(b64_list)
    else:
        compressed = _compress_image(image_bytes, quality)
        b64_payload = base64.b64encode(compressed).decode("ascii")

    _t_done = time.time()
    print(f"[Vision] Encoded {len(b64_payload):,} base64 chars ({(_t_done-_t_enc)*1000:.0f}ms)")

    result = f"__IMG__:{b64_payload}:__IMG__|{user_text}"
    print(f"[Timing] screen_process TOTAL: {(_t_done-_t_start)*1000:.0f}ms")

    if player:
        player.write_log("SYS: Screen captured — sending image to LLM for analysis.")

    return result


# ---------------------------------------------------------------------------
# Locate — Moondream visual element finder
# ---------------------------------------------------------------------------

def _clamp_pixel(x: int, y: int, max_w: int, max_h: int, margin: int = 10) -> tuple[int, int]:
    return max(margin, min(x, max_w - margin)), max(margin, min(y, max_h - margin))


def screen_locate(
    parameters: dict,
    player=None,
    session_memory=None,
) -> str:
    _PAD = 10

    text = (parameters.get("text") or "").strip()
    if not text:
        return "No element description provided."

    if player:
        player.write_log(f"SYS: screen_locate — '{text}'")

    try:
        cx, cy = _get_cursor()
        image_bytes, mime = _capture_screen_annotated(cx, cy)
    except Exception as e:
        return f"Screen capture error: {e}"

    img = PIL.Image.open(io.BytesIO(image_bytes))
    sw, sh = img.size

    # Re-compress at Moondream quality for better locate accuracy
    _md_buf = io.BytesIO()
    img.save(_md_buf, format="JPEG", quality=_get_screenshot_quality("moondream"))
    img = PIL.Image.open(io.BytesIO(_md_buf.getvalue()))

    moondream = _get_moondream_detector()
    if not moondream.loaded:
        return f"Moondream not available: {moondream.error}"

    coords = moondream.locate(img, text)
    if coords is None:
        result = f"Could not locate '{text}' on screen"
        print(f"[Locate] → {result}")
        if player:
            player.write_log(f"SYS: → {result}")
        return result

    abs_x, abs_y = coords
    abs_x, abs_y = _clamp_pixel(abs_x, abs_y, sw, sh, _PAD)
    result = f'"{text}" at pixel ({abs_x}, {abs_y})'
    print(f"[Locate] → {result}")
    if player:
        player.write_log(f"SYS: → {result}")
    return result
