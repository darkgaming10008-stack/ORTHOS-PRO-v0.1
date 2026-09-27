from __future__ import annotations

import base64
import io
import json
import math
import os
import platform
import random
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

from PyQt6.QtCore import (
    QEasingCurve, QEvent, QMimeData, QObject, QPointF, QRectF, QSize, Qt,
    QTimer, QUrl, QSettings, pyqtSignal,
)
from PyQt6.QtGui import (
    QBrush, QColor, QDragEnterEvent, QDropEvent, QFont, QFontDatabase,
    QKeySequence, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap,
    QRadialGradient, QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
    QMainWindow, QPushButton, QScrollArea, QSizePolicy, QTextEdit,
    QSpinBox, QVBoxLayout, QWidget, QProgressBar, QSplitter, QMenu, QStackedLayout,
)

try:
    from mcp_manager_ui import open_mcp_manager, _HAS_MCP_UI
except ImportError:
    _HAS_MCP_UI = False
    def open_mcp_manager(parent=None): pass

try:
    from memory.conversation_db import list_sessions as _db_list_sessions
    from memory.conversation_db import search_sessions as _db_search_sessions
    from memory.conversation_db import set_session_pinned as _db_set_pinned
    from memory.conversation_db import get_session as _db_get_session
    from memory.conversation_db import get_session_turn_count as _db_get_turn_count
    from memory.conversation_db import estimate_session_tokens as _db_est_tokens
    from memory.conversation_db import rename_session as _db_rename_session
except ImportError:
    _db_list_sessions = lambda *a, **kw: []
    _db_search_sessions = lambda *a, **kw: []
    _db_set_pinned = lambda *a, **kw: None
    _db_get_session = lambda *a, **kw: None
    _db_get_turn_count = lambda *a, **kw: 0
    _db_est_tokens = lambda *a, **kw: 0
    _db_rename_session = lambda *a, **kw: None

def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent

BASE_DIR   = _base_dir()
CONFIG_DIR = BASE_DIR / "config"
API_FILE   = CONFIG_DIR / "api_keys.json"

_DEFAULT_W, _DEFAULT_H = 1200, 720
_MIN_W,     _MIN_H     = 960, 580
_LEFT_W  = 220
_RIGHT_W = 400

_OS = platform.system()


# ── 2026 typography system ─────────────────────────────────────────────
# One central place for every font in the app.  Segoe UI Variable is the
# Windows 11 native UI face and resolves to Segoe UI on older systems;
# Cascadia Code is the modern terminal/code font shipped with Windows.
#
# Resolution is deliberately lazy: ``QFontDatabase`` may only be touched
# after a QGuiApplication exists, and ``ui`` is imported long before
# ``OrthosUI()`` builds one.  Each helper resolves (and caches) on its
# first call, which always happens after the QApplication is up.
_font_families_cache: list[str] | None = None
_font_name_cache: dict[str, str] = {}


def _installed_families() -> list[str]:
    global _font_families_cache
    if _font_families_cache is None:
        app = QApplication.instance()
        if app is None:
            # No QApplication yet (e.g. bare ``import ui``): defer, callers
            # fall back to hard-coded faces for this call only.
            return []
        _font_families_cache = list(QFontDatabase.families())
    return _font_families_cache


def _pick_font(candidates: list[str], fallback: str, cache_key: str) -> str:
    if cache_key in _font_name_cache:
        return _font_name_cache[cache_key]
    families = _installed_families()
    chosen = next((name for name in candidates if name in families), fallback)
    if families:
        # Only cache once real font data was available; a pre-app call must
        # not pin the fallback forever.
        _font_name_cache[cache_key] = chosen
    return chosen


def font_family_ui(display: bool = False) -> str:
    if display:
        return _pick_font(
            ["Segoe UI Variable Display", "Segoe UI Variable", "Segoe UI Semibold",
             "Segoe UI"], font_family_ui(), "ui_display"
        )
    return _pick_font(["Segoe UI Variable Text", "Segoe UI Variable", "Segoe UI"],
                      "Segoe UI", "ui")


def font_family_mono() -> str:
    return _pick_font(["Cascadia Code", "Cascadia Mono", "Consolas"],
                      "Consolas", "mono")


def font_ui(size: float = 10, bold: bool = False, display: bool = False) -> QFont:
    """App-wide proportional UI font (sizes in points, 2026 readable scale)."""
    f = QFont(font_family_ui(display), int(round(size)))
    f.setBold(bold)
    return f


def font_mono(size: float = 9.5, bold: bool = False) -> QFont:
    """App-wide monospace font for code, metrics and technical readouts."""
    f = QFont(font_family_mono(), int(round(size)))
    f.setBold(bold)
    return f


class C:
    BG        = "#00060a"
    PANEL     = "#010d14"
    PANEL2    = "#010f18"
    BORDER    = "#0d3347"
    BORDER_B  = "#1a5c7a"
    BORDER_A  = "#0f4060"
    PRI       = "#00d4ff"
    PRI_DIM   = "#007a99"
    PRI_GHO   = "#001f2e"
    ACC       = "#ff6b00"
    ACC2      = "#ffcc00"
    GREEN     = "#00ff88"
    GREEN_D   = "#00aa55"
    RED       = "#ff3355"
    MUTED_C   = "#ff3366"
    TEXT      = "#8ffcff"
    TEXT_DIM  = "#3a8a9a"
    TEXT_MED  = "#5ab8cc"
    WHITE     = "#d8f8ff"
    DARK      = "#000d14"
    BAR_BG    = "#011520"


def qcol(h: str, a: int = 255) -> QColor:
    c = QColor(h); c.setAlpha(a); return c


class _SysMetrics:
    def __init__(self):
        self.cpu  = 0.0
        self.mem  = 0.0
        self.net  = 0.0
        self.gpu  = -1.0
        self.tmp  = -1.0
        self.proc = 0
        self._lock = threading.Lock()
        self._last_net = psutil.net_io_counters()
        self._last_net_t = time.time()
        self._running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def _loop(self):
        while self._running:
            try:
                self._update()
            except Exception:
                pass
            time.sleep(1.5)

    def _update(self):
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent

        nc  = psutil.net_io_counters()
        now = time.time()
        dt  = now - self._last_net_t
        if dt > 0:
            sent = (nc.bytes_sent - self._last_net.bytes_sent) / dt
            recv = (nc.bytes_recv - self._last_net.bytes_recv) / dt
            net  = (sent + recv) / (1024 * 1024)
        else:
            net = 0.0
        self._last_net   = nc
        self._last_net_t = now

        gpu   = self._get_gpu()
        tmp   = self._get_temp()
        proc  = self._get_proc()

        with self._lock:
            self.cpu = cpu
            self.mem = mem
            self.net = net
            self.gpu = gpu
            self.tmp = tmp
            self.proc = proc

    def _get_proc(self) -> int:
        try:
            return len(psutil.pids())
        except Exception:
            return 0

    def _get_gpu(self) -> float:
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2
            )
            if r.returncode == 0:
                vals = [float(v.strip()) for v in r.stdout.strip().split("\n") if v.strip()]
                if vals:
                    return sum(vals) / len(vals)
        except Exception:
            pass

        if _OS == "Linux":
            try:
                r = subprocess.run(
                    ["rocm-smi", "--showuse", "--csv"],
                    capture_output=True, text=True, timeout=2
                )
                if r.returncode == 0:
                    for line in r.stdout.strip().split("\n"):
                        parts = line.split(",")
                        if len(parts) >= 2:
                            try:
                                return float(parts[1].strip().replace("%", ""))
                            except ValueError:
                                pass
            except Exception:
                pass

            try:
                r = subprocess.run(
                    ["intel_gpu_top", "-J", "-s", "500"],
                    capture_output=True, text=True, timeout=1
                )
                if r.returncode == 0 and "Render/3D" in r.stdout:
                    import re
                    m = re.search(r'"busy":\s*([\d.]+)', r.stdout)
                    if m:
                        return float(m.group(1))
            except Exception:
                pass

        if _OS == "Darwin":
            try:
                r = subprocess.run(
                    ["sudo", "-n", "powermetrics", "-n", "1", "-i", "500",
                     "--samplers", "gpu_power"],
                    capture_output=True, text=True, timeout=2
                )
                if r.returncode == 0 and "GPU" in r.stdout:
                    import re
                    m = re.search(r'GPU\s+Active:\s+([\d.]+)%', r.stdout)
                    if m:
                        return float(m.group(1))
            except Exception:
                pass

        return -1.0

    def _get_temp(self) -> float:
        try:
            temps = psutil.sensors_temperatures()
            candidates = ["coretemp", "k10temp", "cpu_thermal", "acpitz",
                          "cpu-thermal", "zenpower", "it8688"]
            for name in candidates:
                if name in temps:
                    entries = temps[name]
                    if entries:
                        return entries[0].current
            for entries in temps.values():
                if entries:
                    return entries[0].current
        except Exception:
            pass
        if _OS == "Darwin":
            try:
                r = subprocess.run(
                    ["osx-cpu-temp"], capture_output=True, text=True, timeout=2
                )
                if r.returncode == 0:
                    import re
                    m = re.search(r"([\d.]+)", r.stdout)
                    if m:
                        return float(m.group(1))
            except Exception:
                pass

        if _OS == "Windows":
            try:
                r = subprocess.run(
                    ["powershell", "-Command",
                     "(Get-WmiObject MSAcpi_ThermalZoneTemperature -Namespace root/wmi).CurrentTemperature"],
                    capture_output=True, text=True, timeout=3
                )
                if r.returncode == 0 and r.stdout.strip():
                    raw = float(r.stdout.strip().split("\n")[0])
                    return (raw / 10.0) - 273.15
            except Exception:
                pass

        return -1.0

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "cpu": self.cpu,
                "mem": self.mem,
                "net": self.net,
                "gpu": self.gpu,
                "tmp": self.tmp,
                "proc": self.proc,
                "boot": psutil.boot_time(),
            }


_metrics = _SysMetrics()


class HudCanvas(QWidget):
    def __init__(self, face_path: str, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.muted    = False
        self.speaking = False
        self.state    = "INITIALISING"
        # Mute arrived while a turn was playing; applied when it ends.
        self._pending_muted_state = False
        # Audio ground truth: set True by notify_audio_activity() whenever
        # a playback batch actually feeds the face. The mouth follows THIS
        # (plus a short grace window), not the state string — PROCESSING
        # arriving mid-reply (tool call) used to freeze the mouth while
        # audio was still playing.
        self._audio_active_until = 0.0

        self._tick       = 0
        self._scale      = 1.0
        self._tgt_scale  = 1.0
        self._halo       = 55.0
        self._tgt_halo   = 55.0
        self._last_t     = time.time()
        self._scan       = 0.0
        self._scan2      = 180.0
        self._rings      = [0.0, 120.0, 240.0]
        self._pulses: list[float] = [0.0, 50.0, 100.0]
        self._blink      = True
        self._blink_tick = 0
        self._particles: list[list[float]] = []
        self._face_px: QPixmap | None = None
        self._face_px_scaled: QPixmap | None = None
        self._face_px_last_fsz = -1
        # Static grid-dot layer, pre-rendered once per size into a pixmap
        # (Mark-LIV style). The old path drew hundreds of drawPoint calls on
        # EVERY paint — the main reason full-rate painting was too costly.
        self._grid_cache: QPixmap | None = None
        self._grid_key: tuple | None = None

        self._load_face_bg(face_path)

        self._avatar = None
        # Prefer the Mark-L style photoreal-mesh head (core/avatar.py +
        # avatar_mesh.py + face_model.obj); fall back to the procedural
        # HoloFace if the mesh assets are missing.
        try:
            from core.avatar import HoloAvatar
            self._avatar = HoloAvatar()
            self._avatar_kind = "mesh"
        except Exception:
            try:
                from core.face_renderer import HoloFace
                self._avatar = HoloFace()
                self._avatar_kind = "proc"
            except Exception:
                self._avatar = None
                self._avatar_kind = None

        self._live_amp = 0.0
        self._amp_disp = 0.0
        self._visemes = None
        self._vis_i = None
        self._step_t = time.time()
        self._paint_tick = 0

        # Cursor gaze: the avatar's eyes follow the mouse while it moves
        # (and for a short hold afterwards), then return to autonomous
        # saccades. setMouseTracking is required to get move events without
        # a button pressed.
        self.setMouseTracking(True)

        self._tmr = QTimer(self)
        self._tmr.timeout.connect(self._step)
        self._tmr.start(16)

        self._steps_since_face_update = 0

    def _make_grid(self, W: int, H: int) -> QPixmap:
        """Render the static grid-dot layer once (Mark-LIV approach)."""
        pm = QPixmap(W, H)
        pm.fill(Qt.GlobalColor.transparent)
        qpp = QPainter(pm)
        qpp.setPen(QPen(qcol(C.PRI_GHO), 1))
        for x in range(0, W, 48):
            for y in range(0, H, 48):
                qpp.drawPoint(x, y)
        qpp.end()
        return pm

    def _load_face_bg(self, path: str):
        def _load():
            try:
                from PIL import Image, ImageDraw
                import io
                img = Image.open(path).convert("RGBA")
                sz  = min(img.size)
                img = img.resize((sz, sz), Image.LANCZOS)
                mk  = Image.new("L", (sz, sz), 0)
                ImageDraw.Draw(mk).ellipse((2, 2, sz - 2, sz - 2), fill=255)
                img.putalpha(mk)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                px = QPixmap(); px.loadFromData(buf.getvalue())
                self._face_px = px
                self._face_px_scaled = None
                self._face_px_last_fsz = -1
            except Exception:
                self._face_px = None
        threading.Thread(target=_load, daemon=True).start()

    def notify_audio_activity(self, grace: float = 0.9) -> None:
        """Mark 'audio is playing now' for `grace` seconds.

        Called from the TTS playback paths on every batch. While active,
        the HUD treats the face as speaking regardless of the state
        string — so a PROCESSING label (tool call) no longer freezes the
        mouth mid-reply.
        """
        self._audio_active_until = max(self._audio_active_until,
                                       time.time() + grace)

    def clear_visemes(self) -> None:
        """Drop any pending viseme schedule (turn ended / barge-in).

        A schedule left over from a killed turn would otherwise keep the
        mouth snapping to frames of audio that will never play.
        """
        self._visemes = None
        self._vis_i = None

    def push_visemes(self, frames, hop: float, at: float) -> None:
        try:
            if not frames:
                return
            hop = max(1e-3, float(hop))
            at = float(at)
            new = list(frames)
            current = self._visemes
            if current is not None:
                old, start, old_hop = current
                if abs(old_hop - hop) < 1e-6:
                    index = int(round((at - start) / hop))
                    if 0 <= index <= len(old) + 1:
                        merged = old[:index] + new
                        elapsed = int((time.time() - start) / hop) - 2
                        if elapsed > 60:
                            merged = merged[elapsed:]
                            start += elapsed * hop
                            if self._vis_i is not None:
                                self._vis_i = max(0, self._vis_i - elapsed)
                        self._visemes = (merged, start, hop)
                        return
            self._visemes = (new, at, hop)
            self._vis_i = None
        except Exception:
            pass

    def set_audio_level(self, level: float) -> None:
        try:
            value = float(level)
        except (TypeError, ValueError):
            return
        value = max(0.0, min(1.0, value))
        if value > self._live_amp:
            self._live_amp = value

    def glance(self, dx: float, dy: float, hold: float = 1.0) -> None:
        try:
            if self._avatar is not None:
                self._avatar.glance(dx, dy, hold)
        except Exception:
            pass

    def set_cursor_gaze(self, dx: float, dy: float, hold: float = 2.5) -> None:
        """Eyes follow the user's cursor (mesh avatar only)."""
        try:
            setter = getattr(self._avatar, "set_cursor_gaze", None)
            if setter is not None:
                setter(dx, dy, hold)
        except Exception:
            pass

    def mouseMoveEvent(self, e):
        """Feed the cursor position to the avatar's gaze as -1..1 offsets."""
        try:
            w, h = max(1, self.width()), max(1, self.height())
            dx = (e.position().x() - w / 2) / (w / 2) * 1.6
            dy = (e.position().y() - h / 2) / (h / 2) * 1.6
            self.set_cursor_gaze(dx, dy, 2.5)
        except Exception:
            pass
        super().mouseMoveEvent(e)

    def _on_screen(self) -> bool:
        try:
            if not self.isVisible():
                return False
            win = self.window()
            return not (win.isMinimized() or win.isHidden())
        except Exception:
            return True

    def _step(self):
        self._tick += 1
        now = time.time()
        dt = max(0.0, now - self._step_t)
        self._step_t = now

        v_open = v_wide = v_level = None
        v_seq = None
        sched = self._visemes
        if sched is not None:
            frames, start, hop = sched
            index = int((now - start) / hop)
            if 0 <= index < len(frames):
                first = self._vis_i if self._vis_i is not None else index
                v_seq = frames[max(0, first):index + 1]
                self._vis_i = max(first, index + 1)
                v_level, v_open, v_wide = frames[index]
                if v_seq:
                    peak = max(frame[0] for frame in v_seq)
                    if peak > self._live_amp:
                        self._live_amp = peak
            elif index >= len(frames):
                self._visemes = None
                self._vis_i = None

        self._live_amp *= 0.86
        self._amp_disp += (self._live_amp - self._amp_disp) * 0.45
        amp = self._amp_disp

        if now - self._last_t > (0.12 if self.speaking else 0.5):
            if self.speaking:
                self._tgt_scale = random.uniform(1.06, 1.14)
                self._tgt_halo  = random.uniform(145, 190)
            elif self.muted:
                self._tgt_scale = random.uniform(0.998, 1.002)
                self._tgt_halo  = random.uniform(15, 28)
            else:
                self._tgt_scale = random.uniform(1.001, 1.008)
                self._tgt_halo  = random.uniform(48, 68)
            self._last_t = now

        sp = 0.38 if self.speaking else (0.30 if amp > 0.02 else 0.15)
        self._scale += (self._tgt_scale - self._scale) * sp
        self._halo  += (self._tgt_halo  - self._halo)  * sp

        speeds = [1.3, -0.9, 2.0] if self.speaking else [0.55, -0.35, 0.9]
        for i, spd in enumerate(speeds):
            self._rings[i] = (self._rings[i] + spd) % 360

        self._scan  = (self._scan  + (3.0 if self.speaking else 1.3)) % 360
        self._scan2 = (self._scan2 + (-2.0 if self.speaking else -0.75)) % 360

        fw  = min(self.width(), self.height())
        lim = fw * 0.74
        # Rings race with the voice: louder speech pushes pulses out faster.
        spd = (2.0 + 2.6 * min(1.0, self._amp_disp)) if self.speaking else 2.0
        self._pulses = [r + spd for r in self._pulses if r + spd < lim]
        _p_rate = (0.05 + 0.45 * min(1.0, self._amp_disp)) if self.speaking else 0.025
        if len(self._pulses) < 3 and random.random() < _p_rate:
            self._pulses.append(0.0)

        # Embers fly with the voice: spawn rate scales with the audio level.
        if self.speaking and random.random() < (0.10 + 0.55 * min(1.0, self._amp_disp)):
            cx, cy = self.width() / 2, self.height() / 2
            ang = random.uniform(0, 2 * math.pi)
            r_s = fw * 0.28
            self._particles.append([
                cx + math.cos(ang) * r_s, cy + math.sin(ang) * r_s,
                math.cos(ang) * random.uniform(0.9, 2.4),
                math.sin(ang) * random.uniform(0.9, 2.4) - 0.4, 1.0,
            ])
        self._particles = [
            [p[0]+p[2], p[1]+p[3], p[2]*0.97, p[3]*0.97, p[4]-0.028]
            for p in self._particles if p[4] > 0
        ]

        if self._avatar is not None:
            try:
                # Mouth lives on AUDIO GROUND TRUTH: state string may read
                # PROCESSING while a reply is still sounding — the mouth
                # must keep moving. State still drives eyes/brows/lids, so
                # the face keeps its thinking look while talking through it.
                live_audio = (self.speaking
                              or time.time() < self._audio_active_until)
                if (self._pending_muted_state and not live_audio
                        and not self.speaking):
                    # Playback ended while muted: now show MUTED.
                    self._pending_muted_state = False
                    self.state = "MUTED"
                self._avatar.step(
                    dt, amp, speaking=live_audio, muted=self.muted,
                    state=self.state, v_open=v_open, v_wide=(v_wide if v_wide is not None else 0.0),
                    v_level=v_level, v_seq=v_seq,
                    v_hop=(sched[2] if sched is not None else 0.02),
                )
            except Exception:
                pass

        self._blink_tick += 1
        if self._blink_tick >= 38:
            self._blink = not self._blink
            self._blink_tick = 0
            blinked = True
        else:
            blinked = False

        # Grid is cached now, so painting at the full 60 Hz tick is cheap
        # (Mark-LIV does exactly this). Skip only when nothing is visible.
        if self._on_screen():
            self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), qcol(C.BG))

        W, H = self.width(), self.height()
        cx, cy = W / 2, H / 2
        fw = min(W, H)

        _gkey = (W, H)
        if self._grid_cache is None or self._grid_key != _gkey:
            self._grid_cache = self._make_grid(W, H)
            self._grid_key = _gkey
        p.drawPixmap(0, 0, self._grid_cache)

        r_face = fw * 0.31

        # Voice-reactive aura: the halo breathes with the actual audio level
        # (amp_disp is the smoothed 0..1 display amplitude), so raising the
        # voice visibly widens the glow while speaking.
        _amp_boost = 1.0 + min(1.0, self._amp_disp) * 0.9
        for i in range(10):
            r   = r_face * (1.8 - i * 0.08)
            frc = 1.0 - i / 10
            a   = max(0, min(255, int(self._halo * 0.085 * frc * _amp_boost)))
            col = qcol(C.MUTED_C if self.muted else C.PRI, a)
            p.setPen(QPen(col, 1.5)); p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QRectF(cx - r, cy - r, r * 2, r * 2))

        for pr in self._pulses:
            a   = max(0, int(230 * (1.0 - pr / (fw * 0.74))))
            col = qcol(C.MUTED_C if self.muted else C.PRI, a)
            p.setPen(QPen(col, 1.5)); p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QRectF(cx - pr, cy - pr, pr * 2, pr * 2))

        for idx, (r_frac, w_r, arc_l, gap) in enumerate(
            [(0.48, 3, 115, 78), (0.40, 2, 78, 55), (0.32, 1, 56, 40)]
        ):
            ring_r = fw * r_frac
            base   = self._rings[idx]
            a_val  = max(0, min(255, int(self._halo * (1.0 - idx * 0.18))))
            col    = qcol(C.MUTED_C if self.muted else C.PRI, a_val)
            p.setPen(QPen(col, w_r)); p.setBrush(Qt.BrushStyle.NoBrush)
            angle = base
            rect  = QRectF(cx - ring_r, cy - ring_r, ring_r * 2, ring_r * 2)
            while angle < base + 360:
                p.drawArc(rect, int(angle * 16), int(arc_l * 16))
                angle += arc_l + gap

        sr = fw * 0.50
        sa = min(255, int(self._halo * 1.5))
        ex = 75 if self.speaking else 44
        p.setPen(QPen(qcol(C.MUTED_C if self.muted else C.PRI, sa), 2.5))
        p.setBrush(Qt.BrushStyle.NoBrush)
        srect = QRectF(cx - sr, cy - sr, sr * 2, sr * 2)
        p.drawArc(srect, int(self._scan * 16), int(ex * 16))
        p.setPen(QPen(qcol(C.ACC, sa // 2), 1.5))
        p.drawArc(srect, int(self._scan2 * 16), int(ex * 16))

        t_out, t_in = fw * 0.497, fw * 0.474
        p.setPen(QPen(qcol(C.PRI, 140), 1))
        for deg in range(0, 360, 10):
            rad = math.radians(deg)
            inn = t_in if deg % 30 == 0 else t_in + 6
            p.drawLine(
                QPointF(cx + t_out * math.cos(rad), cy - t_out * math.sin(rad)),
                QPointF(cx + inn  * math.cos(rad), cy - inn  * math.sin(rad)),
            )

        ch_r, gap_h = fw * 0.51, fw * 0.16
        p.setPen(QPen(qcol(C.PRI, int(self._halo * 0.5)), 1))
        p.drawLine(QPointF(cx - ch_r, cy), QPointF(cx - gap_h, cy))
        p.drawLine(QPointF(cx + gap_h, cy), QPointF(cx + ch_r, cy))
        p.drawLine(QPointF(cx, cy - ch_r), QPointF(cx, cy - gap_h))
        p.drawLine(QPointF(cx, cy + gap_h), QPointF(cx, cy + ch_r))

        bl = 24
        bc = qcol(C.PRI, 210)
        hl, hr = cx - fw // 2, cx + fw // 2
        ht, hb = cy - fw // 2, cy + fw // 2
        p.setPen(QPen(bc, 2))
        for bx, by, dx, dy in [(hl,ht,1,1),(hr,ht,-1,1),(hl,hb,1,-1),(hr,hb,-1,-1)]:
            p.drawLine(QPointF(bx, by), QPointF(bx + dx * bl, by))
            p.drawLine(QPointF(bx, by), QPointF(bx, by + dy * bl))

        if self._avatar is not None:
            try:
                main = qcol(C.MUTED_C if self.muted else C.PRI)
                if self.speaking:
                    accent = qcol(C.MUTED_C if self.muted else C.ACC)
                elif self.state in ("THINKING", "PROCESSING"):
                    accent = qcol(C.ACC2)
                elif self.state == "LISTENING":
                    accent = qcol(C.GREEN)
                else:
                    accent = qcol(C.MUTED_C if self.muted else C.PRI)
                if getattr(self, "_avatar_kind", None) == "mesh":
                    # Mesh head: size from its own vertical span so the chin
                    # and crown always fit, exactly like Mark-L's HUD.
                    _band_t = 12.0
                    _sy_status = cy + fw * 0.40
                    _band_h = max(60.0, _sy_status - _band_t)
                    _r_head = min(fw * 0.355,
                                  _band_h / (self._avatar.SPAN + 0.08))
                    _head_cy = _band_t + (_band_h - self._avatar.SPAN * _r_head) / 2.0 + _r_head
                    self._avatar.paint(p, cx, _head_cy, _r_head,
                                       main, accent, qcol(C.BG))
                else:
                    self._avatar.paint(p, cx, cy - fw * 0.02, fw * 0.31,
                                       main, accent, qcol(C.BG))
            except Exception:
                pass
        elif self._face_px:
            fsz    = int(fw * 0.62 * self._scale)
            if fsz != self._face_px_last_fsz or self._face_px_scaled is None:
                self._face_px_scaled = self._face_px.scaled(
                    fsz, fsz,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self._face_px_last_fsz = fsz
            p.drawPixmap(int(cx - fsz / 2), int(cy - fsz / 2), self._face_px_scaled)
        else:
            orb_r = int(fw * 0.27 * self._scale)
            oc    = (200, 0, 50) if self.muted else (0, 60, 110)
            for i in range(8, 0, -1):
                r2  = int(orb_r * i / 8)
                frc = i / 8
                a   = max(0, min(255, int(self._halo * 1.1 * frc)))
                p.setBrush(QBrush(QColor(int(oc[0]*frc), int(oc[1]*frc), int(oc[2]*frc), a)))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawEllipse(QRectF(cx - r2, cy - r2, r2 * 2, r2 * 2))
            p.setPen(QPen(qcol(C.PRI, min(255, int(self._halo * 2))), 1))
            p.setFont(font_ui(14, True, display=True))
            p.drawText(QRectF(cx - 80, cy - 14, 160, 28),
                       Qt.AlignmentFlag.AlignCenter, "Orthos")

        for pt in self._particles:
            a = max(0, min(255, int(pt[4] * 255)))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(qcol(C.PRI, a)))
            p.drawEllipse(QPointF(pt[0], pt[1]), 2.5, 2.5)

        sy = cy + fw * 0.40
        if self.muted:
            txt, col = "⊘  MUTED",     qcol(C.MUTED_C)
        elif self.speaking:
            txt, col = "●  SPEAKING",  qcol(C.ACC)
        elif self.state == "THINKING":
            sym = "◈" if self._blink else "◇"
            txt, col = f"{sym}  THINKING",   qcol(C.ACC2)
        elif self.state == "PROCESSING":
            sym = "▷" if self._blink else "▶"
            txt, col = f"{sym}  PROCESSING", qcol(C.ACC2)
        elif self.state == "LISTENING":
            sym = "●" if self._blink else "○"
            txt, col = f"{sym}  LISTENING",  qcol(C.GREEN)
        else:
            sym = "●" if self._blink else "○"
            txt, col = f"{sym}  {self.state}", qcol(C.PRI)

        p.setPen(QPen(col, 1))
        p.setFont(font_ui(12, True, display=True))
        p.drawText(QRectF(0, sy, W, 26), Qt.AlignmentFlag.AlignCenter, txt)

        wy = sy + 30
        N, bw = 36, 8
        wx0 = (W - N * bw) / 2
        for i in range(N):
            if self.muted:
                hgt, cl = 2, qcol(C.MUTED_C)
            elif self.speaking:
                hgt = ((self._tick * 3 + i * 7) & 15) + 3
                cl  = qcol(C.PRI) if hgt > 12 else qcol(C.PRI_DIM)
            else:
                hgt = int(3 + 2 * math.sin(self._tick * 0.09 + i * 0.6))
                cl  = qcol(C.BORDER_B)
            p.fillRect(QRectF(wx0 + i * bw, wy + 20 - hgt, bw - 1, hgt), cl)


class MetricBar(QWidget):

    def __init__(self, label: str, color: str = C.PRI, parent=None):
        super().__init__(parent)
        self._label = label
        self._color = color
        self._value = 0.0
        self._text  = "--"
        self.setFixedHeight(38)
        self.setMinimumWidth(80)

    def set_value(self, pct: float, text: str):
        self._value = max(0.0, min(100.0, pct))
        self._text  = text
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()

        p.setBrush(QBrush(qcol(C.PANEL2)))
        p.setPen(QPen(qcol(C.BORDER_A), 1))
        p.drawRoundedRect(QRectF(1, 1, W - 2, H - 2), 4, 4)

        bar_h   = 4
        bar_y   = H - bar_h - 5
        bar_w   = W - 12
        bar_x   = 6
        fill_w  = int(bar_w * self._value / 100)

        p.setBrush(QBrush(qcol(C.BAR_BG)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(QRectF(bar_x, bar_y, bar_w, bar_h), 2, 2)

        if self._value > 85:
            bar_col = qcol(C.RED)
        elif self._value > 65:
            bar_col = qcol(C.ACC)
        else:
            bar_col = qcol(self._color)

        if fill_w > 0:
            p.setBrush(QBrush(bar_col))
            p.drawRoundedRect(QRectF(bar_x, bar_y, fill_w, bar_h), 2, 2)

        p.setFont(font_mono(8.5, True))
        p.setPen(QPen(qcol(C.TEXT_DIM), 1))
        p.drawText(QRectF(8, 5, 50, 14), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self._label)

        p.setFont(font_mono(10, True))
        p.setPen(QPen(bar_col if self._text != "--" else qcol(C.TEXT_DIM), 1))
        p.drawText(QRectF(0, 4, W - 6, 16), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, self._text)


from core.native_chat import NativeRichLogWidget as RichLogWidget


_FILE_ICONS = {
    "image":   ("🖼", "#00d4ff"), "video":   ("🎬", "#ff6b00"),
    "audio":   ("🎵", "#cc44ff"), "pdf":     ("📄", "#ff4444"),
    "word":    ("📝", "#4488ff"), "excel":   ("📊", "#44bb44"),
    "code":    ("💻", "#ffcc00"), "archive": ("📦", "#ff8844"),
    "pptx":    ("📊", "#ff6622"), "text":    ("📃", "#aaaaaa"),
    "data":    ("🔧", "#88ddff"), "unknown": ("📎", "#888888"),
}
_EXT_TO_CAT = {
    **dict.fromkeys(["jpg","jpeg","png","gif","webp","bmp","tiff","svg","ico"], "image"),
    **dict.fromkeys(["mp4","avi","mov","mkv","wmv","flv","webm","m4v"],         "video"),
    **dict.fromkeys(["mp3","wav","ogg","m4a","aac","flac","wma","opus"],        "audio"),
    **dict.fromkeys(["pdf"],                                                     "pdf"),
    **dict.fromkeys(["doc","docx"],                                              "word"),
    **dict.fromkeys(["xls","xlsx","ods"],                                        "excel"),
    **dict.fromkeys(["ppt","pptx"],                                              "pptx"),
    **dict.fromkeys(["py","js","ts","jsx","tsx","html","css","java","c","cpp",
                     "cs","go","rs","rb","php","swift","kt","sh","sql","lua"],   "code"),
    **dict.fromkeys(["zip","rar","tar","gz","7z","bz2","xz"],                   "archive"),
    **dict.fromkeys(["txt","md","rst","log"],                                    "text"),
    **dict.fromkeys(["csv","tsv","json","xml"],                                  "data"),
}

def _file_category(path: Path) -> str:
    return _EXT_TO_CAT.get(path.suffix.lower().lstrip("."), "unknown")

def _fmt_size(size: int) -> str:
    if   size < 1024:    return f"{size} B"
    elif size < 1024**2: return f"{size/1024:.1f} KB"
    elif size < 1024**3: return f"{size/1024**2:.1f} MB"
    else:                return f"{size/1024**3:.1f} GB"


def _attachment_context(path: str) -> dict:
    """Create a small, local preview so documents are ready before a message is sent.

    This intentionally does not transform images; the existing image pipeline owns that
    decision.  The preview is capped to keep a large attachment from consuming context.
    """
    p = Path(path)
    category = _file_category(p)
    info = {
        "status": "ready", "detail": "Ready to analyze", "preview": "", "page_images": [],
        "meta": f"{category.upper()} · {_fmt_size(p.stat().st_size)}",
    }
    try:
        if category in {"text", "code", "data"}:
            content = p.read_text(encoding="utf-8", errors="replace")[:12000]
            info["preview"] = content
            info["detail"] = f"Text extracted · {len(content):,} characters"
        elif category == "word":
            from docx import Document
            content = "\n".join(paragraph.text for paragraph in Document(p).paragraphs)[:12000]
            info["preview"] = content
            info["detail"] = f"Text extracted · {len(content):,} characters"
        elif category == "pdf":
            try:
                from pypdf import PdfReader
            except ImportError:
                from PyPDF2 import PdfReader
            reader = PdfReader(str(p))
            content = "\n".join((page.extract_text() or "") for page in reader.pages)[:12000]
            info["preview"] = content
            if content.strip():
                info["detail"] = f"{len(reader.pages)} pages · text extracted"
            else:
                # Chat-style visual PDF fallback: rasterize pages and hand them to
                # the configured vision model. This avoids opening the PDF or asking
                # the user for screenshots when the PDF is a scan.
                info["page_images"] = _render_scanned_pdf_pages(p)
                if info["page_images"]:
                    info["detail"] = f"{len(reader.pages)} pages · scan rendered for visual reading"
                else:
                    info.update({
                        "status": "warning",
                        "detail": "Scanned PDF needs PyMuPDF vision support (not installed)",
                    })
        elif category == "excel":
            try:
                from openpyxl import load_workbook
                workbook = load_workbook(p, read_only=True, data_only=True)
                sheets = list(workbook.sheetnames)
                info["meta"] += f" · {len(sheets)} sheet{'s' if len(sheets) != 1 else ''}"
                info["preview"] = "Workbook sheets: " + ", ".join(sheets[:12])
                info["detail"] = "Workbook structure read"
            except ImportError:
                info["detail"] = "Ready to analyze (install openpyxl for sheet metadata)"
        elif category == "image":
            info["detail"] = "Image ready for visual analysis"
        else:
            info["detail"] = "Ready for analysis with file tools"
    except Exception as exc:
        info.update({"status": "warning", "detail": f"Could not pre-read: {str(exc)[:80]}"})
    return info


def _render_scanned_pdf_pages(path: Path, max_pages: int = 6) -> list[str]:
    """Return bounded, high-legibility PDF page images for the vision model."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return []
    images: list[str] = []
    try:
        document = fitz.open(path)
        for page_index in range(min(document.page_count, max_pages)):
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            try:
                raw = pixmap.tobytes("jpeg", jpg_quality=80)
            except Exception:
                from PIL import Image
                image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=80, optimize=True)
                raw = buffer.getvalue()
            images.append(base64.b64encode(raw).decode("ascii"))
        document.close()
    except Exception:
        return []
    return images


class _DropCanvas(QWidget):
    def __init__(self, zone):
        super().__init__(zone)
        self._z = zone

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        z    = self._z
        W, H = self.width(), self.height()
        pad  = 6
        rect = QRectF(pad, pad, W - pad * 2, H - pad * 2)

        bg_col = qcol("#001a24" if z._drag_over else ("#001218" if z._hovering else C.PANEL))
        p.setBrush(QBrush(bg_col)); p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(rect, 6, 6)

        if z._current_file:   border_col = qcol(C.GREEN, 200)
        elif z._drag_over:    border_col = qcol(C.PRI, 230)
        elif z._hovering:     border_col = qcol(C.BORDER_B, 200)
        else:                 border_col = qcol(C.BORDER, 160)

        pen = QPen(border_col, 1.5, Qt.PenStyle.DashLine)
        pen.setDashOffset(z._dash_offset)
        p.setPen(pen); p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(rect, 6, 6)

        if z._current_file:   self._paint_file(p, W, H)
        elif z._drag_over:    self._paint_drag_over(p, W, H)
        else:                 self._paint_idle(p, W, H, z._hovering)

    def _paint_idle(self, p, W, H, hover):
        cx, cy = W / 2, H / 2
        col = qcol(C.PRI_DIM if not hover else C.PRI)
        p.setPen(QPen(col, 2)); p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawLine(QPointF(cx, cy - 14), QPointF(cx, cy + 4))
        p.drawLine(QPointF(cx - 8, cy - 6), QPointF(cx, cy - 14))
        p.drawLine(QPointF(cx + 8, cy - 6), QPointF(cx, cy - 14))
        p.drawLine(QPointF(cx - 14, cy + 4), QPointF(cx + 14, cy + 4))
        p.setFont(font_ui(9.5))
        p.setPen(QPen(qcol(C.PRI_DIM if not hover else C.TEXT), 1))
        p.drawText(QRectF(0, cy + 8, W, 16), Qt.AlignmentFlag.AlignCenter,
                   "Drop file here  or  Click to Browse")
        p.setFont(font_ui(8.5))
        p.setPen(QPen(qcol("#1a4a5a"), 1))
        p.drawText(QRectF(0, cy + 24, W, 14), Qt.AlignmentFlag.AlignCenter,
                   "Images · Video · Audio · PDF · Docs · Code · Data")

    def _paint_drag_over(self, p, W, H):
        cx, cy = W / 2, H / 2
        p.setFont(font_ui(22, display=True))
        p.setPen(QPen(qcol(C.PRI), 1))
        p.drawText(QRectF(0, cy - 24, W, 32), Qt.AlignmentFlag.AlignCenter, "⬇")
        p.setFont(font_ui(9.5, True))
        p.setPen(QPen(qcol(C.PRI), 1))
        p.drawText(QRectF(0, cy + 12, W, 16), Qt.AlignmentFlag.AlignCenter, "Release to load")

    def _paint_file(self, p, W, H):
        fi = self._z._file_info
        if fi is None:
            return
        icon, icon_col, name, ext, size_str, par_str = fi

        block_x, block_w = 10, 60
        p.setFont(QFont("Segoe UI Emoji", 22) if _OS == "Windows" else QFont("Arial", 22))
        p.setPen(QPen(qcol(icon_col), 1))
        p.drawText(QRectF(block_x, 0, block_w, H), Qt.AlignmentFlag.AlignCenter, icon)

        tx = block_x + block_w + 6
        tw = W - tx - 38

        p.setFont(font_ui(9.5, True))
        p.setPen(QPen(qcol(C.WHITE), 1))
        p.drawText(QRectF(tx, H * 0.18, tw, 16),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, name)

        p.setFont(font_ui(8.5))
        p.setPen(QPen(qcol(C.TEXT_DIM), 1))
        p.drawText(QRectF(tx, H * 0.18 + 18, tw, 14),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   f"{ext}  ·  {size_str}")

        p.setFont(font_ui(8))
        p.setPen(QPen(qcol("#1e5c6a"), 1))
        p.drawText(QRectF(tx, H * 0.18 + 34, tw, 12),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, par_str)

        p.setFont(font_ui(10, True))
        p.setPen(QPen(qcol(C.RED, 180), 1))
        p.drawText(QRectF(W - 34, 0, 28, H), Qt.AlignmentFlag.AlignCenter, "✕")

    def mousePressEvent(self, e):
        z = self._z
        if z._current_file and e.pos().x() > self.width() - 34:
            z.clear_file()
        else:
            z.mousePressEvent(e)


class FileDropZone(QWidget):
    file_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(100)
        self._current_file: str | None = None
        self._file_info: tuple | None = None
        self._hovering  = False
        self._drag_over = False
        self._dash_offset = 0.0
        self._anim_tmr = QTimer(self)
        self._anim_tmr.timeout.connect(self._animate)
        self._anim_tmr.start(40)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._canvas = _DropCanvas(self)
        layout.addWidget(self._canvas)

    def _animate(self):
        self._dash_offset = (self._dash_offset + 0.8) % 20
        self._canvas.update()

    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self._drag_over = True; self._canvas.update()

    def dragLeaveEvent(self, e):
        self._drag_over = False; self._canvas.update()

    def dropEvent(self, e: QDropEvent):
        self._drag_over = False
        urls = e.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if Path(path).is_file():
                self._set_file(path)
        self._canvas.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._browse()

    def enterEvent(self, e):
        self._hovering = True; self._canvas.update()

    def leaveEvent(self, e):
        self._hovering = False; self._canvas.update()

    def current_file(self) -> str | None:
        return self._current_file

    def clear_file(self):
        self._current_file = None; self._file_info = None; self._canvas.update()

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a file for Orthos", str(Path.home()),
            "All Files (*.*);;"
            "Images (*.jpg *.jpeg *.png *.gif *.webp *.bmp *.svg);;"
            "Documents (*.pdf *.docx *.txt *.md *.pptx);;"
            "Data (*.csv *.xlsx *.json *.xml);;"
            "Code (*.py *.js *.ts *.html *.css *.java *.cpp *.go);;"
            "Audio (*.mp3 *.wav *.ogg *.m4a *.aac *.flac);;"
            "Video (*.mp4 *.avi *.mov *.mkv *.wmv *.webm);;"
            "Archives (*.zip *.rar *.tar *.gz *.7z)",
        )
        if path:
            self._set_file(path)

    def _set_file(self, path: str):
        self._current_file = path
        p = Path(path)
        cat = _file_category(p)
        icon, icon_col = _FILE_ICONS.get(cat, _FILE_ICONS["unknown"])
        size_str = _fmt_size(p.stat().st_size)
        ext_str  = p.suffix.upper().lstrip(".") or "FILE"
        name_display = p.name if len(p.name) <= 34 else p.name[:31] + "..."
        par = str(p.parent)
        if len(par) > 42: par = "..." + par[-41:]
        self._file_info = (icon, icon_col, name_display, ext_str, size_str, par)
        self._canvas.update()
        self.file_selected.emit(path)


class SetupOverlay(QWidget):
    done = pyqtSignal(str)

    _INPUT_STYLE = ""

    def __init__(self, parent=None, initial: dict | None = None, mode: str = "init"):
        super().__init__(parent)
        self._mode = mode
        _init = initial or {}
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(f"""
            SetupOverlay {{
                background: rgba(0, 6, 10, 248);
                border: 1px solid {C.BORDER_B};
                border-radius: 6px;
            }}
        """)

        _INPUT = f"""
            QLineEdit {{
                background: #000d12; color: {C.TEXT};
                border: 1px solid {C.BORDER}; border-radius: 4px; padding: 4px 10px;
                font-family: 'Segoe UI'; font-size: 9pt;
            }}
            QLineEdit:focus {{ border: 1px solid {C.PRI}; background: #001520; }}
            QLineEdit:hover {{ border: 1px solid {C.BORDER_B}; }}
        """

        _COMBO_STYLE = f"""
            QComboBox {{
                background: #000d12; color: {C.TEXT};
                border: 1px solid {C.BORDER}; border-radius: 4px; padding: 4px 10px;
                font-family: 'Segoe UI'; font-size: 9pt;
            }}
            QComboBox:focus {{ border: 1px solid {C.PRI}; background: #001520; }}
            QComboBox:hover {{ border: 1px solid {C.BORDER_B}; }}
            QComboBox::drop-down {{ border: none; width: 20px; }}
            QComboBox::down-arrow {{ image: none; }}
            QComboBox QAbstractItemView {{
                background: #000d12; color: {C.TEXT};
                border: 1px solid {C.BORDER};
                selection-background-color: {C.PRI_GHO};
                font-family: 'Segoe UI'; font-size: 9pt;
                outline: none;
            }}
        """

        self._init            = _init
        self._sel_stt          = _init.get("stt_engine",    "whisper")
        self._sel_tts          = _init.get("tts_engine",    "edgetts")
        self._sel_llm_provider = _init.get("llm_provider",  "ollama")
        self._sel_summarization_provider = (
            _init.get("summarization_provider", "") or "auto"
        )

        # ── Shared helpers ──
        def _lbl(txt, sz=9, bold=False, col=C.PRI, align=Qt.AlignmentFlag.AlignCenter):
            w = QLabel(txt); w.setAlignment(align)
            w.setFont(QFont("Segoe UI", sz,
                            QFont.Weight.Bold if bold else QFont.Weight.Normal))
            w.setStyleSheet(f"color: {col}; background: transparent;")
            return w

        def _sep():
            s = QFrame(); s.setFrameShape(QFrame.Shape.HLine)
            s.setStyleSheet(f"color: {C.BORDER}; margin: 2px 0;")
            return s

        def _input(placeholder="", pw=False, fixed_h=30):
            w = QLineEdit()
            w.setPlaceholderText(placeholder)
            w.setFixedHeight(fixed_h)
            if pw:
                w.setEchoMode(QLineEdit.EchoMode.Password)
            w.setStyleSheet(_INPUT)
            return w

        _PILL_ACTIVE = f"""
            QPushButton {{
                background: {C.PRI}; color: #001a22;
                border: none; border-radius: 13px; font-weight: bold;
                padding: 0 12px;
            }}
        """
        _PILL_INACTIVE = f"""
            QPushButton {{
                background: transparent; color: {C.TEXT_DIM};
                border: 1px solid {C.BORDER}; border-radius: 13px;
                padding: 0 12px;
            }}
            QPushButton:hover {{ color: {C.TEXT}; border: 1px solid {C.BORDER_B}; }}
        """

        def _toggle_row(keys_labels: list, getter, setter):
            row = QHBoxLayout(); row.setSpacing(6)
            btns: dict[str, QPushButton] = {}
            def _click(k):
                setter(k)
                for bk, b in btns.items():
                    _style_btn(b, bk == k)
            for k, lbl in keys_labels:
                b = QPushButton(lbl)
                b.setFixedHeight(26)
                b.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
                b.setCursor(Qt.CursorShape.PointingHandCursor)
                b.clicked.connect(lambda _, kk=k: _click(kk))
                row.addWidget(b)
                btns[k] = b
            _click(getter())
            return row, btns

        def _style_btn(btn: QPushButton, active: bool):
            btn.setStyleSheet(_PILL_ACTIVE if active else _PILL_INACTIVE)

        _TEST_BTN_STYLE = f"""
            QPushButton {{
                background: transparent; color: {C.TEXT_DIM};
                border: 1px solid {C.BORDER}; border-radius: 4px;
                padding: 0 10px; font: 7pt 'Segoe UI'; font-weight: bold;
            }}
            QPushButton:hover {{ color: {C.PRI}; border: 1px solid {C.PRI_DIM}; }}
        """

        def _collapsible_card(title: str, content_widget: QWidget,
                               start_collapsed: bool = False) -> QFrame:
            card = QFrame()
            card.setStyleSheet(f"""
                QFrame {{
                    background: #011520;
                    border: 1px solid {C.BORDER};
                    border-radius: 4px;
                }}
            """)
            card_lay = QVBoxLayout(card)
            card_lay.setContentsMargins(0, 0, 0, 0)
            card_lay.setSpacing(0)

            # Accent line at top
            accent = QFrame()
            accent.setFixedHeight(2)
            accent.setStyleSheet(f"background: {C.PRI_DIM}; border: none;")
            card_lay.addWidget(accent)

            header = QPushButton(f"▼  {title}")
            header.setFlat(True)
            header.setCursor(Qt.CursorShape.PointingHandCursor)
            header.setStyleSheet(f"""
                QPushButton {{
                    text-align: left; padding: 8px 12px;
                    color: {C.TEXT_MED}; font: 7.5pt 'Segoe UI'; font-weight: bold;
                    letter-spacing: 0.5px;
                    background: transparent; border: none;
                    border-bottom: 1px solid {C.BORDER};
                }}
                QPushButton:hover {{ color: {C.TEXT}; }}
            """)

            container = QWidget()
            container.setStyleSheet("background: transparent;")
            c_lay = QVBoxLayout(container)
            c_lay.setContentsMargins(12, 10, 12, 10)
            c_lay.setSpacing(8)
            c_lay.addWidget(content_widget)

            if start_collapsed:
                container.setVisible(False)
                header.setText(f"▶  {title}")

            def toggle():
                collapsed = container.isVisible()
                container.setVisible(not collapsed)
                header.setText(f"{'▶' if collapsed else '▼'}  {title}")

            header.clicked.connect(toggle)
            card_lay.addWidget(header)
            card_lay.addWidget(container)
            return card

        # ── Main layout: scroll + fixed bottom bar ──
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet(f"""
            QScrollArea {{ border: none; background: transparent; }}
            QScrollBar:vertical {{
                background: transparent; width: 6px; margin: 0; border: none;
            }}
            QScrollBar::handle:vertical {{
                background: #0d3347; border-radius: 3px; min-height: 24px;
            }}
            QScrollBar::handle:vertical:hover {{ background: #1a5c7a; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0; border: none; background: none;
            }}
            QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {{
                background: none; border: none;
            }}
        """)

        content_w = QWidget()
        content_w.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(content_w)
        layout.setContentsMargins(24, 18, 24, 12)
        layout.setSpacing(12)
        scroll.setWidget(content_w)

        main_layout.addWidget(scroll, stretch=1)

        # ── Title ──
        if mode == "config":
            layout.addWidget(_lbl("◈  CONFIGURATION", 13, True))
            layout.addWidget(_lbl("Update Orthos settings and click Apply.", 8, col=C.PRI_DIM))
        else:
            layout.addWidget(_lbl("◈  INITIALISATION REQUIRED", 12, True))
            layout.addWidget(_lbl("Configure Orthos before first boot.", 8, col=C.PRI_DIM))

        # ======
        #  CARD 1: SPEECH-TO-TEXT
        # ======
        stt_w = QWidget(); stt_w.setStyleSheet("background: transparent;")
        stt_lay = QVBoxLayout(stt_w); stt_lay.setContentsMargins(0,0,0,0); stt_lay.setSpacing(6)

        stt_row, self._stt_btns = _toggle_row(
            [("whisper","🎙 Whisper"), ("vosk","🔊 Vosk"), ("gemini","☁ Gemini"), ("groq","⚡ Groq"), ("hashim","🎧 Hashim")],
            lambda: self._sel_stt,
            self._set_stt,
        )
        stt_lay.addLayout(stt_row)

        stt_detail = QHBoxLayout(); stt_detail.setSpacing(5)
        stt_detail.addWidget(_lbl("Model:", 7, col=C.TEXT_MED,
                                   align=Qt.AlignmentFlag.AlignRight))

        self._whisper_combo = QComboBox()
        self._whisper_combo.setFixedHeight(28)
        self._whisper_combo.setStyleSheet(_COMBO_STYLE)
        for m in ["tiny", "base", "small", "medium", "large-v3"]:
            self._whisper_combo.addItem(m)
        _cur_model = _init.get("stt_model", "base")
        _idx = self._whisper_combo.findText(_cur_model)
        self._whisper_combo.setCurrentIndex(_idx if _idx >= 0 else 1)
        stt_detail.addWidget(self._whisper_combo)

        self._vosk_model_input = _input("model dir path  (leave empty for auto-download)")
        self._vosk_model_input.setText(_init.get("vosk_model_path", ""))
        stt_detail.addWidget(self._vosk_model_input)

        stt_lay.addLayout(stt_detail)

        self._gemini_key_widget = QWidget()
        self._gemini_key_widget.setStyleSheet("background: transparent;")
        gk_row = QHBoxLayout(self._gemini_key_widget)
        gk_row.setContentsMargins(0, 0, 0, 0)
        gk_row.setSpacing(5)
        gk_row.addWidget(_lbl("Voice API Key:", 7, col=C.TEXT_MED,
                               align=Qt.AlignmentFlag.AlignRight))
        self._gemini_key_input = _input("AIzaSy...  (Voice API key — STT + TTS)", pw=True)
        self._gemini_key_input.setText(_init.get("gemini_voice_api_key", ""))
        gk_row.addWidget(self._gemini_key_input)
        stt_lay.addWidget(self._gemini_key_widget)

        self._groq_key_widget = QWidget()
        self._groq_key_widget.setStyleSheet("background: transparent;")
        gk_row = QHBoxLayout(self._groq_key_widget)
        gk_row.setContentsMargins(0, 0, 0, 0)
        gk_row.setSpacing(5)
        gk_row.addWidget(_lbl("API Key:", 7, col=C.TEXT_MED,
                               align=Qt.AlignmentFlag.AlignRight))
        self._groq_key_input = _input("gsk_...  (Groq API key)", pw=True)
        self._groq_key_input.setText(_init.get("groq_api_key", ""))
        gk_row.addWidget(self._groq_key_input)
        stt_lay.addWidget(self._groq_key_widget)

        self._whisper_combo.setVisible(self._sel_stt == "whisper")
        self._vosk_model_input.setVisible(self._sel_stt == "vosk")
        self._gemini_key_widget.setVisible(self._sel_stt in ("gemini", "hashim"))
        self._groq_key_widget.setVisible(self._sel_stt == "groq")

        stt_lang_row = QHBoxLayout(); stt_lang_row.setSpacing(5)
        stt_lang_row.addWidget(_lbl("Language:", 7, col=C.TEXT_MED,
                                    align=Qt.AlignmentFlag.AlignRight))
        self._stt_lang_input = _input("auto  (or: tr / en / de / fr / es / zh ...)")
        self._stt_lang_input.setText(_init.get("stt_language", "auto"))
        stt_lang_row.addWidget(self._stt_lang_input)
        stt_lay.addLayout(stt_lang_row)

        layout.addWidget(_collapsible_card("SPEECH-TO-TEXT", stt_w))

        # ======
        #  CARD 2: LANGUAGE MODEL
        # ======
        llm_w = QWidget(); llm_w.setStyleSheet("background: transparent;")
        llm_lay = QVBoxLayout(llm_w); llm_lay.setContentsMargins(0,0,0,0); llm_lay.setSpacing(6)

        llm_prov_row, self._llm_prov_btns = _toggle_row(
            [
                ("ollama", "🦙 Ollama"),
                ("openai", "🔌 OpenAI / LM Studio"),
                ("groq", "⚡ Groq"),
                ("cloudflare", "☁ Cloudflare"),
                ("gemini", "🌀 Gemini"),
                ("gemini_live", "⚡ Gemini Live"),
                ("openrouter", "🌐 OpenRouter"),
                ("nvidia", "🎮 NVIDIA"),
                ("kilo", "🛰 Kilo"),
                ("kaggle", "📎 Kaggle"),
            ],
            lambda: self._sel_llm_provider,
            self._set_llm_provider,
        )
        llm_lay.addLayout(llm_prov_row)

        _ollama_hint = "ollama.com  ·  run: ollama pull qwen2.5:3b"
        _openai_hint = "lmstudio.ai  ·  start Local Server first, then pick model"
        _cloudflare_hint = "developers.cloudflare.com/workers-ai/  ·  needs Account ID + API Token"
        _groq_hint = "console.groq.com  ·  needs Groq LLM API key"
        _gemini_hint = "aistudio.google.com  ·  needs Gemini API key"
        _or_hint = "openrouter.ai  ·  needs API key, free models available"
        _nvidia_hint = "build.nvidia.com  ·  needs NVIDIA API key"
        _kilo_hint = "kilo.ai  ·  needs Kilo API key, kilo-auto/free"
        _kaggle_hint = "kaggle.com notebook + Ollama  ·  free 2x T4 GPU  ·  tunnel URL (Tailscale/Cloudflare)"
        self._llm_hint_lbl = _lbl(
            {
                "openai": _openai_hint, "cloudflare": _cloudflare_hint,
                "groq": _groq_hint, "gemini": _gemini_hint,
                "openrouter": _or_hint, "nvidia": _nvidia_hint,
                "kilo": _kilo_hint, "kaggle": _kaggle_hint,
            }.get(self._sel_llm_provider, _ollama_hint),
            7, col=C.TEXT_DIM, align=Qt.AlignmentFlag.AlignLeft
        )
        llm_lay.addWidget(self._llm_hint_lbl)

        llm_row = QHBoxLayout(); llm_row.setSpacing(5)
        llm_row.addWidget(_lbl("URL:", 7, col=C.TEXT_MED,
                                align=Qt.AlignmentFlag.AlignRight))
        _default_url = _init.get("llm_url",
                                  "http://localhost:1234" if self._sel_llm_provider == "openai"
                                  else "http://localhost:11434")
        self._llm_url_input = _input(
            "http://localhost:1234" if self._sel_llm_provider == "openai"
            else "http://localhost:11434"
        )
        self._llm_url_input.setText(_default_url)
        llm_row.addWidget(self._llm_url_input, stretch=2)
        llm_lay.addLayout(llm_row)

        llm_model_row = QHBoxLayout(); llm_model_row.setSpacing(5)
        llm_model_row.addWidget(_lbl("Model:", 7, col=C.TEXT_MED,
                                align=Qt.AlignmentFlag.AlignRight))
        self._llm_model_combo = QComboBox()
        self._llm_model_combo.setFixedHeight(28)
        self._llm_model_combo.setStyleSheet(_COMBO_STYLE)
        self._llm_model_combo.addItem("── Custom ──", userData="__CUSTOM__")
        self._llm_custom_input = _input("e.g.  qwen2.5:3b  /  llama3.2  /  mistral")
        self._llm_custom_input.setText(_init.get("llm_model", ""))
        self._llm_custom_input.setVisible(True)
        llm_model_row.addWidget(self._llm_model_combo, stretch=2)
        llm_model_row.addWidget(self._llm_custom_input, stretch=3)
        self._llm_model_combo.currentIndexChanged.connect(self._on_ollama_model_changed)
        llm_lay.addLayout(llm_model_row)

        ollama_btn_row = QHBoxLayout(); ollama_btn_row.setSpacing(5)
        self._ollama_fetch_btn = QPushButton("🔄 Fetch Models")
        self._ollama_fetch_btn.setFixedHeight(24)
        self._ollama_fetch_btn.setFont(font_ui(8.5, True))
        self._ollama_fetch_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._ollama_fetch_btn.setStyleSheet(_TEST_BTN_STYLE)
        self._ollama_fetch_btn.clicked.connect(lambda: threading.Thread(
            target=self._fetch_ollama_models, daemon=True).start())
        ollama_btn_row.addWidget(self._ollama_fetch_btn)
        self._ollama_test_btn = QPushButton("Test Connection")
        self._ollama_test_btn.setFixedHeight(24)
        self._ollama_test_btn.setFont(font_ui(8.5, True))
        self._ollama_test_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._ollama_test_btn.setStyleSheet(_TEST_BTN_STYLE)
        self._ollama_test_btn.clicked.connect(lambda: threading.Thread(
            target=self._test_ollama, daemon=True).start())
        ollama_btn_row.addWidget(self._ollama_test_btn)
        self._ollama_test_status = _lbl("", 7, col=C.TEXT_DIM)
        ollama_btn_row.addWidget(self._ollama_test_status)
        ollama_btn_row.addStretch()
        llm_lay.addLayout(ollama_btn_row)

        # ── Endpoint profiles (local ↔ Kaggle tunnel one-click switch) ──
        llm_profiles_row = QHBoxLayout(); llm_profiles_row.setSpacing(5)
        llm_profiles_row.addWidget(_lbl("Profile:", 7, col=C.TEXT_MED,
                                align=Qt.AlignmentFlag.AlignRight))
        self._llm_profile_combo = QComboBox()
        self._llm_profile_combo.setFixedHeight(28)
        self._llm_profile_combo.setStyleSheet(_COMBO_STYLE)
        self._llm_profiles_refresh()
        llm_profiles_row.addWidget(self._llm_profile_combo, stretch=2)
        self._llm_profile_save_btn = QPushButton("💾 Save as Profile")
        self._llm_profile_save_btn.setFixedHeight(24)
        self._llm_profile_save_btn.setFont(font_ui(8.5, True))
        self._llm_profile_save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._llm_profile_save_btn.setStyleSheet(_TEST_BTN_STYLE)
        self._llm_profile_save_btn.clicked.connect(self._llm_profile_save_clicked)
        llm_profiles_row.addWidget(self._llm_profile_save_btn)
        self._llm_profile_switch_btn = QPushButton("⚡ Switch")
        self._llm_profile_switch_btn.setFixedHeight(24)
        self._llm_profile_switch_btn.setFont(font_ui(8.5, True))
        self._llm_profile_switch_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._llm_profile_switch_btn.setStyleSheet(_TEST_BTN_STYLE)
        self._llm_profile_switch_btn.clicked.connect(lambda: threading.Thread(
            target=self._llm_profile_switch_clicked, daemon=True).start())
        llm_profiles_row.addWidget(self._llm_profile_switch_btn)
        self._llm_profile_delete_btn = QPushButton("🗑")
        self._llm_profile_delete_btn.setFixedHeight(24)
        self._llm_profile_delete_btn.setFont(font_ui(8.5, True))
        self._llm_profile_delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._llm_profile_delete_btn.setStyleSheet(_TEST_BTN_STYLE)
        self._llm_profile_delete_btn.clicked.connect(self._llm_profile_delete_clicked)
        llm_profiles_row.addWidget(self._llm_profile_delete_btn)
        llm_lay.addLayout(llm_profiles_row)

        # ── Kaggle-specific fields (remote Ollama on free Kaggle GPU) ──
        self._kaggle_widget = QWidget()
        self._kaggle_widget.setStyleSheet("background: transparent;")
        kg_layout = QVBoxLayout(self._kaggle_widget)
        kg_layout.setContentsMargins(0, 0, 0, 0)
        kg_layout.setSpacing(4)
        kg_url_row = QHBoxLayout(); kg_url_row.setSpacing(5)
        kg_url_row.addWidget(_lbl("Tunnel URL:", 7, col=C.TEXT_MED))
        self._kaggle_url_input = _input("http://100.x.y.z:11434  (Tailscale) or https://xxx.trycloudflare.com")
        self._kaggle_url_input.setText(_init.get("kaggle_url", ""))
        kg_url_row.addWidget(self._kaggle_url_input, stretch=2)
        kg_layout.addLayout(kg_url_row)
        kg_model_row = QHBoxLayout(); kg_model_row.setSpacing(5)
        kg_model_row.addWidget(_lbl("Model:", 7, col=C.TEXT_MED))
        self._kaggle_model_combo = QComboBox()
        self._kaggle_model_combo.setFixedHeight(28)
        self._kaggle_model_combo.setStyleSheet(_COMBO_STYLE)
        _KG_MODELS = [
            ("qwen3.8-27b-uncensored-mtp", "Qwen3.8 27B Uncensored + MTP (Q4_K_M, 256K ctx)"),
            ("hf.co/JonathanColetti/Qwen3.8-27B-Uncensored-GGUF:Q4_K_M", "Qwen3.8 27B Uncensored (base tag)"),
        ]
        for val, lbl in _KG_MODELS:
            self._kaggle_model_combo.addItem(lbl, userData=val)
        self._kaggle_model_combo.addItem("── Custom ──", userData="__CUSTOM__")
        self._kaggle_custom_input = _input("Custom ollama model name on the Kaggle server")
        _cur_kg = _init.get("kaggle_model", "qwen3.8-27b-uncensored-mtp")
        _found_kg = False
        for i in range(self._kaggle_model_combo.count()):
            if self._kaggle_model_combo.itemData(i) == _cur_kg:
                self._kaggle_model_combo.setCurrentIndex(i)
                _found_kg = True
                break
        if not _found_kg:
            self._kaggle_model_combo.setCurrentIndex(self._kaggle_model_combo.count() - 1)
            self._kaggle_custom_input.setText(_cur_kg)
        self._kaggle_custom_input.setVisible(not _found_kg and _cur_kg != "qwen3.8-27b-uncensored-mtp")
        self._kaggle_model_combo.currentIndexChanged.connect(self._on_kaggle_model_changed)
        kg_model_row.addWidget(self._kaggle_model_combo, stretch=2)
        kg_model_row.addWidget(self._kaggle_custom_input, stretch=3)
        kg_layout.addLayout(kg_model_row)
        kg_test_row = QHBoxLayout(); kg_test_row.setSpacing(5)
        self._kaggle_test_btn = QPushButton("Test Kaggle Connection")
        self._kaggle_test_btn.setFixedHeight(24)
        self._kaggle_test_btn.setFont(font_ui(8.5, True))
        self._kaggle_test_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._kaggle_test_btn.setStyleSheet(_TEST_BTN_STYLE)
        self._kaggle_test_btn.clicked.connect(lambda: threading.Thread(
            target=self._test_kaggle, daemon=True).start())
        kg_test_row.addWidget(self._kaggle_test_btn)
        self._kaggle_test_status = _lbl("", 7, col=C.TEXT_DIM)
        kg_test_row.addWidget(self._kaggle_test_status)
        kg_test_row.addStretch()
        kg_layout.addLayout(kg_test_row)
        llm_lay.addWidget(self._kaggle_widget)

        # ── Cloudflare-specific fields ──
        self._cf_widget = QWidget()
        self._cf_widget.setStyleSheet("background: transparent;")
        cf_layout = QVBoxLayout(self._cf_widget)
        cf_layout.setContentsMargins(0, 0, 0, 0)
        cf_layout.setSpacing(4)
        cf_row = QHBoxLayout(); cf_row.setSpacing(5)
        cf_row.addWidget(_lbl("Account ID:", 7, col=C.TEXT_MED))
        self._cf_account_input = _input("Cloudflare Account ID")
        self._cf_account_input.setText(_init.get("cf_account_id", ""))
        cf_row.addWidget(self._cf_account_input)
        cf_row.addWidget(_lbl("Token:", 7, col=C.TEXT_MED))
        self._cf_token_input = _input("Cloudflare API Token", pw=True)
        self._cf_token_input.setText(_init.get("cf_api_token", ""))
        cf_row.addWidget(self._cf_token_input)
        cf_layout.addLayout(cf_row)
        cf_model_row = QHBoxLayout(); cf_model_row.setSpacing(5)
        cf_model_row.addWidget(_lbl("Model:", 7, col=C.TEXT_MED))
        self._cf_model_combo = QComboBox()
        self._cf_model_combo.setFixedHeight(28)
        self._cf_model_combo.setStyleSheet(_COMBO_STYLE)
        _CF_MODELS = [
            ("@cf/google/gemma-4-26b-a4b-it", "Gemma 4 26B A4B (MoE, vision, tools)"),
            ("@cf/mistralai/mistral-small-3.1-24b-instruct", "Mistral Small 3.1 24B (vision, tools)"),
            ("@cf/meta/llama-3.2-11b-vision-instruct", "Llama 3.2 11B Vision"),
            ("@cf/moondream/moondream3.1-9B-A2B", "Moondream 3.1 9B A2B (MoE, vision)"),
            ("@cf/moonshotai/kimi-k2.6", "Kimi K2.6 (MoE, vision, tools)"),
            ("@cf/moonshotai/kimi-k2.7-code", "Kimi K2.7 Code (MoE, coding, vision)"),
            ("@cf/meta/llama-3.3-70b-instruct-fp8-fast", "Llama 3.3 70B FP8 Fast"),
            ("@cf/meta/llama-3.1-8b-instruct-fp8-fast", "Llama 3.1 8B FP8 Fast"),
            ("@cf/qwen/qwen3-30b-a3b-fp8", "Qwen3 30B A3B (MoE)"),
            ("@cf/openai/gpt-oss-120b", "GPT-OSS 120B (reasoning, tools)"),
            ("@cf/openai/gpt-oss-20b", "GPT-OSS 20B (fast, tools)"),
            ("@cf/nvidia/nemotron-3-120b-a12b", "Nemotron 3 Super 120B (MoE, tools)"),
            ("@cf/meta/llama-4-scout-17b-16e-instruct", "Llama 4 Scout 17B (MoE, vision)"),
        ]
        for val, lbl in _CF_MODELS:
            self._cf_model_combo.addItem(lbl, userData=val)
        self._cf_model_combo.addItem("── Custom ──", userData="__CUSTOM__")
        self._cf_custom_input = _input("Custom model ID  (e.g. @cf/provider/model)")
        _cur_cf = _init.get("cf_model", "@cf/google/gemma-4-26b-a4b-it")
        _found_cf = False
        for i in range(self._cf_model_combo.count()):
            if self._cf_model_combo.itemData(i) == _cur_cf:
                self._cf_model_combo.setCurrentIndex(i)
                _found_cf = True
                break
        if not _found_cf:
            self._cf_model_combo.setCurrentIndex(self._cf_model_combo.count() - 1)
            self._cf_custom_input.setText(_cur_cf)
        self._cf_custom_input.setVisible(not _found_cf and _cur_cf != "@cf/google/gemma-4-26b-a4b-it")
        cf_model_row.addWidget(self._cf_model_combo, stretch=2)
        cf_model_row.addWidget(self._cf_custom_input, stretch=3)
        self._cf_model_combo.currentIndexChanged.connect(self._on_cf_model_changed)
        cf_layout.addLayout(cf_model_row)
        cf_test_row = QHBoxLayout(); cf_test_row.setSpacing(5)
        self._cf_fetch_btn = QPushButton("🔄 Fetch Models")
        self._cf_fetch_btn.setFixedHeight(24)
        self._cf_fetch_btn.setFont(font_ui(8.5, True))
        self._cf_fetch_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._cf_fetch_btn.setStyleSheet(_TEST_BTN_STYLE)
        self._cf_fetch_btn.clicked.connect(lambda: threading.Thread(
            target=self._fetch_cf_models, daemon=True).start())
        cf_test_row.addWidget(self._cf_fetch_btn)
        self._cf_test_btn = QPushButton("Test Connection")
        self._cf_test_btn.setFixedHeight(24)
        self._cf_test_btn.setFont(font_ui(8.5, True))
        self._cf_test_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._cf_test_btn.setStyleSheet(_TEST_BTN_STYLE)
        self._cf_test_btn.clicked.connect(lambda: threading.Thread(
            target=self._test_cf, daemon=True).start())
        cf_test_row.addWidget(self._cf_test_btn)
        self._cf_test_status = _lbl("", 7, col=C.TEXT_DIM)
        cf_test_row.addWidget(self._cf_test_status)
        cf_layout.addLayout(cf_test_row)
        llm_lay.addWidget(self._cf_widget)

        # ── Groq-specific fields ──
        self._groq_llm_widget = QWidget()
        self._groq_llm_widget.setStyleSheet("background: transparent;")
        gk_layout = QVBoxLayout(self._groq_llm_widget)
        gk_layout.setContentsMargins(0, 0, 0, 0)
        gk_layout.setSpacing(4)
        gk_row = QHBoxLayout(); gk_row.setSpacing(5)
        gk_row.addWidget(_lbl("LLM Key:", 7, col=C.TEXT_MED))
        self._groq_llm_key_input = _input("gsk_...  (Groq LLM key)", pw=True)
        self._groq_llm_key_input.setText(_init.get("groq_llm_key", ""))
        gk_row.addWidget(self._groq_llm_key_input)
        gk_layout.addLayout(gk_row)
        gk_model_row = QHBoxLayout(); gk_model_row.setSpacing(5)
        gk_model_row.addWidget(_lbl("Model:", 7, col=C.TEXT_MED))
        self._groq_model_combo = QComboBox()
        self._groq_model_combo.setFixedHeight(28)
        self._groq_model_combo.setStyleSheet(_COMBO_STYLE)
        _GROQ_MODELS = [
            ("meta-llama/llama-4-scout-17b-16e-instruct", "Llama 4 Scout 17B (MoE, vision)"),
            ("meta-llama/llama-4-maverick-17b-128e-instruct", "Llama 4 Maverick 17B (MoE, vision, preview)"),
            ("llama-3.1-8b-instant", "Llama 3.1 8B Instant (fastest, 14K RPD)"),
            ("llama-3.3-70b-versatile", "Llama 3.3 70B Versatile (best quality)"),
            ("openai/gpt-oss-120b", "GPT-OSS 120B (reasoning)"),
            ("openai/gpt-oss-20b", "GPT-OSS 20B (fast, tools)"),
            ("qwen/qwen3-32b", "Qwen3 32B (reasoning)"),
            ("qwen/qwen3.6-27b", "Qwen3.6 27B (preview)"),
            ("moonshotai/kimi-k2-instruct", "Kimi K2 (MoE, tools)"),
            ("groq/compound", "Groq Compound Router"),
            ("groq/compound-mini", "Groq Compound Mini"),
        ]
        for val, lbl in _GROQ_MODELS:
            self._groq_model_combo.addItem(lbl, userData=val)
        self._groq_model_combo.addItem("── Custom ──", userData="__CUSTOM__")
        self._groq_custom_input = _input("Custom model ID  (e.g. provider/model)")
        _cur_gq = _init.get("groq_llm_model", "meta-llama/llama-4-scout-17b-16e-instruct")
        _found_gq = False
        for i in range(self._groq_model_combo.count()):
            if self._groq_model_combo.itemData(i) == _cur_gq:
                self._groq_model_combo.setCurrentIndex(i)
                _found_gq = True
                break
        if not _found_gq:
            self._groq_model_combo.setCurrentIndex(self._groq_model_combo.count() - 1)
            self._groq_custom_input.setText(_cur_gq)
        self._groq_custom_input.setVisible(not _found_gq and _cur_gq != "meta-llama/llama-4-scout-17b-16e-instruct")
        gk_model_row.addWidget(self._groq_model_combo, stretch=2)
        gk_model_row.addWidget(self._groq_custom_input, stretch=3)
        self._groq_model_combo.currentIndexChanged.connect(self._on_groq_model_changed)
        gk_layout.addLayout(gk_model_row)
        gq_test_row = QHBoxLayout(); gq_test_row.setSpacing(5)
        self._groq_fetch_btn = QPushButton("🔄 Fetch Models")
        self._groq_fetch_btn.setFixedHeight(24)
        self._groq_fetch_btn.setFont(font_ui(8.5, True))
        self._groq_fetch_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._groq_fetch_btn.setStyleSheet(_TEST_BTN_STYLE)
        self._groq_fetch_btn.clicked.connect(lambda: threading.Thread(
            target=self._fetch_groq_models, daemon=True).start())
        gq_test_row.addWidget(self._groq_fetch_btn)
        self._groq_test_btn = QPushButton("Test Connection")
        self._groq_test_btn.setFixedHeight(24)
        self._groq_test_btn.setFont(font_ui(8.5, True))
        self._groq_test_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._groq_test_btn.setStyleSheet(_TEST_BTN_STYLE)
        self._groq_test_btn.clicked.connect(lambda: threading.Thread(
            target=self._test_groq, daemon=True).start())
        gq_test_row.addWidget(self._groq_test_btn)
        self._groq_test_status = _lbl("", 7, col=C.TEXT_DIM)
        gq_test_row.addWidget(self._groq_test_status)
        gk_layout.addLayout(gq_test_row)
        llm_lay.addWidget(self._groq_llm_widget)

        # ── Gemini-specific fields ──
        self._gemini_llm_widget = QWidget()
        self._gemini_llm_widget.setStyleSheet("background: transparent;")
        gm_layout = QVBoxLayout(self._gemini_llm_widget)
        gm_layout.setContentsMargins(0, 0, 0, 0)
        gm_layout.setSpacing(4)
        gm_row = QHBoxLayout(); gm_row.setSpacing(5)
        gm_row.addWidget(_lbl("API Key:", 7, col=C.TEXT_MED))
        self._gemini_llm_key_input = _input("AIzaSy...  (Gemini API key)", pw=True)
        self._gemini_llm_key_input.setText(_init.get("gemini_api_key", ""))
        gm_row.addWidget(self._gemini_llm_key_input)
        gm_layout.addLayout(gm_row)
        gm_model_row = QHBoxLayout(); gm_model_row.setSpacing(5)
        gm_model_row.addWidget(_lbl("Model:", 7, col=C.TEXT_MED))
        self._gemini_model_combo = QComboBox()
        self._gemini_model_combo.setFixedHeight(28)
        self._gemini_model_combo.setStyleSheet(_COMBO_STYLE)
        _GEMINI_MODELS = [
            ("gemini-2.5-flash", "Gemini 2.5 Flash (vision, 1,500 RPD free, stable)"),
            ("gemini-2.5-flash-lite", "Gemini 2.5 Flash Lite (1,500 RPD free)"),
            ("gemini-3.1-flash-lite", "Gemini 3.1 Flash Lite (5,000 RPD free, text)"),
            ("gemini-3-flash-preview", "Gemini 3 Flash Preview (1M ctx, free tier)"),
            ("gemini-3.5-flash-preview", "Gemini 3.5 Flash Preview (free tier, vision)"),
            ("gemini-3.1-flash-live-preview", "Gemini 3.1 Flash Live Preview (WebSocket)"),
            ("gemini-2.5-flash-native-audio-preview-12-2025", "Gemini 2.5 Flash Audio (Live, Mark-L stable)"),
        ]
        for val, lbl in _GEMINI_MODELS:
            self._gemini_model_combo.addItem(lbl, userData=val)
        self._gemini_model_combo.addItem("── Custom ──", userData="__CUSTOM__")
        self._gemini_custom_input = _input("Custom model ID")
        _cur_ge = _init.get("gemini_model", "gemini-2.5-flash")
        _found_ge = False
        for i in range(self._gemini_model_combo.count()):
            if self._gemini_model_combo.itemData(i) == _cur_ge:
                self._gemini_model_combo.setCurrentIndex(i)
                _found_ge = True
                break
        if not _found_ge:
            self._gemini_model_combo.setCurrentIndex(self._gemini_model_combo.count() - 1)
            self._gemini_custom_input.setText(_cur_ge)
        self._gemini_custom_input.setVisible(not _found_ge and _cur_ge != "gemini-2.5-flash")
        gm_model_row.addWidget(self._gemini_model_combo, stretch=2)
        gm_model_row.addWidget(self._gemini_custom_input, stretch=3)
        self._gemini_model_combo.currentIndexChanged.connect(self._on_gemini_model_changed)
        gm_layout.addLayout(gm_model_row)
        self._live_voice_widget = QWidget()
        self._live_voice_widget.setStyleSheet("background: transparent;")
        _lvw_lay = QHBoxLayout(self._live_voice_widget)
        _lvw_lay.setContentsMargins(0, 0, 0, 0)
        _lvw_lay.setSpacing(5)
        _lvw_lay.addWidget(_lbl("Live Voice:", 7, col=C.TEXT_MED))
        self._live_voice_combo = QComboBox()
        self._live_voice_combo.setFixedHeight(28)
        self._live_voice_combo.setStyleSheet(_COMBO_STYLE)
        _LIVE_VOICES = [
            "Puck", "Charon", "Kore", "Fenrir", "Leda", "Aoede", "Zephyr",
            "Orus", "Autonoe", "Umbriel", "Erinome", "Laomedeia", "Schedar",
            "Achird", "Sadachbia", "Enceladus", "Algieba", "Algenib", "Achernar",
            "Gacrux", "Zubenelgenubi", "Sadaltager", "Callirrhoe", "Iapetus",
            "Despina", "Rasalgethi", "Alnilam", "Pulcherrima", "Vindemiatrix",
            "Sulafat",
        ]
        _live_cur = _init.get("gemini_live_voice", "Charon")
        for _li, _lv in enumerate(_LIVE_VOICES):
            _tag = "  (recommended)" if _lv == "Charon" else ""
            self._live_voice_combo.addItem(f"{_lv}{_tag}", userData=_lv)
            if _lv == _live_cur:
                self._live_voice_combo.setCurrentIndex(_li)
        _lvw_lay.addWidget(self._live_voice_combo, stretch=2)
        self._live_voice_widget.setVisible(False)
        gm_layout.addWidget(self._live_voice_widget)
        ge_test_row = QHBoxLayout(); ge_test_row.setSpacing(5)
        self._gemini_fetch_btn = QPushButton("🔄 Fetch Models")
        self._gemini_fetch_btn.setFixedHeight(24)
        self._gemini_fetch_btn.setFont(font_ui(8.5, True))
        self._gemini_fetch_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._gemini_fetch_btn.setStyleSheet(_TEST_BTN_STYLE)
        self._gemini_fetch_btn.clicked.connect(lambda: threading.Thread(
            target=self._fetch_gemini_models, daemon=True).start())
        ge_test_row.addWidget(self._gemini_fetch_btn)
        self._gemini_test_btn = QPushButton("Test Connection")
        self._gemini_test_btn.setFixedHeight(24)
        self._gemini_test_btn.setFont(font_ui(8.5, True))
        self._gemini_test_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._gemini_test_btn.setStyleSheet(_TEST_BTN_STYLE)
        self._gemini_test_btn.clicked.connect(lambda: threading.Thread(
            target=self._test_gemini, daemon=True).start())
        ge_test_row.addWidget(self._gemini_test_btn)
        self._gemini_test_status = _lbl("", 7, col=C.TEXT_DIM)
        ge_test_row.addWidget(self._gemini_test_status)
        gm_layout.addLayout(ge_test_row)
        llm_lay.addWidget(self._gemini_llm_widget)

        # ── OpenRouter-specific fields ──
        self._or_widget = QWidget()
        self._or_widget.setStyleSheet("background: transparent;")
        or_layout = QVBoxLayout(self._or_widget)
        or_layout.setContentsMargins(0, 0, 0, 0)
        or_layout.setSpacing(6)

        or_key_row = QHBoxLayout(); or_key_row.setSpacing(5)
        or_key_row.addWidget(_lbl("API Key:", 7, col=C.TEXT_MED,
                                   align=Qt.AlignmentFlag.AlignRight))
        self._or_key_input = _input("sk-or-v1-...  (OpenRouter API key)", pw=True)
        self._or_key_input.setText(_init.get("or_api_key", ""))
        or_key_row.addWidget(self._or_key_input)
        or_layout.addLayout(or_key_row)

        or_model_row = QHBoxLayout(); or_model_row.setSpacing(5)
        or_model_row.addWidget(_lbl("Model:", 7, col=C.TEXT_MED,
                                    align=Qt.AlignmentFlag.AlignRight))
        self._or_model_combo = QComboBox()
        self._or_model_combo.setFixedHeight(28)
        self._or_model_combo.setStyleSheet(_COMBO_STYLE)
        for val, label in [
            ("openrouter/free", "Auto-router (free, vision, tools)"),
            ("google/gemma-4-31b-it:free", "Gemma 4 31B (262K, vision)"),
            ("nvidia/nemotron-3-super-120b-a12b:free", "Nemotron 3 Super 120B (1M, tools)"),
            ("openai/gpt-oss-120b:free", "GPT-OSS 120B (131K, tools)"),
            ("google/gemma-4-26b-a4b-it:free", "Gemma 4 26B A4B (262K, vision, tools)"),
            ("qwen/qwen3-coder:free", "Qwen3 Coder (1M, coding)"),
            ("openai/gpt-oss-20b:free", "GPT-OSS 20B (131K, tools)"),
            ("nvidia/nemotron-3-nano-30b-a3b:free", "Nemotron 3 Nano 30B (256K, tools)"),
            ("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", "Nemotron 3 Nano Omni (256K, vision, reasoning)"),
            ("qwen/qwen3-next-80b-a3b-instruct:free", "Qwen3 Next 80B (262K, tools)"),
            ("nvidia/nemotron-nano-12b-v2-vl:free", "Nemotron Nano 12B V2 VL (128K, vision)"),
            ("meta-llama/llama-3.3-70b-instruct:free", "Llama 3.3 70B (131K, tools)"),
            ("google/lyria-3-pro-preview", "Lyria 3 Pro Preview (1M, vision)"),
            ("google/lyria-3-clip-preview", "Lyria 3 Clip Preview (1M, vision)"),
            ("nvidia/nemotron-3-ultra-550b-a55b:free", "Nemotron 3 Ultra 550B (1M, tools)"),
            ("poolside/laguna-m.1:free", "Laguna M.1 (262K, coding)"),
            ("poolside/laguna-xs-2.1:free", "Laguna XS.2 (262K, coding)"),
            ("tencent/hy3:free", "Tencent HY3 (262K, tools)"),
            ("cohere/north-mini-code:free", "Cohere North Mini Code (256K, coding)"),
            ("nousresearch/hermes-3-llama-3.1-405b:free", "Hermes 3 405B (131K)"),
        ]:
            self._or_model_combo.addItem(label, userData=val)
        self._or_model_combo.addItem("── Custom ──", userData="__CUSTOM__")
        self._or_custom_input = _input("Custom model ID  (e.g. provider/model:free)")
        _cur_or = _init.get("or_model", "openrouter/free")
        _found_or = False
        for i in range(self._or_model_combo.count()):
            if self._or_model_combo.itemData(i) == _cur_or:
                self._or_model_combo.setCurrentIndex(i)
                _found_or = True
                break
        if not _found_or:
            self._or_model_combo.setCurrentIndex(self._or_model_combo.count() - 1)
            self._or_custom_input.setText(_cur_or)
        self._or_custom_input.setVisible(not _found_or and _cur_or != "openrouter/free")
        or_model_row.addWidget(self._or_model_combo, stretch=2)
        or_model_row.addWidget(self._or_custom_input, stretch=3)
        self._or_model_combo.currentIndexChanged.connect(self._on_or_model_changed)
        or_layout.addLayout(or_model_row)

        or_test_row = QHBoxLayout(); or_test_row.setSpacing(5)
        self._or_fetch_btn = QPushButton("🔄 Fetch Models")
        self._or_fetch_btn.setFixedHeight(24)
        self._or_fetch_btn.setFont(font_ui(8.5, True))
        self._or_fetch_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._or_fetch_btn.setStyleSheet(_TEST_BTN_STYLE)
        self._or_fetch_btn.clicked.connect(lambda: threading.Thread(
            target=self._fetch_or_models, daemon=True).start())
        or_test_row.addWidget(self._or_fetch_btn)
        self._or_test_btn = QPushButton("Test Connection")
        self._or_test_btn.setFixedHeight(24)
        self._or_test_btn.setFont(font_ui(8.5, True))
        self._or_test_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._or_test_btn.setStyleSheet(_TEST_BTN_STYLE)
        self._or_test_btn.clicked.connect(lambda: threading.Thread(
            target=self._test_or, daemon=True).start())
        or_test_row.addWidget(self._or_test_btn)
        self._or_test_status = _lbl("", 7, col=C.TEXT_DIM)
        or_test_row.addWidget(self._or_test_status)
        or_layout.addLayout(or_test_row)

        or_layout.addWidget(_sep())
        llm_lay.addWidget(self._or_widget)

        # ── NVIDIA-specific fields ──
        self._nvidia_widget = QWidget()
        self._nvidia_widget.setStyleSheet("background: transparent;")
        nv_layout = QVBoxLayout(self._nvidia_widget)
        nv_layout.setContentsMargins(0, 0, 0, 0)
        nv_layout.setSpacing(6)

        nv_key_row = QHBoxLayout(); nv_key_row.setSpacing(5)
        nv_key_row.addWidget(_lbl("API Key:", 7, col=C.TEXT_MED,
                                   align=Qt.AlignmentFlag.AlignRight))
        self._nv_key_input = _input("nvapi-...  (NVIDIA API key)", pw=True)
        self._nv_key_input.setText(_init.get("nvidia_api_key", ""))
        nv_key_row.addWidget(self._nv_key_input)
        nv_layout.addLayout(nv_key_row)

        nv_model_row = QHBoxLayout(); nv_model_row.setSpacing(5)
        nv_model_row.addWidget(_lbl("Model:", 7, col=C.TEXT_MED,
                                    align=Qt.AlignmentFlag.AlignRight))
        self._nv_model_combo = QComboBox()
        self._nv_model_combo.setFixedHeight(28)
        self._nv_model_combo.setStyleSheet(_COMBO_STYLE)
        for val, label in [
            ("qwen/qwen3.5-397b-a17b", "Qwen 3.5 397B (131K, tools, top model)"),
            ("nvidia/nemotron-3-super-120b-a12b", "Nemotron 3 Super 120B (1M, tools)"),
            ("nvidia/nemotron-3-ultra-550b-a55b", "Nemotron 3 Ultra 550B (1M, tools)"),
            ("nvidia/nemotron-3-nano-30b-a3b", "Nemotron 3 Nano 30B (256K, tools)"),
            ("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning", "Nemotron 3 Nano Omni (256K, vision, reasoning)"),
            ("nvidia/nemotron-nano-12b-v2-vl", "Nemotron Nano 12B V2 VL (128K, vision)"),
        ]:
            self._nv_model_combo.addItem(label, userData=val)
        self._nv_model_combo.addItem("── Custom ──", userData="__CUSTOM__")
        self._nv_custom_input = _input("Custom model ID  (e.g. provider/model)")
        _cur_nv = _init.get("nvidia_model", "qwen/qwen3.5-397b-a17b")
        _found_nv = False
        for i in range(self._nv_model_combo.count()):
            if self._nv_model_combo.itemData(i) == _cur_nv:
                self._nv_model_combo.setCurrentIndex(i)
                _found_nv = True
                break
        if not _found_nv:
            self._nv_model_combo.setCurrentIndex(self._nv_model_combo.count() - 1)
            self._nv_custom_input.setText(_cur_nv)
        self._nv_custom_input.setVisible(not _found_nv and _cur_nv != "qwen/qwen3.5-397b-a17b")
        nv_model_row.addWidget(self._nv_model_combo, stretch=2)
        nv_model_row.addWidget(self._nv_custom_input, stretch=3)
        self._nv_model_combo.currentIndexChanged.connect(self._on_nv_model_changed)
        nv_layout.addLayout(nv_model_row)

        nv_btn_row = QHBoxLayout(); nv_btn_row.setSpacing(5)
        self._nv_fetch_btn = QPushButton("🔄 Fetch Models")
        self._nv_fetch_btn.setFixedHeight(24)
        self._nv_fetch_btn.setFont(font_ui(8.5, True))
        self._nv_fetch_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._nv_fetch_btn.setStyleSheet(_TEST_BTN_STYLE)
        self._nv_fetch_btn.clicked.connect(lambda: threading.Thread(
            target=self._fetch_nvidia_models, daemon=True).start())
        nv_btn_row.addWidget(self._nv_fetch_btn)
        self._nv_test_btn = QPushButton("Test Connection")
        self._nv_test_btn.setFixedHeight(24)
        self._nv_test_btn.setFont(font_ui(8.5, True))
        self._nv_test_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._nv_test_btn.setStyleSheet(_TEST_BTN_STYLE)
        self._nv_test_btn.clicked.connect(lambda: threading.Thread(
            target=self._test_nvidia, daemon=True).start())
        nv_btn_row.addWidget(self._nv_test_btn)
        self._nv_test_status = _lbl("", 7, col=C.TEXT_DIM)
        nv_btn_row.addWidget(self._nv_test_status)
        nv_layout.addLayout(nv_btn_row)

        nv_layout.addWidget(_sep())
        llm_lay.addWidget(self._nvidia_widget)

        # ── Kilo-specific fields ──
        self._kilo_widget = QWidget()
        self._kilo_widget.setStyleSheet("background: transparent;")
        kilo_layout = QVBoxLayout(self._kilo_widget)
        kilo_layout.setContentsMargins(0, 0, 0, 0)
        kilo_layout.setSpacing(6)

        kilo_key_row = QHBoxLayout(); kilo_key_row.setSpacing(5)
        kilo_key_row.addWidget(_lbl("API Key:", 7, col=C.TEXT_MED,
                                   align=Qt.AlignmentFlag.AlignRight))
        self._kilo_key_input = _input("JWT  (Kilo API key)", pw=True)
        self._kilo_key_input.setText(_init.get("kilo_api_key", ""))
        kilo_key_row.addWidget(self._kilo_key_input)
        kilo_layout.addLayout(kilo_key_row)

        kilo_model_row = QHBoxLayout(); kilo_model_row.setSpacing(5)
        kilo_model_row.addWidget(_lbl("Model:", 7, col=C.TEXT_MED,
                                    align=Qt.AlignmentFlag.AlignRight))
        self._kilo_model_combo = QComboBox()
        self._kilo_model_combo.setFixedHeight(28)
        self._kilo_model_combo.setStyleSheet(_COMBO_STYLE)
        for val, label in [
            ("kilo-auto/free", "Kilo Auto Free (best free, vision)"),
            ("openrouter/free", "OpenRouter Free (best available)"),
            ("nvidia/nemotron-3-ultra-550b-a55b:free", "Nemotron 3 Ultra 550B (1M)"),
            ("stepfun/step-3.7-flash:free", "StepFun 3.7 Flash (free)"),
            ("tencent/hy3:free", "Tencent Hy3 (free)"),
            ("poolside/laguna-s-2.1:free", "Laguna S 2.1 (free)"),
        ]:
            self._kilo_model_combo.addItem(label, userData=val)
        self._kilo_model_combo.addItem("── Custom ──", userData="__CUSTOM__")
        self._kilo_custom_input = _input("Custom model ID  (e.g. provider/model:free)")
        _cur_kilo = _init.get("kilo_model", "kilo-auto/free")
        _found_kilo = False
        for i in range(self._kilo_model_combo.count()):
            if self._kilo_model_combo.itemData(i) == _cur_kilo:
                self._kilo_model_combo.setCurrentIndex(i)
                _found_kilo = True
                break
        if not _found_kilo:
            self._kilo_model_combo.setCurrentIndex(self._kilo_model_combo.count() - 1)
            self._kilo_custom_input.setText(_cur_kilo)
        self._kilo_custom_input.setVisible(not _found_kilo and _cur_kilo != "kilo-auto/free")
        kilo_model_row.addWidget(self._kilo_model_combo, stretch=2)
        kilo_model_row.addWidget(self._kilo_custom_input, stretch=3)
        self._kilo_model_combo.currentIndexChanged.connect(self._on_kilo_model_changed)
        kilo_layout.addLayout(kilo_model_row)

        kilo_btn_row = QHBoxLayout(); kilo_btn_row.setSpacing(5)
        self._kilo_fetch_btn = QPushButton("🔄 Fetch Models")
        self._kilo_fetch_btn.setFixedHeight(24)
        self._kilo_fetch_btn.setFont(font_ui(8.5, True))
        self._kilo_fetch_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._kilo_fetch_btn.setStyleSheet(_TEST_BTN_STYLE)
        self._kilo_fetch_btn.clicked.connect(lambda: threading.Thread(
            target=self._fetch_kilo_models, daemon=True).start())
        kilo_btn_row.addWidget(self._kilo_fetch_btn)
        self._kilo_test_btn = QPushButton("Test Connection")
        self._kilo_test_btn.setFixedHeight(24)
        self._kilo_test_btn.setFont(font_ui(8.5, True))
        self._kilo_test_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._kilo_test_btn.setStyleSheet(_TEST_BTN_STYLE)
        self._kilo_test_btn.clicked.connect(lambda: threading.Thread(
            target=self._test_kilo, daemon=True).start())
        kilo_btn_row.addWidget(self._kilo_test_btn)
        self._kilo_test_status = _lbl("", 7, col=C.TEXT_DIM)
        kilo_btn_row.addWidget(self._kilo_test_status)
        kilo_layout.addLayout(kilo_btn_row)

        kilo_layout.addWidget(_sep())
        llm_lay.addWidget(self._kilo_widget)

        # ── Web Search Provider ──
        self._websearch_widget = QWidget()
        self._websearch_widget.setStyleSheet("background: transparent;")
        ws_layout = QVBoxLayout(self._websearch_widget)
        ws_layout.setContentsMargins(0, 0, 0, 0)
        ws_layout.setSpacing(6)

        ws_provider_row = QHBoxLayout(); ws_provider_row.setSpacing(5)
        ws_provider_row.addWidget(_lbl("Provider:", 7, col=C.TEXT_MED,
                                       align=Qt.AlignmentFlag.AlignRight))
        self._websearch_provider_combo = QComboBox()
        self._websearch_provider_combo.setFixedHeight(28)
        self._websearch_provider_combo.setStyleSheet(_COMBO_STYLE)
        self._websearch_provider_combo.addItem("Exa (AI-optimized, with DuckDuckGo fallback)", userData="exa")
        self._websearch_provider_combo.addItem("DuckDuckGo Only (free, no API key)", userData="duckduckgo")
        _cur_provider = _init.get("web_search_provider", "exa")
        for i in range(self._websearch_provider_combo.count()):
            if self._websearch_provider_combo.itemData(i) == _cur_provider:
                self._websearch_provider_combo.setCurrentIndex(i)
                break
        ws_provider_row.addWidget(self._websearch_provider_combo)
        ws_layout.addLayout(ws_provider_row)

        ws_key_row = QHBoxLayout(); ws_key_row.setSpacing(5)
        ws_key_row.addWidget(_lbl("Exa API Key:", 7, col=C.TEXT_MED,
                                  align=Qt.AlignmentFlag.AlignRight))
        self._exa_key_input = _input("API key from dashboard.exa.ai", pw=True)
        self._exa_key_input.setText(_init.get("exa_api_key", ""))
        ws_key_row.addWidget(self._exa_key_input)
        ws_layout.addLayout(ws_key_row)

        ws_type_row = QHBoxLayout(); ws_type_row.setSpacing(5)
        ws_type_row.addWidget(_lbl("Search Type:", 7, col=C.TEXT_MED,
                                   align=Qt.AlignmentFlag.AlignRight))
        self._exa_type_combo = QComboBox()
        self._exa_type_combo.setFixedHeight(28)
        self._exa_type_combo.setStyleSheet(_COMBO_STYLE)
        self._exa_type_combo.addItem("Fast (quick results)", userData="fast")
        self._exa_type_combo.addItem("Auto (balanced) — default", userData="auto")
        self._exa_type_combo.addItem("Deep (thorough research)", userData="deep")
        _cur_type = _init.get("exa_default_type", "auto")
        for i in range(self._exa_type_combo.count()):
            if self._exa_type_combo.itemData(i) == _cur_type:
                self._exa_type_combo.setCurrentIndex(i)
                break
        ws_type_row.addWidget(self._exa_type_combo)
        ws_layout.addLayout(ws_type_row)

        ws_info_row = QHBoxLayout(); ws_info_row.setSpacing(5)
        ws_info_row.addWidget(_lbl("ℹ️ Exa: ~$7/1000 searches | Free tier: $10/month | Auto-fallback to DDG", 7, col=C.TEXT_DIM))
        ws_layout.addLayout(ws_info_row)

        ws_layout.addWidget(_sep())
        llm_lay.addWidget(self._websearch_widget)

        # Show/hide provider-specific fields based on current selection
        _sel = self._sel_llm_provider
        self._cf_widget.setVisible(_sel == "cloudflare")
        self._groq_llm_widget.setVisible(_sel == "groq")
        self._gemini_llm_widget.setVisible(_sel == "gemini")
        self._or_widget.setVisible(_sel == "openrouter")
        self._nvidia_widget.setVisible(_sel == "nvidia")
        self._kilo_widget.setVisible(_sel == "kilo")
        self._llm_url_input.setVisible(_sel in ("ollama", "openai"))
        self._llm_model_combo.setVisible(_sel in ("ollama", "openai"))
        self._llm_custom_input.setVisible(_sel in ("ollama", "openai"))
        if hasattr(self, "_ollama_fetch_btn"):
            self._ollama_fetch_btn.setVisible(_sel == "ollama")
            self._ollama_test_btn.setVisible(_sel == "ollama")
            self._ollama_test_status.setVisible(_sel == "ollama")

        layout.addWidget(_collapsible_card("LANGUAGE MODEL", llm_w))

        # ======
        #  CARD: SUMMARIZATION MODEL (dedicated, separate from main LLM)
        # ======
        summ_w = QWidget(); summ_w.setStyleSheet("background: transparent;")
        summ_lay = QVBoxLayout(summ_w); summ_lay.setContentsMargins(0,0,0,0); summ_lay.setSpacing(6)

        summ_lay.addWidget(_lbl(
            "Dedicated model for conversation summaries (session compression, "
            "rolling context). Runs separately from the main LLM.",
            7, col=C.TEXT_DIM, align=Qt.AlignmentFlag.AlignLeft
        ))

        summ_prov_row, self._summ_prov_btns = _toggle_row(
            [
                ("ollama", "🦙 Ollama"),
                ("gemini", "🌀 Gemini"),
                ("auto", "⚡ Auto (main)"),
            ],
            lambda: self._sel_summarization_provider,
            self._set_summarization_provider,
        )
        summ_lay.addLayout(summ_prov_row)

        summ_model_row = QHBoxLayout(); summ_model_row.setSpacing(5)
        summ_model_row.addWidget(_lbl("Model:", 7, col=C.TEXT_MED,
                                      align=Qt.AlignmentFlag.AlignRight))
        self._summ_model_input = _input("e.g.  gemma4:31b-cloud")
        self._summ_model_input.setText(_init.get("summarization_model", ""))
        summ_model_row.addWidget(self._summ_model_input, stretch=2)
        summ_lay.addLayout(summ_model_row)

        summ_url_row = QHBoxLayout(); summ_url_row.setSpacing(5)
        summ_url_row.addWidget(_lbl("URL:", 7, col=C.TEXT_MED,
                                    align=Qt.AlignmentFlag.AlignRight))
        self._summ_url_input = _input("http://localhost:11434")
        self._summ_url_input.setText(_init.get("summarization_url", "http://localhost:11434"))
        summ_url_row.addWidget(self._summ_url_input, stretch=2)
        self._summ_key_input = _input("Gemini API key (summarization)", pw=True)
        self._summ_key_input.setText(_init.get("summarization_api_key", ""))
        summ_url_row.addWidget(self._summ_key_input, stretch=2)
        summ_lay.addLayout(summ_url_row)

        _summ_sel = self._sel_summarization_provider
        self._summ_url_input.setVisible(_summ_sel == "ollama")
        self._summ_key_input.setVisible(_summ_sel == "gemini")

        layout.addWidget(_collapsible_card("SUMMARIZATION MODEL", summ_w))

        # ======
        #  CARD 3: VISION & SCREENSHOT
        # ======
        vis_w = QWidget(); vis_w.setStyleSheet("background: transparent;")
        vis_lay = QVBoxLayout(vis_w); vis_lay.setContentsMargins(0,0,0,0); vis_lay.setSpacing(6)

        md_row = QHBoxLayout(); md_row.setSpacing(5)
        md_row.addWidget(_lbl("Moondream API Key:", 7, col=C.TEXT_MED))
        self._moondream_key_input = _input("Moondream API key", pw=True)
        self._moondream_key_input.setText(_init.get("moondream_api_key", ""))
        md_row.addWidget(self._moondream_key_input)
        vis_lay.addLayout(md_row)

        vis_lay.addWidget(_sep())

        sq_row = QHBoxLayout(); sq_row.setSpacing(5)
        sq_row.addWidget(_lbl("LLM Vision (screen_process):", 7, col=C.TEXT_MED))
        self._llm_quality_spin = QSpinBox()
        self._llm_quality_spin.setRange(1, 100)
        self._llm_quality_spin.setValue(int(_init.get("screenshot_quality_llm", 65)))
        self._llm_quality_spin.setSuffix("%")
        self._llm_quality_spin.setFixedWidth(80)
        sq_row.addWidget(self._llm_quality_spin)
        vis_lay.addLayout(sq_row)

        sq_row2 = QHBoxLayout(); sq_row2.setSpacing(5)
        sq_row2.addWidget(_lbl("Moondream (locate/find):", 7, col=C.TEXT_MED))
        self._md_quality_spin = QSpinBox()
        self._md_quality_spin.setRange(1, 100)
        self._md_quality_spin.setValue(int(_init.get("screenshot_quality_moondream", 85)))
        self._md_quality_spin.setSuffix("%")
        self._md_quality_spin.setFixedWidth(80)
        sq_row2.addWidget(self._md_quality_spin)
        vis_lay.addLayout(sq_row2)

        layout.addWidget(_collapsible_card("VISION & SCREENSHOT", vis_w))

        # ======
        #  CARD 4: TEXT-TO-SPEECH
        # ======
        tts_w = QWidget(); tts_w.setStyleSheet("background: transparent;")
        tts_lay = QVBoxLayout(tts_w); tts_lay.setContentsMargins(0,0,0,0); tts_lay.setSpacing(6)

        tts_row, self._tts_btns = _toggle_row(
            [("edgetts","🔊 EdgeTTS"), ("kokoro","🫥 Kokoro"), ("elevenlabs","⚡ ElevenLabs"), ("googlecloud","🗣️ Google Cloud"), ("gtts","🌐 Google Translate"), ("gemini","🎤 Gemini"), ("gemini_live","⚡ Gemini Live"), ("hashim_live","⚡ Hashim Live")],
            lambda: self._sel_tts,
            self._set_tts,
        )
        tts_lay.addLayout(tts_row)

        voice_row = QHBoxLayout(); voice_row.setSpacing(5)
        self._voice_lbl = QLabel("Voice:")
        self._voice_lbl.setFont(font_ui(8.5))
        self._voice_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._voice_lbl.setStyleSheet(f"color: {C.TEXT_MED}; background: transparent;")
        voice_row.addWidget(self._voice_lbl)

        self._gcloud_combo = QComboBox()
        self._gcloud_combo.setFixedHeight(28)
        self._gcloud_combo.setStyleSheet(_COMBO_STYLE)
        _GCLOUD_VOICES = [
            ("en-US-Chirp-HD-F", "en-US-Chirp-HD-F (Female)"),
            ("en-US-Chirp-HD-D", "en-US-Chirp-HD-D (Male)"),
            ("en-US-Chirp-HD-O", "en-US-Chirp-HD-O (Narration)"),
            ("hi-IN-Chirp3-HD-Kore", "hi-IN-Chirp3-HD-Kore (Female)"),
            ("bn-IN-Chirp3-HD-Kore", "bn-IN-Chirp3-HD-Kore (Female)"),
            ("ar-XA-Chirp3-HD-Kore", "ar-XA-Chirp3-HD-Kore (Female)"),
            ("en-US-Studio-O", "en-US-Studio-O (Narration)"),
            ("hi-IN-Neural2-A", "hi-IN-Neural2-A (Female)"),
            ("hi-IN-Neural2-B", "hi-IN-Neural2-B (Male)"),
            ("hi-IN-Neural2-C", "hi-IN-Neural2-C (Female)"),
            ("hi-IN-Neural2-D", "hi-IN-Neural2-D (Male)"),
            ("en-US-Neural2-D", "en-US-Neural2-D (Male)"),
            ("en-US-Neural2-F", "en-US-Neural2-F (Female)"),
            ("en-GB-Neural2-A", "en-GB-Neural2-A (Female)"),
            ("en-GB-Neural2-B", "en-GB-Neural2-B (Male)"),
            ("en-GB-Neural2-C", "en-GB-Neural2-C (Female)"),
            ("en-GB-Neural2-D", "en-GB-Neural2-D (Male)"),
            ("en-GB-Neural2-F", "en-GB-Neural2-F (Female)"),
            ("en-AU-Neural2-A", "en-AU-Neural2-A (Female)"),
            ("en-AU-Neural2-B", "en-AU-Neural2-B (Male)"),
            ("fr-FR-Neural2-A", "fr-FR-Neural2-A (Female)"),
            ("fr-FR-Neural2-C", "fr-FR-Neural2-C (Female)"),
            ("ja-JP-Neural2-B", "ja-JP-Neural2-B (Male)"),
            ("ja-JP-Neural2-C", "ja-JP-Neural2-C (Female)"),
            ("ko-KR-Neural2-C", "ko-KR-Neural2-C (Female)"),
            ("ko-KR-Neural2-D", "ko-KR-Neural2-D (Male)"),
            ("en-US-Wavenet-A", "en-US-Wavenet-A (Female)"),
            ("en-US-Wavenet-B", "en-US-Wavenet-B (Male)"),
            ("en-US-Wavenet-C", "en-US-Wavenet-C (Female)"),
            ("en-US-Wavenet-D", "en-US-Wavenet-D (Male)"),
        ]
        for val, display in _GCLOUD_VOICES:
            self._gcloud_combo.addItem(display, userData=val)
        _gcloud_cur = _init.get("tts_voice", "hi-IN-Neural2-A")
        for i in range(self._gcloud_combo.count()):
            if self._gcloud_combo.itemData(i) == _gcloud_cur:
                self._gcloud_combo.setCurrentIndex(i)
                break
        self._gcloud_combo.setVisible(False)
        voice_row.addWidget(self._gcloud_combo)


        self._kokoro_combo = QComboBox()
        self._kokoro_combo.setFixedHeight(28)
        self._kokoro_combo.setStyleSheet(_COMBO_STYLE)
        _KOKORO_VOICES = [
            ("af_heart",    "af_heart  — EN-F warm (recommended)"),
            ("af_sky",      "af_sky  — EN-F clear"),
            ("af_bella",    "af_bella  — EN-F bella"),
            ("af_sarah",    "af_sarah  — EN-F sarah"),
            ("am_adam",     "am_adam  — EN-M adam"),
            ("am_michael",  "am_michael  — EN-M michael"),
            ("bf_emma",     "bf_emma  — UK-F emma"),
            ("bf_isabella", "bf_isabella  — UK-F isabella"),
            ("bm_george",   "bm_george  — UK-M george"),
            ("bm_lewis",    "bm_lewis  — UK-M lewis"),
            ("jf_alpha",    "jf_alpha  — Japanese Female"),
            ("jm_kumo",     "jm_kumo  — Japanese Male"),
        ]
        for val, display in _KOKORO_VOICES:
            self._kokoro_combo.addItem(display, userData=val)
        _cur_voice = _init.get("tts_voice", "af_heart")
        for i in range(self._kokoro_combo.count()):
            if self._kokoro_combo.itemData(i) == _cur_voice:
                self._kokoro_combo.setCurrentIndex(i)
                break
        self._kokoro_combo.setVisible(False)
        voice_row.addWidget(self._kokoro_combo)

        self._gemini_voice_combo = QComboBox()
        self._gemini_voice_combo.setFixedHeight(28)
        self._gemini_voice_combo.setStyleSheet(_COMBO_STYLE)
        _GEMINI_VOICES = [
            "Puck", "Charon", "Kore", "Fenrir", "Leda", "Aoede", "Zephyr",
            "Orus", "Autonoe", "Umbriel", "Erinome", "Laomedeia", "Schedar",
            "Achird", "Sadachbia", "Enceladus", "Algieba", "Algenib", "Achernar",
            "Gacrux", "Zubenelgenubi", "Sadaltager", "Callirrhoe", "Iapetus",
            "Despina", "Rasalgethi", "Alnilam", "Pulcherrima", "Vindemiatrix",
            "Sulafat",
        ]
        _gem_cur = _init.get("gemini_tts_voice", "Puck")
        for _gi, _gv in enumerate(_GEMINI_VOICES):
            _tag = ""
            if _gv == "Puck":
                _tag = "  (default)"
            self._gemini_voice_combo.addItem(f"{_gv}{_tag}", userData=_gv)
            if _gv == _gem_cur:
                self._gemini_voice_combo.setCurrentIndex(_gi)
        self._gemini_voice_combo.setVisible(False)
        voice_row.addWidget(self._gemini_voice_combo)

        self._gtts_combo = QComboBox()
        self._gtts_combo.setFixedHeight(28)
        self._gtts_combo.setStyleSheet(_COMBO_STYLE)
        _GTTS_VOICES = [
            ("en",   "English (Default)"),
            ("en-US","English (US)"),
            ("en-GB","English (UK)"),
            ("en-AU","English (Australia)"),
            ("hi",   "Hindi"),
            ("fr",   "French"),
            ("es",   "Spanish"),
            ("de",   "German"),
            ("ja",   "Japanese"),
            ("ko",   "Korean"),
            ("zh-CN","Chinese (Simplified)"),
            ("ar",   "Arabic"),
            ("pt",   "Portuguese"),
            ("ru",   "Russian"),
            ("tr",   "Turkish"),
            ("bn",   "Bengali"),
            ("id",   "Indonesian"),
            ("nl",   "Dutch"),
            ("pl",   "Polish"),
            ("sv",   "Swedish"),
            ("uk",   "Ukrainian"),
            ("vi",   "Vietnamese"),
            ("it",   "Italian"),
        ]
        _gtts_cur = _init.get("tts_voice", "hi")
        for _gi, (_code, _label) in enumerate(_GTTS_VOICES):
            self._gtts_combo.addItem(_label, userData=_code)
            if _code == _gtts_cur:
                self._gtts_combo.setCurrentIndex(_gi)
        self._gtts_combo.setVisible(False)
        voice_row.addWidget(self._gtts_combo)

        _tts_voice_default = _init.get("tts_voice", "en-US-GuyNeural")
        self._tts_voice_input = _input("en-US-GuyNeural  /  en-GB-RyanNeural  /  tr-TR-AhmetNeural  ...")
        self._tts_voice_input.setText(_tts_voice_default)
        voice_row.addWidget(self._tts_voice_input)

        tts_lay.addLayout(voice_row)

        self._barge_widget = QWidget()
        self._barge_widget.setStyleSheet("background: transparent;")
        br_row = QHBoxLayout(self._barge_widget)
        br_row.setContentsMargins(0, 0, 0, 0)
        br_row.setSpacing(5)
        self._barge_check = QCheckBox("🗣 Barge-in  (interrupt while speaking — streaming)")
        self._barge_check.setFont(font_ui(8.5))
        self._barge_check.setChecked(bool(_init.get("tts_barge_in", True)))
        self._barge_check.setStyleSheet(
            f"QCheckBox {{ color: {C.TEXT_MED}; }}"
            f"QCheckBox::indicator {{ width: 15px; height: 15px; }}"
            f"QCheckBox::indicator:checked {{ background: {C.ACC2}; border-radius: 3px; }}"
            f"QCheckBox::indicator:unchecked {{ background: #000d12; border: 1px solid {C.BORDER}; border-radius: 3px; }}"
        )
        br_row.addWidget(self._barge_check)
        self._barge_widget.setVisible(False)
        tts_lay.addWidget(self._barge_widget)

        # Echo guard: ON = mic echo filtering (gate + transcript filter);
        # OFF = raw mic, no echo rejection (user trust: TRRS splitter,
        # headphones, or external AEC already handles it).
        self._echo_guard_check = QCheckBox("🛡 Echo filter  (drop speech that matches the TTS reply — OFF if your setup has no echo)")
        self._echo_guard_check.setFont(font_ui(8.5))
        self._echo_guard_check.setChecked(
            str(_init.get("echo_guard_mode", "auto") or "auto").lower() != "off")
        self._echo_guard_check.setStyleSheet(
            f"QCheckBox {{ color: {C.TEXT_MED}; }}"
            f"QCheckBox::indicator {{ width: 15px; height: 15px; }}"
            f"QCheckBox::indicator:checked {{ background: {C.ACC2}; border-radius: 3px; }}"
            f"QCheckBox::indicator:unchecked {{ background: #000d12; border: 1px solid {C.BORDER}; border-radius: 3px; }}"
        )
        tts_lay.addWidget(self._echo_guard_check)

        self._kokoro_speed_widget = QWidget()
        self._kokoro_speed_widget.setStyleSheet("background: transparent;")
        ks_row = QHBoxLayout(self._kokoro_speed_widget)
        ks_row.setContentsMargins(0, 0, 0, 0)
        ks_row.setSpacing(5)
        ks_row.addWidget(_lbl("Speed:", 7, col=C.TEXT_MED,
                               align=Qt.AlignmentFlag.AlignRight))
        self._kokoro_speed_combo = QComboBox()
        self._kokoro_speed_combo.setFixedHeight(28)
        self._kokoro_speed_combo.setStyleSheet(_COMBO_STYLE)
        for val, label in [
            ("0.8",  "0.8x  — Slow"),
            ("1.0",  "1.0x  — Normal"),
            ("1.1",  "1.1x  — Slightly fast"),
            ("1.2",  "1.2x  — Fast (recommended)"),
            ("1.3",  "1.3x  — Faster"),
            ("1.5",  "1.5x  — Very fast"),
        ]:
            self._kokoro_speed_combo.addItem(label, userData=val)
        _cur_speed = str(_init.get("tts_speed", "1.2"))
        for i in range(self._kokoro_speed_combo.count()):
            if self._kokoro_speed_combo.itemData(i) == _cur_speed:
                self._kokoro_speed_combo.setCurrentIndex(i)
                break
        ks_row.addWidget(self._kokoro_speed_combo)
        tts_lay.addWidget(self._kokoro_speed_widget)

        self._el_key_widget = QWidget()
        self._el_key_widget.setStyleSheet("background: transparent;")
        el_row = QHBoxLayout(self._el_key_widget)
        el_row.setContentsMargins(0, 0, 0, 0)
        el_row.setSpacing(5)
        el_row.addWidget(_lbl("API Key:", 7, col=C.TEXT_MED,
                               align=Qt.AlignmentFlag.AlignRight))
        self._el_key_input = _input("ElevenLabs API key", pw=True)
        self._el_key_input.setText(_init.get("elevenlabs_api_key", ""))
        el_row.addWidget(self._el_key_input)
        tts_lay.addWidget(self._el_key_widget)

        self._tts_voice_key_widget = QWidget()
        self._tts_voice_key_widget.setStyleSheet("background: transparent;")
        tvk_row = QHBoxLayout(self._tts_voice_key_widget)
        tvk_row.setContentsMargins(0, 0, 0, 0)
        tvk_row.setSpacing(5)
        tvk_row.addWidget(_lbl("Voice API Key:", 7, col=C.TEXT_MED,
                               align=Qt.AlignmentFlag.AlignRight))
        self._tts_gemini_key_input = _input("AIzaSy...  (Voice API key — STT + TTS)", pw=True)
        self._tts_gemini_key_input.setText(_init.get("gemini_voice_api_key", ""))
        tvk_row.addWidget(self._tts_gemini_key_input)
        self._tts_voice_key_widget.setVisible(False)
        tts_lay.addWidget(self._tts_voice_key_widget)

        layout.addWidget(_collapsible_card("TEXT-TO-SPEECH", tts_w))

        self._update_tts_ui(self._sel_tts)

        # ── Fixed bottom bar ──
        bottom_bar = QWidget()
        bottom_bar.setFixedHeight(60)
        bottom_bar.setStyleSheet(f"background: #000d14; border-top: 1px solid {C.BORDER};")
        bottom_lay = QHBoxLayout(bottom_bar)
        bottom_lay.setContentsMargins(22, 10, 22, 10)
        bottom_lay.setSpacing(8)

        if mode == "config":
            cancel_btn = QPushButton("✕  CANCEL")
            cancel_btn.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            cancel_btn.setFixedHeight(36)
            cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            cancel_btn.setStyleSheet(f"""
                QPushButton {{
                    background: transparent; color: {C.TEXT_DIM};
                    border: 1px solid {C.BORDER}; border-radius: 4px;
                    padding: 0 16px;
                }}
                QPushButton:hover {{
                    color: {C.RED}; border: 1px solid {C.RED};
                }}
            """)
            cancel_btn.clicked.connect(self.hide)
            bottom_lay.addWidget(cancel_btn)

        btn_label = "▸  APPLY CHANGES" if mode == "config" else "▸  INITIALISE SYSTEMS"
        init_btn = QPushButton(btn_label)
        init_btn.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        init_btn.setFixedHeight(36)
        init_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        init_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {C.PRI};
                border: 1px solid {C.PRI_DIM}; border-radius: 4px;
                padding: 0 18px;
            }}
            QPushButton:hover {{
                background: {C.PRI_GHO}; border: 1px solid {C.PRI};
            }}
        """)
        init_btn.clicked.connect(self._submit)
        bottom_lay.addWidget(init_btn)

        main_layout.addWidget(bottom_bar)

        self._main_window = parent.window() if parent else None
        if self._main_window:
            self._main_window.installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj == self._main_window and event.type() == QEvent.Type.Resize:
            cw = self._main_window
            if cw:
                ow = max(820, int(cw.width() * 0.9))
                oh = max(600, int(cw.height() * 0.9))
                self.resize(ow, oh)
                self.move(
                    (cw.width() - ow) // 2,
                    (cw.height() - oh) // 2,
                )
        return super().eventFilter(obj, event)

    def hide(self):
        if self._main_window:
            self._main_window.removeEventFilter(self)
        super().hide()

    def _update_tts_ui(self, key: str) -> None:
        if not hasattr(self, "_voice_lbl"):
            return

        is_kokoro = (key == "kokoro")
        is_google = (key in ("googlecloud", "google"))
        is_el     = (key == "elevenlabs")
        is_gemini = (key == "gemini")
        is_gemini_live = (key == "gemini_live")
        is_hashim_live = (key == "hashim_live")
        is_gtts   = (key == "gtts")

        if hasattr(self, "_tts_voice_input"):
            self._tts_voice_input.setVisible(not (is_kokoro or is_gemini or is_gemini_live or is_hashim_live or is_gtts or is_google))
        if hasattr(self, "_kokoro_combo"):
            self._kokoro_combo.setVisible(is_kokoro)
        if hasattr(self, "_gemini_voice_combo"):
            self._gemini_voice_combo.setVisible(is_gemini or is_gemini_live or is_hashim_live)
            if is_gemini_live or is_hashim_live:
                try:
                    _live_cur = self._init.get("gemini_live_voice", "Charon")
                    for _i in range(self._gemini_voice_combo.count()):
                        if self._gemini_voice_combo.itemData(_i) == _live_cur:
                            self._gemini_voice_combo.setCurrentIndex(_i)
                            break
                except Exception:
                    pass
        if hasattr(self, "_gtts_combo"):
            self._gtts_combo.setVisible(is_gtts)
        if hasattr(self, "_gcloud_combo"):
            self._gcloud_combo.setVisible(is_google)
        if hasattr(self, "_barge_widget"):
            self._barge_widget.setVisible(is_gemini)
        if hasattr(self, "_el_key_widget"):
            self._el_key_widget.setVisible(is_el)
        if hasattr(self, "_tts_voice_key_widget"):
            self._tts_voice_key_widget.setVisible(is_gemini or is_gemini_live or is_hashim_live or is_google)

        if is_el:
            self._voice_lbl.setText("Voice ID:")
            if hasattr(self, "_tts_voice_input"):
                self._tts_voice_input.setPlaceholderText("ElevenLabs voice ID")
        elif is_gemini or is_gemini_live or is_hashim_live:
            self._voice_lbl.setText("Voice:")
        elif is_kokoro:
            self._voice_lbl.setText("Voice:")
        elif is_gtts:
            self._voice_lbl.setText("Language:")
        elif is_google:
            self._voice_lbl.setText("Voice:")
        else:
            self._voice_lbl.setText("Voice:")
            if hasattr(self, "_tts_voice_input"):
                self._tts_voice_input.setPlaceholderText(
                    "en-US-GuyNeural  /  en-GB-RyanNeural  /  tr-TR-AhmetNeural  ..."
                )

        if hasattr(self, "_kokoro_speed_widget"):
            self._kokoro_speed_widget.setVisible(is_kokoro)

    def _set_llm_provider(self, key: str):
        self._sel_llm_provider = key
        if not hasattr(self, "_llm_prov_btns"):
            return
        for k, btn in self._llm_prov_btns.items():
            active = (k == key)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {'#00d4ff' if active else '#000d12'};
                    color: {'#001a22' if active else C.TEXT_DIM};
                    border: {'none' if active else f'1px solid {C.BORDER}'};
                    border-radius: 3px; font-weight: {'bold' if active else 'normal'};
                }}
                QPushButton:hover {{ color: {C.TEXT}; border: 1px solid {C.BORDER_B}; }}
            """)
        if hasattr(self, "_llm_url_input"):
            visible = key in ("ollama", "openai")
            self._llm_url_input.setVisible(visible)
        if hasattr(self, "_llm_model_combo"):
            self._llm_model_combo.setVisible(key in ("ollama", "openai"))
            self._llm_custom_input.setVisible(key in ("ollama", "openai"))
        if hasattr(self, "_ollama_fetch_btn"):
            self._ollama_fetch_btn.setVisible(key == "ollama")
        if hasattr(self, "_ollama_test_btn"):
            self._ollama_test_btn.setVisible(key == "ollama")
        if hasattr(self, "_kaggle_widget"):
            self._kaggle_widget.setVisible(key == "kaggle")
        # Endpoint profiles only apply to ollama provider
        if hasattr(self, "_llm_profile_combo"):
            _profile_vis = key == "ollama"
            self._llm_profile_combo.setVisible(_profile_vis)
            self._llm_profile_save_btn.setVisible(_profile_vis)
            self._llm_profile_switch_btn.setVisible(_profile_vis)
            self._llm_profile_delete_btn.setVisible(_profile_vis)
        if hasattr(self, "_ollama_test_status"):
            self._ollama_test_status.setVisible(key == "ollama")
        if hasattr(self, "_cf_widget"):
            self._cf_widget.setVisible(key == "cloudflare")
        if hasattr(self, "_groq_llm_widget"):
            self._groq_llm_widget.setVisible(key == "groq")
        if hasattr(self, "_gemini_llm_widget"):
            self._gemini_llm_widget.setVisible(key in ("gemini", "gemini_live"))
        if hasattr(self, "_live_voice_widget"):
            self._live_voice_widget.setVisible(key == "gemini_live")
        if hasattr(self, "_or_widget"):
            self._or_widget.setVisible(key == "openrouter")
        if hasattr(self, "_nvidia_widget"):
            self._nvidia_widget.setVisible(key == "nvidia")
        if hasattr(self, "_kilo_widget"):
            self._kilo_widget.setVisible(key == "kilo")
        if key == "kilo" and hasattr(self, "_kilo_model_combo"):
            _def = "kilo-auto/free"
            for i in range(self._kilo_model_combo.count()):
                if self._kilo_model_combo.itemData(i) == _def:
                    self._kilo_model_combo.setCurrentIndex(i)
                    break
        if key == "nvidia" and hasattr(self, "_nv_model_combo"):
            _def = "qwen/qwen3.5-397b-a17b"
            for i in range(self._nv_model_combo.count()):
                if self._nv_model_combo.itemData(i) == _def:
                    self._nv_model_combo.setCurrentIndex(i)
                    break
        # Sync model combo to provider's default when switching
        if key == "cloudflare" and hasattr(self, "_cf_model_combo"):
            _def = "@cf/google/gemma-4-26b-a4b-it"
            for i in range(self._cf_model_combo.count()):
                if self._cf_model_combo.itemData(i) == _def:
                    self._cf_model_combo.setCurrentIndex(i)
                    break
        elif key == "groq" and hasattr(self, "_groq_model_combo"):
            _def = "meta-llama/llama-4-scout-17b-16e-instruct"
            for i in range(self._groq_model_combo.count()):
                if self._groq_model_combo.itemData(i) == _def:
                    self._groq_model_combo.setCurrentIndex(i)
                    break
        elif key == "gemini" and hasattr(self, "_gemini_model_combo"):
            _def = "gemini-2.5-flash"
            for i in range(self._gemini_model_combo.count()):
                if self._gemini_model_combo.itemData(i) == _def:
                    self._gemini_model_combo.setCurrentIndex(i)
                    break
        elif key == "gemini_live" and hasattr(self, "_gemini_model_combo"):
            _def = "gemini-2.5-flash-native-audio-preview-12-2025"
            for i in range(self._gemini_model_combo.count()):
                if self._gemini_model_combo.itemData(i) == _def:
                    self._gemini_model_combo.setCurrentIndex(i)
                    break
        elif key == "openrouter" and hasattr(self, "_or_model_combo"):
            _def = "openrouter/free"
            for i in range(self._or_model_combo.count()):
                if self._or_model_combo.itemData(i) == _def:
                    self._or_model_combo.setCurrentIndex(i)
                    break
        if hasattr(self, "_llm_hint_lbl"):
            hints = {
                "openai": "lmstudio.ai  ·  start Local Server first, then pick model",
                "cloudflare": "developers.cloudflare.com/workers-ai/  ·  needs Account ID + API Token",
                "groq": "console.groq.com  ·  needs Groq LLM API key",
                "gemini": "aistudio.google.com  ·  needs Gemini API key",
                "gemini_live": "aistudio.google.com  ·  WebSocket-based, needs Gemini API key",
                "openrouter": "openrouter.ai  ·  needs API key, free models available",
                "nvidia": "build.nvidia.com  ·  needs NVIDIA API key",
                "kilo": "kilo.ai  ·  needs Kilo API key, kilo-auto/free",
            }
            self._llm_hint_lbl.setText(hints.get(key, "ollama.com  ·  run: ollama pull qwen2.5:3b"))

    def _set_summarization_provider(self, key: str):
        self._sel_summarization_provider = key
        if not hasattr(self, "_summ_prov_btns"):
            return
        for k, btn in self._summ_prov_btns.items():
            active = (k == key)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {'#00d4ff' if active else '#000d12'};
                    color: {'#001a22' if active else C.TEXT_DIM};
                    border: {'none' if active else f'1px solid {C.BORDER}'};
                    border-radius: 3px; font-weight: {'bold' if active else 'normal'};
                }}
                QPushButton:hover {{ color: {C.TEXT}; border: 1px solid {C.BORDER_B}; }}
            """)
        if hasattr(self, "_summ_url_input"):
            self._summ_url_input.setVisible(key == "ollama")
        if hasattr(self, "_summ_key_input"):
            self._summ_key_input.setVisible(key == "gemini")

    def _on_cf_model_changed(self, idx: int):
        is_custom = self._cf_model_combo.itemData(idx) == "__CUSTOM__"
        self._cf_custom_input.setVisible(is_custom)

    def _get_cf_model(self) -> str:
        idx = self._cf_model_combo.currentIndex()
        val = self._cf_model_combo.itemData(idx)
        if val == "__CUSTOM__":
            return self._cf_custom_input.text().strip()
        return val

    def _on_groq_model_changed(self, idx: int):
        is_custom = self._groq_model_combo.itemData(idx) == "__CUSTOM__"
        self._groq_custom_input.setVisible(is_custom)

    def _get_groq_model(self) -> str:
        idx = self._groq_model_combo.currentIndex()
        val = self._groq_model_combo.itemData(idx)
        if val == "__CUSTOM__":
            return self._groq_custom_input.text().strip()
        return val

    def _on_gemini_model_changed(self, idx: int):
        is_custom = self._gemini_model_combo.itemData(idx) == "__CUSTOM__"
        self._gemini_custom_input.setVisible(is_custom)

    def _get_gemini_model(self) -> str:
        idx = self._gemini_model_combo.currentIndex()
        val = self._gemini_model_combo.itemData(idx)
        if val == "__CUSTOM__":
            return self._gemini_custom_input.text().strip()
        return val

    def _on_or_model_changed(self, idx: int):
        val = self._or_model_combo.itemData(idx)
        self._or_custom_input.setVisible(val in ("__CUSTOM__", "__HEADING__"))

    def _get_or_model(self) -> str:
        idx = self._or_model_combo.currentIndex()
        val = self._or_model_combo.itemData(idx)
        if val in ("__CUSTOM__", "__HEADING__", None):
            return self._or_custom_input.text().strip()
        return val

    # ── Test Cloud Connections ──
    def _test_cf(self):
        self._cf_test_status.setText("Testing...")
        self._cf_test_status.setStyleSheet(f"color: {C.TEXT_MED}; background: transparent;")
        try:
            token = self._cf_token_input.text().strip()
            aid   = self._cf_account_input.text().strip()
            model = self._get_cf_model()
            if not token or not aid:
                self._cf_test_status.setText("✗ API token and Account ID required")
                self._cf_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")
                return
            import requests
            resp = requests.post(
                f"https://api.cloudflare.com/client/v4/accounts/{aid}/ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json={"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "stream": False},
                timeout=15,
            )
            if resp.status_code == 200:
                self._cf_test_status.setText(f"✓ OK  ({model})")
                self._cf_test_status.setStyleSheet(f"color: {C.GREEN}; background: transparent;")
            else:
                self._cf_test_status.setText(f"✗ HTTP {resp.status_code}")
                self._cf_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")
        except Exception as e:
            self._cf_test_status.setText(f"✗ {e}")
            self._cf_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")

    def _test_groq(self):
        self._groq_test_status.setText("Testing...")
        self._groq_test_status.setStyleSheet(f"color: {C.TEXT_MED}; background: transparent;")
        try:
            key   = self._groq_llm_key_input.text().strip()
            model = self._get_groq_model()
            if not key:
                self._groq_test_status.setText("✗ API key required")
                self._groq_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")
                return
            import requests
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "stream": False},
                timeout=15,
            )
            if resp.status_code == 200:
                self._groq_test_status.setText(f"✓ OK  ({model})")
                self._groq_test_status.setStyleSheet(f"color: {C.GREEN}; background: transparent;")
            else:
                self._groq_test_status.setText(f"✗ HTTP {resp.status_code}")
                self._groq_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")
        except Exception as e:
            self._groq_test_status.setText(f"✗ {e}")
            self._groq_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")

    def _test_gemini(self):
        self._gemini_test_status.setText("Testing...")
        self._gemini_test_status.setStyleSheet(f"color: {C.TEXT_MED}; background: transparent;")
        try:
            key   = self._gemini_llm_key_input.text().strip()
            model = self._get_gemini_model()
            if not key:
                self._gemini_test_status.setText("✗ API key required")
                self._gemini_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")
                return
            import requests
            resp = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
                headers={"Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": "hi"}]}]},
                timeout=15,
            )
            if resp.status_code == 200:
                self._gemini_test_status.setText(f"✓ OK  ({model})")
                self._gemini_test_status.setStyleSheet(f"color: {C.GREEN}; background: transparent;")
            else:
                self._gemini_test_status.setText(f"✗ HTTP {resp.status_code}")
                self._gemini_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")
        except Exception as e:
            self._gemini_test_status.setText(f"✗ {e}")
            self._gemini_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")

    def _test_or(self):
        self._or_test_status.setText("Testing...")
        self._or_test_status.setStyleSheet(f"color: {C.TEXT_MED}; background: transparent;")
        try:
            key   = self._or_key_input.text().strip()
            model = self._get_or_model()
            if not key:
                self._or_test_status.setText("✗ API key required")
                self._or_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")
                return
            import requests
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "stream": False},
                timeout=15,
            )
            if resp.status_code == 200:
                self._or_test_status.setText(f"✓ OK  ({model})")
                self._or_test_status.setStyleSheet(f"color: {C.GREEN}; background: transparent;")
            else:
                self._or_test_status.setText(f"✗ HTTP {resp.status_code}")
                self._or_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")
        except Exception as e:
            self._or_test_status.setText(f"✗ {e}")
            self._or_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")

    def _on_nv_model_changed(self, idx: int):
        is_custom = self._nv_model_combo.itemData(idx) == "__CUSTOM__"
        self._nv_custom_input.setVisible(is_custom)

    def _get_nv_model(self) -> str:
        idx = self._nv_model_combo.currentIndex()
        val = self._nv_model_combo.itemData(idx)
        if val == "__CUSTOM__":
            return self._nv_custom_input.text().strip()
        return val

    def _on_kilo_model_changed(self, idx: int):
        is_custom = self._kilo_model_combo.itemData(idx) == "__CUSTOM__"
        self._kilo_custom_input.setVisible(is_custom)

    def _get_kilo_model(self) -> str:
        idx = self._kilo_model_combo.currentIndex()
        val = self._kilo_model_combo.itemData(idx)
        if val == "__CUSTOM__":
            return self._kilo_custom_input.text().strip()
        return val

    def _test_kilo(self):
        self._kilo_test_status.setText("Testing...")
        self._kilo_test_status.setStyleSheet(f"color: {C.TEXT_MED}; background: transparent;")
        try:
            key   = self._kilo_key_input.text().strip()
            model = self._get_kilo_model()
            if not key:
                self._kilo_test_status.setText("✗ API key required")
                self._kilo_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")
                return
            import requests
            resp = requests.post(
                "https://api.kilo.ai/api/gateway/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "stream": False},
                timeout=15,
            )
            if resp.status_code == 200:
                self._kilo_test_status.setText(f"✓ OK  ({model})")
                self._kilo_test_status.setStyleSheet(f"color: {C.GREEN}; background: transparent;")
            else:
                self._kilo_test_status.setText(f"✗ HTTP {resp.status_code}")
                self._kilo_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")
        except Exception as e:
            self._kilo_test_status.setText(f"✗ {e}")
            self._kilo_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")

    def _fetch_kilo_models(self):
        key = self._kilo_key_input.text().strip()
        self._kilo_test_status.setText("Fetching...")
        self._kilo_test_status.setStyleSheet(f"color: {C.TEXT_MED}; background: transparent;")
        self._kilo_fetch_btn.setEnabled(False)
        try:
            import requests
            headers = {"Authorization": f"Bearer {key}"} if key else {}
            resp = requests.get(
                "https://api.kilo.ai/api/gateway/models",
                headers=headers,
                timeout=15,
            )
            if resp.status_code != 200:
                self._kilo_test_status.setText(f"✗ HTTP {resp.status_code}")
                self._kilo_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")
                return
            data = resp.json()
            models = [m["id"] for m in data.get("data", [])]
            # Ensure virtual auto models present even if API omits them
            for virt in ("kilo-auto/free", "openrouter/free"):
                if virt not in models:
                    models.insert(0, virt)
            models.sort(key=lambda m: (m != "kilo-auto/free", m))
            self._kilo_model_combo.blockSignals(True)
            self._kilo_model_combo.clear()
            for mid in models:
                self._kilo_model_combo.addItem(mid, userData=mid)
            self._kilo_model_combo.addItem("── Custom ──", userData="__CUSTOM__")
            self._kilo_model_combo.blockSignals(False)
            self._kilo_custom_input.setVisible(False)
            self._kilo_test_status.setText(f"✓ {len(models)} models loaded")
            self._kilo_test_status.setStyleSheet(f"color: {C.GREEN}; background: transparent;")
        except Exception as e:
            self._kilo_test_status.setText(f"✗ {e}")
            self._kilo_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")
        finally:
            self._kilo_fetch_btn.setEnabled(True)

    def _test_nvidia(self):
        self._nv_test_status.setText("Testing...")
        self._nv_test_status.setStyleSheet(f"color: {C.TEXT_MED}; background: transparent;")
        try:
            key   = self._nv_key_input.text().strip()
            model = self._get_nv_model()
            if not key:
                self._nv_test_status.setText("✗ API key required")
                self._nv_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")
                return
            import requests
            resp = requests.post(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "stream": False},
                timeout=15,
            )
            if resp.status_code == 200:
                self._nv_test_status.setText(f"✓ OK  ({model})")
                self._nv_test_status.setStyleSheet(f"color: {C.GREEN}; background: transparent;")
            else:
                self._nv_test_status.setText(f"✗ HTTP {resp.status_code}")
                self._nv_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")
        except Exception as e:
            self._nv_test_status.setText(f"✗ {e}")
            self._nv_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")

    def _fetch_nvidia_models(self):
        key = self._nv_key_input.text().strip()
        if not key:
            self._nv_test_status.setText("✗ API key required")
            self._nv_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")
            return
        self._nv_test_status.setText("Fetching...")
        self._nv_test_status.setStyleSheet(f"color: {C.TEXT_MED}; background: transparent;")
        self._nv_fetch_btn.setEnabled(False)
        try:
            import requests
            resp = requests.get(
                "https://integrate.api.nvidia.com/v1/models",
                headers={"Authorization": f"Bearer {key}"},
                timeout=15,
            )
            if resp.status_code != 200:
                self._nv_test_status.setText(f"✗ HTTP {resp.status_code}")
                self._nv_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")
                return
            data = resp.json()
            models = [m["id"] for m in data.get("data", [])]
            models.sort()
            self._nv_model_combo.blockSignals(True)
            self._nv_model_combo.clear()
            for mid in models:
                self._nv_model_combo.addItem(mid, userData=mid)
            self._nv_model_combo.addItem("── Custom ──", userData="__CUSTOM__")
            self._nv_model_combo.blockSignals(False)
            self._nv_custom_input.setVisible(False)
            self._nv_test_status.setText(f"✓ {len(models)} models loaded")
            self._nv_test_status.setStyleSheet(f"color: {C.GREEN}; background: transparent;")
        except Exception as e:
            self._nv_test_status.setText(f"✗ {e}")
            self._nv_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")
        finally:
            self._nv_fetch_btn.setEnabled(True)

    # ── Ollama model helpers ──

    def _on_ollama_model_changed(self, idx: int):
        is_custom = self._llm_model_combo.itemData(idx) == "__CUSTOM__"
        self._llm_custom_input.setVisible(is_custom)

    # ── Kaggle (remote Ollama) helpers ──

    def _on_kaggle_model_changed(self, idx: int):
        is_custom = self._kaggle_model_combo.itemData(idx) == "__CUSTOM__"
        self._kaggle_custom_input.setVisible(is_custom)

    def _get_kaggle_model(self) -> str:
        idx = self._kaggle_model_combo.currentIndex()
        val = self._kaggle_model_combo.itemData(idx)
        if val in ("__CUSTOM__", None):
            return self._kaggle_custom_input.text().strip()
        return val

    def _test_kaggle(self):
        url = self._kaggle_url_input.text().strip().rstrip("/")
        model = self._get_kaggle_model()
        if not url:
            self._kaggle_test_status.setText("✗ Tunnel URL required")
            self._kaggle_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")
            return
        if not url.startswith(("http://", "https://")):
            url = "http://" + url
        self._kaggle_test_status.setText("Testing...")
        self._kaggle_test_status.setStyleSheet(f"color: {C.TEXT_MED}; background: transparent;")
        try:
            import requests
            resp = requests.get(f"{url}/api/tags", timeout=8)
            if resp.status_code != 200:
                self._kaggle_test_status.setText(f"✗ HTTP {resp.status_code}")
                self._kaggle_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")
                return
            names = [m.get("name", "") for m in resp.json().get("models", [])]
            base = model.split(":")[0]
            if not any(n == model or n.startswith(base + ":") for n in names):
                self._kaggle_test_status.setText(
                    f"⚠ Server reachable, but model '{model}' not pulled on Kaggle")
                self._kaggle_test_status.setStyleSheet(f"color: {C.ACC2}; background: transparent;")
                return
            self._kaggle_test_status.setText(f"✓ Kaggle OK  ({len(names)} models)")
            self._kaggle_test_status.setStyleSheet(f"color: {C.GREEN}; background: transparent;")
        except Exception as e:
            self._kaggle_test_status.setText(f"✗ {e}  — tunnel/Kaggle session running?")
            self._kaggle_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")

    def _get_ollama_model(self) -> str:
        idx = self._llm_model_combo.currentIndex()
        val = self._llm_model_combo.itemData(idx)
        if val in ("__CUSTOM__", None):
            return self._llm_custom_input.text().strip()
        return val

    def _fetch_ollama_models(self):
        url = self._llm_url_input.text().strip() or "http://localhost:11434"
        self._ollama_test_status.setText("Fetching...")
        self._ollama_test_status.setStyleSheet(f"color: {C.TEXT_MED}; background: transparent;")
        self._ollama_fetch_btn.setEnabled(False)
        try:
            import requests
            resp = requests.get(f"{url}/api/tags", timeout=10)
            if resp.status_code != 200:
                self._ollama_test_status.setText(f"✗ HTTP {resp.status_code}")
                self._ollama_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")
                return
            data = resp.json()
            models = [m["name"] for m in data.get("models", [])]
            models.sort()
            if not models:
                self._ollama_test_status.setText("✗ No models installed. Run: ollama pull <model>")
                self._ollama_test_status.setStyleSheet(f"color: {C.ACC2}; background: transparent;")
                return
            self._llm_model_combo.blockSignals(True)
            self._llm_model_combo.clear()
            for mid in models:
                self._llm_model_combo.addItem(mid, userData=mid)
            self._llm_model_combo.addItem("── Custom ──", userData="__CUSTOM__")
            self._llm_model_combo.blockSignals(False)
            self._llm_custom_input.setVisible(False)
            self._ollama_test_status.setText(f"✓ {len(models)} models loaded")
            self._ollama_test_status.setStyleSheet(f"color: {C.GREEN}; background: transparent;")
        except Exception as e:
            self._ollama_test_status.setText(f"✗ {e}")
            self._ollama_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")
        finally:
            self._ollama_fetch_btn.setEnabled(True)

    def _test_ollama(self):
        url = self._llm_url_input.text().strip() or "http://localhost:11434"
        self._ollama_test_status.setText("Testing...")
        self._ollama_test_status.setStyleSheet(f"color: {C.TEXT_MED}; background: transparent;")
        try:
            import requests
            resp = requests.get(f"{url}/api/tags", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                count = len(data.get("models", []))
                self._ollama_test_status.setText(f"✓ OK  ({count} models)")
                self._ollama_test_status.setStyleSheet(f"color: {C.GREEN}; background: transparent;")
            else:
                self._ollama_test_status.setText(f"✗ HTTP {resp.status_code}")
                self._ollama_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")
        except Exception as e:
            self._ollama_test_status.setText(f"✗ {e}")
            self._ollama_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")

    # ── LLM endpoint profiles (local ↔ Kaggle tunnel one-click switch) ──

    def _llm_profiles_refresh(self):
        """Rebuild the profile dropdown from stored config."""
        from core import llm_profiles as _lp
        self._llm_profile_combo.blockSignals(True)
        self._llm_profile_combo.clear()
        self._llm_profile_combo.addItem("— No profile (top-level config) —", userData="")
        self._llm_profile_combo.addItem("🏠 local", userData=_lp.LOCAL_PROFILE)
        for name in _lp.get_profiles():
            if name == _lp.LOCAL_PROFILE:
                continue
            self._llm_profile_combo.addItem(name, userData=name)
        active = _lp.get_active_profile()
        if active:
            idx = self._llm_profile_combo.findData(active)
            if idx >= 0:
                self._llm_profile_combo.setCurrentIndex(idx)
        self._llm_profile_combo.blockSignals(False)

    def _llm_profile_save_clicked(self):
        from core import llm_profiles as _lp
        name, ok = QInputDialog.getText(self, "Save Profile", "Profile name (e.g. kaggle):")
        if not ok or not name.strip():
            return
        url = self._llm_url_input.text().strip() or "http://localhost:11434"
        model = self._get_ollama_model()
        try:
            stored = _lp.save_profile(name, url, model)
        except ValueError as e:
            self._ollama_test_status.setText(f"✗ {e}")
            self._ollama_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")
            return
        self._llm_profiles_refresh()
        idx = self._llm_profile_combo.findData(stored)
        if idx >= 0:
            self._llm_profile_combo.setCurrentIndex(idx)
        self._ollama_test_status.setText(f"✓ Profile '{stored}' saved")
        self._ollama_test_status.setStyleSheet(f"color: {C.GREEN}; background: transparent;")

    def _llm_profile_switch_clicked(self):
        from core import llm_profiles as _lp
        name = self._llm_profile_combo.currentData() or ""
        self._ollama_test_status.setText("Switching...")
        self._ollama_test_status.setStyleSheet(f"color: {C.TEXT_MED}; background: transparent;")
        try:
            if not _lp.switch_profile(name):
                self._ollama_test_status.setText("✗ Profile not found")
                self._ollama_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")
                return
            ok, msg = _lp.test_profile(name)
        except Exception as e:
            ok, msg = False, str(e)
        if ok:
            self._ollama_test_status.setText(f"⚡ Switched → {name or 'no profile'}  ({msg})")
            self._ollama_test_status.setStyleSheet(f"color: {C.GREEN}; background: transparent;")
        else:
            self._ollama_test_status.setText(f"⚠ Switched to {name or 'no profile'}, but: {msg}")
            self._ollama_test_status.setStyleSheet(f"color: {C.ACC2}; background: transparent;")

    def _llm_profile_delete_clicked(self):
        from core import llm_profiles as _lp
        name = self._llm_profile_combo.currentData() or ""
        if not name:
            return
        _lp.delete_profile(name)
        self._llm_profiles_refresh()
        self._ollama_test_status.setText(f"🗑 Profile '{name}' deleted")
        self._ollama_test_status.setStyleSheet(f"color: {C.TEXT_DIM}; background: transparent;")

    # ── Cloud provider fetch methods ──

    def _fetch_cf_models(self):
        token = self._cf_token_input.text().strip()
        aid = self._cf_account_input.text().strip()
        if not token or not aid:
            self._cf_test_status.setText("✗ API token and Account ID required")
            self._cf_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")
            return
        self._cf_test_status.setText("Fetching...")
        self._cf_test_status.setStyleSheet(f"color: {C.TEXT_MED}; background: transparent;")
        self._cf_fetch_btn.setEnabled(False)
        try:
            import requests
            resp = requests.get(
                f"https://api.cloudflare.com/client/v4/accounts/{aid}/ai/models/search",
                headers={"Authorization": f"Bearer {token}"},
                params={"task": "Text Generation", "hide_experimental": "true"},
                timeout=15,
            )
            if resp.status_code != 200:
                self._cf_test_status.setText(f"✗ HTTP {resp.status_code}")
                self._cf_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")
                return
            data = resp.json()
            models = []
            for m in data.get("result", []):
                mid = m.get("model_id", "")
                if not mid:
                    continue
                name = m.get("name", mid)
                desc = m.get("task", {})
                task_slug = desc.get("slug", "") if isinstance(desc, dict) else ""
                display = f"{mid}  ({name})" if name != mid else mid
                models.append((mid, display))
            models.sort(key=lambda x: x[0])
            if not models:
                self._cf_test_status.setText("✗ No text generation models found")
                self._cf_test_status.setStyleSheet(f"color: {C.ACC2}; background: transparent;")
                return
            self._cf_model_combo.blockSignals(True)
            self._cf_model_combo.clear()
            for mid, display in models:
                self._cf_model_combo.addItem(display, userData=mid)
            self._cf_model_combo.addItem("── Custom ──", userData="__CUSTOM__")
            self._cf_model_combo.blockSignals(False)
            self._cf_custom_input.setVisible(False)
            self._cf_test_status.setText(f"✓ {len(models)} models loaded")
            self._cf_test_status.setStyleSheet(f"color: {C.GREEN}; background: transparent;")
        except Exception as e:
            self._cf_test_status.setText(f"✗ {e}")
            self._cf_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")
        finally:
            self._cf_fetch_btn.setEnabled(True)

    def _fetch_groq_models(self):
        key = self._groq_llm_key_input.text().strip()
        if not key:
            self._groq_test_status.setText("✗ API key required")
            self._groq_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")
            return
        self._groq_test_status.setText("Fetching...")
        self._groq_test_status.setStyleSheet(f"color: {C.TEXT_MED}; background: transparent;")
        self._groq_fetch_btn.setEnabled(False)
        try:
            import requests
            resp = requests.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {key}"},
                timeout=15,
            )
            if resp.status_code != 200:
                self._groq_test_status.setText(f"✗ HTTP {resp.status_code}")
                self._groq_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")
                return
            data = resp.json()
            models = [m["id"] for m in data.get("data", []) if m.get("active", True)]
            models.sort()
            if not models:
                self._groq_test_status.setText("✗ No active models")
                self._groq_test_status.setStyleSheet(f"color: {C.ACC2}; background: transparent;")
                return
            self._groq_model_combo.blockSignals(True)
            self._groq_model_combo.clear()
            for mid in models:
                self._groq_model_combo.addItem(mid, userData=mid)
            self._groq_model_combo.addItem("── Custom ──", userData="__CUSTOM__")
            self._groq_model_combo.blockSignals(False)
            self._groq_custom_input.setVisible(False)
            self._groq_test_status.setText(f"✓ {len(models)} models loaded")
            self._groq_test_status.setStyleSheet(f"color: {C.GREEN}; background: transparent;")
        except Exception as e:
            self._groq_test_status.setText(f"✗ {e}")
            self._groq_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")
        finally:
            self._groq_fetch_btn.setEnabled(True)

    def _fetch_gemini_models(self):
        key = self._gemini_llm_key_input.text().strip()
        if not key:
            self._gemini_test_status.setText("✗ API key required")
            self._gemini_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")
            return
        self._gemini_test_status.setText("Fetching...")
        self._gemini_test_status.setStyleSheet(f"color: {C.TEXT_MED}; background: transparent;")
        self._gemini_fetch_btn.setEnabled(False)
        try:
            import requests
            resp = requests.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                params={"key": key, "pageSize": 1000},
                timeout=15,
            )
            if resp.status_code != 200:
                self._gemini_test_status.setText(f"✗ HTTP {resp.status_code}")
                self._gemini_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")
                return
            data = resp.json()
            models = []
            for m in data.get("models", []):
                name = m.get("name", "")
                if name.startswith("models/"):
                    name = name[len("models/"):]
                gen_methods = m.get("supportedGenerationMethods", [])
                if "generateContent" in gen_methods and name:
                    models.append(name)
            models.sort()
            if not models:
                self._gemini_test_status.setText("✗ No models found")
                self._gemini_test_status.setStyleSheet(f"color: {C.ACC2}; background: transparent;")
                return
            self._gemini_model_combo.blockSignals(True)
            self._gemini_model_combo.clear()
            for mid in models:
                self._gemini_model_combo.addItem(mid, userData=mid)
            self._gemini_model_combo.addItem("── Custom ──", userData="__CUSTOM__")
            self._gemini_model_combo.blockSignals(False)
            self._gemini_custom_input.setVisible(False)
            self._gemini_test_status.setText(f"✓ {len(models)} models loaded")
            self._gemini_test_status.setStyleSheet(f"color: {C.GREEN}; background: transparent;")
        except Exception as e:
            self._gemini_test_status.setText(f"✗ {e}")
            self._gemini_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")
        finally:
            self._gemini_fetch_btn.setEnabled(True)

    def _fetch_or_models(self):
        key = self._or_key_input.text().strip()
        self._or_test_status.setText("Fetching...")
        self._or_test_status.setStyleSheet(f"color: {C.TEXT_MED}; background: transparent;")
        self._or_fetch_btn.setEnabled(False)
        try:
            import requests
            headers = {"Content-Type": "application/json"}
            if key:
                headers["Authorization"] = f"Bearer {key}"
            resp = requests.get(
                "https://openrouter.ai/api/v1/models",
                headers=headers,
                timeout=15,
            )
            if resp.status_code != 200:
                self._or_test_status.setText(f"✗ HTTP {resp.status_code}")
                self._or_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")
                return
            data = resp.json()
            all_models = data.get("data", [])
            free_models = []
            for m in all_models:
                mid = m.get("id", "")
                pricing = m.get("pricing", {})
                prompt_price = pricing.get("prompt", "1")
                completion_price = pricing.get("completion", "1")
                try:
                    is_free = float(prompt_price) == 0 and float(completion_price) == 0
                except (ValueError, TypeError):
                    is_free = False
                if is_free and mid:
                    free_models.append(mid)
            free_models.sort()
            all_ids = [m.get("id", "") for m in all_models if m.get("id")]
            all_ids.sort()
            display = free_models if free_models else all_ids
            if not display:
                self._or_test_status.setText("✗ No models found")
                self._or_test_status.setStyleSheet(f"color: {C.ACC2}; background: transparent;")
                return
            self._or_model_combo.blockSignals(True)
            self._or_model_combo.clear()
            if free_models:
                self._or_model_combo.addItem(f"── Free models ({len(free_models)}) ──", userData="__HEADING__")
            for mid in display:
                label = mid
                if mid in free_models:
                    label = f"{mid}  (free)"
                self._or_model_combo.addItem(label, userData=mid)
            self._or_model_combo.addItem("── All models ──", userData="__HEADING__")
            for mid in all_ids:
                if mid not in free_models:
                    pricing = next((m.get("pricing", {}) for m in all_models if m.get("id") == mid), {})
                    prompt_p = pricing.get("prompt", "?")
                    self._or_model_combo.addItem(f"{mid}  (${prompt_p}/tok)", userData=mid)
            self._or_model_combo.addItem("── Custom ──", userData="__CUSTOM__")
            self._or_model_combo.blockSignals(False)
            self._or_custom_input.setVisible(False)
            self._or_test_status.setText(f"✓ {len(free_models)} free / {len(all_ids)} total")
            self._or_test_status.setStyleSheet(f"color: {C.GREEN}; background: transparent;")
        except Exception as e:
            self._or_test_status.setText(f"✗ {e}")
            self._or_test_status.setStyleSheet(f"color: {C.RED}; background: transparent;")
        finally:
            self._or_fetch_btn.setEnabled(True)

    def _set_stt(self, key: str):
        self._sel_stt = key
        if not hasattr(self, "_stt_btns"):
            return
        for k, btn in self._stt_btns.items():
            active = (k == key)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {'#00d4ff' if active else '#000d12'};
                    color: {'#001a22' if active else C.TEXT_DIM};
                    border: {'none' if active else f'1px solid {C.BORDER}'};
                    border-radius: 3px; font-weight: {'bold' if active else 'normal'};
                }}
                QPushButton:hover {{ color: {C.TEXT}; border: 1px solid {C.BORDER_B}; }}
            """)
        if hasattr(self, "_whisper_combo"):
            self._whisper_combo.setVisible(key == "whisper")
        if hasattr(self, "_vosk_model_input"):
            self._vosk_model_input.setVisible(key == "vosk")
        if hasattr(self, "_gemini_key_widget"):
            self._gemini_key_widget.setVisible(key in ("gemini", "hashim"))
        if hasattr(self, "_groq_key_widget"):
            self._groq_key_widget.setVisible(key == "groq")

    def _set_tts(self, key: str):
        self._sel_tts = key
        if not hasattr(self, "_tts_btns"):
            return
        for k, btn in self._tts_btns.items():
            active = (k == key)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {'#00d4ff' if active else '#000d12'};
                    color: {'#001a22' if active else C.TEXT_DIM};
                    border: {'none' if active else f'1px solid {C.BORDER}'};
                    border-radius: 3px; font-weight: {'bold' if active else 'normal'};
                }}
                QPushButton:hover {{ color: {C.TEXT}; border: 1px solid {C.BORDER_B}; }}
            """)
        self._update_tts_ui(key)

    def _submit(self):
        _provider = getattr(self, "_sel_llm_provider", "ollama")

        # Validate required fields per provider
        if _provider in ("ollama", "openai"):
            llm_model = self._get_ollama_model()
            if not llm_model:
                self._llm_custom_input.setStyleSheet(
                    self._llm_custom_input.styleSheet() +
                    f" QLineEdit {{ border: 1px solid {C.RED}; }}"
                )
                return
        else:
            llm_model = ""

        if self._sel_stt == "whisper":
            stt_model = self._whisper_combo.currentText()
        else:
            stt_model = self._vosk_model_input.text().strip()

        if self._sel_tts == "kokoro":
            tts_voice = self._kokoro_combo.currentData() or "af_heart"
            tts_speed = self._kokoro_speed_combo.currentData() or "1.2"
        elif self._sel_tts == "gemini":
            tts_voice = self._gemini_voice_combo.currentData() or "Puck"
            tts_speed = "1.0"
        elif self._sel_tts in ("gemini_live", "hashim_live"):
            tts_voice = self._gemini_voice_combo.currentData() or "Charon"
            tts_speed = "1.0"
        elif self._sel_tts == "gtts":
            tts_voice = self._gtts_combo.currentData() or "hi"
            tts_speed = "1.0"
        elif self._sel_tts == "googlecloud":
            tts_voice = self._gcloud_combo.currentData() or "hi-IN-Neural2-A"
            tts_speed = "1.0"
        else:
            tts_voice = self._tts_voice_input.text().strip() or "en-US-GuyNeural"
            tts_speed = "1.0"

        _default_url = "http://localhost:1234" if _provider == "openai" else "http://localhost:11434"
        _stt_key = self._gemini_key_input.text().strip() if hasattr(self, "_gemini_key_input") else ""
        _tts_key = self._tts_gemini_key_input.text().strip() if hasattr(self, "_tts_gemini_key_input") else ""
        _voice_key = _tts_key or _stt_key
        _groq_key = self._groq_key_input.text().strip() if hasattr(self, "_groq_key_input") else ""

        # ── Voice API Key is required when Gemini/Hashim STT or Gemini/Google Cloud TTS is used ──
        _needs_voice_key = (self._sel_stt in ("gemini", "hashim") or self._sel_tts in ("gemini", "gemini_live", "hashim_live", "googlecloud"))
        if _needs_voice_key and not _voice_key:
            _err_field = self._tts_gemini_key_input if (self._sel_tts in ("gemini", "googlecloud") and hasattr(self, "_tts_gemini_key_input")) else self._gemini_key_input
            _err_field.setStyleSheet(
                _err_field.styleSheet() +
                f" QLineEdit {{ border: 1px solid {C.RED}; }}"
            )
            return

        # ── Groq STT key must be present (and look like a Groq key) ──
        if self._sel_stt == "groq":
            if not _groq_key:
                self._groq_key_input.setStyleSheet(
                    self._groq_key_input.styleSheet() +
                    f" QLineEdit {{ border: 1px solid {C.RED}; }}"
                )
                return
            if not _groq_key.startswith("gsk_"):
                self._groq_key_input.setStyleSheet(
                    self._groq_key_input.styleSheet() +
                    f" QLineEdit {{ border: 1px solid {C.RED}; }}"
                )
                return

        cfg = {
            "stt_engine":           self._sel_stt,
            "stt_model":            stt_model,
            "stt_language":         self._stt_lang_input.text().strip() or "auto",
            "llm_provider":         _provider,
            "llm_url":              self._llm_url_input.text().strip() or _default_url,
            "llm_model":            llm_model,
            "tts_engine":           self._sel_tts,
            "tts_voice":            tts_voice,
            "tts_speed":            tts_speed,
            "elevenlabs_api_key":   self._el_key_input.text().strip(),
            "gemini_voice_api_key": _voice_key,
            "groq_api_key":         _groq_key,
            "moondream_api_key":    self._moondream_key_input.text().strip(),
            "screenshot_quality_llm":     self._llm_quality_spin.value(),
            "screenshot_quality_moondream": self._md_quality_spin.value(),
            "tts_barge_in":         bool(self._barge_check.isChecked()),
            "echo_guard_mode":      "auto" if self._echo_guard_check.isChecked() else "off",
        }
        if self._sel_tts == "gemini":
            cfg["gemini_tts_voice"] = tts_voice
        if self._sel_tts in ("gemini_live", "hashim_live"):
            cfg["gemini_live_voice"] = tts_voice

        # Cloudflare fields
        if hasattr(self, "_cf_account_input"):
            cfg["cf_account_id"] = self._cf_account_input.text().strip()
        if hasattr(self, "_cf_token_input"):
            cfg["cf_api_token"] = self._cf_token_input.text().strip()
        if hasattr(self, "_get_cf_model"):
            cfg["cf_model"] = self._get_cf_model()

        # Groq LLM fields
        if hasattr(self, "_groq_llm_key_input"):
            cfg["groq_llm_key"] = self._groq_llm_key_input.text().strip()
        if hasattr(self, "_get_groq_model"):
            cfg["groq_llm_model"] = self._get_groq_model()

        # Kaggle fields
        if hasattr(self, "_kaggle_url_input"):
            cfg["kaggle_url"] = self._kaggle_url_input.text().strip()
        if hasattr(self, "_get_kaggle_model"):
            cfg["kaggle_model"] = self._get_kaggle_model()

        # Gemini LLM fields
        if hasattr(self, "_gemini_llm_key_input"):
            cfg["gemini_api_key"] = self._gemini_llm_key_input.text().strip()

        # Summarization model fields (dedicated, separate from main LLM)
        if hasattr(self, "_summ_model_input"):
            _summ_prov = self._sel_summarization_provider if hasattr(self, "_sel_summarization_provider") else "auto"
            cfg["summarization_provider"] = "" if _summ_prov == "auto" else _summ_prov
            cfg["summarization_model"] = self._summ_model_input.text().strip()
            cfg["summarization_url"] = self._summ_url_input.text().strip() or "http://localhost:11434"
            cfg["summarization_api_key"] = self._summ_key_input.text().strip()

        if _provider == "gemini_live":
            if hasattr(self, "_get_gemini_model"):
                cfg["gemini_live_model"] = "gemini-2.5-flash-native-audio-preview-12-2025"
            if hasattr(self, "_live_voice_combo"):
                cfg["gemini_live_voice"] = self._live_voice_combo.currentData() or "Charon"
        elif hasattr(self, "_get_gemini_model"):
            cfg["gemini_model"] = self._get_gemini_model()

        # OpenRouter fields
        if hasattr(self, "_or_key_input"):
            cfg["or_api_key"] = self._or_key_input.text().strip()
        if hasattr(self, "_get_or_model"):
            cfg["or_model"] = self._get_or_model()

        # NVIDIA fields
        if hasattr(self, "_nv_key_input"):
            cfg["nvidia_api_key"] = self._nv_key_input.text().strip()
        if hasattr(self, "_get_nv_model"):
            cfg["nvidia_model"] = self._get_nv_model()

        # Kilo fields
        if hasattr(self, "_kilo_key_input"):
            cfg["kilo_api_key"] = self._kilo_key_input.text().strip()
        if hasattr(self, "_get_kilo_model"):
            cfg["kilo_model"] = self._get_kilo_model()

        # Web Search Provider fields
        if hasattr(self, "_websearch_provider_combo"):
            cfg["web_search_provider"] = self._websearch_provider_combo.currentData()
        if hasattr(self, "_exa_key_input"):
            cfg["exa_api_key"] = self._exa_key_input.text().strip()
        if hasattr(self, "_exa_type_combo"):
            cfg["exa_default_type"] = self._exa_type_combo.currentData()

        if self._sel_stt == "vosk" and stt_model:
            cfg["vosk_model_path"] = stt_model

        self.done.emit(json.dumps(cfg))


class StartupPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(f"""
            StartupPanel {{
                background: rgba(0, 6, 10, 235);
                border: 1px solid {C.BORDER_B};
                border-radius: 8px;
            }}
        """)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 20, 28, 20)
        lay.setSpacing(10)

        title = QLabel("◈  SYSTEMS INITIALISING")
        title.setFont(font_ui(12, True))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"color: {C.PRI}; background: transparent;")
        lay.addWidget(title)

        lay.addSpacing(2)

        self._rows: dict[str, dict] = {}
        _COMPS = [
            ("stt", "SPEECH RECOGNITION  (STT)", C.GREEN),
            ("llm", "LANGUAGE MODEL  (LLM)",      C.ACC2),
            ("tts", "VOICE SYNTHESIS  (TTS)",      C.PRI),
        ]
        for key, label, color in _COMPS:
            box = QWidget()
            box.setStyleSheet(
                f"background: {C.PANEL2}; border: 1px solid {C.BORDER}; border-radius: 4px;"
            )
            box_lay = QVBoxLayout(box)
            box_lay.setContentsMargins(10, 6, 10, 6)
            box_lay.setSpacing(4)

            top = QHBoxLayout()
            nm = QLabel(label)
            nm.setFont(font_ui(9.5, True))
            nm.setStyleSheet(f"color: {C.TEXT_MED}; background: transparent; border: none;")
            top.addWidget(nm)
            top.addStretch()

            st = QLabel("LOADING...")
            st.setFont(font_ui(9.5, True))
            st.setStyleSheet(f"color: {C.TEXT_DIM}; background: transparent; border: none;")
            top.addWidget(st)
            box_lay.addLayout(top)

            bar = QProgressBar()
            bar.setFixedHeight(4)
            bar.setRange(0, 0)
            bar.setTextVisible(False)
            bar.setStyleSheet(f"""
                QProgressBar {{
                    background: {C.BAR_BG}; border: none; border-radius: 2px;
                }}
                QProgressBar::chunk {{
                    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                        stop:0 {C.BORDER}, stop:1 {color});
                    border-radius: 2px; width: 60px; margin: 0px;
                }}
            """)
            box_lay.addWidget(bar)
            lay.addWidget(box)
            self._rows[key] = {"bar": bar, "status": st, "color": color}

        lay.addSpacing(4)

        self._status_lbl = QLabel("Initialising components...")
        self._status_lbl.setFont(font_ui(9.5))
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_lbl.setStyleSheet(f"color: {C.TEXT_DIM}; background: transparent;")
        self._status_lbl.setWordWrap(True)
        lay.addWidget(self._status_lbl)

        tip = QLabel("All AI models run 100% locally . No data leaves your device")
        tip.setFont(font_ui(8.5))
        tip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tip.setStyleSheet(f"color: {C.BORDER}; background: transparent;")
        lay.addWidget(tip)

    def update_component(self, key: str, status: str) -> None:
        if key not in self._rows:
            return
        row = self._rows[key]
        ok     = status == "ready"
        color  = row["color"] if ok else C.RED
        label  = "READY  ✓" if ok else "ERROR  ✗"

        bar = row["bar"]
        bar.setRange(0, 100)
        bar.setValue(100)
        bar.setStyleSheet(f"""
            QProgressBar {{
                background: {C.BAR_BG}; border: none; border-radius: 2px;
            }}
            QProgressBar::chunk {{
                background: {color}; border-radius: 2px;
            }}
        """)
        st = row["status"]
        st.setText(label)
        st.setStyleSheet(f"color: {color}; background: transparent; border: none;")

    def set_status(self, text: str) -> None:
        self._status_lbl.setText(text)
        col = C.GREEN if "online" in text.lower() else C.TEXT_DIM
        self._status_lbl.setStyleSheet(f"color: {col}; background: transparent;")


class MainWindow(QMainWindow):
    _log_sig           = pyqtSignal(str)
    _state_sig         = pyqtSignal(str)
    _startup_sig       = pyqtSignal(str, str)
    _session_refresh_sig = pyqtSignal()
    _attachment_ready_sig = pyqtSignal(str, object)
    _audio_level_sig = pyqtSignal(float)
    _visemes_sig = pyqtSignal(object, float, float)

    def __init__(self, face_path: str):
        super().__init__()
        self.setWindowTitle("Orthos")
        self.setMinimumSize(_MIN_W, _MIN_H)
        self.resize(_DEFAULT_W, _DEFAULT_H)

        screen = QApplication.primaryScreen().availableGeometry()
        self.move(
            (screen.width()  - _DEFAULT_W) // 2,
            (screen.height() - _DEFAULT_H) // 2,
        )

        self.setAcceptDrops(True)
        self.on_text_command  = None
        self._muted           = False
        self._screen_vision   = self._load_screen_vision_setting()
        self._current_file: str | None = None
        self._sidebar_visible = True
        self._cur_project_id: str | None = None
        self._sessions_pin_state: dict[str, bool] = {}
        self._recent_files: list[str] = []
        self._MAX_RECENT_FILES = 10
        self._MAX_FILE_SIZE_MB = 25
        self._pending_files: list[dict] = []
        self.on_text_command_with_files = None
        self.on_query_typing = None   # E: typing-time query prefetch (main.py)

        central = QWidget()
        central.setStyleSheet(f"background: {C.BG};")
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_header())

        # Splitter for resizable sidebar
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setHandleWidth(2)
        self._splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background: {C.BORDER};
                width: 2px;
            }}
            QSplitter::handle:hover {{
                background: {C.PRI_DIM};
            }}
        """)

        self._left_panel = self._build_session_panel()
        self._left_panel.setMinimumWidth(160)
        self._left_panel.setMaximumWidth(400)
        self._splitter.addWidget(self._left_panel)

        # Independent splitter: drag the divider beside Activity Log to make
        # the chat wider/narrower without changing the sessions sidebar.
        self._chat_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._chat_splitter.setChildrenCollapsible(False)
        self._chat_splitter.setHandleWidth(6)
        self._chat_splitter.setStyleSheet(f"""
            QSplitter::handle {{ background: {C.BORDER}; width: 6px; }}
            QSplitter::handle:hover {{ background: {C.PRI_DIM}; }}
        """)
        self.hud = HudCanvas(face_path)
        self.hud.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.hud.setMinimumWidth(360)
        self._audio_level_sig.connect(self.hud.set_audio_level)
        self._visemes_sig.connect(self.hud.push_visemes)
        self._chat_splitter.addWidget(self.hud)

        self._right_panel = self._build_right_panel()
        self._chat_splitter.addWidget(self._right_panel)
        self._chat_splitter.setStretchFactor(0, 1)
        self._chat_splitter.setStretchFactor(1, 0)
        self._chat_splitter.setSizes([_DEFAULT_W - _LEFT_W - _RIGHT_W, _RIGHT_W])
        self._chat_splitter.splitterMoved.connect(self._save_activity_panel_width)
        QTimer.singleShot(0, self._restore_activity_panel_width)

        self._splitter.addWidget(self._chat_splitter)
        self._splitter.setSizes([_LEFT_W, _DEFAULT_W - _LEFT_W])
        root.addWidget(self._splitter, stretch=1)
        root.addWidget(self._build_footer())

        self._clock_tmr = QTimer(self)
        self._clock_tmr.timeout.connect(self._tick_clock)
        self._clock_tmr.start(1000)
        self._tick_clock()

        self._metric_tmr = QTimer(self)
        self._metric_tmr.timeout.connect(self._update_metrics)
        self._metric_tmr.start(2000)
        self._update_metrics()

        self._log_sig.connect(self._log.append_log)
        self._state_sig.connect(self._apply_state)
        self._startup_sig.connect(self._on_startup_sig)
        self._session_refresh_sig.connect(self._refresh_session_list)
        self._attachment_ready_sig.connect(self._on_attachment_ready)

        self._overlay: SetupOverlay | None = None
        self._startup_panel: StartupPanel | None = None
        self._on_reconfigure_cb = None
        self._ready = self._check_config()
        if not self._ready:
            self._show_setup()

        sc_mute = QShortcut(QKeySequence("F4"), self)
        sc_mute.activated.connect(self._toggle_mute)
        sc_full = QShortcut(QKeySequence("F11"), self)
        sc_full.activated.connect(self._toggle_fullscreen)

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _restore_activity_panel_width(self):
        """Restore the user-selected Activity Log width after the window lays out."""
        settings = QSettings("Orthos", "Orthos")
        width = settings.value("layout/activity_panel_width", _RIGHT_W, type=int)
        width = max(280, min(2400, width))
        total = max(self._chat_splitter.width(), self.hud.minimumWidth() + width)
        self._chat_splitter.setSizes([max(self.hud.minimumWidth(), total - width), width])

    def _save_activity_panel_width(self, _position: int, _index: int):
        sizes = self._chat_splitter.sizes()
        if len(sizes) == 2:
            QSettings("Orthos", "Orthos").setValue("layout/activity_panel_width", sizes[1])

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cw = self.centralWidget()
        if self._overlay and self._overlay.isVisible():
            ow, oh = 820, 720
            self._overlay.setGeometry(
                (cw.width()  - ow) // 2,
                (cw.height() - oh) // 2,
                ow, oh,
            )


        if self._startup_panel and self._startup_panel.isVisible():
            pw, ph = 400, 310
            self._startup_panel.setGeometry(
                (cw.width()  - pw) // 2,
                (cw.height() - ph) // 2,
                pw, ph,
            )

        # Keep the scroll FAB anchored to the chat's corner.
        self._position_fab()

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts: Ctrl+V for clipboard paste."""
        if event.matches(QKeySequence.StandardKey.Paste):
            clipboard = QApplication.clipboard()
            mime = clipboard.mimeData()
            if mime.hasImage():
                image = clipboard.image()
                if not image.isNull():
                    import tempfile
                    path = tempfile.mktemp(suffix=".png", dir=str(_CAPTURES_DIR))
                    image.save(path)
                    self._on_file_selected(path)
                    self._show_paste_toast("Image attached!")
                    return
            elif mime.hasUrls():
                urls = mime.urls()
                if urls:
                    path = urls[0].toLocalFile()
                    import os as _os
                    if _os.path.isfile(path):
                        self._on_file_selected(path)
                        self._show_paste_toast("File attached!")
                        return
        super().keyPressEvent(event)


    # -- Drag & Drop on entire window -------------------------------------------
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._show_drop_overlay(True)

    def dragLeaveEvent(self, event):
        self._show_drop_overlay(False)

    def dropEvent(self, event):
        self._show_drop_overlay(False)
        urls = event.mimeData().urls()
        for url in urls:
            path = url.toLocalFile()
            if Path(path).is_file():
                self._on_file_selected(path)

    def _show_drop_overlay(self, show: bool):
        if not hasattr(self, "_drop_overlay"):
            from PyQt6.QtWidgets import QLabel as _QLbl
            self._drop_overlay = _QLbl("Drop files here", self)
            self._drop_overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._drop_overlay.setFont(font_ui(15, True, display=True))
            self._drop_overlay.setStyleSheet("background: rgba(0,20,40,200); color: #00d4ff; border: 2px dashed #00d4ff; border-radius: 8px;")
            self._drop_overlay.hide()
        if show:
            self._drop_overlay.setGeometry(self.rect())
            self._drop_overlay.raise_()
            self._drop_overlay.show()
        else:
            self._drop_overlay.hide()

    # -- Paste toast notification ----------------------------------------------
    def _show_paste_toast(self, msg: str = "File attached!"):
        from PyQt6.QtWidgets import QLabel as _QLbl
        from PyQt6.QtCore import QTimer as _QTimer
        toast = _QLbl(msg, self)
        toast.setAlignment(Qt.AlignmentFlag.AlignCenter)
        toast.setFont(font_ui(10))
        toast.setStyleSheet("background: rgba(0,40,20,220); color: #00ff88; border: 1px solid #00ff88; border-radius: 6px; padding: 6px 16px;")
        toast.adjustSize()
        toast.move((self.width() - toast.width()) // 2, self.height() - 80)
        toast.show()
        _QTimer.singleShot(2000, toast.deleteLater)

    def _on_startup_sig(self, action: str, data: str) -> None:
        if action == "show":
            self._create_startup_panel()
        elif action in ("ready", "error"):
            if self._startup_panel:
                self._startup_panel.update_component(data, action)
        elif action == "status":
            if self._startup_panel:
                self._startup_panel.set_status(data)
        elif action == "hide":
            if self._startup_panel:
                QTimer.singleShot(1200, self._destroy_startup_panel)

    def _create_startup_panel(self) -> None:
        if self._startup_panel and self._startup_panel.isVisible():
            return
        cw = self.centralWidget()
        pw, ph = 400, 310
        panel = StartupPanel(cw)
        panel.setGeometry(
            (cw.width()  - pw) // 2,
            (cw.height() - ph) // 2,
            pw, ph,
        )
        panel.show()
        panel.raise_()
        self._startup_panel = panel

    def _destroy_startup_panel(self) -> None:
        if self._startup_panel:
            self._startup_panel.hide()
            self._startup_panel.deleteLater()
            self._startup_panel = None

    def _update_metrics(self):
        snap = _metrics.snapshot()

        cpu = snap["cpu"]
        self._bar_cpu.set_value(cpu, f"{cpu:.0f}%")

        mem = snap["mem"]
        self._bar_mem.set_value(mem, f"{mem:.0f}%")

        net = snap["net"]
        if net < 1.0:
            net_str = f"{net*1024:.0f}KB/s"
        else:
            net_str = f"{net:.1f}MB/s"
        net_pct = min(100, net * 10)
        self._bar_net.set_value(net_pct, net_str)

        gpu = snap["gpu"]
        if gpu >= 0:
            self._bar_gpu.set_value(gpu, f"{gpu:.0f}%")
        else:
            self._bar_gpu.set_value(0, "N/A")

        tmp = snap["tmp"]
        if tmp >= 0:
            tmp_pct = min(100, (tmp / 100) * 100)
            self._bar_tmp.set_value(tmp_pct, f"{tmp:.0f}°C")
        else:
            self._bar_tmp.set_value(0, "N/A")

        try:
            boot_t  = snap["boot"]
            elapsed = time.time() - boot_t
            h = int(elapsed // 3600)
            m = int((elapsed % 3600) // 60)
            self._uptime_lbl.setText(f"UP  {h:02d}:{m:02d}")
        except Exception:
            self._uptime_lbl.setText("UP  --:--")

        try:
            proc_count = snap["proc"]
            self._proc_lbl.setText(f"PROC  {proc_count}")
        except Exception:
            self._proc_lbl.setText("PROC  --")


    def _build_header(self) -> QWidget:
        w = QWidget()
        w.setFixedHeight(54)
        w.setStyleSheet(f"background: {C.DARK}; border-bottom: 1px solid {C.BORDER_B};")
        lay = QHBoxLayout(w)
        lay.setContentsMargins(16, 0, 16, 0)

        # Sidebar toggle
        self._sidebar_btn = QPushButton("☰")
        self._sidebar_btn.setFixedSize(30, 30)
        self._sidebar_btn.setFont(font_ui(13, True))
        self._sidebar_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._sidebar_btn.setToolTip("Toggle session sidebar")
        self._sidebar_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {C.TEXT_DIM};
                border: 1px solid {C.BORDER}; border-radius: 4px;
            }}
            QPushButton:hover {{ color: {C.PRI}; border: 1px solid {C.PRI_DIM}; }}
        """)
        self._sidebar_btn.clicked.connect(self._toggle_sidebar)
        lay.addWidget(self._sidebar_btn)

        def _badge(txt, color=C.TEXT_MED):
            l = QLabel(txt)
            l.setFont(font_ui(9.5))
            l.setStyleSheet(f"color: {color}; background: transparent;")
            return l

        lay.addWidget(_badge("ORTHOS", C.PRI_DIM))
        lay.addStretch()

        mid = QVBoxLayout(); mid.setSpacing(1)
        title = QLabel("<span style='font-weight:bold; letter-spacing:3px;'>O<span style='color:#ff6b00'>R</span>THOS</span>")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(font_ui(20, True, display=True))
        title.setStyleSheet("color: #00d4ff; background: transparent;")
        mid.addWidget(title)
        sub = QLabel("AI Assistant")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setFont(font_ui(8.5))
        sub.setStyleSheet(f"color: {C.PRI_DIM}; background: transparent;")
        mid.addWidget(sub)
        lay.addLayout(mid)
        lay.addStretch()

        right_col = QVBoxLayout(); right_col.setSpacing(2)
        self._clock_lbl = QLabel("00:00:00")
        self._clock_lbl.setFont(font_ui(15, True, display=True))
        self._clock_lbl.setStyleSheet(f"color: {C.PRI}; background: transparent;")
        self._clock_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        right_col.addWidget(self._clock_lbl)
        self._date_lbl = QLabel("")
        self._date_lbl.setFont(font_ui(8.5))
        self._date_lbl.setStyleSheet(f"color: {C.TEXT_DIM}; background: transparent;")
        self._date_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        right_col.addWidget(self._date_lbl)
        lay.addLayout(right_col)
        return w

    def _tick_clock(self):
        self._clock_lbl.setText(time.strftime("%H:%M:%S"))
        self._date_lbl.setText(time.strftime("%a %d %b %Y"))

    def _toggle_sidebar(self):
        self._sidebar_visible = not self._sidebar_visible
        self._left_panel.setVisible(self._sidebar_visible)

    def _build_session_panel(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet(f"background: {C.DARK}; border-right: 1px solid {C.BORDER};")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 10, 8, 6)
        lay.setSpacing(4)

        # ── Header ──
        ses_hdr = QLabel("☰  SESSIONS")
        ses_hdr.setFont(font_ui(9.5, True))
        ses_hdr.setStyleSheet(f"color: {C.PRI}; background: transparent; "
                              f"border-bottom: 1px solid {C.BORDER}; padding-bottom: 4px;")
        lay.addWidget(ses_hdr)

        # ── New Chat button ──
        self._new_chat_btn = QPushButton("✦  New Chat")
        self._new_chat_btn.setFixedHeight(28)
        self._new_chat_btn.setFont(font_ui(9.5, True))
        self._new_chat_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._new_chat_btn.setStyleSheet(f"""
            QPushButton {{
                background: {C.PRI_GHO}; color: {C.PRI};
                border: 1px solid {C.PRI_DIM}; border-radius: 4px;
                padding: 0 8px;
            }}
            QPushButton:hover {{ background: {C.BORDER_B}; color: {C.WHITE}; }}
        """)
        self._new_chat_btn.clicked.connect(self._on_new_chat)
        lay.addWidget(self._new_chat_btn)

        # ── Branch indicator ──
        self._branch_lbl = QLabel("")
        self._branch_lbl.setFont(font_ui(8))
        self._branch_lbl.setStyleSheet(f"color: {C.TEXT_DIM}; background: transparent; padding: 0 2px;")
        self._branch_lbl.setVisible(False)
        lay.addWidget(self._branch_lbl)

        # ── Project switcher ──
        proj_row = QHBoxLayout()
        proj_row.setSpacing(4)
        self._project_combo = QComboBox()
        self._project_combo.setFont(font_ui(8.5))
        self._project_combo.setStyleSheet(f"""
            QComboBox {{
                background: #000d14; color: {C.WHITE};
                border: 1px solid {C.BORDER}; border-radius: 3px;
                padding: 1px 4px; min-height: 20px;
            }}
            QComboBox:hover {{ border: 1px solid {C.PRI_DIM}; }}
            QComboBox::drop-down {{ border: none; width: 16px; }}
            QComboBox QAbstractItemView {{
                background: #001a24; color: {C.WHITE};
                selection-background-color: {C.PRI_DIM};
                border: 1px solid {C.BORDER};
            }}
        """)
        self._project_combo.currentIndexChanged.connect(self._on_project_combo_changed)
        proj_row.addWidget(self._project_combo, stretch=1)

        self._add_project_btn = QPushButton("+")
        self._add_project_btn.setFixedSize(22, 22)
        self._add_project_btn.setFont(font_ui(9.5, True))
        self._add_project_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._add_project_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {C.ACC2};
                border: 1px solid {C.BORDER}; border-radius: 3px;
            }}
            QPushButton:hover {{ color: {C.WHITE}; border: 1px solid {C.ACC2}; }}
        """)
        self._add_project_btn.clicked.connect(self._on_add_project)
        proj_row.addWidget(self._add_project_btn)
        lay.addLayout(proj_row)

        # ── Session search bar ──
        self._session_search = QLineEdit()
        self._session_search.setPlaceholderText("🔍 Search sessions…")
        self._session_search.setClearButtonEnabled(True)
        self._session_search.setFixedHeight(24)
        self._session_search.setFont(font_ui(8.5))
        self._session_search.setStyleSheet(f"""
            QLineEdit {{
                background: #000d14; color: {C.WHITE};
                border: 1px solid {C.BORDER}; border-radius: 3px;
                padding: 2px 8px;
            }}
            QLineEdit:focus {{ border: 1px solid {C.PRI}; }}
            QLineEdit::clear-button {{
                color: {C.TEXT_DIM}; border: none;
            }}
            QLineEdit::clear-button:hover {{ color: {C.WHITE}; }}
        """)
        self._session_search.setToolTip(
            "Search session titles and conversation content")
        self._session_search.textChanged.connect(self._on_session_search)
        lay.addWidget(self._session_search)

        # ── Scrollable session list ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setStyleSheet(f"""
            QScrollArea {{ border: none; background: transparent; }}
            QScrollBar:vertical {{
                background: transparent; width: 4px; margin: 0; border: none;
            }}
            QScrollBar::handle:vertical {{
                background: {C.BORDER_B}; border-radius: 2px; min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{ background: {C.PRI_DIM}; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0; border: none; background: none;
            }}
        """)

        self._session_container = QWidget()
        self._session_container.setStyleSheet("background: transparent;")
        self._session_list_layout = QVBoxLayout(self._session_container)
        self._session_list_layout.setContentsMargins(0, 0, 0, 0)
        self._session_list_layout.setSpacing(2)
        scroll.setWidget(self._session_container)
        lay.addWidget(scroll, stretch=1)

        # ── Bottom status info ──
        self._session_status_lbl = QLabel("No sessions")
        self._session_status_lbl.setFont(font_ui(8))
        self._session_status_lbl.setStyleSheet(f"color: {C.TEXT_DIM}; background: transparent; "
                                                f"border-top: 1px solid {C.BORDER}; padding-top: 4px;")
        lay.addWidget(self._session_status_lbl)

        # ── Metrics moved to footer ──

        return w
    def _build_right_panel(self) -> QWidget:
        w = QWidget()
        w.setMinimumWidth(260)
        w.setStyleSheet(f"background: {C.DARK}; border-left: 1px solid {C.BORDER};")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        def _sec(txt):
            l = QLabel(f"▸ {txt}")
            l.setFont(font_ui(8.5, True))
            l.setStyleSheet(f"color: {C.TEXT_MED}; background: transparent;")
            return l

        # ── Session Info ──
        self._session_info_frame = QFrame()
        self._session_info_frame.setStyleSheet(f"""
            QFrame {{
                background: #010d14;
                border: 1px solid {C.BORDER};
                border-radius: 4px;
            }}
        """)
        info_lay = QVBoxLayout(self._session_info_frame)
        info_lay.setContentsMargins(10, 6, 10, 6)
        info_lay.setSpacing(1)

        self._info_title = QLabel("No session")
        self._info_title.setFont(font_ui(10.5, True))
        self._info_title.setStyleSheet(f"color: {C.PRI}; background: transparent;")
        info_lay.addWidget(self._info_title)

        info_detail = QWidget()
        info_detail.setStyleSheet("background: transparent;")
        info_det_lay = QHBoxLayout(info_detail)
        info_det_lay.setContentsMargins(0, 0, 0, 0)
        info_det_lay.setSpacing(8)

        self._info_turns = QLabel("")
        self._info_turns.setFont(font_ui(10))
        self._info_turns.setStyleSheet(f"color: {C.TEXT_DIM}; background: transparent;")
        info_det_lay.addWidget(self._info_turns)

        self._info_tokens = QLabel("")
        self._info_tokens.setFont(font_ui(10))
        self._info_tokens.setStyleSheet(f"color: {C.TEXT_DIM}; background: transparent;")
        info_det_lay.addWidget(self._info_tokens)

        self._info_created = QLabel("")
        self._info_created.setFont(font_ui(10))
        self._info_created.setStyleSheet(f"color: {C.TEXT_DIM}; background: transparent;")
        info_det_lay.addWidget(self._info_created)

        info_det_lay.addStretch()
        info_lay.addWidget(info_detail)

        sep_info = QFrame()
        sep_info.setFrameShape(QFrame.Shape.HLine)
        sep_info.setStyleSheet(f"color: {C.BORDER}; margin: 2px 0;")
        info_lay.addWidget(sep_info)

        self._session_info_frame.setVisible(False)
        lay.addWidget(self._session_info_frame)

        lay.addWidget(_sec("ACTIVITY LOG"))
        self._log = RichLogWidget()

        # ── Scroll-to-bottom FAB (ChatGPT-style) ─────────────────────
        # Child of the log: floats over the chat's bottom-right corner.
        self._scroll_fab = QPushButton("↓")
        self._scroll_fab.setParent(self._log)
        self._scroll_fab.setFixedSize(34, 34)
        self._scroll_fab.setCursor(Qt.CursorShape.PointingHandCursor)
        self._scroll_fab.setToolTip("Jump to latest messages")
        self._scroll_fab.setFont(font_ui(12, True))
        self._scroll_fab.setStyleSheet(f"""
            QPushButton {{
                background: {C.PANEL2}; color: {C.PRI};
                border: 1px solid {C.BORDER_B}; border-radius: 17px;
            }}
            QPushButton:hover {{ background: {C.PRI_GHO}; border: 1px solid {C.PRI}; }}
        """)
        self._scroll_fab.clicked.connect(self._on_scroll_fab_clicked)
        self._scroll_fab.hide()

        # Badge showing unread message count while browsing history.
        self._fab_badge = QLabel("")
        self._fab_badge.setParent(self._log)
        self._fab_badge.setFont(font_ui(7.5, True))
        self._fab_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._fab_badge.setStyleSheet(f"""
            QLabel {{
                background: {C.RED}; color: #ffffff;
                border-radius: 8px; min-width: 16px; min-height: 16px;
                padding: 1px 3px;
            }}
        """)
        self._fab_badge.hide()

        # Floating "Orthos is thinking" badge — visible ONLY while the
        # reader is browsing history and the live indicator is off-screen
        # at the document tail (decision: 2026-09-06 chat UX review).
        self._thinking_badge = QLabel("● Orthos is thinking…")
        self._thinking_badge.setParent(self._log)
        self._thinking_badge.setFont(font_ui(8.5, True))
        self._thinking_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thinking_badge.setCursor(Qt.CursorShape.PointingHandCursor)
        self._thinking_badge.setStyleSheet(f"""
            QLabel {{
                background: {C.PANEL2}; color: {C.PRI};
                border: 1px solid {C.BORDER_B}; border-radius: 12px;
                padding: 3px 12px;
            }}
            QLabel:hover {{ background: {C.PRI_GHO}; }}
        """)
        self._thinking_badge.setToolTip("Jump to the live reply")
        # Click = come down to the indicator: routed through the FAB logic.
        self._thinking_badge.mousePressEvent = lambda ev: (
            self._on_scroll_fab_clicked())
        self._thinking_badge.hide()

        # Track unread from the log; show the FAB when reading history.
        if hasattr(self._log, "_on_unread_changed_cb"):
            self._log._on_unread_changed_cb = self._on_log_unread
        if hasattr(self._log, "verticalScrollBar"):
            self._log.verticalScrollBar().valueChanged.connect(
                self._on_log_scrolled)

        lay.addWidget(self._log, stretch=1)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color: {C.BORDER}; margin: 2px 0;")
        lay.addWidget(sep)

        # Attachment tray: readable cards instead of tiny, overflowing chips.
        self._file_chips_container = QScrollArea()
        self._file_chips_container.setWidgetResizable(True)
        self._file_chips_container.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._file_chips_container.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._file_chips_container.setFixedHeight(126)
        self._file_chips_container.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        self._file_chips_content = QWidget()
        self._file_chips_content.setStyleSheet("background: transparent;")
        self._file_chips_layout = QVBoxLayout(self._file_chips_content)
        self._file_chips_layout.setContentsMargins(0, 4, 4, 4)
        self._file_chips_layout.setSpacing(5)
        self._file_chips_layout.addStretch()
        self._file_chips_container.setWidget(self._file_chips_content)
        self._file_chips_container.setVisible(False)
        lay.addWidget(self._file_chips_container)

        lay.addWidget(_sec("COMMAND INPUT"))
        lay.addLayout(self._build_input_row())

        self._mute_btn = QPushButton("🎙  MICROPHONE ACTIVE")
        self._mute_btn.setFixedHeight(30)
        self._mute_btn.setFont(font_ui(9.5, True))
        self._mute_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._mute_btn.clicked.connect(self._toggle_mute)
        self._style_mute_btn()
        lay.addWidget(self._mute_btn)

        self._screen_vision_btn = QPushButton("◉  SCREEN VISION ON")
        self._screen_vision_btn.setFixedHeight(30)
        self._screen_vision_btn.setFont(font_ui(9.5, True))
        self._screen_vision_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._screen_vision_btn.clicked.connect(self._toggle_screen_vision)
        self._style_screen_vision_btn()
        lay.addWidget(self._screen_vision_btn)

        # Utility buttons compressed into ONE icon row (tooltips carry the
        # labels) instead of four stacked full-width buttons — ~80 px of
        # vertical space goes back to the chat/HUD.
        util_row = QHBoxLayout()
        util_row.setSpacing(3)

        def _util_btn(icon: str, tip: str, cb, hover: str) -> None:
            b = QPushButton(icon)
            b.setFixedSize(42, 24)
            b.setFont(font_ui(9))
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setToolTip(tip)
            b.setStyleSheet(f"""
                QPushButton {{
                    background: transparent; color: {C.TEXT_MED};
                    border: 1px solid {C.BORDER}; border-radius: 3px;
                }}
                QPushButton:hover {{
                    color: {hover}; border: 1px solid {hover};
                }}
            """)
            b.clicked.connect(cb)
            util_row.addWidget(b)

        _util_btn("⛶", "Fullscreen [F11]", self._toggle_fullscreen, C.PRI)
        _util_btn("⚙", "Configure", self._show_config, C.ACC2)
        _util_btn("◈", "MCP Manager", self._show_mcp_manager, C.PRI)
        _util_btn("📋", "Memory versions", self._show_memory_versions, C.ACC2)
        lay.addLayout(util_row)

        return w

    def _build_input_row(self) -> QHBoxLayout:
        row = QHBoxLayout(); row.setSpacing(3)

        # Attach button (paperclip icon)
        attach_btn = QPushButton("📎")
        attach_btn.setText("ATTACH")
        attach_btn.setFixedSize(72, 30)
        attach_btn.setFont(font_ui(10.5))
        attach_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        attach_btn.setToolTip("Attach files or images")
        attach_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {C.TEXT_MED};
                border: 1px solid {C.BORDER}; border-radius: 3px;
            }}
            QPushButton:hover {{ color: {C.PRI}; border: 1px solid {C.PRI_DIM}; }}
        """)
        attach_menu = QMenu(attach_btn)
        attach_menu.setStyleSheet(f"QMenu {{ background: {C.PANEL}; color: {C.TEXT}; border: 1px solid {C.BORDER}; }} QMenu::item {{ padding: 7px 18px; }} QMenu::item:selected {{ background: {C.PRI_GHO}; }}")
        attach_menu.addAction("Files or documents…", self._attach_file)
        attach_menu.addAction("Image with preview…", self._attach_image)
        attach_btn.setMenu(attach_menu)
        row.addWidget(attach_btn)

        # Image button
        img_btn = QPushButton("🖼")
        img_btn.setFixedSize(28, 28)
        img_btn.setFont(font_ui(10.5))
        img_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        img_btn.setToolTip("Attach image")
        img_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {C.TEXT_MED};
                border: 1px solid {C.BORDER}; border-radius: 3px;
            }}
            QPushButton:hover {{ color: {C.ACC2}; border: 1px solid {C.ACC2}; }}
        """)
        img_btn.clicked.connect(self._attach_image)
        # Kept as a keyboard-accessible implementation detail; the attachment menu
        # above is now the single visible entry point.
        img_btn.hide()

        # Input field
        self._input = QLineEdit()
        self._input.setPlaceholderText("Type a command or question...")
        self._input.setFont(font_ui(10))
        self._input.setFixedHeight(30)
        self._input.setStyleSheet(f"""
            QLineEdit {{
                background: #000d14; color: {C.WHITE};
                border: 1px solid {C.BORDER}; border-radius: 3px; padding: 3px 7px;
            }}
            QLineEdit:focus {{ border: 1px solid {C.PRI}; }}
        """)
        self._input.returnPressed.connect(self._send)
        self._input.textChanged.connect(self._on_query_typing)   # E: debounce → prefetch
        row.addWidget(self._input, stretch=1)

        # Send button
        send = QPushButton("▸")
        send.setFixedSize(30, 30)
        send.setFont(font_ui(12, True))
        send.setCursor(Qt.CursorShape.PointingHandCursor)
        send.setStyleSheet(f"""
            QPushButton {{
                background: {C.PANEL}; color: {C.PRI};
                border: 1px solid {C.PRI_DIM}; border-radius: 3px;
            }}
            QPushButton:hover {{ background: {C.PRI_GHO}; border: 1px solid {C.PRI}; }}
        """)
        send.clicked.connect(self._send)
        row.addWidget(send)
        return row

    def _attach_file(self):
        """Open file dialog for multiple files — shows grid preview for images."""
        from PyQt6.QtGui import QPixmap
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGridLayout, QScrollArea
        
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Files", "",
            "All Files (*);;Documents (*.pdf *.doc *.docx *.txt);;Code (*.py *.js *.ts *.html *.css);;Images (*.png *.jpg *.jpeg *.gif *.bmp *.webp)"
        )
        if not paths:
            return
        
        # Single file — direct attach
        if len(paths) == 1:
            self._on_file_selected(paths[0])
            return
        
        # Multiple files — show grid preview
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Attach {len(paths)} files")
        dlg.setMinimumSize(420, 350)
        dlg.setStyleSheet(f"QDialog {{ background: {C.BG}; }} QLabel {{ color: {C.TEXT}; background: transparent; }} QPushButton {{ background: {C.PANEL}; color: {C.TEXT_MED}; border: 1px solid {C.BORDER}; border-radius: 3px; padding: 6px 16px; font-family: 'Segoe UI'; font-size: 8pt; }} QPushButton:hover {{ color: {C.PRI}; border: 1px solid {C.PRI}; }}")
        
        lay = QVBoxLayout(dlg)
        lay.setSpacing(8)
        
        title = QLabel(f"{len(paths)} files selected")
        title.setFont(font_ui(10.5, True))
        title.setStyleSheet(f"color: {C.PRI}; background: transparent;")
        lay.addWidget(title)
        
        # Grid of file previews (scrollable)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        grid_w = QWidget()
        grid = QGridLayout(grid_w)
        grid.setSpacing(6)
        
        for idx, path in enumerate(paths[:12]):  # max 12 previews
            p = Path(path)
            cat = _file_category(p)
            cell = QVBoxLayout()
            
            if cat == "image" and p.exists():
                try:
                    pix = QPixmap(str(p))
                    if not pix.isNull():
                        thumb = pix.scaled(80, 80, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                        lbl = QLabel()
                        lbl.setPixmap(thumb)
                        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                        lbl.setStyleSheet(f"border: 1px solid {C.BORDER}; border-radius: 4px; background: #000;")
                        cell.addWidget(lbl)
                except Exception:
                    pass
            
            name_lbl = QLabel(p.name[:15] + ("..." if len(p.name) > 15 else ""))
            name_lbl.setFont(font_ui(8.5))
            name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cell.addWidget(name_lbl)
            
            grid.addLayout(cell, idx // 4, idx % 4)
        
        if len(paths) > 12:
            more = QLabel(f"+{len(paths) - 12} more...")
            more.setFont(font_ui(9.5))
            more.setStyleSheet(f"color: {C.TEXT_DIM}; background: transparent;")
            grid.addWidget(more, len(paths) // 4, 0, 1, 4)
        
        scroll.setWidget(grid_w)
        lay.addWidget(scroll)
        
        def _attach_all():
            for path in paths:
                self._on_file_selected(path)
            dlg.accept()
        
        btn_row = QHBoxLayout()
        attach_btn = QPushButton(f"Attach All ({len(paths)})")
        attach_btn.clicked.connect(_attach_all)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dlg.reject)
        btn_row.addStretch()
        btn_row.addWidget(attach_btn)
        btn_row.addWidget(cancel_btn)
        lay.addLayout(btn_row)
        
        dlg.exec()

    def _attach_image(self):
        """Open file dialog for image files — shows preview before attach."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Images (*.png *.jpg *.jpeg *.gif *.bmp *.webp)"
        )
        if path:
            self._show_file_preview(path)

    def _show_file_preview(self, path: str):
        """Show a thumbnail preview of the selected file."""
        from PyQt6.QtGui import QPixmap
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton
        
        p = Path(path)
        cat = _file_category(p)
        
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Preview: {p.name}")
        dlg.setMinimumSize(350, 300)
        dlg.setStyleSheet(f"""
            QDialog {{ background: {C.BG}; }}
            QLabel {{ color: {C.TEXT}; background: transparent; }}
            QPushButton {{ background: {C.PANEL}; color: {C.TEXT_MED}; 
                           border: 1px solid {C.BORDER}; border-radius: 3px;
                           padding: 6px 16px; font-family: 'Segoe UI'; font-size: 8pt; }}
            QPushButton:hover {{ color: {C.PRI}; border: 1px solid {C.PRI}; }}
        """)
        
        lay = QVBoxLayout(dlg)
        lay.setSpacing(10)
        
        if cat == "image":
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                scaled = pixmap.scaled(320, 240, Qt.AspectRatioMode.KeepAspectRatio, 
                                       Qt.TransformationMode.SmoothTransformation)
                preview = QLabel()
                preview.setPixmap(scaled)
                preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lay.addWidget(preview)
        
        size = _fmt_size(p.stat().st_size)
        info = QLabel()
        info.setText(f"<b>Name:</b> {p.name}<br><b>Size:</b> {size}<br><b>Type:</b> {cat.upper()}")
        info.setFont(font_ui(10))
        lay.addWidget(info)
        
        def _use_file():
            self._on_file_selected(path)
            dlg.accept()
        
        btn_row = QHBoxLayout()
        use_btn = QPushButton("Use This File")
        use_btn.clicked.connect(_use_file)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dlg.reject)
        btn_row.addStretch()
        btn_row.addWidget(use_btn)
        btn_row.addWidget(cancel_btn)
        lay.addLayout(btn_row)
        
        dlg.exec()

    def _build_footer(self) -> QWidget:
        w = QWidget()
        w.setFixedHeight(52)
        w.setStyleSheet(f"background: {C.DARK}; border-top: 1px solid {C.BORDER};")
        lay = QHBoxLayout(w); lay.setContentsMargins(14, 0, 14, 0)

        def _fl(txt, color=C.TEXT_MED):
            l = QLabel(txt); l.setFont(font_ui(10.5))
            l.setStyleSheet(f"color: {color}; background: transparent;")
            return l

        # Left: shortcuts
        lay.addWidget(_fl("[F4] Mute  ·  [F11] Fullscreen"))
        lay.addSpacing(12)

        # Center: system metrics (compact)
        self._bar_cpu = MetricBar("CPU", C.PRI)
        self._bar_mem = MetricBar("MEM", C.ACC2)
        self._bar_net = MetricBar("NET", C.ACC)
        self._bar_gpu = MetricBar("GPU", C.ACC)
        self._bar_tmp = MetricBar("TMP", C.GREEN)
        for bar in [self._bar_cpu, self._bar_mem, self._bar_net, self._bar_gpu, self._bar_tmp]:
            bar.setFixedHeight(28)
            lay.addWidget(bar)

        lay.addSpacing(12)

        # Right: uptime + process count
        self._uptime_lbl = QLabel("UP  --:--")
        self._uptime_lbl.setFont(font_mono(9.5))
        self._uptime_lbl.setStyleSheet(f"color: {C.TEXT_DIM}; background: transparent;")
        lay.addWidget(self._uptime_lbl)

        self._proc_lbl = QLabel("PROC  --")
        self._proc_lbl.setFont(font_mono(9.5))
        self._proc_lbl.setStyleSheet(f"color: {C.TEXT_DIM}; background: transparent;")
        lay.addWidget(self._proc_lbl)

        return w

    # ── Session sidebar methods ─────────────────────────────────────────

    def _refresh_session_info(self):
        if not hasattr(self, "_info_title"):
            return
        sid = getattr(self, "_cur_session_id", None)
        if not sid:
            self._session_info_frame.setVisible(False)
            return
        self._session_info_frame.setVisible(True)
        session = _db_get_session(sid)
        if not session:
            return
        title = session.get("title", "New Chat")
        self._info_title.setText(f"◈  {title}")

        turn_count = _db_get_turn_count(sid)
        self._info_turns.setText(f"TURNS {turn_count}")

        tokens = _db_est_tokens(sid)
        if tokens >= 1000:
            self._info_tokens.setText(f"TOKENS ~{tokens/1000:.1f}K")
        else:
            self._info_tokens.setText(f"TOKENS ~{tokens}")

        created = session.get("created_at", "")[:10]
        self._info_created.setText(f"CREATED {created}")

    def _on_new_chat(self):
        if hasattr(self, "_session_cb_new"):
            self._session_cb_new()

    def _on_session_click(self, session_id: str):
        if hasattr(self, "_session_cb_switch"):
            self._session_cb_switch(session_id)
        self._refresh_session_info()

    def _on_session_context_menu(self, session_id: str, pos):
        """Right-click context menu for rename/delete/fork."""
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background: {C.PANEL}; color: {C.TEXT};
                border: 1px solid {C.BORDER}; padding: 4px;
                font-family: 'Segoe UI'; font-size: 8pt;
            }}
            QMenu::item {{
                padding: 4px 16px; border-radius: 2px;
            }}
            QMenu::item:selected {{
                background: {C.PRI_GHO}; color: {C.PRI};
            }}
        """)

        rename_action = menu.addAction("✏  Rename")
        fork_action = menu.addAction("␢  Fork conversation")
        delete_action = menu.addAction("✕  Delete")

        action = menu.exec(pos)
        if action == rename_action:
            if hasattr(self, "_session_cb_rename"):
                self._session_cb_rename(session_id)
        elif action == fork_action:
            if hasattr(self, "_session_cb_fork"):
                self._session_cb_fork(session_id)
        elif action == delete_action:
            if hasattr(self, "_session_cb_delete"):
                self._session_cb_delete(session_id)

    def _on_session_search(self, _text: str):
        """Live-filter the session list by the search bar."""
        self._refresh_session_list()

    def _refresh_session_list(self):
        """Rebuild session list widget from DB with active highlight."""
        try:
            pid = getattr(self, "_cur_project_id", None)
            query = (self._session_search.text() if hasattr(self, "_session_search")
                     else "").strip()
            if query:
                sessions = _db_search_sessions(
                    query, project_id=pid)
            else:
                sessions = _db_list_sessions(project_id=pid)
        except Exception:
            sessions = []
        # Also refresh project combo
        try:
            self._refresh_project_combo()
        except Exception:
            pass

        while self._session_list_layout.count():
            item = self._session_list_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

        if not sessions:
            if hasattr(self, "_session_status_lbl"):
                query = (self._session_search.text()
                         if hasattr(self, "_session_search") else "").strip()
                if query:
                    self._session_status_lbl.setText(f"No matches for “{query[:24]}”")
                else:
                    self._session_status_lbl.setText("No past sessions")
            self._session_list_layout.addStretch()
            return

        cur_id = getattr(self, "_cur_session_id", None)
        # Move current session to top of list
        if cur_id:
            try:
                idx = next(i for i, s in enumerate(sessions) if s["id"] == cur_id)
                sessions.insert(0, sessions.pop(idx))
            except (StopIteration, IndexError):
                pass
        active_count = sum(1 for s in sessions if s.get("status") == "active")

        # Update branch indicator
        if hasattr(self, "_branch_lbl"):
            cur_session = next((s for s in sessions if s["id"] == cur_id), None)
            if cur_session and cur_session.get("parent_session_id"):
                parent = next((s for s in sessions if s["id"] == cur_session["parent_session_id"]), None)
                parent_title = parent["title"][:20] if parent else cur_session["parent_session_id"][:8]
                self._branch_lbl.setText(f"␢  Fork of: {parent_title}")
                self._branch_lbl.setVisible(True)
            else:
                self._branch_lbl.setVisible(False)
        if hasattr(self, "_session_status_lbl"):
            self._session_status_lbl.setText(
                f"{len(sessions)} session{'s' if len(sessions)!=1 else ''}"
                f" ({active_count} active)"
            )

        # Build parent map for tree display
        parent_map = {}
        for s in sessions:
            pid = s.get("parent_session_id")
            if pid:
                parent_map.setdefault(pid, []).append(s["id"])

        root_sessions = [s for s in sessions if not s.get("parent_session_id")]
        child_sessions = [s for s in sessions if s.get("parent_session_id")]

        class _SessionRow(QWidget):
            def __init__(self, sid, title, status, is_current, pinned, depth, parent_ref):
                super().__init__()
                self.sid = sid
                self._parent = parent_ref
                self._editing = False
                self._orig_title = title
                lay = QHBoxLayout(self)
                lay.setContentsMargins(0, 0, 0, 0)
                lay.setSpacing(2)

                prefix = "  " * depth + ("↳ " if depth > 0 else "")
                max_len = 22 - depth * 2
                display = title if len(title) <= max_len else title[:max_len-3] + "..."

                self.btn = QPushButton(prefix + display)
                self.btn.setFixedHeight(22)
                self.btn.setCursor(Qt.CursorShape.PointingHandCursor)
                self.btn.setStyleSheet(self._btn_style(status, is_current))
                self.btn.clicked.connect(lambda: self._parent._on_session_click(sid))
                self.btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
                self.btn.customContextMenuRequested.connect(
                    lambda pos: self._parent._on_session_context_menu(sid, self.btn.mapToGlobal(pos))
                )
                self.btn.installEventFilter(self)
                lay.addWidget(self.btn, stretch=1)

                self.star_btn = QPushButton("★" if pinned else "☆")
                self.star_btn.setFixedSize(18, 22)
                self.star_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                self.star_btn.setFont(font_ui(9.5))
                star_col = C.ACC2 if pinned else C.TEXT_DIM
                self.star_btn.setStyleSheet(f"""
                    QPushButton {{
                        background: transparent; color: {star_col};
                        border: none; border-radius: 2px; padding: 0;
                    }}
                    QPushButton:hover {{ color: {C.ACC2}; }}
                """)
                self.star_btn.clicked.connect(self._toggle_pin)
                lay.addWidget(self.star_btn)

            def _btn_style(self, status, is_current):
                if is_current:
                    return f"""
                        QPushButton {{
                            background: {C.PRI_GHO}; color: {C.PRI};
                            border: 1px solid {C.PRI}; border-radius: 3px;
                            text-align: left; padding: 3px 6px; font-size: 7pt;
                        }}
                    """
                elif status == "active":
                    return f"""
                        QPushButton {{
                            background: {C.PANEL2}; color: {C.TEXT_MED};
                            border: 1px solid {C.BORDER}; border-radius: 3px;
                            text-align: left; padding: 3px 6px; font-size: 7pt;
                        }}
                    """
                else:
                    return f"""
                        QPushButton {{
                            background: transparent; color: {C.TEXT_DIM};
                            border: 1px solid {C.BORDER}; border-radius: 3px;
                            text-align: left; padding: 3px 6px; font-size: 7pt;
                        }}
                    """

            def _toggle_pin(self):
                current = self._parent._sessions_pin_state.get(self.sid, False)
                new_state = not current
                _db_set_pinned(self.sid, new_state)
                self._parent._sessions_pin_state[self.sid] = new_state
                self._parent._refresh_session_list()

            def eventFilter(self, obj, event):
                if obj is self.btn and event.type() == event.Type.MouseButtonDblClick:
                    self._start_rename()
                    return True
                return super().eventFilter(obj, event)

            def _start_rename(self):
                if self._editing:
                    return
                self._editing = True
                self.btn.setVisible(False)
                self.star_btn.setVisible(False)

                self._edit = QLineEdit(self._orig_title)
                self._edit.setFont(font_ui(8.5))
                self._edit.setStyleSheet(f"""
                    QLineEdit {{
                        background: #001a24; color: {C.WHITE};
                        border: 1px solid {C.PRI}; border-radius: 3px;
                        padding: 2px 6px; font-size: 7pt;
                    }}
                """)
                lay = self.layout()
                lay.insertWidget(0, self._edit, stretch=1)
                self._edit.setFocus()
                self._edit.selectAll()
                self._edit.returnPressed.connect(self._finish_rename)
                self._edit.editingFinished.connect(self._finish_rename)

            def _finish_rename(self):
                if not self._editing:
                    return
                self._editing = False
                new_title = self._edit.text().strip()
                if new_title and new_title != self._orig_title:
                    _db_rename_session(self.sid, new_title)
                self._parent._refresh_session_list()

            def cancel_rename(self):
                if self._editing:
                    self._editing = False
                    self._parent._refresh_session_list()

        def _render_session(s, depth=0):
            # child_sessions gets REASSIGNED below — needs nonlocal (without
            # it the function body treats it as local and crashes before
            # the first assignment on any forked session: UnboundLocalError).
            nonlocal child_sessions
            sid = s["id"]
            title = s.get("title", "New Chat")
            status = s.get("status", "active")
            is_current = (sid == cur_id)
            pinned = s.get("pinned", 0) or s.get("pinned", False)
            self._sessions_pin_state[sid] = bool(pinned)

            row = _SessionRow(sid, title, status, is_current, bool(pinned), depth, self)
            self._session_list_layout.addWidget(row)

            # Render children
            for child_id in parent_map.get(sid, []):
                child = next((x for x in child_sessions if x["id"] == child_id), None)
                if child:
                    _render_session(child, depth + 1)
                    child_sessions = [x for x in child_sessions if x["id"] != child_id]

        for s in root_sessions:
            _render_session(s, 0)
        for s in child_sessions:
            _render_session(s, 0)

        self._session_list_layout.addStretch()
        self._refresh_session_info()

    # ── Project switcher ───────────────────────────────────────────────

    def _refresh_project_combo(self):
        """Populate project dropdown from DB, preserving current selection."""
        if not hasattr(self, "_project_combo"):
            return
        try:
            projects = (self._project_cb_list() if hasattr(self, "_project_cb_list") else [])
        except Exception:
            projects = []
        if not projects:
            projects = [{"id": "default_project", "name": "default"}]

        current_id = getattr(self, "_cur_project_id", None)
        self._project_combo.blockSignals(True)
        self._project_combo.clear()
        for p in projects:
            self._project_combo.addItem(p.get("name", "?"), p.get("id", ""))
        # Try to restore current
        if current_id:
            idx = self._project_combo.findData(current_id)
            if idx >= 0:
                self._project_combo.setCurrentIndex(idx)
        self._project_combo.blockSignals(False)

    def _on_project_combo_changed(self, index: int):
        pid = self._project_combo.itemData(index)
        if pid and pid != getattr(self, "_cur_project_id", None) and hasattr(self, "_project_cb_switch"):
            self._project_cb_switch(pid)

    def _on_add_project(self):
        from PyQt6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(None, "New Project", "Project name:")
        if ok and name and name.strip() and hasattr(self, "_project_cb_create"):
            self._project_cb_create(name.strip())

    # ── Memory version browser ──────────────────────────────────────────

    def _show_memory_versions(self):
        """Open a dialog showing memory version history."""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton, QHBoxLayout
        versions = []
        try:
            if hasattr(self, "_version_cb_browse"):
                versions = self._version_cb_browse()
        except Exception:
            versions = []
        dlg = QDialog(self)
        dlg.setWindowTitle("Memory Version History")
        dlg.setMinimumSize(600, 400)
        dlg.setStyleSheet(f"""
            QDialog {{ background: {C.BG}; }}
            QTextEdit {{ background: #000d14; color: {C.TEXT};
                         border: 1px solid {C.BORDER}; font-family: 'Segoe UI'; font-size: 8pt; }}
            QPushButton {{ background: transparent; color: {C.TEXT_MED};
                           border: 1px solid {C.BORDER}; border-radius: 3px;
                           padding: 4px 16px; font-family: 'Segoe UI'; font-size: 7pt; }}
            QPushButton:hover {{ color: {C.PRI}; border: 1px solid {C.PRI}; }}
        """)
        lay = QVBoxLayout(dlg)
        txt = QTextEdit()
        txt.setReadOnly(True)
        if versions:
            lines = []
            for v in versions[:100]:
                vid = v.get("id", "")
                mid = v.get("memory_item_id", "")
                ver = v.get("version", 0)
                content = (v.get("content") or "")[:200]
                imp = v.get("importance", 0)
                src = v.get("source_quality", "")
                ts = v.get("created_at", "")
                lines.append(f"v{ver} | item:{mid} | imp:{imp:.2f} | src:{src} | {ts}")
                lines.append(f"  {content}")
                lines.append("")
            txt.setPlainText("\n".join(lines) if lines else "No memory versions found.")
        else:
            txt.setPlainText("No memory versions recorded yet.\n\nVersions are created when memory items are updated.")
        lay.addWidget(txt)

        btn_row = QHBoxLayout()
        close_btn = QPushButton("CLOSE")
        close_btn.clicked.connect(dlg.accept)
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        lay.addLayout(btn_row)
        dlg.exec()
        dlg.exec()

    def _update_file_chips(self):
        """Rebuild the file chips UI from _pending_files list.
        
        For images: shows a 32x32 thumbnail preview (ChatGPT style).
        For other files: shows file type icon.
        """
        from PyQt6.QtGui import QPixmap
        # Clear existing chips
        while self._file_chips_layout.count():
            item = self._file_chips_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        if not self._pending_files:
            self._file_chips_container.setVisible(False)
            return
        
        self._file_chips_container.setVisible(True)
        
        for i, file_info in enumerate(self._pending_files):
            chip = QWidget()
            chip.setFixedHeight(52)
            chip.setStyleSheet(f"""
                QWidget {{
                    background: #010f18;
                    border: 1px solid #0d3347;
                    border-radius: 4px;
                    padding: 2px 4px;
                }}
            """)
            chip_layout = QHBoxLayout(chip)
            chip_layout.setContentsMargins(4, 2, 4, 2)
            chip_layout.setSpacing(4)
            
            cat = file_info.get("type", "unknown")
            
            # Thumbnail for images, icon for others
            if cat == "image":
                fpath = file_info.get("path", "")
                if fpath and Path(fpath).exists():
                    try:
                        pixmap = QPixmap(fpath)
                        if not pixmap.isNull():
                            thumb = pixmap.scaled(32, 32, Qt.AspectRatioMode.KeepAspectRatio,
                                                   Qt.TransformationMode.SmoothTransformation)
                            thumb_lbl = QLabel()
                            thumb_lbl.setPixmap(thumb)
                            thumb_lbl.setFixedSize(32, 32)
                            thumb_lbl.setStyleSheet("border: 1px solid #0d3347; border-radius: 3px; background: #000;")
                            chip_layout.addWidget(thumb_lbl)
                        else:
                            raise ValueError("null pixmap")
                    except Exception:
                        icon_lbl = QLabel("🖼")
                        icon_lbl.setFont(font_ui(10))
                        icon_lbl.setStyleSheet("background: transparent; border: none;")
                        chip_layout.addWidget(icon_lbl)
                else:
                    icon_lbl = QLabel("🖼")
                    icon_lbl.setFont(font_ui(10))
                    icon_lbl.setStyleSheet("background: transparent; border: none;")
                    chip_layout.addWidget(icon_lbl)
            else:
                _icons = {"pdf": "📄", "code": "💻", "text": "—"}
                icon_lbl = QLabel(_icons.get(cat, "📎"))
                icon_lbl.setFont(font_ui(10))
                icon_lbl.setStyleSheet("background: transparent; border: none;")
                chip_layout.addWidget(icon_lbl)
            
            # File name and processing state.  The full name stays available on hover.
            name = file_info.get("name", "file")
            full_name = name
            if len(name) > 20:
                name = name[:17] + "..."
            name_lbl = QLabel(name)
            name_lbl.setFont(font_ui(8.5))
            name_lbl.setStyleSheet("color: #8ffcff; background: transparent; border: none;")
            name_lbl.setToolTip(full_name)
            chip_layout.addWidget(name_lbl)

            detail = file_info.get("detail", "Preparing for analysis…")
            status = file_info.get("status", "preparing")
            status_lbl = QLabel(detail)
            status_lbl.setFont(font_ui(8.5))
            status_color = {"ready": C.GREEN, "preparing": C.PRI, "warning": "#ffcc66", "error": C.RED}.get(status, C.TEXT_DIM)
            status_lbl.setStyleSheet(f"color: {status_color}; background: transparent; border: none;")
            status_lbl.setToolTip(detail)
            chip_layout.addWidget(status_lbl, stretch=1)
            
            # Remove button
            remove_btn = QPushButton("✕")
            remove_btn.setFixedSize(16, 16)
            remove_btn.setFont(font_ui(8.5))
            remove_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            remove_btn.setStyleSheet(f"""
                QPushButton {{ background: transparent; color: #3a8a9a; border: none; }}
                QPushButton:hover {{ color: #ff3355; }}
            """)
            idx = i
            remove_btn.clicked.connect(lambda checked, idx=idx: self._remove_pending_file(idx))
            chip_layout.addWidget(remove_btn)
            
            self._file_chips_layout.insertWidget(self._file_chips_layout.count(), chip)

        self._file_chips_layout.addStretch()

    def _remove_pending_file(self, index: int):
        """Remove a file from pending files list."""
        if 0 <= index < len(self._pending_files):
            removed = self._pending_files.pop(index)
            self._log.append_log(f"REMOVED: {removed['name']}")
            self._update_file_chips()

    def _clear_log(self):
        self._log.clear()

    def _populate_log(self, turns: list[dict]):
        """Restore a session into the chat (F6: newest window + lazy older)."""
        entries: list[str] = []
        timestamps: list[str] = []
        for t in turns:
            role = t.get("role", "")
            content = (t.get("content") or "").strip()
            tr = (t.get("tool_result") or "").strip()
            ts = t.get("timestamp", "") or ""
            if role == "user" and content:
                entries.append(f"You: {content}")
                timestamps.append(ts)
            elif role == "assistant" and content:
                entries.append(f"Orthos: {content}")
                timestamps.append(ts)
            elif role == "tool" and (tr or content):
                # Empty silent tool turns render as blank rows — skip them.
                entries.append(f"SYS: ▶ Tool result — {(tr or content)[:100]}")
                timestamps.append(ts)
        log = self._log
        if hasattr(log, "set_history_window"):
            # F6: pass only the newest page; older turns load on demand.
            from core.native_chat import _MAX_RENDERED_MESSAGES as _CAP
            shown = entries[-_CAP:]
            shown_ts = timestamps[-_CAP:]

            def _fetch_older(shown_count: int, want: int) -> tuple[list[str], list[str]]:
                """Load older raw entries (+ their timestamps) from the DB."""
                try:
                    sid = getattr(self, "_cur_session_id", None)
                    if not sid:
                        return [], []
                    from memory.conversation_db import get_turns_by_session
                    all_turns = get_turns_by_session(sid)
                    all_entries: list[str] = []
                    all_ts: list[str] = []
                    for t in all_turns:
                        role = t.get("role", "")
                        content = (t.get("content") or "").strip()
                        tr = (t.get("tool_result") or "").strip()
                        ts = t.get("timestamp", "") or ""
                        if role == "user" and content:
                            all_entries.append(f"You: {content}")
                            all_ts.append(ts)
                        elif role == "assistant" and content:
                            all_entries.append(f"Orthos: {content}")
                            all_ts.append(ts)
                        elif role == "tool" and (tr or content):
                            all_entries.append(
                                f"SYS: ▶ Tool result — {(tr or content)[:100]}")
                            all_ts.append(ts)
                    # Older page: everything before the currently shown window.
                    boundary = max(0, len(all_entries) - shown_count - want)
                    return (all_entries[boundary:len(all_entries) - shown_count],
                            all_ts[boundary:len(all_ts) - shown_count])
                except Exception:
                    return [], []

            log.set_history_window(shown, shown_ts, len(entries),
                                   load_older_cb=_fetch_older)
        elif hasattr(log, "set_logs_with_timestamps"):
            log.set_logs_with_timestamps(entries, timestamps)
        else:
            log.set_logs(entries)

    def _on_file_selected(self, path: str):
        """Attach file to pending list (ChatGPT style — send with next message)."""
        p = Path(path)
        
        # File size check (25MB limit)
        size_bytes = p.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        if size_mb > self._MAX_FILE_SIZE_MB:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "File Too Large",
                f"<b>{p.name}</b> is <b>{size_mb:.1f}MB</b><br><br>"
                f"Maximum allowed size is <b>{self._MAX_FILE_SIZE_MB}MB</b><br><br>"
                f"Please compress the file and try again."
            )
            return
        
        cat = _file_category(p)
        size = _fmt_size(size_bytes)
        
        # Windows paths are case-insensitive: C:\A.pdf == c:\a.pdf (m10).
        _norm = os.path.normcase(os.path.abspath(path))
        if any(os.path.normcase(os.path.abspath(e.get("path", ""))) == _norm
               for e in self._pending_files):
            self._show_paste_toast(f"{p.name} is already attached")
            return

        # Add to pending files.  A background preflight then reads lightweight
        # document context before sending, without changing image handling.
        self._pending_files.append({
            "path": path,
            "name": p.name,
            "type": cat,
            "size": size,
            "status": "preparing",
            "detail": "Preparing for analysis…",
            "preview": "",
        })
        
        # Add to recent files
        if path not in self._recent_files:
            self._recent_files.insert(0, path)
            if len(self._recent_files) > self._MAX_RECENT_FILES:
                self._recent_files = self._recent_files[:self._MAX_RECENT_FILES]
        
        # Update chips UI
        self._update_file_chips()
        self._log.append_log(f"ATTACHED: {p.name} ({size}) [{cat.upper()}]")
        
        # Show toast
        self._show_paste_toast(f"{p.name} attached ({size})")
        threading.Thread(target=self._prepare_attachment, args=(path,), daemon=True).start()

    def _prepare_attachment(self, path: str):
        """Read safe local metadata/content off the UI thread."""
        self._attachment_ready_sig.emit(path, _attachment_context(path))

    def _on_attachment_ready(self, path: str, info: dict):
        for attachment in self._pending_files:
            if attachment.get("path") == path:
                attachment.update(info)
                self._update_file_chips()
                if info.get("status") == "ready":
                    self._log.append_log(f"READY: {attachment['name']} — {info.get('detail')}")
                return

    def _toggle_mute(self):
        self._muted = not self._muted
        self.hud.muted = self._muted
        self._style_mute_btn()
        if self._muted:
            # Mute = the USER's microphone is off. It must NOT clobber a
            # SPEAKING turn in progress: TTS keeps talking (text input
            # works while muted), and the face must keep animating. Only
            # flip to MUTED when nothing is being spoken.
            if not (self.hud.speaking or self.hud._audio_active_until > time.time()):
                self._apply_state("MUTED")
            self._log.append_log("SYS: Microphone muted.")
        else:
            if not (self.hud.speaking or self.hud._audio_active_until > time.time()):
                self._apply_state("LISTENING")
            self._log.append_log("SYS: Microphone active.")

    def _style_mute_btn(self):
        if self._muted:
            self._mute_btn.setText("🔇  MICROPHONE MUTED")
            self._mute_btn.setStyleSheet(f"""
                QPushButton {{
                    background: #140006; color: {C.MUTED_C};
                    border: 1px solid {C.MUTED_C}; border-radius: 3px;
                }}
            """)
        else:
            self._mute_btn.setText("🎙  MICROPHONE ACTIVE")
            self._mute_btn.setStyleSheet(f"""
                QPushButton {{
                    background: #00140a; color: {C.GREEN};
                    border: 1px solid {C.GREEN}; border-radius: 3px;
                }}
                QPushButton:hover {{ background: #001f10; }}
            """)

    def _load_screen_vision_setting(self) -> bool:
        try:
            if not API_FILE.exists():
                return True
            d = json.loads(API_FILE.read_text(encoding="utf-8"))
            return bool(d.get("screen_vision_auto", True))
        except Exception:
            return True

    def _toggle_screen_vision(self):
        self._screen_vision = not self._screen_vision
        self._style_screen_vision_btn()
        try:
            cfg = json.loads(API_FILE.read_text(encoding="utf-8")) if API_FILE.exists() else {}
            cfg["screen_vision_auto"] = self._screen_vision
            os.makedirs(CONFIG_DIR, exist_ok=True)
            API_FILE.write_text(json.dumps(cfg, indent=4), encoding="utf-8")
        except Exception:
            pass
        self._log.append_log(
            "SYS: Screen vision " + ("ON — auto screenshots enabled."
                                     if self._screen_vision
                                     else "OFF — auto screenshots disabled.")
        )

    def _style_screen_vision_btn(self):
        if not self._screen_vision:
            self._screen_vision_btn.setText("◉  SCREEN VISION OFF")
            self._screen_vision_btn.setStyleSheet(f"""
                QPushButton {{
                    background: #140006; color: {C.MUTED_C};
                    border: 1px solid {C.MUTED_C}; border-radius: 3px;
                }}
            """)
        else:
            self._screen_vision_btn.setText("◉  SCREEN VISION ON")
            self._screen_vision_btn.setStyleSheet(f"""
                QPushButton {{
                    background: #00140a; color: {C.GREEN};
                    border: 1px solid {C.GREEN}; border-radius: 3px;
                }}
                QPushButton:hover {{ background: #001f10; }}
            """)

    def _on_query_typing(self, text: str) -> None:
        """E — debounce typing (500ms pause) then prefetch query blocks."""
        try:
            from PyQt6.QtCore import QTimer
            if getattr(self, "_qpref_timer", None) is None:
                t = QTimer(self)
                t.setSingleShot(True)
                t.setInterval(500)
                t.timeout.connect(self._run_query_prefetch)
                self._qpref_timer = t
            self._qpref_timer.start()   # restart on each keystroke
        except Exception:
            pass

    def _run_query_prefetch(self) -> None:
        try:
            txt = (self._input.text() or "").strip()
            if len(txt) < 4 or not self.on_query_typing:
                return
            import threading
            threading.Thread(target=self.on_query_typing, args=(txt,), daemon=True).start()
        except Exception:
            pass

    def _send(self):
        try:
            txt = self._input.text().strip()
            if not txt and not self._pending_files:
                return
            preparing = [f["name"] for f in self._pending_files if f.get("status") == "preparing"]
            if preparing:
                self._show_paste_toast("Preparing attachment(s)… please send again in a moment")
                return
            self._input.clear()
            
            # Build message with attached files info
            files_info = ""
            if self._pending_files:
                files_list = []
                for f in self._pending_files:
                    files_list.append(f"{f['name']} ({f['type']}, {f['size']})")
                files_info = chr(10) + "[ATTACHED FILES: " + "; ".join(files_list) + "]"
            
            # File-only sends must keep the attachment block too (m9).
            full_msg = (txt + files_info).strip() if txt else ("Please analyze the attached files." + files_info)
            
            # Collect file paths
            file_paths = [f["path"] for f in self._pending_files]
            image_paths = [f["path"] for f in self._pending_files if f["type"] == "image"]
            
            # Send via callback with files
            if self.on_text_command_with_files and file_paths:
                extra = {
                    "files": file_paths,
                    "image_files": image_paths,
                    "document_images": [
                        image
                        for f in self._pending_files if f["type"] == "pdf"
                        for image in f.get("page_images", [])
                    ],
                    "attachment_context": [
                        {
                            "name": f["name"], "type": f["type"], "size": f["size"],
                            "status": f.get("status", "ready"), "detail": f.get("detail", ""),
                            "preview": f.get("preview", ""),
                        }
                        for f in self._pending_files if f["type"] != "image"
                    ],
                }
                try:
                    threading.Thread(
                        target=self.on_text_command_with_files, 
                        args=(full_msg, extra),
                        daemon=True
                    ).start()
                except Exception as _thread_err:
                    print(f"[Orthos] WARN: file thread failed ({_thread_err}), falling back to text-only")
                    threading.Thread(target=self.on_text_command, args=(full_msg,), daemon=True).start()
            elif self.on_text_command:
                threading.Thread(target=self.on_text_command, args=(full_msg,), daemon=True).start()
            
            # Clear pending files + chips
            self._pending_files.clear()
            self._update_file_chips()
        except Exception as e:
            print(f"[Orthos] ERROR in _send: {e}")
            import traceback
            traceback.print_exc()

    def _apply_state(self, state: str):
        # A MUTED state must never interrupt an active speaking turn:
        # mute silences the USER's mic, not the assistant's mouth. If
        # audio is actually playing (speaking flag or the ground-truth
        # window), preserve the SPEAKING state and just remember that a
        # mute is pending — it takes over the moment playback ends.
        if state == "MUTED":
            self._pending_muted_state = True
            if self.hud.speaking or self.hud._audio_active_until > time.time():
                return
        else:
            self._pending_muted_state = False
        self.hud.state    = state
        self.hud.speaking = (state == "SPEAKING")
        # Do NOT drop speaking while audio is still sounding: a PROCESSING
        # state arriving mid-reply (tool call) used to freeze the mouth.
        # The HUD's audio-ground-truth window keeps it live instead.
        # Thinking/processing states drive the chat typing indicator.
        if hasattr(self._log, "set_typing"):
            self._log.set_typing(state in ("THINKING", "PROCESSING"))
        # The floating badge mirrors indicator state for history readers.
        self._update_fab_visibility()

    # ── Scroll FAB + typing indicator plumbing ───────────────────────

    def _position_fab(self):
        """Float the FAB over the chat's bottom-right corner."""
        log = self._log
        fab = getattr(self, "_scroll_fab", None)
        if fab is None:
            return
        m, b = 10, 8
        fab.move(max(m, log.width() - fab.width() - m),
                 max(m, log.height() - fab.height() - m))
        badge = getattr(self, "_fab_badge", None)
        if badge is not None and not badge.isHidden():
            badge.move(fab.x() + fab.width() - 10, fab.y() - 4)
        # Thinking badge: top-center pill (visible only for history readers).
        tbadge = getattr(self, "_thinking_badge", None)
        if tbadge is not None and not tbadge.isHidden():
            tbadge.adjustSize()
            tbadge.move(max(6, (log.width() - tbadge.width()) // 2), 6)

    def _on_scroll_fab_clicked(self):
        # Layout-safe 3-pass snap: a raw setValue(maximum()) reads a stale
        # maximum while the typing ticker/streaming mutates the document.
        if hasattr(self._log, "scroll_to_bottom_hard"):
            self._log.scroll_to_bottom_hard()
        else:
            self._log._snap_to_bottom()
        self._log._unread = 0
        self._scroll_fab.hide()
        self._fab_badge.hide()

    def _on_log_unread(self, count: int):
        self._update_fab_visibility()

    def _on_log_scrolled(self, _value: int):
        self._update_fab_visibility()

    def _update_fab_visibility(self):
        """Show the FAB when the reader is away from the bottom edge."""
        log = getattr(self, "_log", None)
        fab = getattr(self, "_scroll_fab", None)
        if fab is None or log is None:
            return
        bar = log.verticalScrollBar()
        away = bar.maximum() - bar.value() > 60
        fab.setVisible(away)
        unread = getattr(log, "_unread", 0)
        badge = getattr(self, "_fab_badge", None)
        if badge is not None:
            badge.setText(str(unread) if unread > 0 else "")
            badge.setVisible(away and unread > 0)
        # Thinking badge: history reader + live indicator running.
        tbadge = getattr(self, "_thinking_badge", None)
        if tbadge is not None:
            busy = bool(getattr(log, "_typing", False)) or bool(
                getattr(log, "_streaming_active", False))
            tbadge.setVisible(away and busy)
            if tbadge.isVisible():
                self._position_fab()
        if away:
            self._position_fab()

    def _check_config(self) -> bool:
        if not API_FILE.exists(): return False
        try:
            d = json.loads(API_FILE.read_text(encoding="utf-8"))
            return bool(d.get("stt_engine")) and bool(d.get("tts_engine"))
        except Exception:
            return False

    def _show_setup(self):
        current: dict = {}
        try:
            current = json.loads(API_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
        ov = SetupOverlay(self.centralWidget(), initial=current)
        cw = self.centralWidget()
        ow = max(820, int(cw.width() * 0.9))
        oh = max(600, int(cw.height() * 0.9))
        ov.setGeometry(
            (cw.width()  - ow) // 2,
            (cw.height() - oh) // 2,
            ow, oh,
        )
        ov.done.connect(self._on_setup_done)
        ov.show()
        self._overlay = ov

    def _on_setup_done(self, config_json: str):
        try:
            cfg = json.loads(config_json)
        except Exception:
            cfg = {}
        os.makedirs(CONFIG_DIR, exist_ok=True)
        API_FILE.write_text(
            json.dumps(cfg, indent=4),
            encoding="utf-8",
        )
        self._ready = True
        if self._overlay:
            self._overlay.hide()
            self._overlay = None
        self._apply_state("LISTENING")
        llm = cfg.get("llm_model", "")
        stt = cfg.get("stt_engine", "")
        tts = cfg.get("tts_engine", "")
        self._log.append_log(
            f"SYS: Initialised. LLM={llm} | STT={stt} | TTS={tts}"
        )

    def _show_config(self):
        if self._overlay and self._overlay.isVisible():
            return
        current: dict = {}
        try:
            current = json.loads(API_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
        ov = SetupOverlay(self.centralWidget(), initial=current, mode="config")
        cw = self.centralWidget()
        ow = max(820, int(cw.width() * 0.9))
        oh = max(600, int(cw.height() * 0.9))
        ov.setGeometry(
            (cw.width()  - ow) // 2,
            (cw.height() - oh) // 2,
            ow, oh,
        )
        ov.done.connect(self._on_config_done)
        ov.show()
        self._overlay = ov

    def _show_mcp_manager(self):
        if not _HAS_MCP_UI:
            return
        try:
            open_mcp_manager(self)
        except Exception as e:
            print(f"[MCP Manager] Error: {e}")
            import traceback
            traceback.print_exc()

    def _on_config_done(self, config_json: str):
        try:
            cfg = json.loads(config_json)
        except Exception:
            cfg = {}
        os.makedirs(CONFIG_DIR, exist_ok=True)
        API_FILE.write_text(json.dumps(cfg, indent=4), encoding="utf-8")
        if self._overlay:
            self._overlay.hide()
            self._overlay = None
        llm = cfg.get("llm_model", "")
        stt = cfg.get("stt_engine", "")
        tts = cfg.get("tts_engine", "")
        self._log.append_log(f"SYS: Config updated. LLM={llm} | STT={stt} | TTS={tts}")
        if self._on_reconfigure_cb:
            self._on_reconfigure_cb(cfg)


class _RootShim:
    def __init__(self, app: QApplication):
        self._app = app
    def mainloop(self):
        self._app.exec()
    def protocol(self, *_):
        pass


class OrthosUI:
    def __init__(self, face_path: str, size=None):
        self._app = QApplication.instance() or QApplication(sys.argv)
        self._app.setStyle("Fusion")
        self._win = MainWindow(face_path)
        self._win.show()
        self.root = _RootShim(self._app)

    @property
    def muted(self) -> bool:
        return self._win._muted

    @muted.setter
    def muted(self, v: bool):
        if v != self._win._muted:
            self._win._toggle_mute()

    @property
    def screen_vision(self) -> bool:
        return self._win._screen_vision

    @property
    def current_file(self) -> str | None:
        return getattr(self._win, '_current_file', None)

    @property
    def on_text_command(self):
        return self._win.on_text_command

    @on_text_command.setter
    def on_text_command(self, cb):
        self._win.on_text_command = cb

    @property
    def on_text_command_with_files(self):
        return self._win.on_text_command_with_files

    @on_text_command_with_files.setter
    def on_text_command_with_files(self, cb):
        self._win.on_text_command_with_files = cb

    @property
    def on_reconfigure(self):
        return self._win._on_reconfigure_cb

    @on_reconfigure.setter
    def on_reconfigure(self, cb):
        self._win._on_reconfigure_cb = cb

    def set_state(self, state: str):
        self._win._state_sig.emit(state)

    def set_audio_level(self, level: float):
        self._win._audio_level_sig.emit(level)

    def notify_audio_activity(self, grace: float = 0.9):
        """Thread-safe: playback batch is sounding now (mouth ground truth)."""
        try:
            self._win.hud.notify_audio_activity(grace)
        except Exception:
            pass

    def push_visemes(self, frames, hop: float, at: float):
        self._win._visemes_sig.emit(frames, hop, at)

    def clear_visemes(self):
        """Thread-safe-ish: drop pending visemes (stale schedule purge)."""
        try:
            self._win.hud.clear_visemes()
        except Exception:
            pass

    def write_log(self, text: str):
        self._win._log_sig.emit(text)

    def show_startup_panel(self) -> None:
        self._win._startup_sig.emit("show", "")

    def mark_startup_ready(self, key: str, error: bool = False) -> None:
        self._win._startup_sig.emit("error" if error else "ready", key)

    def set_startup_status(self, text: str) -> None:
        self._win._startup_sig.emit("status", text)

    def hide_startup_panel(self) -> None:
        self._win._startup_sig.emit("hide", "")

    def wait_for_api_key(self):
        while not self._win._ready:
            QApplication.processEvents()
            time.sleep(0.05)

    def start_speaking(self):
        self.set_state("SPEAKING")

    def stop_speaking(self):
        if not self.muted:
            self.set_state("LISTENING")
