"""Clean-room animated holographic face for the Orthos HUD."""

from __future__ import annotations

import math
import random
import time

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)

# ── Realism constants (ported from Mark-LIV) ──────────────────────────────
# Loudness normalization: int16 RMS below _LEVEL_FLOOR reads as room silence
# and above _LEVEL_FULL as a fully-open mouth. Ordinary speech then lands
# mid-range, so the mouth moves through its full range like Mark-LIV's face
# instead of flickering in the bottom 20 % (raw rms*1.35 gave 0.07-0.27).
_LEVEL_FLOOR = 60.0
_LEVEL_FULL = 2600.0
# 5-band formant analysis: F1 low/high (jaw), F2 back/front (lip rounding),
# hiss (fricatives). Windows sized for a 24 kHz stream.
_VIS_F1_LO = (150.0, 450.0)
_VIS_F1_HI = (450.0, 1100.0)
_VIS_F2_BK = (600.0, 1300.0)
_VIS_F2_FR = (1700.0, 3200.0)
_VIS_HISS = (3800.0, 8000.0)
# Playback-timing cursor: how early the mouth runs ahead of the sound (a lead
# reads as sync; a lag reads as dubbing) and how far the cursor may drift
# before it is re-anchored.
_FIRST_SOUND = 0.043
_CURSOR_SLACK = 0.15

_TAU_OPEN = 0.022
_TAU_SHUT = 0.012
_TAU_REST = 0.055
_TAU_SHAPE = 0.018
_TAU_BROW = 0.05
_TAU_GAZE = 0.12
_TAU_AMP = 0.045
_AMP_DECAY = 0.86
_VIS_WIN = 1024
_VIS_HOP = 480


def _rate(dt: float, tau: float) -> float:
    if tau <= 0.0:
        return 1.0
    return 1.0 - math.exp(-dt / tau)


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else (high if value > high else value)


def _mix(bg: QColor, col: QColor, alpha: float) -> QColor:
    factor = _clamp(alpha, 0.0, 1.0)
    return QColor(
        int(bg.red() + (col.red() - bg.red()) * factor),
        int(bg.green() + (col.green() - bg.green()) * factor),
        int(bg.blue() + (col.blue() - bg.blue()) * factor),
    )


def pcm_level(samples: np.ndarray) -> float:
    """Loudness 0..1, normalized to speech range (Mark-LIV mapping).

    The old rms*1.35 mapping left ordinary speech at 0.07-0.27 — technically
    moving, visually near-dead. With the floor/full mapping normal speech
    lands mid-range and the mouth reads clearly at HUD size. Accepts both
    float and int16 input.
    """
    if samples is None or len(samples) == 0:
        return 0.0
    x = np.asarray(samples, dtype=np.float32)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 2.0:                      # int16 input
        x = x / 32768.0
    if x.max() < 1e-6 and x.min() > -1e-6:
        return 0.0
    rms16 = math.sqrt(float(np.mean(x * x))) * 32768.0
    if rms16 <= _LEVEL_FLOOR:
        return 0.0
    return _clamp((rms16 - _LEVEL_FLOOR) / (_LEVEL_FULL - _LEVEL_FLOOR), 0.0, 1.0)


def pcm_visemes(samples: np.ndarray, sample_rate: int = 24000):
    """Return (level, openness, width) frames from Orthos-owned FFT analysis."""
    values = np.asarray(samples, dtype=np.float64)
    if values.size == 0:
        return []
    if values.max() > 2.0 or values.min() < -2.0:
        values = values / 32768.0
    if values.size < _VIS_WIN:
        values = np.pad(values, (0, _VIS_WIN - values.size))
    frames = []
    window = np.hanning(_VIS_WIN)
    freqs = np.fft.rfftfreq(_VIS_WIN, 1.0 / max(1, sample_rate))
    f1l = (freqs >= _VIS_F1_LO[0]) & (freqs < _VIS_F1_LO[1])
    f1h = (freqs >= _VIS_F1_HI[0]) & (freqs < _VIS_F1_HI[1])
    f2b = (freqs >= _VIS_F2_BK[0]) & (freqs < _VIS_F2_BK[1])
    f2f = (freqs >= _VIS_F2_FR[0]) & (freqs < _VIS_F2_FR[1])
    hiss_b = (freqs >= _VIS_HISS[0]) & (freqs < _VIS_HISS[1])
    for start in range(0, max(1, values.size - _VIS_WIN + 1), _VIS_HOP):
        # Level over exactly this hop (no look-ahead), spectrum over the
        # longer zero-padded window — same split as Mark-LIV.
        level = pcm_level(values[start:start + _VIS_HOP])
        segment = values[start:start + _VIS_WIN] * window
        spectrum = np.abs(np.fft.rfft(segment))
        e_f1l = float(spectrum[f1l].sum())
        e_f1h = float(spectrum[f1h].sum())
        e_f2b = float(spectrum[f2b].sum())
        e_f2f = float(spectrum[f2f].sum())
        hiss = float(spectrum[hiss_b].sum())
        if level <= 0.0:
            frames.append((0.0, 0.0, 0.0))
            continue
        # F1 climbs as the jaw drops: /a/ open, /i/ /u/ closed.
        openness = e_f1h / (e_f1l + e_f1h + 1e-6)
        # F2 high for spread (/i/ /e/), low for rounded (/u/ /o/).
        width = (e_f2f - e_f2b) / (e_f2f + e_f2b + 1e-6)
        # Physics: a wide-open jaw cannot purse; /a/'s low F2 would
        # otherwise read as "rounded".
        width *= (1.0 - openness) ** 0.8
        # Fricatives (s, f, sh) are formed with a nearly closed mouth.
        h = hiss / (e_f1l + e_f1h + e_f2b + e_f2f + hiss + 1e-6)
        openness *= 1.0 - 0.65 * min(1.0, h * 2.5)
        frames.append((
            level,
            _clamp(openness, 0.0, 1.0),
            _clamp(width, -1.0, 1.0),
        ))
    return frames


class HoloFace:
    """Procedural holographic head driven by state and TTS audio."""

    SPAN = 2.0

    def __init__(self) -> None:
        self._t = 0.0
        self._phase = 0.0
        self._tilt = 0.0
        self._mouth = 0.0
        self._mouth_w = 0.0
        self._glow = 0.0
        self._amp_disp = 0.0
        self._live_amp = 0.0
        self._brow = 0.0
        self._blink = 0.0
        self._blink_at = 2.7
        self._lids = 1.0
        self._gaze = [0.0, 0.0]
        self._gaze_target = [0.0, 0.0]
        self._bias = [0.0, 0.0]
        self._visemes = None
        self._vis_i = None

    def glance(self, dx: float, dy: float, hold: float = 1.0) -> None:
        self._gaze_target = [_clamp(dx, -1.0, 1.0), _clamp(dy, -1.0, 1.0)]

    def push_visemes(self, frames, hop: float, at: float) -> None:
        try:
            if not frames:
                return
            hop = max(1e-3, float(hop))
            at = float(at)
            current = self._visemes
            if current is not None:
                old, start, old_hop = current
                if abs(old_hop - hop) < 1e-6:
                    index = int(round((at - start) / hop))
                    if 0 <= index <= len(old) + 1:
                        merged = old[:index] + list(frames)
                        elapsed = int((time.time() - start) / hop) - 2
                        if elapsed > 60:
                            merged = merged[elapsed:]
                            start += elapsed * hop
                            if self._vis_i is not None:
                                self._vis_i = max(0, self._vis_i - elapsed)
                        self._visemes = (merged, start, hop)
                        return
            self._visemes = (list(frames), at, hop)
            self._vis_i = None
        except Exception:
            pass

    def set_audio_level(self, level: float) -> None:
        try:
            value = _clamp(float(level), 0.0, 1.0)
        except (TypeError, ValueError):
            return
        if value > self._live_amp:
            self._live_amp = value

    def step(self, dt: float, amp: float, *, speaking: bool, muted: bool,
             state: str, v_open=None, v_wide=0.0, v_level=None,
             v_seq=None, v_hop=0.02) -> None:
        now = time.time()
        schedule = self._visemes
        v_o = v_w = v_l = None
        if schedule is not None:
            frames, start, hop = schedule
            index = int((now - start) / hop)
            if 0 <= index < len(frames):
                first = self._vis_i if self._vis_i is not None else index
                sequence = frames[max(0, first):index + 1]
                self._vis_i = max(first, index + 1)
                v_l, v_o, v_w = frames[index]
                if sequence:
                    peak = max(frame[0] for frame in sequence)
                    if peak > self._live_amp:
                        self._live_amp = peak
            elif index >= len(frames):
                self._visemes = None
                self._vis_i = None

        self._live_amp *= _AMP_DECAY
        self._amp_disp += (self._live_amp - self._amp_disp) * _rate(dt, _TAU_AMP)
        level = self._amp_disp if v_l is None else max(self._amp_disp, v_l)
        openness = level if v_o is None else max(level, v_o)
        width = 0.0 if v_w is None else v_w
        # `muted` = user's mic off, NOT the assistant's mouth (see avatar.py):
        # text input while muted must still animate the face.
        if speaking:
            open_tau = _TAU_OPEN if openness > self._mouth else _TAU_SHUT
            self._mouth += (openness - self._mouth) * _rate(dt, open_tau)
            self._mouth_w += (width - self._mouth_w) * _rate(dt, _TAU_SHAPE)
        else:
            self._mouth += (0.0 - self._mouth) * _rate(dt, _TAU_REST)
            self._mouth_w += (0.0 - self._mouth_w) * _rate(dt, _TAU_SHAPE)

        self._glow += (level - self._glow) * _rate(dt, _TAU_AMP)
        brow_target = 0.75 * level + (0.25 if speaking else 0.0)
        self._brow += (brow_target - self._brow) * _rate(dt, _TAU_BROW)
        self._phase += _clamp(dt, 0.0, 0.10)
        self._tilt = math.sin(self._phase * 0.8) * (1.2 + 4.0 * level)
        if muted:
            bias = [0.0, -0.18]
        elif state in ("THINKING", "PROCESSING"):
            bias = [-0.24, 0.04]
        elif state == "LISTENING":
            bias = [0.0, 0.02]
        else:
            bias = [0.0, 0.0]
        self._bias[0] += (bias[0] - self._bias[0]) * _rate(dt, 0.25)
        self._bias[1] += (bias[1] - self._bias[1]) * _rate(dt, 0.25)
        self._gaze_target = list(self._bias)
        self._gaze[0] += (self._gaze_target[0] - self._gaze[0]) * _rate(dt, _TAU_GAZE)
        self._gaze[1] += (self._gaze_target[1] - self._gaze[1]) * _rate(dt, _TAU_GAZE)
        self._blink_at -= dt
        if self._blink_at <= 0.0:
            self._blink = 1.0
            self._blink_at = random.uniform(0.08, 0.16)
        elif self._blink > 0.0:
            self._blink -= dt * 7.0
            if self._blink < 0.0:
                self._blink = 0.0
        lid_target = 0.65 if state in ("THINKING", "PROCESSING") else 1.0
        self._lids += (lid_target - self._lids) * _rate(dt, 0.12)
        self._t = now

    def paint(self, painter: QPainter, cx: float, cy: float, radius: float,
              main: QColor, accent: QColor, bg: QColor) -> None:
        if radius <= 0.0:
            return
        painter.save()
        painter.translate(cx, cy)
        painter.scale(radius, radius)
        painter.rotate(self._tilt)
        self._paint_body(painter, main, accent, bg)
        self._paint_lattice(painter, main, bg)
        self._paint_features(painter, main, accent, bg)
        painter.restore()

    def _paint_body(self, painter, main, accent, bg):
        lift = 1.0 + 0.55 * self._glow + (0.18 if self._mouth > 0.03 else 0.0)
        for gradient_radius, start_alpha in ((1.05, 0.30), (0.55, 0.22)):
            gradient = QRadialGradient(0.0, -0.08, gradient_radius)
            gradient.setColorAt(0.00, _mix(bg, main, min(0.95, start_alpha * lift)))
            gradient.setColorAt(0.48, _mix(bg, main, min(0.95, start_alpha * lift * 0.45)))
            gradient.setColorAt(1.00, _mix(bg, main, 0.0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(gradient))
            painter.drawEllipse(QRectF(-gradient_radius, -gradient_radius,
                                       gradient_radius * 2.0, gradient_radius * 2.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        neck = QPainterPath()
        neck.moveTo(-0.28, 0.72)
        neck.lineTo(0.28, 0.72)
        neck.lineTo(0.42, 1.28)
        neck.lineTo(-0.42, 1.28)
        neck.closeSubpath()
        neck_gradient = QRadialGradient(0.0, 0.85, 0.65)
        neck_gradient.setColorAt(0.0, _mix(bg, main, 0.24 * lift))
        neck_gradient.setColorAt(1.0, _mix(bg, main, 0.0))
        painter.setBrush(QBrush(neck_gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(neck)

        outline = self._head_path()
        fill_gradient = QRadialGradient(0.0, -0.18, 1.15)
        fill_gradient.setColorAt(0.00, _mix(bg, accent, 0.42 + 0.28 * self._glow))
        fill_gradient.setColorAt(0.48, _mix(bg, main, 0.48 + 0.22 * self._glow))
        fill_gradient.setColorAt(1.00, _mix(bg, main, 0.18))
        painter.setBrush(QBrush(fill_gradient))
        painter.setPen(QPen(_mix(bg, accent, 0.62 + 0.28 * self._glow), 0.018))
        painter.drawPath(outline)

    def _head_path(self):
        path = QPainterPath()
        path.moveTo(0.0, -0.98)
        path.cubicTo(0.48, -0.98, 0.72, -0.65, 0.72, -0.20)
        path.cubicTo(0.72, 0.22, 0.55, 0.55, 0.30, 0.82)
        path.cubicTo(0.16, 0.96, -0.16, 0.96, -0.30, 0.82)
        path.cubicTo(-0.55, 0.55, -0.72, 0.22, -0.72, -0.20)
        path.cubicTo(-0.72, -0.65, -0.48, -0.98, 0.0, -0.98)
        path.closeSubpath()
        return path

    def _paint_lattice(self, painter, main, bg):
        outline = self._head_path()
        painter.save()
        painter.setClipPath(outline)
        painter.setPen(QPen(_mix(bg, main, 0.34), 0.008))
        for index in range(-7, 8):
            y = index * 0.13 - 0.03
            half_width = 0.72 * math.sqrt(max(0.0, 1.0 - ((y + 0.03) / 0.92) ** 2))
            painter.drawLine(QPointF(-half_width, y), QPointF(half_width, y))
        for offset in (-0.55, -0.28, 0.0, 0.28, 0.55):
            curve = QPainterPath()
            curve.moveTo(offset * 0.62, -0.94)
            curve.cubicTo(offset, -0.55, offset, 0.45, offset * 0.48, 0.90)
            painter.drawPath(curve)
        painter.restore()

    def _paint_features(self, painter, main, accent, bg):
        eye_y = -0.24
        eye_x = 0.27
        eye_w = 0.115
        eye_h = 0.055 * max(0.08, self._lids) * (1.0 - self._blink * 0.92)
        for side in (-1.0, 1.0):
            x = side * eye_x + self._gaze[0] * 0.025
            y = eye_y + self._gaze[1] * 0.02
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(_mix(bg, main, 0.92)))
            painter.drawEllipse(QRectF(x - eye_w, y - eye_h,
                                       eye_w * 2.0, eye_h * 2.0))
            pupil = 0.035
            painter.setBrush(QBrush(_mix(bg, accent, 0.96)))
            painter.drawEllipse(QRectF(x + self._gaze[0] * 0.025 - pupil,
                                       y + self._gaze[1] * 0.025 - pupil,
                                       pupil * 2.0, pupil * 2.0))
        brow_lift = self._brow * 0.09
        painter.setPen(QPen(_mix(bg, accent, 0.72 + 0.22 * self._brow), 0.016))
        for side in (-1.0, 1.0):
            x = side * eye_x
            painter.drawLine(QPointF(x - 0.10, eye_y - 0.12 + brow_lift),
                             QPointF(x + 0.10, eye_y - 0.085 - brow_lift * 0.45))
        painter.setPen(QPen(_mix(bg, main, 0.42), 0.012))
        painter.drawLine(QPointF(0.0, -0.05), QPointF(0.0, 0.18))
        width = 0.30 + 0.12 * self._mouth_w
        height = 0.018 + 0.20 * self._mouth
        mouth = QPainterPath()
        mouth.moveTo(-width, 0.36)
        mouth.cubicTo(-width * 0.42, 0.36 - height,
                      width * 0.42, 0.36 - height, width, 0.36)
        mouth.cubicTo(width * 0.42, 0.36 + height,
                      -width * 0.42, 0.36 + height, -width, 0.36)
        mouth.closeSubpath()
        painter.setBrush(QBrush(_mix(bg, bg, 0.0)))
        painter.setPen(QPen(_mix(bg, accent, 0.72 + 0.25 * self._glow), 0.018))
        painter.drawPath(mouth)
        if height > 0.035:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(_mix(bg, bg, 0.0)))
            painter.drawEllipse(QRectF(-width * 0.62, 0.36 - height * 0.72,
                                       width * 1.24, height * 1.44))
        for side in (-1.0, 1.0):
            x = side * 0.50
            gradient = QRadialGradient(x, 0.10, 0.22)
            gradient.setColorAt(0.0, _mix(bg, accent, 0.10 + 0.16 * self._glow))
            gradient.setColorAt(1.0, _mix(bg, accent, 0.0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(gradient))
            painter.drawEllipse(QRectF(x - 0.22, -0.12, 0.44, 0.44))
        sweep_y = -0.90 + (self._phase % 1.0) * 1.80
        painter.setPen(QPen(_mix(bg, accent, 0.10 + 0.12 * self._glow), 0.010))
        painter.drawLine(QPointF(-0.68, sweep_y), QPointF(0.68, sweep_y))
