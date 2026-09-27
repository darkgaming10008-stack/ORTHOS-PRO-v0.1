"""Reusable UI components for MCP Manager.

Components:
- ToastNotification      Slide-in toast messages
- StatusPill             Animated status indicator
- ShimmerLabel           Loading shimmer placeholder
- AnimatedProgressBar    Smooth progress indicator
- ServerCard             Card widget for a single MCP server
- CategoryBadge          Category icon + label badge
- ToolChip               Compact tool name chip
- HealthIndicator        Mini health graph widget
- PermissionDialog       Permission prompt for sensitive actions
"""

from __future__ import annotations

import random
import time
from typing import Optional

from PyQt6.QtCore import (
    QEasingCurve, QMargins, QParallelAnimationGroup, QPoint, QPropertyAnimation,
    QRect, QSize, QTimer,
)
from PyQt6.QtGui import QColor, QPainter, QPaintEvent
from PyQt6.QtWidgets import (
    QCheckBox, QFrame, QGraphicsOpacityEffect, QHBoxLayout, QLabel,
    QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from .styles import (
    BG, BG2, BORDER, BORDER_B, CARD, CARD_HOVER, GREEN, GREEN_D,
    PULSE_INTERVAL, RADIUS_M, RADIUS_S, SPACING_M, SPACING_S, SPACING_XS,
    TEXT, TEXT_ACC, TEXT_DIM, TEXT_MED, YELLOW, _animate_property,
    _color, _font, _pulsing_color,
)


# ── Toast Notification ──────────────────────────────────────────────────

class ToastNotification(QWidget):
    """Modern toast notification that slides in from the right.
    
    Features:
    - Auto-dismiss after duration
    - Queue support
    - Success/Error/Warning/Info types
    - Smooth slide + fade animation
    """

    _instances: list["ToastNotification"] = []
    _MAX_VISIBLE = 3

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        title: str = "",
        message: str = "",
        toast_type: str = "info",
        duration: int = 4000,
        action_text: str = "",
        action_callback=None,
    ):
        super().__init__(parent)
        self._title = title
        self._message = message
        self._toast_type = toast_type
        self._duration = duration
        self._action_callback = action_callback

        self.setWindowFlags(
            self.windowFlags() |
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.Tool |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        type_colors = {
            "success": (GREEN, GREEN_D),
            "error":   (RED, RED_D),
            "warning": (YELLOW, YELLOW_D),
            "info":    (TEXT_ACC, BORDER_B),
        }
        self._accent, self._accent_dark = type_colors.get(toast_type, type_colors["info"])

        self._setup_ui()
        self._setup_animations()

        # Queue management
        ToastNotification._instances.append(self)
        self._adjust_positions()

    def _setup_ui(self):
        self.setFixedWidth(380)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(SPACING_M, SPACING_M, SPACING_M, SPACING_M)
        layout.setSpacing(SPACING_M)

        # Accent bar
        accent_bar = QFrame()
        accent_bar.setFixedWidth(4)
        accent_bar.setStyleSheet(f"background: {self._accent}; border-radius: 2px;")
        layout.addWidget(accent_bar)

        # Content
        content = QWidget()
        content_lay = QVBoxLayout(content)
        content_lay.setContentsMargins(0, 0, 0, 0)
        content_lay.setSpacing(SPACING_XS)

        if self._title:
            title_lbl = QLabel(self._title)
            title_lbl.setFont(_font(10, bold=True))
            title_lbl.setStyleSheet(f"color: {TEXT};")
            content_lay.addWidget(title_lbl)

        msg_lbl = QLabel(self._message)
        msg_lbl.setFont(_font(FONT_SIZE_S))
        msg_lbl.setStyleSheet(f"color: {TEXT_MED};")
        msg_lbl.setWordWrap(True)
        content_lay.addWidget(msg_lbl)

        layout.addWidget(content, stretch=1)

        # Close button
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(20, 20)
        close_btn.setStyleSheet(f"""
            QPushButton {{ background: transparent; color: {TEXT_DIM};
                          border: none; font-size: 10pt; }}
            QPushButton:hover {{ color: {TEXT}; }}
        """)
        close_btn.clicked.connect(self._dismiss)
        layout.addWidget(close_btn)

        # Container frame
        self.setStyleSheet(f"""
            ToastNotification {{
                background: {CARD};
                border: 1px solid {BORDER};
                border-radius: {RADIUS_M}px;
            }}
            ToastNotification:hover {{
                border: 1px solid {self._accent};
            }}
        """)

    def _setup_animations(self):
        self._opacity = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._opacity)
        self._opacity.setOpacity(0.0)

        self._slide_anim = QPropertyAnimation(self, b"pos")
        self._slide_anim.setDuration(ANIM_DURATION_NORMAL)
        self._slide_anim.setEasingCurve(ANIM_EASING)

        self._fade_anim = QPropertyAnimation(self._opacity, b"opacity")
        self._fade_anim.setDuration(ANIM_DURATION_NORMAL)
        self._fade_anim.setEasingCurve(ANIM_EASING)

        self._dismiss_timer = QTimer(self)
        self._dismiss_timer.setSingleShot(True)
        self._dismiss_timer.timeout.connect(self._dismiss)

    def showEvent(self, event):
        super().showEvent(event)
        self._fade_anim.setStartValue(0.0)
        self._fade_anim.setEndValue(1.0)
        self._fade_anim.start()
        self._dismiss_timer.start(self._duration)

    def _dismiss(self):
        self._dismiss_timer.stop()
        self._fade_anim.finished.connect(self._on_dismissed)
        self._fade_anim.setStartValue(self._opacity.opacity())
        self._fade_anim.setEndValue(0.0)
        self._fade_anim.start()

    def _on_dismissed(self):
        self._fade_anim.finished.disconnect(self._on_dismissed)
        self.close()
        if self in ToastNotification._instances:
            ToastNotification._instances.remove(self)
        self._adjust_positions()

    def _adjust_positions(self):
        """Stack toasts vertically on the right side of parent."""
        if not self.parent():
            return
        parent_geo = self.parent().geometry()
        y = parent_geo.height() - 20
        for t in reversed(ToastNotification._instances):
            if t.isVisible() or t._opacity.opacity() > 0:
                geo = t.geometry()
                t.move(parent_geo.right() - geo.width() - 16, y - geo.height())
                y -= geo.height() + SPACING_S


# ── Status Pill ─────────────────────────────────────────────────────────

class StatusPill(QWidget):
    """Animated status indicator with pulsing dot.
    
    Modes: running, connecting, error, stopped
    """

    def __init__(self, status: str = "stopped", parent=None):
        super().__init__(parent)
        self._status = status
        self._pulse_t = 0.0
        self._pulse_dir = 1.0

        self.setFixedHeight(20)
        self.setMinimumWidth(80)

        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._tick)
        self._pulse_timer.start(PULSE_INTERVAL)

        self._update_appearance()

    def set_status(self, status: str):
        self._status = status
        self._update_appearance()
        self.update()

    def _update_appearance(self):
        color = STATUS_COLORS.get(self._status, TEXT_DIM)
        self._base_color = color
        self.setText(self._status.title() if self._status != "stopped" else "Offline")

    def _tick(self):
        if self._status == "connecting":
            self._pulse_t += 0.05 * self._pulse_dir
            if self._pulse_t >= 1.0:
                self._pulse_t = 1.0
                self._pulse_dir = -1.0
            elif self._pulse_t <= 0.0:
                self._pulse_t = 0.0
                self._pulse_dir = 1.0
            self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        color = self._base_color

        if self._status == "connecting":
            color = _pulsing_color(self._base_color, self._pulse_t)

        # Dot
        dot_r = 6
        dot_y = h // 2
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(color))
        p.drawEllipse(QPoint(8, dot_y), dot_r, dot_r)

        # Text
        p.setPen(QColor(TEXT_MED))
        p.setFont(_font(FONT_SIZE_S))
        text = self._status.title() if self._status != "stopped" else "Offline"
        p.drawText(QRect(20, 0, w - 24, h), Qt.AlignmentFlag.AlignVCenter, text)


# ── Shimmer Label ────────────────────────────────────────────────────────

class ShimmerLabel(QLabel):
    """Loading placeholder with shimmer animation."""

    def __init__(self, width_chars: int = 20, parent=None):
        super().__init__(parent)
        self._width_chars = width_chars
        self._shimmer_pos = -1.0
        self._placeholder = "█" * width_chars

        self.setText(self._placeholder)
        self.setFont(_font(FONT_SIZE_S, mono=True))
        self.setStyleSheet(f"color: {TEXT_DIM};")

        self._shimmer_timer = QTimer(self)
        self._shimmer_timer.timeout.connect(self._tick)
        self._shimmer_timer.start(30)

    def _tick(self):
        self._shimmer_pos = (self._shimmer_pos + 0.05) % 2.0
        if self._shimmer_pos > 1.0:
            t = 2.0 - self._shimmer_pos
        else:
            t = self._shimmer_pos

        # Build shimmer text
        chars = list(self._placeholder)
        center = int(len(chars) * t)
        for i in range(max(0, center - 3), min(len(chars), center + 3)):
            chars[i] = "▓"
        self.setText("".join(chars))

    def stop_shimmer(self):
        self._shimmer_timer.stop()
        self.setText("")


# ── Animated Progress Bar ────────────────────────────────────────────────

class AnimatedProgressBar(QWidget):
    """Smooth progress bar with gradient fill."""

    def __init__(self, parent=None, color: str = TEXT_ACC, height: int = 4):
        super().__init__(parent)
        self._value = 0.0
        self._target = 0.0
        self._color = color
        self._bar_height = height
        self.setFixedHeight(height + 8)
        self.setMinimumWidth(100)

        self._anim = QPropertyAnimation(self, b"_value")
        self._anim.setDuration(ANIM_DURATION_NORMAL)
        self._anim.setEasingCurve(ANIM_EASING)
        self._anim.finished.connect(self.update)

    def set_value(self, value: float):
        self._target = max(0.0, min(100.0, value))
        self._anim.stop()
        self._anim.setStartValue(self._value)
        self._anim.setEndValue(self._target)
        self._anim.start()

    def get_value(self):
        return self._value

    def _set_value(self, value):
        self._value = value
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        bar_y = (h - self._bar_height) // 2
        bar_w = w - 12
        bar_x = 6

        # Background
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(BORDER))
        p.drawRoundedRect(QRect(bar_x, bar_y, bar_w, self._bar_height), 2, 2)

        # Fill
        fill_w = int(bar_w * self._value / 100)
        if fill_w > 0:
            p.setBrush(QColor(self._color))
            p.drawRoundedRect(QRect(bar_x, bar_y, fill_w, self._bar_height), 2, 2)


# ── Category Badge ──────────────────────────────────────────────────────

class CategoryBadge(QLabel):
    """Badge showing category icon + name."""

    def __init__(self, category: str = "default", parent=None):
        super().__init__(parent)
        cat = category.lower().strip() if category else "default"
        meta = CATEGORY_META.get(cat, CATEGORY_META["default"])
        icon = meta["icon"]
        color = meta["color"]
        display = f"{icon} {category.title() if category else 'Other'}"

        self.setText(display)
        self.setFont(_font(FONT_SIZE_S, bold=True))
        self.setStyleSheet(f"""
            QLabel {{
                color: {color};
                background: {color}15;
                border: 1px solid {color}40;
                border-radius: {RADIUS_S}px;
                padding: 2px 8px;
            }}
        """)
        self.setContentsMargins(SPACING_S, SPACING_XS, SPACING_S, SPACING_XS)


# ── Tool Chip ────────────────────────────────────────────────────────────

class ToolChip(QWidget):
    """Compact chip showing a tool name with access type indicator."""

    def __init__(self, name: str, read_only: bool = False, parent=None):
        super().__init__(parent)
        self.setFixedHeight(22)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(SPACING_S, SPACING_XS, SPACING_S, SPACING_XS)
        layout.setSpacing(SPACING_XS)

        icon = "🔒" if read_only else "✏️"
        color = GREEN if read_only else YELLOW

        icon_lbl = QLabel(icon)
        icon_lbl.setFont(_font(FONT_SIZE_S - 1))
        layout.addWidget(icon_lbl)

        name_lbl = QLabel(name)
        name_lbl.setFont(_font(FONT_SIZE_S - 1, mono=True))
        name_lbl.setStyleSheet(f"color: {TEXT_MED};")
        layout.addWidget(name_lbl)

        self.setStyleSheet(f"""
            ToolChip {{
                background: {BG2};
                border: 1px solid {BORDER};
                border-radius: {RADIUS_S}px;
            }}
            QToolTip {{
                background: {CARD};
                color: {TEXT};
                border: 1px solid {BORDER};
            }}
        """)


# ── Permission Dialog ────────────────────────────────────────────────────

class PermissionDialog(QDialog):
    """Dialog asking user for permission for sensitive MCP actions."""

    PERMISSION_LEVELS = {
        "read":    ("📖 Read", "Read-only access to data"),
        "write":   ("✏️ Write", "Can modify data and files"),
        "admin":   ("🔧 Admin", "Full system access"),
        "network": ("🌐 Network", "Can make external network requests"),
        "system":  ("⚙️ System", "Can execute system commands"),
    }

    def __init__(self, server_name: str, permission: str, details: str = "", parent=None):
        super().__init__(parent)
        self._permission = permission
        self.setWindowTitle(f"Permission Request — {server_name}")
        self.setModal(True)
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACING_L, SPACING_L, SPACING_L, SPACING_L)
        layout.setSpacing(SPACING_M)

        # Header
        header = QLabel(f"🔐 {server_name} requests access")
        header.setFont(_font(FONT_SIZE_L, bold=True))
        header.setStyleSheet(f"color: {TEXT_ACC};")
        layout.addWidget(header)

        # Permission info
        perm_info = self.PERMISSION_LEVELS.get(permission, ("Unknown", permission))
        perm_lbl = QLabel(f"<b>{perm_info[0]}</b>: {perm_info[1]}")
        perm_lbl.setFont(_font(FONT_SIZE))
        perm_lbl.setStyleSheet(f"color: {TEXT};")
        layout.addWidget(perm_lbl)

        if details:
            details_lbl = QLabel(details)
            details_lbl.setFont(_font(FONT_SIZE_S))
            details_lbl.setStyleSheet(f"color: {TEXT_MED};")
            details_lbl.setWordWrap(True)
            layout.addWidget(details_lbl)

        layout.addSpacing(SPACING_M)

        # Remember choice
        self._remember = QCheckBox("Remember this choice for this session")
        self._remember.setFont(_font(FONT_SIZE_S))
        self._remember.setStyleSheet(f"color: {TEXT_MED};")
        layout.addWidget(self._remember)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        deny_btn = QPushButton("Deny")
        deny_btn.setFont(_font(FONT_SIZE, bold=True))
        deny_btn.setStyleSheet(f"""
            QPushButton {{ background: {RED_D}; color: {TEXT}; border: none;
                          border-radius: {RADIUS_S}px; padding: 8px 16px; }}
            QPushButton:hover {{ background: {RED}; }}
        """)
        deny_btn.clicked.connect(lambda: self.done(QDialog.DialogCode.Rejected))
        btn_row.addWidget(deny_btn)

        allow_btn = QPushButton("Allow")
        allow_btn.setFont(_font(FONT_SIZE, bold=True))
        allow_btn.setStyleSheet(f"""
            QPushButton {{ background: {GREEN_D}; color: {TEXT}; border: none;
                          border-radius: {RADIUS_S}px; padding: 8px 16px; }}
            QPushButton:hover {{ background: {GREEN}; }}
        """)
        allow_btn.clicked.connect(lambda: self.done(QDialog.DialogCode.Accepted))
        btn_row.addWidget(allow_btn)

        layout.addLayout(btn_row)

    def remember_choice(self) -> bool:
        return self._remember.isChecked()
