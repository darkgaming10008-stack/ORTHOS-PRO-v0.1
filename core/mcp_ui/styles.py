"""Centralized theme and design tokens for MCP UI.

All colors, fonts, spacing, and animation constants live here.
Inspired by: GitHub Dark, VS Code Dark+, August 2026 design standards.
"""

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, QTimer
from PyQt6.QtGui import QColor, QFont

# ── Palette ─────────────────────────────────────────────────────────────
BG          = "#0a0e14"
BG2         = "#0d1117"
CARD        = "#161b22"
CARD_HOVER  = "#1c2129"
BORDER      = "#30363d"
BORDER_B    = "#58a6ff"
BORDER_A    = "#1f6feb"

TEXT        = "#e6edf3"
TEXT_MED    = "#8b949e"
TEXT_DIM    = "#484f58"
TEXT_ACC    = "#58a6ff"

GREEN       = "#3fb950"
GREEN_D     = "#238636"
RED         = "#f85149"
RED_D       = "#da3633"
YELLOW      = "#d29922"
YELLOW_D    = "#9e6a03"
PURPLE      = "#bc8cff"
PURPLE_D    = "#8957e5"
CYAN        = "#39c5cf"
MAGENTA     = "#f778ba"

# ── Typography ──────────────────────────────────────────────────────────
FONT        = "Segoe UI"
FONT_MONO   = "Consolas"
FONT_SIZE   = 9
FONT_SIZE_S = 8
FONT_SIZE_L = 11
FONT_SIZE_XL= 14

# ── Spacing ─────────────────────────────────────────────────────────────
SPACING_XS  = 2
SPACING_S   = 4
SPACING_M   = 8
SPACING_L   = 12
SPACING_XL  = 16
SPACING_XXL = 24

RADIUS_S    = 3
RADIUS_M    = 6
RADIUS_L    = 8
RADIUS_XL   = 12

# ── Animation ───────────────────────────────────────────────────────────
ANIM_DURATION_FAST   = 120   # ms
ANIM_DURATION_NORMAL = 200   # ms
ANIM_DURATION_SLOW   = 350   # ms
ANIM_EASING          = QEasingCurve.Type.OutCubic
PULSE_INTERVAL       = 80    # ms between pulse frames

# ── Status Colors ───────────────────────────────────────────────────────
STATUS_COLORS = {
    "running":    GREEN,
    "connecting": YELLOW,
    "error":      RED,
    "stopped":    TEXT_DIM,
}

# ── Category Icons & Colors ─────────────────────────────────────────────
CATEGORY_META = {
    "web":          {"icon": "🌐", "color": "#39c5cf"},
    "web search":   {"icon": "🔍", "color": "#39c5cf"},
    "database":     {"icon": "🗄️", "color": "#58a6ff"},
    "ai":           {"icon": "🧠", "color": "#bc8cff"},
    "ai & thinking":{"icon": "🧠", "color": "#bc8cff"},
    "ai & media":   {"icon": "🎨", "color": "#bc8cff"},
    "file system":  {"icon": "📁", "color": "#3fb950"},
    "development":  {"icon": "💻", "color": "#58a6ff"},
    "browser":      {"icon": "🎮", "color": "#f778ba"},
    "desktop":      {"icon": "🖥️", "color": "#d29922"},
    "version control": {"icon": "📦", "color": "#3fb950"},
    "social media": {"icon": "📱", "color": "#f778ba"},
    "communication":{"icon": "💬", "color": "#3fb950"},
    "productivity": {"icon": "📋", "color": "#d29922"},
    "security":     {"icon": "🔐", "color": "#f85149"},
    "cloud":        {"icon": "☁️", "color": "#58a6ff"},
    "media":        {"icon": "🎬", "color": "#f778ba"},
    "default":      {"icon": "⚡", "color": "#58a6ff"},
}

# ── Helper Functions ────────────────────────────────────────────────────

def _font(size=None, bold=False, mono=False):
    """Create a QFont with consistent styling."""
    size = size or FONT_SIZE
    family = FONT_MONO if mono else FONT
    weight = QFont.Weight.Bold if bold else QFont.Weight.Normal
    return QFont(family, size, weight)


def _color(hex_str: str, alpha: int = 255) -> QColor:
    """Create QColor from hex string with optional alpha."""
    c = QColor(hex_str)
    c.setAlpha(alpha)
    return c


def _animate_property(widget, prop_name: str, start, end, duration: int = ANIM_DURATION_NORMAL):
    """Create and start a QPropertyAnimation on a widget."""
    anim = QPropertyAnimation(widget, prop_name.encode() if isinstance(prop_name, str) else prop_name)
    anim.setDuration(duration)
    anim.setStartValue(start)
    anim.setEndValue(end)
    anim.setEasingCurve(ANIM_EASING)
    anim.start()
    return anim


def _pulsing_color(base_hex: str, t: float) -> str:
    """Generate a pulsing color variant. t should be 0..1."""
    c = QColor(base_hex)
    h, s, l, a = c.getHsl()
    new_l = max(0, min(255, int(l + (255 - l) * 0.4 * t)))
    c.setHsl(h, s, new_l, a)
    return c.name()
