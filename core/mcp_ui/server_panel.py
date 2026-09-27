"""Card-based server panel with modern hover effects, health indicators, and bulk actions.

Replaces dense QTableWidget with responsive card grid.
"""

from __future__ import annotations

import time
from typing import Any, Optional

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
    _color, _font, _pulsing_color, STATUS_COLORS, CATEGORY_META,
)
from .components import CategoryBadge, StatusPill, ToolChip, ToastNotification


class ServerCard(QFrame):
    """Modern card representing a single MCP server.

    Features:
    - Hover lift effect
    - Category badge with icon
    - Animated status dot
    - Tool count + quick toggle
    - Action buttons (Start/Stop, Details, Remove)
    - Health indicator on hover
    """

    clicked = None  # signal
    actions_requested = None  # signal

    def __init__(self, name: str, config: dict, manager, parent=None):
        super().__init__(parent)
        self._name = name
        self._config = config
        self._manager = manager
        self._hovered = False
        self._health_visible = False
        self._last_ping = -1.0
        self._uptime_start = time.time() if config.get("enabled") else 0

        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setFixedHeight(110)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(f"""
            ServerCard {{
                background: {CARD};
                border: 1px solid {BORDER};
                border-radius: {RADIUS_M}px;
            }}
            ServerCard:hover {{
                background: {CARD_HOVER};
                border: 1px solid {BORDER_B};
            }}
        """)

        self._setup_ui()
        self._setup_animations()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(SPACING_M, SPACING_M, SPACING_M, SPACING_M)
        layout.setSpacing(SPACING_M)

        # Icon area
        icon_frame = QFrame()
        icon_frame.setFixedSize(48, 48)
        icon_frame.setStyleSheet(f"""
            QFrame {{
                background: {BG2};
                border: 1px solid {BORDER};
                border-radius: {RADIUS_M}px;
            }}
        """)
        icon_lay = QVBoxLayout(icon_frame)
        icon_lay.setContentsMargins(0, 0, 0, 0)
        icon_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)

        category = self._config.get("category", "default").lower()
        cat_meta = CATEGORY_META.get(category, CATEGORY_META["default"])
        icon_lbl = QLabel(cat_meta["icon"])
        icon_lbl.setFont(_font(20))
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_lay.addWidget(icon_lbl)
        layout.addWidget(icon_frame)

        # Main content
        content = QWidget()
        content_lay = QVBoxLayout(content)
        content_lay.setContentsMargins(0, 0, 0, 0)
        content_lay.setSpacing(SPACING_XS)

        # Name + status row
        name_row = QHBoxLayout()
        name_row.setSpacing(SPACING_S)

        name_lbl = QLabel(self._name)
        name_lbl.setFont(_font(FONT_SIZE_L, bold=True))
        name_lbl.setStyleSheet(f"color: {TEXT};")
        name_lbl.setToolTip(self._config.get("description", ""))
        name_row.addWidget(name_lbl)

        self._status_pill = StatusPill(
            self._manager.get_server_status(self._name) if self._manager else "stopped"
        )
        name_row.addWidget(self._status_pill, stretch=1)

        content_lay.addLayout(name_row)

        # Description
        desc = self._config.get("description", "")
        if desc:
            desc_lbl = QLabel(desc[:80] + "..." if len(desc) > 80 else desc)
            desc_lbl.setFont(_font(FONT_SIZE_S))
            desc_lbl.setStyleSheet(f"color: {TEXT_MED};")
            desc_lbl.setWordWrap(False)
            content_lay.addWidget(desc_lbl)

        # Tools + category row
        tools_row = QHBoxLayout()
        tools_row.setSpacing(SPACING_S)

        cat_badge = CategoryBadge(self._config.get("category", "default"))
        tools_row.addWidget(cat_badge)

        n_tools = len(self._manager.get_server_tools_meta(self._name)) if self._manager else 0
        tools_lbl = QLabel(f"🛠 {n_tools} tools")
        tools_lbl.setFont(_font(FONT_SIZE_S))
        tools_lbl.setStyleSheet(f"color: {TEXT_DIM};")
        tools_row.addWidget(tools_lbl)

        self._health_lbl = QLabel("")
        self._health_lbl.setFont(_font(FONT_SIZE_S))
        self._health_lbl.setStyleSheet(f"color: {TEXT_DIM};")
        tools_row.addWidget(self._health_lbl)

        tools_row.addStretch()
        content_lay.addLayout(tools_row)

        layout.addWidget(content, stretch=1)

        # Actions column
        actions = QWidget()
        actions_lay = QVBoxLayout(actions)
        actions_lay.setContentsMargins(0, 0, 0, 0)
        actions_lay.setSpacing(SPACING_XS)

        # Quick toggle
        self._toggle = QCheckBox()
        self._toggle.setChecked(self._config.get("enabled", False))
        self._toggle.setStyleSheet(f"""
            QCheckBox::indicator {{
                width: 16px; height: 16px;
            }}
            QCheckBox::indicator:checked {{
                background: {GREEN}; border-radius: 3px;
            }}
            QCheckBox::indicator:unchecked {{
                background: {BG2}; border: 1px solid {BORDER}; border-radius: 3px;
            }}
        """)
        self._toggle.toggled.connect(self._on_toggle)
        actions_lay.addWidget(self._toggle)

        # Buttons
        status = self._manager.get_server_status(self._name) if self._manager else "stopped"
        if status == "running":
            btn_text = "⏹"
            btn_color = RED
            btn_callback = self._on_stop
        else:
            btn_text = "▶"
            btn_color = GREEN
            btn_callback = self._on_start

        action_btn = QPushButton(btn_text)
        action_btn.setFixedSize(28, 28)
        action_btn.setFont(_font(10, bold=True))
        action_btn.setStyleSheet(f"""
            QPushButton {{ background: {btn_color}30; color: {btn_color};
                          border: 1px solid {btn_color}60; border-radius: 4px; }}
            QPushButton:hover {{ background: {btn_color}; color: {TEXT}; }}
        """)
        action_btn.clicked.connect(btn_callback)
        actions_lay.addWidget(action_btn)

        details_btn = QPushButton("≡")
        details_btn.setFixedSize(28, 28)
        details_btn.setFont(_font(9))
        details_btn.setStyleSheet(f"""
            QPushButton {{ background: {BG2}; color: {TEXT_MED};
                          border: 1px solid {BORDER}; border-radius: 4px; }}
            QPushButton:hover {{ color: {TEXT_ACC}; border-color: {BORDER_B}; }}
        """)
        details_btn.clicked.connect(self._on_details)
        actions_lay.addWidget(details_btn)

        remove_btn = QPushButton("✕")
        remove_btn.setFixedSize(28, 28)
        remove_btn.setFont(_font(8))
        remove_btn.setStyleSheet(f"""
            QPushButton {{ background: transparent; color: {TEXT_DIM};
                          border: none; border-radius: 4px; }}
            QPushButton:hover {{ color: {RED}; background: {RED}20; }}
        """)
        remove_btn.clicked.connect(self._on_remove)
        actions_lay.addWidget(remove_btn)

        actions_lay.addStretch()
        layout.addWidget(actions)

    def _setup_animations(self):
        self._hover_anim = QPropertyAnimation(self, b"maximumHeight")
        self._hover_anim.setDuration(ANIM_DURATION_NORMAL)
        self._hover_anim.setEasingCurve(ANIM_EASING)

    def enterEvent(self, event):
        self._hovered = True
        self._hover_anim.stop()
        self._hover_anim.setStartValue(self.height())
        self._hover_anim.setEndValue(125)
        self._hover_anim.start()
        self._health_visible = True
        self._update_health()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self._hover_anim.stop()
        self._hover_anim.setStartValue(self.height())
        self._hover_anim.setEndValue(110)
        self._hover_anim.start()
        self._health_visible = False
        self._health_lbl.setText("")
        super().leaveEvent(event)

    def _update_health(self):
        if not self._health_visible or not self._manager:
            return
        try:
            result = self._manager.probe_server(self._name, force=False)
            if result.get("ok"):
                latency = result.get("latency_ms", -1)
                tools = result.get("tools", 0)
                self._health_lbl.setText(f"✅ {latency}ms · {tools}t")
                self._health_lbl.setStyleSheet(f"color: {GREEN};")
            else:
                self._health_lbl.setText("⚠️ " + (result.get("error", "")[:20]))
                self._health_lbl.setStyleSheet(f"color: {YELLOW};")
        except Exception:
            pass

    def _on_toggle(self, checked: bool):
        if self._manager:
            result = self._manager.toggle_server(self._name)
            ToastNotification.show(
                self.window(),
                message=result,
                toast_type="success" if "enabled" in result else "error",
                duration=3000,
            )

    def _on_start(self):
        if self._manager:
            result = self._manager.start_server(self._name)
            ToastNotification.show(self.window(), message=result, toast_type="success")

    def _on_stop(self):
        if self._manager:
            result = self._manager.stop_server(self._name)
            ToastNotification.show(self.window(), message=result, toast_type="info")

    def _on_details(self):
        from .install_flow import ServerDialog
        dlg = ServerDialog(self.window(), self._manager, self._name)
        dlg.exec()

    def _on_remove(self):
        from PyQt6.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self, "Remove Server",
            f"Remove '{self._name}' from config?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes and self._manager:
            result = self._manager.uninstall_server(self._name)
            ToastNotification.show(self.window(), message=result, toast_type="info")

    def refresh_status(self):
        if self._manager:
            status = self._manager.get_server_status(self._name)
            self._status_pill.set_status(status)


# ── Server Grid Panel ────────────────────────────────────────────────────

class ServerGridPanel(QWidget):
    """Responsive card grid for MCP servers.

    Features:
    - Auto-flow grid layout
    - Search/filter
    - Bulk enable/disable
    - Category filtering
    - Sort by name/status/category
    """

    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self._manager = manager
        self._cards: dict[str, ServerCard] = {}
        self._filter_text = ""
        self._filter_category = "all"
        self._sort_by = "name"

        self.setStyleSheet(f"background: {BG};")
        self._setup_ui()
        self.refresh()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACING_M, SPACING_M, SPACING_M, SPACING_M)
        layout.setSpacing(SPACING_M)

        # Toolbar
        toolbar = QWidget()
        toolbar.setStyleSheet(f"background: {BG2}; border-radius: {RADIUS_M}px;")
        tb_lay = QHBoxLayout(toolbar)
        tb_lay.setContentsMargins(SPACING_M, SPACING_M, SPACING_M, SPACING_M)
        tb_lay.setSpacing(SPACING_M)

        # Search
        from PyQt6.QtWidgets import QLineEdit
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("🔍 Search servers...")
        self._search_input.setFont(_font(FONT_SIZE))
        self._search_input.setStyleSheet(f"""
            QLineEdit {{
                background: {BG}; color: {TEXT}; border: 1px solid {BORDER};
                border-radius: {RADIUS_S}px; padding: 6px 10px;
            }}
            QLineEdit:focus {{ border: 1px solid {BORDER_B}; }}
        """)
        self._search_input.textChanged.connect(self._on_filter_changed)
        tb_lay.addWidget(self._search_input, stretch=1)

        # Category filter
        from PyQt6.QtWidgets import QComboBox
        self._cat_filter = QComboBox()
        self._cat_filter.addItems(["All", "Web", "Browser", "Database", "AI & Media",
                                    "Development", "Desktop", "Version Control", "Social"])
        self._cat_filter.setStyleSheet(f"""
            QComboBox {{
                background: {BG}; color: {TEXT}; border: 1px solid {BORDER};
                border-radius: {RADIUS_S}px; padding: 4px 8px;
            }}
            QComboBox:focus {{ border: 1px solid {BORDER_B}; }}
        """)
        self._cat_filter.currentTextChanged.connect(self._on_filter_changed)
        tb_lay.addWidget(self._cat_filter)

        # Sort
        self._sort_combo = QComboBox()
        self._sort_combo.addItems(["Name", "Status", "Category", "Last Used"])
        self._sort_combo.setStyleSheet(f"""
            QComboBox {{
                background: {BG}; color: {TEXT}; border: 1px solid {BORDER};
                border-radius: {RADIUS_S}px; padding: 4px 8px;
            }}
        """)
        self._sort_combo.currentTextChanged.connect(self._on_sort_changed)
        tb_lay.addWidget(self._sort_combo)

        # Bulk actions
        start_all_btn = QPushButton("▶ Start All")
        start_all_btn.setFont(_font(FONT_SIZE_S, bold=True))
        start_all_btn.setStyleSheet(f"""
            QPushButton {{ background: {GREEN_D}; color: {TEXT}; border: none;
                          border-radius: {RADIUS_S}px; padding: 6px 12px; }}
            QPushButton:hover {{ background: {GREEN}; }}
        """)
        start_all_btn.clicked.connect(self._start_all_disabled)
        tb_lay.addWidget(start_all_btn)

        stop_all_btn = QPushButton("⏹ Stop All")
        stop_all_btn.setFont(_font(FONT_SIZE_S, bold=True))
        stop_all_btn.setStyleSheet(f"""
            QPushButton {{ background: {RED_D}; color: {TEXT}; border: none;
                          border-radius: {RADIUS_S}px; padding: 6px 12px; }}
            QPushButton:hover {{ background: {RED}; }}
        """)
        stop_all_btn.clicked.connect(self._stop_all)
        tb_lay.addWidget(stop_all_btn)

        layout.addWidget(toolbar)

        # Grid area
        self._grid_widget = QWidget()
        self._grid_widget.setStyleSheet(f"background: transparent;")
        self._grid_layout = QVBoxLayout(self._grid_widget)
        self._grid_layout.setContentsMargins(0, 0, 0, 0)
        self._grid_layout.setSpacing(SPACING_M)
        layout.addWidget(self._grid_widget, stretch=1)

        # Scroll area for grid
        from PyQt6.QtWidgets import QScrollArea
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet(f"""
            QScrollArea {{ background: transparent; border: none; }}
            QScrollBar:vertical {{
                background: {BG}; width: 8px; border: none;
            }}
            QScrollBar::handle:vertical {{
                background: {BORDER_B}; border-radius: 4px; min-height: 20px;
            }}
        """)
        scroll.setWidget(self._grid_widget)
        layout.addWidget(scroll, stretch=1)

    def refresh(self):
        if not self._manager:
            return

        # Clear existing
        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._cards.clear()

        config = self._manager.get_all_servers_config()
        servers = [(k, v) for k, v in config.items() if not k.startswith("_")]

        # Apply filters
        filtered = []
        for name, cfg in servers:
            if self._filter_text and self._filter_text.lower() not in name.lower():
                continue
            cat = cfg.get("category", "").lower()
            if self._filter_category != "all" and cat != self._filter_category.lower():
                continue
            filtered.append((name, cfg))

        # Sort
        sort_key = {"name": 0, "status": 1, "category": 2}.get(self._sort_by, 0)
        if sort_key == 0:
            filtered.sort(key=lambda x: x[0].lower())
        elif sort_key == 1:
            filtered.sort(key=lambda x: self._manager.get_server_status(x[0]))
        elif sort_key == 2:
            filtered.sort(key=lambda x: x[1].get("category", "").lower())

        # Create cards in pairs (two per row)
        row_widget = None
        row_layout = None
        for i, (name, cfg) in enumerate(filtered):
            if i % 2 == 0:
                row_widget = QWidget()
                row_widget.setStyleSheet("background: transparent;")
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(SPACING_M)
                self._grid_layout.addWidget(row_widget)

            card = ServerCard(name, cfg, self._manager, self)
            self._cards[name] = card
            row_layout.addWidget(card, stretch=1)

        if not filtered:
            empty = QLabel("No MCP servers configured.\nBrowse tab se naye servers install karo!")
            empty.setFont(_font(FONT_SIZE_L))
            empty.setStyleSheet(f"color: {TEXT_DIM};")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._grid_layout.addWidget(empty, stretch=1)

    def _on_filter_changed(self, _):
        self._filter_text = self._search_input.text().strip()
        cat_text = self._cat_filter.currentText()
        self._filter_category = cat_text if cat_text != "All" else "all"
        self.refresh()

    def _on_sort_changed(self, text):
        self._sort_by = text.lower()
        self.refresh()

    def _start_all_disabled(self):
        if not self._manager:
            return
        config = self._manager.get_all_servers_config()
        count = 0
        for name, cfg in config.items():
            if name.startswith("_"):
                continue
            if not cfg.get("enabled", False):
                self._manager.toggle_server(name)
                count += 1
        ToastNotification.show(
            self.window(),
            title="Bulk Action",
            message=f"Started {count} disabled servers",
            toast_type="success",
        )
        self.refresh()

    def _stop_all(self):
        if not self._manager:
            return
        config = self._manager.get_all_servers_config()
        count = 0
        for name, cfg in config.items():
            if name.startswith("_"):
                continue
            if cfg.get("enabled", False) and self._manager.get_server_status(name) == "running":
                self._manager.toggle_server(name)
                count += 1
        ToastNotification.show(
            self.window(),
            title="Bulk Action",
            message=f"Stopped {count} servers",
            toast_type="info",
        )
        self.refresh()

    def refresh_statuses(self):
        for card in self._cards.values():
            card.refresh_status()
