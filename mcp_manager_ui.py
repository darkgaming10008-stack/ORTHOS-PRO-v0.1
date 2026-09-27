"""MCP Manager UI — PyQt6 window for browsing, installing, and managing MCP servers.

ChatGPT-style flow: search -> detail dialog -> Scan Tools -> Connect & Enable.
Everything runs in background threads so the UI never freezes.
"""
import json
import shutil
import threading
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QFormLayout, QFrame,
    QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit,
    QMainWindow, QPushButton, QTableWidget, QTableWidgetItem,
    QTabWidget, QTextEdit, QVBoxLayout, QWidget, QMessageBox,
    QFileDialog, QScrollArea, QSizePolicy,
)

from core.mcp_manager import get_global_manager

_HAS_MCP_UI = True

BASE_DIR = Path(__file__).resolve().parent

FONT = "Segoe UI"
BG = "#0d1117"
CARD = "#161b22"
BORDER = "#30363d"
GREEN = "#3fb950"
RED = "#f85149"
YELLOW = "#d29922"
TEXT = "#e6edf3"
TEXT_MED = "#8b949e"
ACCENT = "#58a6ff"
PURPLE = "#bc8cff"


def _font(size=9, bold=False):
    w = QFont.Weight.Bold if bold else QFont.Weight.Normal
    return QFont(FONT, size, w)


def _compact_downloads(n) -> str:
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "?"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}k"
    return str(n)


def _btn(text, color, bg=CARD, size=8, bold=False, padding="4px 10px"):
    btn = QPushButton(text)
    btn.setFont(_font(size, bold))
    btn.setStyleSheet(f"""
        QPushButton {{ background: {bg}; color: {color}; border: 1px solid {BORDER};
                      padding: {padding}; border-radius: 4px; }}
        QPushButton:hover {{ background: {BORDER}; }}
    """)
    return btn


def _label(text, color=TEXT, size=9, bold=False, wrap=True):
    lbl = QLabel(text)
    lbl.setFont(_font(size, bold))
    lbl.setStyleSheet(f"color: {color};")
    if wrap:
        lbl.setWordWrap(True)
    return lbl


# ─────────────────────────────────────────────────────────────────────
# Install / details dialog — the ChatGPT-style single install surface
# ─────────────────────────────────────────────────────────────────────

class InstallDialog(QDialog):
    """One dialog for the whole install flow: info -> env keys -> Scan Tools -> Connect."""

    def __init__(self, parent, manager, result: dict, prefill: dict | None = None):
        super().__init__(parent)
        self._manager = manager
        self._result = result or {}
        self._prefill = prefill or {}
        self._name = self._prefill.get("name") or self._result.get("name", "server")
        self._tools_meta = []
        self._scan_out = {"done": False, "tools": [], "error": None, "info": None}
        self._install_out = {"done": False, "result": None}
        self._busy = False
        self._setup_url = ""

        self.setWindowTitle(f"Install MCP Server — {self._name}")
        self.setMinimumSize(640, 620)
        self.resize(720, 680)
        self.setStyleSheet(f"background: {BG}; color: {TEXT};")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)

        # Header — name + badges
        head = QHBoxLayout()
        title = QLabel(self._name)
        title.setFont(_font(15, True))
        title.setStyleSheet(f"color: {ACCENT};")
        head.addWidget(title)

        source = self._prefill.get("source") or self._result.get("source", "?")
        src_badge = QLabel(f"  {source.upper()}  ")
        src_badge.setFont(_font(8, True))
        src_badge.setStyleSheet(
            f"background: {CARD}; color: {PURPLE}; border: 1px solid {BORDER}; border-radius: 8px; padding: 1px 4px;")
        head.addWidget(src_badge)

        if self._result.get("verified") or self._prefill.get("verified"):
            v_badge = QLabel("  ✓ VERIFIED  ")
            v_badge.setFont(_font(8, True))
            v_badge.setStyleSheet(
                f"background: {CARD}; color: {GREEN}; border: 1px solid {BORDER}; border-radius: 8px; padding: 1px 4px;")
            head.addWidget(v_badge)

        score = self._result.get("downloads")
        if score is not None:
            s_badge = QLabel(f"  ▲ {_compact_downloads(score)}/mo  ")
            s_badge.setFont(_font(8, True))
            s_badge.setStyleSheet(
                f"background: {CARD}; color: {YELLOW}; border: 1px solid {BORDER}; border-radius: 8px; padding: 1px 4px;")
            head.addWidget(s_badge)
        head.addStretch()
        lay.addLayout(head)

        desc = self._prefill.get("description") or self._result.get("description", "")
        if desc:
            lay.addWidget(_label(desc, TEXT_MED, 9))

        # Command line
        remote = bool(self._result.get("url"))
        transport = self._result.get("transport")
        if transport == "smithery":
            lay.addWidget(_label(
                "Transport: Smithery Connect  ·  hosted cloud server (OAuth / API-key)",
                TEXT_MED, 8))
        elif remote:
            lay.addWidget(_label(f"Transport: Remote HTTP  ·  {self._result.get('url', '')}",
                                 TEXT_MED, 8))
        else:
            cmd = self._prefill.get("command", "npx")
            args = self._prefill.get("args") or self._result.get("args_base") or ["-y", self._name]
            if isinstance(args, list):
                args_str = " ".join(str(a) for a in args)
            else:
                args_str = str(args)
            lay.addWidget(_label(f"Command: {cmd} {args_str}   ·   Transport: stdio (local process)",
                                 TEXT_MED, 8))
            runtime = "Node.js / npx" if cmd == "npx" else "Python / uvx" if cmd == "uvx" else cmd
            available = shutil.which(cmd) is not None
            lay.addWidget(_label(
                f"Runtime check: {'✓' if available else '⚠'} {runtime} "
                f"{'is ready' if available else 'was not found — install it before connecting'}",
                GREEN if available else YELLOW, 8))

        # Category
        cat_row = QHBoxLayout()
        cat_row.addWidget(_label("Category:", TEXT_MED, 9, bold=False))
        self.cat_input = QLineEdit(self._prefill.get("category", "Development"))
        self.cat_input.setStyleSheet(f"""
            QLineEdit {{ background: {CARD}; color: {TEXT}; border: 1px solid {BORDER};
                        padding: 5px 8px; border-radius: 4px; }}
        """)
        cat_row.addWidget(self.cat_input, stretch=1)
        lay.addLayout(cat_row)

        # Env variables (auto-detected from registry metadata or prefill)
        env_entries: list[dict] = []
        env_placeholder: dict[str, str] = {}
        for k, v in (self._prefill.get("env") or {}).items():
            env_entries.append({
                "name": str(k),
                "required": False,
                "secret": any(s in str(k).upper() for s in ("KEY", "TOKEN", "SECRET", "PASSWORD")),
                "description": "",
            })
            env_placeholder[str(k)] = str(v)
        for e in (self._result.get("env") or []):
            if isinstance(e, dict):
                nm = str(e.get("name", ""))
                entry = e
            else:
                nm = str(e)
                entry = {"required": False, "secret": False, "description": ""}
            if nm and nm not in env_placeholder and nm not in {x["name"] for x in env_entries}:
                env_entries.append({
                    "name": nm,
                    "required": bool(entry.get("required", False)),
                    "secret": bool(entry.get("secret", False)),
                    "description": entry.get("description", ""),
                })

        self.env_inputs: dict[str, QLineEdit] = {}
        if env_entries:
            env_label = _label("Required credentials:", TEXT, 10, True)
            lay.addWidget(env_label)
            env_form = QFormLayout()
            env_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
            for entry in env_entries:
                key = entry["name"]
                edit = QLineEdit()
                edit.setPlaceholderText("Optional — leave empty if not needed")
                if key in env_placeholder:
                    edit.setText(env_placeholder[key])
                if entry.get("secret"):
                    edit.setEchoMode(QLineEdit.EchoMode.Password)
                edit.setStyleSheet(f"""
                    QLineEdit {{ background: {CARD}; color: {TEXT}; border: 1px solid {BORDER};
                                padding: 5px 8px; border-radius: 4px; }}
                """)
                self.env_inputs[key] = edit
                label = key + (" *" if entry.get("required") else "")
                lbl = _label(label, TEXT_MED, 8, False)
                if entry.get("description"):
                    lbl.setToolTip(entry["description"])
                env_form.addRow(lbl, edit)
            lay.addLayout(env_form)

        # Auth headers (remote servers / Smithery config keys)
        self.header_inputs: dict[str, tuple[QLineEdit, dict]] = {}
        self._headers_form: QFormLayout | None = None
        self._headers_label: QLabel | None = None
        headers_meta = self._result.get("headers") or []
        if self._result.get("transport") == "smithery":
            self._headers_label = _label("Server authentication:", TEXT, 10, True)
            self._headers_label.hide()
            lay.addWidget(self._headers_label)
            h_form = QFormLayout()
            self._headers_form = h_form
            h_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
            lay.addLayout(h_form)
        if headers_meta:
            lay.addWidget(_label("Server authentication:", TEXT, 10, True))
            h_form = QFormLayout()
            self._headers_form = h_form
            h_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
            for h in headers_meta:
                edit = QLineEdit()
                edit.setPlaceholderText(f"e.g. {h.get('value_template', 'Bearer YOUR_KEY')}")
                if h.get("secret"):
                    edit.setEchoMode(QLineEdit.EchoMode.Password)
                if h.get("default_value"):
                    edit.setText(h["default_value"])
                edit.setStyleSheet(f"""
                    QLineEdit {{ background: {CARD}; color: {TEXT}; border: 1px solid {BORDER};
                                padding: 5px 8px; border-radius: 4px; }}
                """)
                self.header_inputs[h.get("name", "Authorization")] = (edit, h)
                lbl = _label(h.get("name", "Authorization") + (" *" if h.get("required") else ""),
                             TEXT_MED, 8, False)
                if h.get("description"):
                    lbl.setToolTip(h["description"])
                h_form.addRow(lbl, edit)
            lay.addLayout(h_form)

        # Tool list (filled by Scan Tools)
        self.tools_label = _label("Tools (click Scan Tools to discover):", TEXT, 10, True)
        lay.addWidget(self.tools_label)

        self.tools_table = QTableWidget(0, 3)
        self.tools_table.setHorizontalHeaderLabels(["Enable", "Tool", "Access"])
        self.tools_table.setStyleSheet(f"""
            QTableWidget {{ background: {CARD}; border: 1px solid {BORDER};
                           gridline-color: {BORDER}; }}
            QTableWidget::item {{ padding: 3px 6px; }}
            QHeaderView::section {{ background: {BG}; color: {TEXT_MED};
                                   border: 1px solid {BORDER}; padding: 3px; }}
        """)
        self.tools_table.horizontalHeader().setStretchLastSection(False)
        self.tools_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.tools_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.tools_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.tools_table.verticalHeader().setVisible(False)
        self.tools_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.tools_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        lay.addWidget(self.tools_table, stretch=1)

        # Status line
        self.status = _label("", TEXT_MED, 9)
        lay.addWidget(self.status)

        # Buttons
        btn_row = QHBoxLayout()
        self.scan_btn = _btn("Test & Scan Tools", ACCENT, bg=CARD, size=9, bold=True, padding="6px 16px")
        self.scan_btn.clicked.connect(self._do_scan)
        btn_row.addWidget(self.scan_btn)

        self.open_btn = _btn("Open OAuth Page", YELLOW, bg=CARD, size=9, bold=True, padding="6px 16px")
        self.open_btn.clicked.connect(self._open_setup)
        self.open_btn.hide()
        btn_row.addWidget(self.open_btn)

        btn_row.addStretch()

        self.cancel_btn = _btn("Cancel", TEXT_MED, bg=CARD, size=9, padding="6px 16px")
        self.cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self.cancel_btn)

        self.install_btn = _btn("Connect & Enable", BG, bg=GREEN, size=9, bold=True, padding="6px 16px")
        self.install_btn.clicked.connect(self._do_install)
        self.install_btn.setEnabled(False)
        btn_row.addWidget(self.install_btn)
        lay.addLayout(btn_row)

        self._poll = QTimer(self)
        self._poll.timeout.connect(self._poll_tick)

        # Auto-start scan when opened from a quick-install button
        if self._prefill.get("auto_scan", False):
            QTimer.singleShot(150, self._do_scan)

    # ── helpers ──────────────────────────────────────────────────────

    def _build_config(self) -> dict:
        if self._result.get("transport") == "smithery":
            config = {
                "enabled": True,
                "category": self.cat_input.text().strip() or "Development",
                "transport": "smithery",
                "server_qname": self._result.get("server_qname") or self._name,
                "description": self._prefill.get("description") or self._result.get("description", ""),
            }
            env = {}
            for key, edit in self.env_inputs.items():
                val = edit.text().strip()
                if val:
                    env[key] = val
            if env:
                config["env"] = env
            headers = {}
            for hname, (edit, _hmeta) in self.header_inputs.items():
                val = edit.text().strip()
                if val:
                    headers[hname] = val
            if headers:
                config["headers"] = headers
            return config

        remote = bool(self._result.get("url"))
        if remote:
            cmd = None
            args = []
        else:
            cmd = self._prefill.get("command", "npx")
            args = self._prefill.get("args") or self._result.get("args_base") or ["-y", self._name]
            if isinstance(args, str):
                args = [args]
        config = {
            "enabled": True,
            "category": self.cat_input.text().strip() or "Development",
            "command": cmd,
            "args": list(args),
            "description": self._prefill.get("description") or self._result.get("description", ""),
        }
        if self._result.get("url"):
            config["url"] = self._result["url"]
            config["transport"] = "streamableHttp"
        env = {}
        for key, edit in self.env_inputs.items():
            val = edit.text().strip()
            if val:
                env[key] = val
        if env:
            config["env"] = env
        if self.header_inputs:
            config["headers"] = {}
            for hname, (edit, hmeta) in self.header_inputs.items():
                val = edit.text().strip()
                if not val:
                    continue
                vt = hmeta.get("value_template") or ""
                var = hmeta.get("var")
                if var and "{" in vt:
                    config["headers"][hname] = vt.replace("{" + var + "}", val)
                else:
                    config["headers"][hname] = val
        return config

    def _set_status(self, text, color=TEXT_MED):
        self.status.setText(text)
        self.status.setStyleSheet(f"color: {color};")

    def _set_busy(self, busy: bool):
        self._busy = busy
        self.scan_btn.setEnabled(not busy)
        self.install_btn.setEnabled(not busy and bool(self._tools_meta))
        self.cancel_btn.setEnabled(not busy)

    # ── Scan Tools ───────────────────────────────────────────────────

    def _do_scan(self):
        if self._busy:
            return
        self._set_busy(True)
        self._scan_out = {"done": False, "tools": [], "error": None, "info": None}
        self._setup_url = ""
        self.open_btn.hide()
        self.tools_table.setRowCount(0)
        self._set_status("Scanning tools from server… (first run downloads the package)", ACCENT)

        cfg = self._build_config()

        def work():
            tools, err, info = self._manager.preview_server(cfg)
            self._scan_out = {"done": True, "tools": tools, "error": err, "info": info}

        threading.Thread(target=work, daemon=True).start()
        self._poll.start(120)

    def _open_setup(self):
        if self._setup_url:
            import webbrowser
            webbrowser.open(self._setup_url)

    def _render_input_required(self, info: dict):
        """Render missing Smithery config keys so the user can fill them in."""
        if self._result.get("transport") != "smithery":
            return
        fields = info.get("fields") or {}
        missing = info.get("missing") or {}
        missing_h = missing.get("headers") or []
        missing_q = missing.get("query") or []
        if missing_q:
            self._setup_url = info.get("setup_url", "") or self._setup_url
            if self._setup_url:
                self.open_btn.setText("Open Setup Page")
                self.open_btn.show()
        if missing_h and self._headers_form is not None:
            if self._headers_label is not None:
                self._headers_label.show()
            for name in missing_h:
                if name in self.header_inputs:
                    continue
                meta = fields.get(name) or {}
                edit = QLineEdit()
                edit.setEchoMode(QLineEdit.EchoMode.Password)
                edit.setPlaceholderText("required value")
                edit.setStyleSheet(f"""
                    QLineEdit {{ background: {CARD}; color: {TEXT}; border: 1px solid {BORDER};
                                padding: 5px 8px; border-radius: 4px; }}
                """)
                self.header_inputs[name] = (edit, {
                    "name": name,
                    "required": bool(meta.get("required")),
                    "description": meta.get("description", ""),
                    "value_template": "",
                })
                lbl = _label(name + (" *" if meta.get("required") else ""), TEXT_MED, 8, False)
                if meta.get("description"):
                    lbl.setToolTip(str(meta["description"]))
                self._headers_form.addRow(lbl, edit)

    # ── Install ──────────────────────────────────────────────────────

    def _do_install(self):
        if self._busy or not self._tools_meta:
            return
        self._set_busy(True)
        self._install_out = {"done": False, "result": None}
        self._set_status(f"Installing & connecting '{self._name}'…", ACCENT)

        cfg = self._build_config()
        enabled = [t["short"] for t in self._tools_meta if self._tool_checked(t["short"])]
        name = self._name

        def work():
            result = self._manager.install_server_detailed(name, cfg, enabled_tools=enabled or None)
            self._install_out = {"done": True, "result": result}

        threading.Thread(target=work, daemon=True).start()
        self._poll.start(120)

    # ── poll worker results (UI thread) ──────────────────────────────

    def _poll_tick(self):
        if self._scan_out.get("done"):
            self._poll.stop()
            self._set_busy(False)
            tools = self._scan_out.get("tools") or []
            error = self._scan_out.get("error")
            info = self._scan_out.get("info") or {}
            if error:
                if info.get("kind") == "auth_required":
                    self._setup_url = info.get("setup_url", "")
                    if self._setup_url:
                        self.open_btn.setText("Open OAuth Page")
                        self.open_btn.show()
                    self._set_status(
                        "OAuth required — complete authorization in your browser, "
                        "then click Scan Tools again.", YELLOW)
                    return
                if info.get("kind") == "input_required":
                    self._render_input_required(info)
                    self._set_status(
                        "Server needs the keys above — fill them in, "
                        "then click Scan Tools again.", YELLOW)
                    return
                self._set_status(f"Scan failed: {error}", RED)
                return
            self.open_btn.hide()
            self._tools_meta = []
            self.tools_table.setRowCount(len(tools))
            for row, t in enumerate(tools):
                self._tools_meta.append({
                    "short": t["name"].replace(f"mcp_{self._name}_", "", 1),
                    "full": t["name"],
                    "description": t.get("description", ""),
                    "read_only": bool(t.get("read_only", False)),
                })
                check = QCheckBox()
                check.setChecked(True)
                check.setStyleSheet(f"""
                    QCheckBox::indicator {{ width: 15px; height: 15px; }}
                    QCheckBox::indicator:checked {{ background: {GREEN}; border-radius: 3px; }}
                    QCheckBox::indicator:unchecked {{ background: {CARD};
                        border: 1px solid {BORDER}; border-radius: 3px; }}
                """)
                w = QWidget()
                h = QHBoxLayout(w)
                h.setContentsMargins(6, 2, 6, 2)
                h.addWidget(check)
                h.addStretch()
                self.tools_table.setCellWidget(row, 0, w)
                self.tools_table.setItem(row, 1, QTableWidgetItem(self._tools_meta[-1]["short"]))
                access = "🔒 Read-only" if t.get("read_only") else "✏️ Write"
                acc_item = QTableWidgetItem(access)
                acc_item.setForeground(Qt.GlobalColor.green if t.get("read_only") else Qt.GlobalColor.yellow)
                self.tools_table.setItem(row, 2, acc_item)
                self.tools_table.item(row, 1).setToolTip(self._tools_meta[-1]["description"])
            self._set_status(
                f"{len(tools)} tool(s) discovered. Untick any you don't need, then Connect & Enable.",
                GREEN)
            self.install_btn.setEnabled(True)
            self.install_btn.setText("Connect & Enable")
            return

        if self._install_out.get("done"):
            self._poll.stop()
            self._set_busy(False)
            result = self._install_out.get("result") or {}
            ok = result.get("ok", False)
            self._set_status(result.get("message", ""), GREEN if ok else RED)
            if ok:
                self.install_btn.setEnabled(False)
                self.install_btn.setText("✓ Installed")
                self.scan_btn.setEnabled(False)
            return

    def _tool_checked(self, short: str) -> bool:
        for row in range(self.tools_table.rowCount()):
            if row < len(self._tools_meta) and self._tools_meta[row]["short"] == short:
                w = self.tools_table.cellWidget(row, 0)
                if w:
                    cb = w.layout().itemAt(0).widget()
                    if isinstance(cb, QCheckBox):
                        return cb.isChecked()
        return True


# ─────────────────────────────────────────────────────────────────────
# Server details dialog — per-tool toggles, env editor, refresh, backup
# ─────────────────────────────────────────────────────────────────────

class ServerDialog(QDialog):
    """Manage one installed server: per-tool toggles, env, refresh, rollback."""

    def __init__(self, parent, manager, name: str):
        super().__init__(parent)
        self._manager = manager
        self._name = name
        self.setWindowTitle(f"Server — {name}")
        self.setMinimumSize(560, 620)
        self.resize(640, 680)
        self.setStyleSheet(f"background: {BG}; color: {TEXT};")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)

        cfg = manager.load_config().get(name, {})
        status = manager.get_server_status(name)

        head = QHBoxLayout()
        title = QLabel(name)
        title.setFont(_font(15, True))
        title.setStyleSheet(f"color: {ACCENT};")
        head.addWidget(title)
        dot = QLabel("●")
        dot.setFont(_font(11))
        dot.setStyleSheet(f"color: {GREEN if status == 'running' else RED if status == 'error' else TEXT_MED};")
        head.addWidget(dot)
        head.addWidget(_label(status, TEXT_MED, 9))
        head.addStretch()
        lay.addLayout(head)

        if cfg.get("description"):
            lay.addWidget(_label(cfg["description"], TEXT_MED, 9))

        lay.addWidget(_label(f"Command: {cfg.get('command', 'npx')} "
                             f"{' '.join(str(a) for a in cfg.get('args', []))}",
                             TEXT_MED, 8))

        # ── Tools ──
        lay.addWidget(_label("Tools (toggle individual access):", TEXT, 10, True))
        self.tools_table = QTableWidget(0, 3)
        self.tools_table.setHorizontalHeaderLabels(["Enable", "Tool", "Access"])
        self.tools_table.setStyleSheet(f"""
            QTableWidget {{ background: {CARD}; border: 1px solid {BORDER};
                           gridline-color: {BORDER}; }}
            QTableWidget::item {{ padding: 3px 6px; }}
            QHeaderView::section {{ background: {BG}; color: {TEXT_MED};
                                   border: 1px solid {BORDER}; padding: 3px; }}
        """)
        self.tools_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.tools_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.tools_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.tools_table.verticalHeader().setVisible(False)
        self.tools_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        lay.addWidget(self.tools_table, stretch=1)

        self._load_tools()

        # ── Env vars ──
        env = cfg.get("env") or {}
        headers = cfg.get("headers") or {}
        if env or headers:
            lay.addWidget(_label("Environment variables:", TEXT, 10, True))
            env_form = QFormLayout()
            env_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
            self.env_inputs: dict[str, QLineEdit] = {}
            for key, val in env.items():
                edit = QLineEdit(str(val))
                secretish = any(s in key.upper() for s in ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASS"))
                if secretish:
                    edit.setEchoMode(QLineEdit.EchoMode.Password)
                edit.setStyleSheet(f"""
                    QLineEdit {{ background: {CARD}; color: {TEXT}; border: 1px solid {BORDER};
                                padding: 5px 8px; border-radius: 4px; }}
                """)
                self.env_inputs[key] = edit
                env_form.addRow(_label(key, TEXT_MED, 8, False), edit)
            for key, val in headers.items():
                edit = QLineEdit(str(val))
                edit.setEchoMode(QLineEdit.EchoMode.Password)
                edit.setStyleSheet(f"""
                    QLineEdit {{ background: {CARD}; color: {TEXT}; border: 1px solid {BORDER};
                                padding: 5px 8px; border-radius: 4px; }}
                """)
                self.env_inputs[f"__header__{key}"] = edit
                env_form.addRow(_label(f"{key} (header)", TEXT_MED, 8, False), edit)
            lay.addLayout(env_form)

        # ── Backups ──
        backups = manager.list_backups()
        if backups:
            bk_row = QHBoxLayout()
            bk_row.addWidget(_label("Config backups:", TEXT_MED, 8))
            self.bk_combo = QComboBox()
            self.bk_combo.addItems(backups)
            self.bk_combo.setStyleSheet(f"""
                QComboBox {{ background: {CARD}; color: {TEXT}; border: 1px solid {BORDER};
                            padding: 3px 6px; border-radius: 4px; }}
                QComboBox::drop-down {{ border: none; }}
                QComboBox QAbstractItemView {{ background: {CARD}; color: {TEXT};
                                              selection-background-color: {BORDER}; }}
            """)
            bk_row.addWidget(self.bk_combo, stretch=1)
            rollback_btn = _btn("Rollback", YELLOW)
            rollback_btn.clicked.connect(self._do_rollback)
            bk_row.addWidget(rollback_btn)
            lay.addLayout(bk_row)

        # ── Actions ──
        btn_row = QHBoxLayout()
        save_env_btn = _btn("Save Env", ACCENT, bold=True)
        save_env_btn.clicked.connect(self._save_env)
        btn_row.addWidget(save_env_btn)

        refresh_btn = _btn("Refresh Tools", ACCENT)
        refresh_btn.clicked.connect(self._do_refresh)
        btn_row.addWidget(refresh_btn)

        test_btn = _btn("Test Connection", GREEN)
        test_btn.clicked.connect(self._do_test)
        btn_row.addWidget(test_btn)

        btn_row.addStretch()

        uninstall_btn = _btn("Uninstall", RED)
        uninstall_btn.clicked.connect(self._do_uninstall)
        btn_row.addWidget(uninstall_btn)

        close_btn = _btn("Close", TEXT_MED)
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        lay.addLayout(btn_row)

        self.status = _label("", TEXT_MED, 9)
        lay.addWidget(self.status)

        self._refresh_out = {"done": False, "result": None}
        self._poll = QTimer(self)
        self._poll.timeout.connect(self._poll_tick)

    def _load_tools(self):
        metas = self._manager.get_server_tools_meta(self._name)
        self.tools_table.setRowCount(len(metas))
        for row, t in enumerate(metas):
            check = QCheckBox()
            check.setChecked(bool(t["enabled"]))
            check.stateChanged.connect(lambda st, n=t["name"]: self._on_toggle(n, st))
            check.setStyleSheet(f"""
                QCheckBox::indicator {{ width: 15px; height: 15px; }}
                QCheckBox::indicator:checked {{ background: {GREEN}; border-radius: 3px; }}
                QCheckBox::indicator:unchecked {{ background: {CARD};
                    border: 1px solid {BORDER}; border-radius: 3px; }}
            """)
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(6, 2, 6, 2)
            h.addWidget(check)
            h.addStretch()
            self.tools_table.setCellWidget(row, 0, w)
            item = QTableWidgetItem(t["short"])
            item.setToolTip(t["description"])
            self.tools_table.setItem(row, 1, item)
            access = "🔒 Read-only" if t["read_only"] else "✏️ Write"
            acc_item = QTableWidgetItem(access)
            acc_item.setForeground(Qt.GlobalColor.green if t["read_only"] else Qt.GlobalColor.yellow)
            self.tools_table.setItem(row, 2, acc_item)

    def _on_toggle(self, tool_name, state):
        msg = self._manager.set_tool_enabled(self._name, tool_name, bool(state))
        _append_log(msg)
        self.status.setText(msg)
        self.status.setStyleSheet(f"color: {GREEN if state else YELLOW};")

    def _save_env(self):
        cfg = self._manager.load_config().get(self._name)
        if not cfg:
            self.status.setText("Server not found")
            return
        env = {}
        headers = {}
        for key, edit in self.env_inputs.items():
            val = edit.text().strip()
            if not val:
                continue
            if key.startswith("__header__"):
                headers[key[len("__header__"):]] = val
            else:
                env[key] = val
        cfg["env"] = env or None
        cfg["headers"] = headers or None
        config = self._manager.load_config()
        config[self._name] = cfg
        self._manager.save_config(config)
        msg = self._manager.restart_server(self._name)
        _append_log(msg)
        self.status.setText(f"Env saved. {msg}")
        self.status.setStyleSheet(f"color: {GREEN};")

    def _do_refresh(self):
        self.status.setText("Refreshing tools…")
        self.status.setStyleSheet(f"color: {ACCENT};")

        def work():
            self._refresh_out = {"done": True, "result": self._manager.update_server(self._name)}

        threading.Thread(target=work, daemon=True).start()
        self._poll.start(120)

    def _do_test(self):
        result = self._manager.probe_server(self._name, force=True)
        if result["ok"]:
            self.status.setText(f"✓ Connection OK — {result['tools']} tools reachable")
            self.status.setStyleSheet(f"color: {GREEN};")
        else:
            self.status.setText(f"✗ Failed: {result['error']}")
            self.status.setStyleSheet(f"color: {RED};")
        _append_log(self.status.text())

    def _do_rollback(self):
        backup = self.bk_combo.currentText()
        if not backup:
            return
        reply = QMessageBox.question(self, "Rollback",
            f"Restore config from '{backup}'? This stops all servers.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            msg = self._manager.rollback_backup(backup)
            self.status.setText(msg)
            self.status.setStyleSheet(f"color: {YELLOW};")
            _append_log(msg)

    def _do_uninstall(self):
        reply = QMessageBox.question(self, "Uninstall",
            f"Remove '{self._name}' and its config?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            msg = self._manager.uninstall_server(self._name)
            _append_log(msg)
            self.accept()

    def _poll_tick(self):
        if self._refresh_out.get("done"):
            self._poll.stop()
            result = self._refresh_out.get("result") or {}
            self.status.setText(result.get("message", ""))
            self.status.setStyleSheet(
                f"color: {GREEN if result.get('ok') else RED};")
            _append_log(result.get("message", ""))
            self._load_tools()


# ─────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────

class McpManagerWindow(QMainWindow):
    """Main MCP Manager window with tabs for servers, browse, and logs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MCP Manager")
        self.setMinimumSize(940, 620)
        self.resize(1120, 720)
        self.setStyleSheet(f"background: {BG}; color: {TEXT};")

        central = QWidget()
        self.setCentralWidget(central)
        lay = QVBoxLayout(central)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)

        title = QLabel("MCP Control Center")
        title.setFont(_font(14, True))
        title.setStyleSheet(f"color: {ACCENT};")
        lay.addWidget(title)

        subtitle = _label(
            "Connect, inspect, and control your AI tools. Changes are backed up automatically.",
            TEXT_MED, 9)
        lay.addWidget(subtitle)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(f"""
            QTabWidget::pane {{ border: 1px solid {BORDER}; background: {BG}; }}
            QTabBar::tab {{ background: {CARD}; color: {TEXT_MED}; padding: 6px 18px;
                           border: 1px solid {BORDER}; border-bottom: none;
                           border-top-left-radius: 4px; border-top-right-radius: 4px; }}
            QTabBar::tab:selected {{ background: {BG}; color: {ACCENT}; }}
        """)
        lay.addWidget(self.tabs, stretch=1)

        self._servers_tab = _ServersTab()
        self._browse_tab = _BrowseTab()
        self._log_tab = _LogTab()

        self.tabs.addTab(self._servers_tab, "My servers")
        self.tabs.addTab(self._browse_tab, "Discover")
        self.tabs.addTab(self._log_tab, "Activity")

        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._tick)
        # Status changes are also pushed after each action. A relaxed fallback
        # refresh keeps idle CPU/GPU work negligible on older machines.
        self._refresh_timer.start(6000)

    def _tick(self):
        if self.tabs.currentWidget() is self._servers_tab:
            self._servers_tab.refresh()


class ServerCard(QFrame):
    """Small, responsive server control surface.  Lifecycle work stays in the tab."""

    CATEGORY_ICONS = {
        "web": "🌐", "browser": "🎭", "database": "🗄", "development": "💻",
        "ai": "🧠", "media": "🎬", "file": "📁", "desktop": "🖥", "social": "💬",
    }

    def __init__(self, tab, name: str, cfg: dict, status: str, error: str, tools: int, busy: bool):
        super().__init__()
        self._tab = tab
        self._name = name
        self.setObjectName("serverCard")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(142)
        self.setStyleSheet(f"""
            QFrame#serverCard {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 9px; }}
            QFrame#serverCard:hover {{ border-color: {ACCENT}; background: #1b2230; }}
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)

        head = QHBoxLayout()
        category = str(cfg.get("category", "Other"))
        icon = next((v for k, v in self.CATEGORY_ICONS.items() if k in category.lower()), "🔌")
        head.addWidget(_label(f"{icon}  {name}", TEXT, 10, True, False))
        head.addStretch()
        toggle = QCheckBox("Enabled")
        toggle.setChecked(bool(cfg.get("enabled", False)))
        toggle.setEnabled(not busy)
        toggle.setStyleSheet(f"QCheckBox {{ color: {TEXT_MED}; }}")
        toggle.toggled.connect(lambda _: tab._on_toggle(name))
        head.addWidget(toggle)
        lay.addLayout(head)

        state = "Working…" if busy else status.replace("_", " ").title()
        state_color = YELLOW if busy or status == "connecting" else (
            GREEN if status == "running" else RED if status == "error" else TEXT_MED)
        dot = "●" if not busy else "◌"
        detail = error or str(cfg.get("description", "No description provided."))
        lay.addWidget(_label(
            f"<span style='color:{state_color}'>{dot} {state}</span>  ·  {category}  ·  {tools} tools",
            TEXT_MED, 8))
        description = _label(detail, RED if error else TEXT_MED, 8)
        description.setToolTip(detail)
        description.setMaximumHeight(30)
        lay.addWidget(description)

        actions = QHBoxLayout()
        if status == "running" and not busy:
            action = _btn("Restart", YELLOW, size=8)
            action.clicked.connect(lambda: tab._on_restart(name))
        else:
            action = _btn("Start", GREEN, size=8)
            action.clicked.connect(lambda: tab._on_start(name))
        action.setEnabled(not busy)
        actions.addWidget(action)
        details = _btn("Manage tools", ACCENT, size=8)
        details.setEnabled(not busy)
        details.clicked.connect(lambda: tab._open_details_name(name))
        actions.addWidget(details)
        actions.addStretch()
        lay.addLayout(actions)


class _ServersTab(QWidget):
    """Tab showing all configured servers with toggles, status, and actions."""

    def __init__(self):
        super().__init__()
        self.setStyleSheet(f"background: {BG};")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        top_row = QHBoxLayout()
        refresh_btn = _btn("Refresh", ACCENT)
        refresh_btn.clicked.connect(self.refresh)
        top_row.addWidget(refresh_btn)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Filter servers, categories, or status…")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.setMinimumWidth(260)
        self.search_input.setStyleSheet(
            f"background: {CARD}; color: {TEXT}; border: 1px solid {BORDER}; "
            "border-radius: 5px; padding: 5px 8px;")
        self.search_input.textChanged.connect(self.refresh)
        top_row.addWidget(self.search_input)

        self.status_filter = QComboBox()
        self.status_filter.addItems(["All health", "Healthy", "Needs attention", "Disabled"])
        self.status_filter.setStyleSheet(
            f"background: {CARD}; color: {TEXT}; border: 1px solid {BORDER}; padding: 3px 6px;")
        self.status_filter.currentIndexChanged.connect(self.refresh)
        top_row.addWidget(self.status_filter)

        self.write_check = QCheckBox("Ask before write actions")
        self.write_check.setFont(_font(9))
        self.write_check.setStyleSheet(f"""
            QCheckBox {{ color: {TEXT_MED}; }}
            QCheckBox::indicator {{ width: 15px; height: 15px; }}
            QCheckBox::indicator:checked {{ background: {YELLOW}; border-radius: 3px; }}
            QCheckBox::indicator:unchecked {{ background: {CARD};
                border: 1px solid {BORDER}; border-radius: 3px; }}
        """)
        mgr = get_manager()
        self.write_check.setChecked(mgr.confirm_write())
        self.write_check.toggled.connect(
            lambda v: get_manager().save_settings(confirm_write=bool(v)))
        top_row.addWidget(self.write_check)

        top_row.addStretch()

        toggle_all_btn = _btn("Start All Disabled", GREEN)
        toggle_all_btn.clicked.connect(self._toggle_all_disabled)
        top_row.addWidget(toggle_all_btn)
        lay.addLayout(top_row)

        self.summary = QLabel()
        self.summary.setFont(_font(9))
        self.summary.setStyleSheet(
            f"background: {CARD}; border: 1px solid {BORDER}; border-radius: 6px; "
            f"color: {TEXT_MED}; padding: 7px 10px;")
        lay.addWidget(self.summary)

        utility_row = QHBoxLayout()
        export_btn = _btn("Export setup", TEXT_MED)
        export_btn.setToolTip("Export your server definitions. The file may contain credentials.")
        export_btn.clicked.connect(self._export_config)
        utility_row.addWidget(export_btn)
        import_btn = _btn("Import setup", TEXT_MED)
        import_btn.setToolTip("Review a saved MCP setup before adding its servers.")
        import_btn.clicked.connect(self._import_config)
        utility_row.addWidget(import_btn)
        utility_row.addStretch()
        lay.addLayout(utility_row)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Server", "Category", "Status", "Tools", "Actions", "API Key"])
        self.table.setStyleSheet(f"""
            QTableWidget {{ background: {CARD}; border: 1px solid {BORDER};
                           gridline-color: {BORDER}; }}
            QTableWidget::item {{ padding: 4px 8px; }}
            QHeaderView::section {{ background: {BG}; color: {TEXT_MED};
                                   border: 1px solid {BORDER}; padding: 4px; }}
        """)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.itemDoubleClicked.connect(lambda item: self._open_details(item.row()))
        # Kept as a hidden accessibility/fallback table while the primary view is
        # rendered as cards below. This also preserves existing double-click wiring.
        self.table.hide()

        self.cards_scroll = QScrollArea()
        self.cards_scroll.setWidgetResizable(True)
        self.cards_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.cards_scroll.setStyleSheet(f"QScrollArea {{ background: {BG}; }}")
        self.cards_host = QWidget()
        self.cards_host.setStyleSheet(f"background: {BG};")
        self.cards_layout = QVBoxLayout(self.cards_host)
        self.cards_layout.setContentsMargins(0, 2, 6, 2)
        self.cards_layout.setSpacing(8)
        self.cards_layout.addStretch()
        self.cards_scroll.setWidget(self.cards_host)
        lay.addWidget(self.cards_scroll, stretch=1)

        self._busy_names = set()
        self._jobs = []
        self._cards_signature = None

        self.refresh()

    def refresh(self):
        try:
            mgr = get_manager()
            self.table.clearSpans()
            config = mgr.get_all_servers_config()
            servers = [(k, v) for k, v in config.items() if not k.startswith("_")]
            servers.sort(key=lambda x: (0 if x[1].get("enabled", False) else 1, x[0]))
            total = len(servers)
            running = sum(mgr.get_server_status(name) == "running" for name, _ in servers)
            errors = sum(bool(mgr.get_last_error(name)) or mgr.get_server_status(name) == "error"
                         for name, _ in servers)
            enabled = sum(bool(cfg.get("enabled", False)) for _, cfg in servers)
            self.summary.setText(
                f"{total} configured  ·  {enabled} enabled  ·  "
                f"<span style='color:{GREEN}'>{running} connected</span>  ·  "
                f"<span style='color:{RED}'>{errors} need attention</span>")

            query = self.search_input.text().strip().lower()
            selection = self.status_filter.currentText()
            def visible(entry):
                name, cfg = entry
                status = mgr.get_server_status(name)
                broken = bool(mgr.get_last_error(name)) or status == "error"
                searchable = f"{name} {cfg.get('category', '')} {status}".lower()
                if query and query not in searchable:
                    return False
                if selection == "Healthy":
                    return status == "running" and not broken
                if selection == "Needs attention":
                    return broken
                if selection == "Disabled":
                    return not cfg.get("enabled", False)
                return True
            servers = [entry for entry in servers if visible(entry)]
            # The old table is retained only for backward-compatible object access;
            # avoid constructing dozens of hidden widgets every refresh.
            self.table.setRowCount(0)
            signature = (
                self.search_input.text(), self.status_filter.currentText(),
                tuple((name, bool(cfg.get("enabled", False)), cfg.get("category", ""),
                       mgr.get_server_status(name), mgr.get_last_error(name),
                       len(mgr.get_server_tools_meta(name)), name in self._busy_names)
                      for name, cfg in servers),
            )
            if signature != self._cards_signature:
                self._cards_signature = signature
                self._render_cards(servers, mgr)
            return

        except Exception as e:
            self.table.setRowCount(1)
            self.table.setItem(0, 0, QTableWidgetItem(f"Error: {e}"))
            self._render_error(str(e))

    def _clear_cards(self):
        while self.cards_layout.count():
            item = self.cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _render_cards(self, servers, mgr):
        self._clear_cards()
        if not servers:
            self.cards_layout.addWidget(_label(
                "No servers match this filter. Try All health, clear the search, or discover a new server.",
                TEXT_MED, 10))
        for name, cfg in servers:
            self.cards_layout.addWidget(ServerCard(
                self, name, cfg, mgr.get_server_status(name), mgr.get_last_error(name),
                len(mgr.get_server_tools_meta(name)), name in self._busy_names))
        self.cards_layout.addStretch()

    def _render_error(self, error):
        self._clear_cards()
        self.cards_layout.addWidget(_label(f"Could not render MCP servers: {error}", RED, 10))
        self.cards_layout.addStretch()

    def _run_action(self, name, label, work):
        """Run lifecycle calls away from the Qt event loop to prevent freezes."""
        if name in self._busy_names:
            return
        self._busy_names.add(name)
        self.refresh()
        job = {"done": False, "result": "", "error": None}
        self._jobs.append(job)

        def runner():
            try:
                job["result"] = work()
            except Exception as exc:
                job["error"] = str(exc)
            finally:
                job["done"] = True

        threading.Thread(target=runner, daemon=True, name=f"mcp-{label}-{name}").start()
        poll = QTimer(self)
        poll.setInterval(120)

        def finish():
            if not job["done"]:
                return
            poll.stop()
            poll.deleteLater()
            self._busy_names.discard(name)
            self._jobs.remove(job)
            message = job["result"] or f"[MCP] {label} failed for '{name}': {job['error']}"
            _append_log(message)
            self.refresh()

        poll.timeout.connect(finish)
        poll.start()

    def _on_toggle(self, name):
        self._run_action(name, "toggle", lambda: get_manager().toggle_server(name))

    def _on_start(self, name):
        self._run_action(name, "start", lambda: get_manager().start_server(name))

    def _on_restart(self, name):
        self._run_action(name, "restart", lambda: get_manager().restart_server(name))

    def _on_uninstall(self, name):
        reply = QMessageBox.question(self, "Remove Server",
            f"Remove '{name}' from config?", QMessageBox.StandardButton.Yes |
            QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self._run_action(name, "remove", lambda: get_manager().uninstall_server(name))

    def _open_details(self, row):
        item = self.table.item(row, 0)
        if item and item.data(Qt.ItemDataRole.UserRole):
            self._open_details_name(item.data(Qt.ItemDataRole.UserRole))

    def _open_details_name(self, name):
        dlg = ServerDialog(self, get_manager(), name)
        dlg.exec()

    def _toggle_all_disabled(self):
        names = [name for name, cfg in get_manager().get_all_servers_config().items()
                 if not name.startswith("_") and not cfg.get("enabled", False)]
        if not names:
            return
        # A single worker keeps process/resource pressure predictable while Qt stays responsive.
        self._run_action("bulk-start", "start", lambda: self._start_disabled(names))

    @staticmethod
    def _start_disabled(names):
        mgr = get_manager()
        results = [mgr.toggle_server(name) for name in names]
        return f"[MCP] Started {len(results)} previously disabled servers"

    def _export_config(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export MCP setup", str(BASE_DIR / "mcp-setup.json"), "JSON files (*.json)")
        if not path:
            return
        config = get_manager().get_all_servers_config()
        try:
            Path(path).write_text(json.dumps(config, indent=2), encoding="utf-8")
            _append_log(f"[MCP] Exported {len(config)} server definitions")
            QMessageBox.information(self, "Setup exported",
                "Your MCP setup was exported. Keep this file private: it may include API keys.")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", f"Could not save the setup:\n{exc}")

    def _import_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import MCP setup", "", "JSON files (*.json)")
        if not path:
            return
        try:
            incoming = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(incoming, dict) or not all(isinstance(v, dict) for v in incoming.values()):
                raise ValueError("The file must contain a JSON object of named server configurations.")
            incoming = {k: v for k, v in incoming.items() if not k.startswith("_")}
            if not incoming:
                raise ValueError("No server configurations found.")
            mgr = get_manager()
            current = mgr.get_all_servers_config()
            existing = sorted(set(current) & set(incoming))
            message = f"Import {len(incoming)} server(s)?"
            if existing:
                message += "\n\nExisting servers will be replaced: " + ", ".join(existing)
            message += "\n\nImported servers stay disabled until you explicitly start them."
            reply = QMessageBox.question(self, "Review import", message,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return
            for cfg in incoming.values():
                cfg["enabled"] = False
            current.update(incoming)
            if not mgr.save_config(current):
                raise RuntimeError("The configuration could not be saved.")
            _append_log(f"[MCP] Imported {len(incoming)} server definitions (disabled for review)")
            self.refresh()
            QMessageBox.information(self, "Setup imported",
                "Imported servers are disabled for safety. Review credentials, then start the ones you trust.")
        except Exception as exc:
            QMessageBox.warning(self, "Import failed", f"Could not import this setup:\n{exc}")


class _BrowseTab(QWidget):
    """Tab for searching and installing MCP servers from registries."""

    QUICK = [
        ("GitHub", "@modelcontextprotocol/server-github", "Version Control", "npx",
         {"GITHUB_PERSONAL_ACCESS_TOKEN": "YOUR_GITHUB_TOKEN_HERE"}),
        ("Playwright", "@modelcontextprotocol/server-playwright", "Browser", "npx", None),
        ("Brave Search", "@modelcontextprotocol/server-brave-search", "Web Search", "npx",
         {"BRAVE_API_KEY": "YOUR_BRAVE_API_KEY_HERE"}),
        ("PostgreSQL", "@modelcontextprotocol/server-postgres", "Database", "npx",
         {"DATABASE_URL": "postgresql://user:pass@localhost:5432/dbname"}),
        ("Context7", "@upstash/context7-mcp", "Development", "npx",
         {"CONTEXT7_API_KEY": "YOUR_API_KEY_HERE"}),
        ("Firecrawl", "firecrawl-mcp", "Web Scraping", "npx",
         {"FIRECRAWL_API_KEY": "YOUR_API_KEY_HERE"}),
        ("Exa Search", "exa-mcp-server", "Web Search", "npx",
         {"EXA_API_KEY": "YOUR_API_KEY_HERE"}),
        ("MiniMax", "minimax-mcp", "AI & Media", "npx",
         {"MINIMAX_API_KEY": "YOUR_API_KEY_HERE", "MINIMAX_GROUP_ID": "YOUR_MINIMAX_GROUP_ID_HERE"}),
        ("Sequential Thinking", "@modelcontextprotocol/server-sequential-thinking",
         "AI & Thinking", "npx", None),
        ("SQLite", "mcp-server-sqlite", "Database", "uvx", None),
        ("Filesystem", "@modelcontextprotocol/server-filesystem", "Development", "npx",
         {"ALLOWED_PATH": r"C:\Users\harsh"}),
    ]

    def __init__(self):
        super().__init__()
        self.setStyleSheet(f"background: {BG};")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        search_row = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search MCP servers… (e.g. postgres, github, browser)")
        self.search_input.setStyleSheet(f"""
            QLineEdit {{ background: {CARD}; color: {TEXT}; border: 1px solid {BORDER};
                        padding: 6px 10px; border-radius: 4px; }}
        """)
        self.search_input.textChanged.connect(self._on_text_changed)
        search_row.addWidget(self.search_input, stretch=1)

        self.search_btn = _btn("Search", BG, bg=ACCENT, size=9, bold=True, padding="6px 18px")
        self.search_btn.clicked.connect(self._do_search)
        search_row.addWidget(self.search_btn)

        self.source_filter = QComboBox()
        self.source_filter.addItems(["All", "Official", "Smithery", "npm", "PyPI", "Popular"])
        self.source_filter.setStyleSheet(f"""
            QComboBox {{ background: {CARD}; color: {TEXT}; border: 1px solid {BORDER};
                        padding: 4px 8px; border-radius: 4px; }}
            QComboBox::drop-down {{ border: none; }}
            QComboBox QAbstractItemView {{ background: {CARD}; color: {TEXT};
                                          selection-background-color: {BORDER}; }}
        """)
        self.source_filter.currentIndexChanged.connect(self._apply_filters)
        search_row.addWidget(self.source_filter)
        lay.addLayout(search_row)

        key_row = QHBoxLayout()
        key_lbl = _label("Smithery key:", TEXT_MED, 8, False)
        key_row.addWidget(key_lbl)
        self.key_input = QLineEdit()
        self.key_input.setPlaceholderText("free key — smithery.ai/account/api-keys")
        self.key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_input.setText(get_manager().get_smithery_key())
        self.key_input.setStyleSheet(f"""
            QLineEdit {{ background: {CARD}; color: {TEXT}; border: 1px solid {BORDER};
                        padding: 4px 8px; border-radius: 4px; }}
        """)
        key_row.addWidget(self.key_input, stretch=1)
        self.key_eye_btn = QPushButton("👁")
        self.key_eye_btn.setFixedSize(26, 24)
        self.key_eye_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.key_eye_btn.setToolTip("Show / hide the API key")
        self.key_eye_btn.setStyleSheet(f"""
            QPushButton {{ background: {CARD}; color: {TEXT_MED};
                           border: 1px solid {BORDER}; border-radius: 4px; }}
            QPushButton:hover {{ color: {TEXT}; border-color: {ACCENT}; }}
        """)
        self.key_eye_btn.setCheckable(True)
        self.key_eye_btn.toggled.connect(self._toggle_key_visibility)
        key_row.addWidget(self.key_eye_btn)
        self.key_save_btn = _btn("Save", GREEN, size=8)
        self.key_save_btn.clicked.connect(self._save_smithery_key)
        key_row.addWidget(self.key_save_btn)
        key_hint = _label("Unlocks search of 8,000+ Smithery hosted servers", TEXT_MED, 8)
        key_row.addWidget(key_hint)
        lay.addLayout(key_row)

        info = QLabel("Type to search live — results appear as you type. "
                      "Double-click a result or press Install for the setup dialog.")
        info.setFont(_font(8))
        info.setStyleSheet(f"color: {TEXT_MED};")
        lay.addWidget(info)
        self.intent_hint = _label("Tip: search naturally — PDF parsing, browser automation, GitHub, or databases.",
                                  PURPLE, 8)
        lay.addWidget(self.intent_hint)

        self.results_table = QTableWidget(0, 5)
        self.results_table.setHorizontalHeaderLabels(["Server", "Description", "Source", "Quality", "Install"])
        self.results_table.setStyleSheet(f"""
            QTableWidget {{ background: {CARD}; border: 1px solid {BORDER};
                           gridline-color: {BORDER}; }}
            QTableWidget::item {{ padding: 4px 8px; }}
            QHeaderView::section {{ background: {BG}; color: {TEXT_MED};
                                   border: 1px solid {BORDER}; padding: 4px; }}
        """)
        self.results_table.horizontalHeader().setStretchLastSection(False)
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.results_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.itemDoubleClicked.connect(lambda item: self._open_install(item.row()))
        lay.addWidget(self.results_table, stretch=1)

        bottom_row = QHBoxLayout()
        self.load_more_btn = _btn("Load more…", ACCENT, size=9, bold=True, padding="5px 16px")
        self.load_more_btn.clicked.connect(self._load_more)
        self.load_more_btn.hide()
        bottom_row.addWidget(self.load_more_btn)
        self.results_label = _label("", TEXT_MED, 8, wrap=False)
        bottom_row.addWidget(self.results_label)
        bottom_row.addStretch()
        lay.addLayout(bottom_row)

        quick_label = QLabel("Quick Install — one click opens setup, tools pre-filled")
        quick_label.setFont(_font(10, True))
        quick_label.setStyleSheet(f"color: {TEXT};")
        lay.addWidget(quick_label)

        quick_grid = QHBoxLayout()
        for label, pkg, cat, cmd, env in self.QUICK:
            btn = QPushButton(label)
            btn.setToolTip(f"{pkg} ({cat}) — {cmd}")
            btn.setStyleSheet(f"""
                QPushButton {{ background: {CARD}; color: {ACCENT}; border: 1px solid {BORDER};
                              padding: 4px 10px; border-radius: 4px; font-size: 8pt; }}
                QPushButton:hover {{ background: {BORDER}; }}
            """)
            btn.clicked.connect(
                lambda _, p=pkg, l=label, c=cat, m=cmd, e=env:
                    self._quick_install(l, p, c, m, e))
            quick_grid.addWidget(btn)
        lay.addLayout(quick_grid)

        self._results: list[dict] = []
        self._page = 1
        self._has_more = False
        self._loading_more = False
        self._search_out = {"done": True, "results": [], "page": 1, "has_more": False}
        self._more_out = {"done": False, "results": [], "page": 1, "has_more": False}
        self._poll = QTimer(self)
        self._poll.timeout.connect(self._poll_tick)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._do_search)

        self._do_search()

    # ── live search ──────────────────────────────────────────────────

    def _on_text_changed(self):
        self._debounce.start(350)

    def _do_search(self, force: bool = False):
        query = self.search_input.text().strip()
        registry_query, hint = self._smart_query(query)
        self.intent_hint.setText(hint)
        self._page = 1
        self._has_more = False
        self._search_btn_state(True)
        self._search_out = {"done": False, "results": [], "page": 1, "has_more": False}

        def work():
            try:
                mgr = get_manager()
                res = mgr.search_registry_page(registry_query, 1, force=force)
            except Exception as e:
                print(f"[MCP] Search error: {e}")
                res = {"results": [], "page": 1, "has_more": False}
            self._search_out = {
                "done": True,
                "results": res.get("results") or [],
                "page": int(res.get("page", 1)),
                "has_more": bool(res.get("has_more")),
            }

        threading.Thread(target=work, daemon=True).start()
        self._poll.start(100)

    @staticmethod
    def _smart_query(query: str) -> tuple[str, str]:
        """Local intent matching makes natural Hindi/English searches useful offline."""
        text = query.lower()
        intents = (
            (("pdf", "parse", "document"), "pdf", "Smart search: showing PDF and document tools."),
            (("browser", "chrome", "automation", "website"), "browser", "Smart search: showing browser automation tools."),
            (("github", "git", "repo", "pull request"), "github", "Smart search: showing GitHub and code tools."),
            (("database", "postgres", "sql", "sqlite"), "database", "Smart search: showing database tools."),
            (("search", "research", "internet"), "search", "Smart search: showing web-search tools."),
            (("file", "folder", "filesystem"), "filesystem", "Smart search: showing file-system tools."),
        )
        for words, keyword, hint in intents:
            if any(word in text for word in words):
                return keyword, hint
        return query, "Tip: search naturally — PDF parsing, browser automation, GitHub, or databases."

    def _toggle_key_visibility(self, shown: bool):
        self.key_input.setEchoMode(
            QLineEdit.EchoMode.Normal if shown else QLineEdit.EchoMode.Password)

    def _save_smithery_key(self):
        val = self.key_input.text().strip()
        mgr = get_manager()
        mgr.save_settings(smithery_api_key=val)
        _append_log(f"[MCP] Smithery API key {'saved' if val else 'removed'}")
        self.key_save_btn.setText("✓ Saved" if val else "Removed")
        QTimer.singleShot(1500, lambda: self.key_save_btn.setText("Save"))
        self._do_search(force=True)

    def _load_more(self):
        if self._loading_more or not self._has_more:
            return
        self._loading_more = True
        self.load_more_btn.setEnabled(False)
        self.load_more_btn.setText("Loading…")
        query = self.search_input.text().strip()
        next_page = self._page + 1
        self._more_out = {"done": False, "results": [], "page": next_page, "has_more": False}

        def work():
            try:
                mgr = get_manager()
                res = mgr.search_registry_page(query, next_page)
            except Exception as e:
                print(f"[MCP] Load-more error: {e}")
                res = {"results": [], "page": next_page, "has_more": False}
            self._more_out = {
                "done": True,
                "results": res.get("results") or [],
                "page": int(res.get("page", next_page)),
                "has_more": bool(res.get("has_more")),
            }

        threading.Thread(target=work, daemon=True).start()
        self._poll.start(100)

    def _update_load_more(self):
        self.load_more_btn.setText("Loading…" if self._loading_more else "Load more…")
        self.load_more_btn.setEnabled(self._has_more and not self._loading_more)
        self.load_more_btn.setVisible(self._has_more)

    def _search_btn_state(self, busy: bool):
        self.search_btn.setText("Searching…" if busy else "Search")
        self.search_btn.setEnabled(not busy)

    def _poll_tick(self):
        if self._more_out.get("done"):
            self._poll.stop()
            self._loading_more = False
            more = self._more_out
            self._more_out = {"done": False, "results": [], "page": 1, "has_more": False}
            existing = {r["name"] for r in self._results}
            for r in more.get("results") or []:
                if r.get("name") and r["name"] not in existing:
                    self._results.append(r)
            self._page = int(more.get("page", self._page))
            self._has_more = bool(more.get("has_more"))
            self._update_load_more()
            self._apply_filters()
            return

        if self._search_out.get("done"):
            self._poll.stop()
            self._search_btn_state(False)
            self._results = self._search_out.get("results") or []
            self._page = int(self._search_out.get("page", 1))
            self._has_more = bool(self._search_out.get("has_more"))
            self._update_load_more()
            self._apply_filters()

    def _apply_filters(self):
        src_filter = self.source_filter.currentText().lower()
        mgr = get_manager()
        installed = set(mgr.get_all_servers_config().keys())
        filtered = [
            r for r in self._results
            if src_filter == "all" or r.get("source", "").lower() == src_filter
        ]
        self.results_label.setText(f"{len(self._results)} results across 4 sources")
        self.results_table.setRowCount(len(filtered))
        for row, r in enumerate(filtered):
            name = r.get("name", "")
            name_item = QTableWidgetItem(name)
            name_item.setFont(_font(9, True))
            if name in installed:
                name_item.setText(f"✓ {name}")
                name_item.setForeground(Qt.GlobalColor.green)
                name_item.setToolTip("Already installed — double-click to manage")
            self.results_table.setItem(row, 0, name_item)

            desc_item = QTableWidgetItem((r.get("description", "") or "")[:110])
            desc_item.setFont(_font(8))
            desc_item.setForeground(Qt.GlobalColor.gray)
            self.results_table.setItem(row, 1, desc_item)

            src = r.get("source", "?")
            src_item = QTableWidgetItem(src.upper() if src != "popular" else "★ POPULAR")
            src_item.setFont(_font(7, True))
            if src == "pypi":
                src_item.setForeground(Qt.GlobalColor.magenta)
            elif src == "npm":
                src_item.setForeground(Qt.GlobalColor.cyan)
            elif src == "smithery":
                src_item.setForeground(QColor(PURPLE))
            else:
                src_item.setForeground(Qt.GlobalColor.yellow)
            self.results_table.setItem(row, 2, src_item)

            downloads = r.get("downloads")
            if r.get("verified"):
                qual = "✓ VERIFIED"
            elif downloads is not None:
                qual = f"▲ {_compact_downloads(downloads)}/mo"
            else:
                qual = "—"
            qual_item = QTableWidgetItem(qual)
            qual_item.setFont(_font(8))
            qual_item.setForeground(Qt.GlobalColor.green if r.get("verified") else Qt.GlobalColor.yellow)
            self.results_table.setItem(row, 3, qual_item)

            if name in installed:
                manage_btn = _btn("Manage", GREEN, size=8)
                manage_btn.clicked.connect(lambda _, n=name: self._open_manage(n))
            else:
                manage_btn = _btn("Install", BG, bg=ACCENT, size=8, bold=True)
                manage_btn.clicked.connect(lambda _, r=r: self._open_install_result(r))
            self.results_table.setCellWidget(row, 4, manage_btn)

    # ── install flows ────────────────────────────────────────────────

    def _open_install(self, row):
        if row < len(self._results):
            self._open_install_result(self._results[row])

    def _open_install_result(self, result: dict):
        dlg = InstallDialog(self, get_manager(), result)
        dlg.exec()
        self._apply_filters()

    def _open_manage(self, name: str):
        dlg = ServerDialog(self, get_manager(), name)
        dlg.exec()

    def _quick_install(self, label: str, package: str, category: str,
                       command: str = "npx", env: dict | None = None):
        args = [package] if command == "uvx" else ["-y", package]
        prefill = {
            "name": package,
            "description": f"MCP server: {package} ({label})",
            "category": category,
            "command": command,
            "args": args,
            "source": "popular",
            "env": env or {},
            "auto_scan": True,
        }
        dlg = InstallDialog(self, get_manager(), {}, prefill=prefill)
        dlg.exec()
        self._apply_filters()


class _LogTab(QWidget):
    """Tab showing MCP-related log messages."""

    def __init__(self):
        super().__init__()
        self.setStyleSheet(f"background: {BG};")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        top = QHBoxLayout()
        clear_btn = _btn("Clear", TEXT_MED)
        clear_btn.clicked.connect(self._clear)
        top.addWidget(clear_btn)
        top.addStretch()
        lay.addLayout(top)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet(f"""
            QTextEdit {{ background: {CARD}; color: {TEXT}; border: 1px solid {BORDER};
                        font-family: 'Cascadia Code', 'Cascadia Mono', Consolas; font-size: 9pt; }}
        """)
        _log_tab_instance.set_log_widget(self.log_area)
        lay.addWidget(self.log_area, stretch=1)

    def _clear(self):
        self.log_area.clear()


class _LogCollector:
    """Collects log messages and sends them to the log tab."""

    def __init__(self):
        self._widget = None
        self._buffer = []

    def set_log_widget(self, widget):
        self._widget = widget
        if widget is None:
            return
        for msg in self._buffer:
            widget.append(msg)
        self._buffer.clear()

    def append(self, msg: str):
        if self._widget is not None:
            try:
                self._widget.append(msg)
                return
            except RuntimeError:
                # Underlying C++ object deleted (window closed).
                self._widget = None
        self._buffer.append(msg)


_log_tab_instance = _LogCollector()


def _append_log(msg: str):
    _log_tab_instance.append(msg)


_MGR_SINGLETON = None
_MGR_LOCK = threading.Lock()


def get_manager(refresh: bool = False):
    """Return the shared process-wide MCP manager."""
    global _MGR_SINGLETON
    if _MGR_SINGLETON is None:
        with _MGR_LOCK:
            if _MGR_SINGLETON is None:
                _MGR_SINGLETON = get_global_manager()
    return _MGR_SINGLETON


def open_mcp_manager(parent=None):
    """Open the MCP Manager window. If one exists, bring it to front."""
    if not hasattr(open_mcp_manager, "_window") or open_mcp_manager._window is None:
        window = McpManagerWindow(parent)
        open_mcp_manager._window = window
        # Detach the shared log sink when the window goes away so appends
        # never target a destroyed QTextEdit.
        try:
            window.destroyed.connect(
                lambda *_: _log_tab_instance.set_log_widget(None))
        except Exception:
            pass
    window = open_mcp_manager._window
    window.show()
    window.raise_()
    window.activateWindow()
    return window
