"""Real regression tests for the 2026 UI modernization.

Covers the three shipped changes:
1. Activity Log free-resize (no artificial width cap, persisted splitter).
2. The typography system (lazy font resolution — ``import ui`` must work
   even before a QApplication exists, and helpers must resolve real
   installed families afterwards).
3. Refined-cyber chat colors that keep every role visually distinct.

These tests boot real Qt widgets offscreen; they fail if any of the
above regresses.
"""

import os
import subprocess
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

import pytest


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ──────────────────────────────────────────────────────────────────────
# 1. Import safety (the QFontDatabase-at-import crash)
# ──────────────────────────────────────────────────────────────────────

def test_import_ui_without_qapplication_does_not_crash():
    """Bare ``import ui`` must never touch QFontDatabase (needs QGuiApplication).

    main.py imports ui before OrthosUI() builds the QApplication, so this
    runs in a clean subprocess to reproduce that exact ordering.
    """
    code = (
        "import os, sys; os.environ['QT_QPA_PLATFORM']='offscreen';"
        f"sys.path.insert(0, {PROJECT_ROOT!r});"
        "import ui; print('IMPORT-NO-APP-OK')"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, timeout=90, cwd=PROJECT_ROOT)
    assert r.returncode == 0, f"import ui crashed:\n{r.stderr[-2000:]}"
    assert "IMPORT-NO-APP-OK" in r.stdout
    assert "QFontDatabase" not in r.stderr


def test_import_main_module_does_not_crash():
    """``import main`` (the app's import graph) must stay clean offscreen."""
    code = (
        "import os, sys; os.environ['QT_QPA_PLATFORM']='offscreen';"
        f"sys.path.insert(0, {PROJECT_ROOT!r});"
        "import main; print('MAIN-IMPORT-OK')"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, timeout=180, cwd=PROJECT_ROOT)
    assert r.returncode == 0, f"import main crashed:\n{r.stderr[-2000:]}"
    assert "MAIN-IMPORT-OK" in r.stdout
    assert "QFontDatabase" not in r.stderr


# ──────────────────────────────────────────────────────────────────────
# 2. Typography system
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_font_helpers_resolve_real_installed_families(qapp):
    import ui
    from PyQt6.QtGui import QFontDatabase

    installed = set(QFontDatabase.families())
    # The offscreen QPA may report an empty family list; the helpers then
    # deliberately fall back to hard-coded faces instead of crashing.
    acceptable = installed if installed else {
        "Segoe UI", "Segoe UI Semibold", "Segoe UI Variable Text",
        "Segoe UI Variable Display", "Cascadia Code", "Cascadia Mono",
        "Consolas",
    }
    for family in (ui.font_family_ui(), ui.font_family_mono()):
        assert isinstance(family, str) and family
        assert family in acceptable, f"{family!r} is neither installed nor a known fallback"


def test_font_helpers_return_requested_sizes_and_weight(qapp):
    import ui

    f = ui.font_ui(10, bold=True)
    assert f.pointSize() == 10 and f.bold()
    m = ui.font_mono(9.5)
    # Python round(): banker's rounding — 9.5 -> 10, 8.5 -> 8.
    assert m.pointSize() == 10
    assert ui.font_mono(8.5).pointSize() == 8
    d = ui.font_ui(12, display=True)
    assert d.pointSize() == 12


def test_no_courier_new_remains_in_ui(qapp):
    """The retro face is fully retired from ui.py and native_chat.py."""
    for rel in ("ui.py", os.path.join("core", "native_chat.py")):
        path = os.path.join(PROJECT_ROOT, rel)
        text = open(path, encoding="utf-8").read()
        assert "QFont(\"Courier New\"" not in text, f"{rel} still hard-codes Courier New"
        assert "'Courier New', Consolas" not in text, f"{rel} stylesheet still prefers Courier New"


# ──────────────────────────────────────────────────────────────────────
# 3. Activity Log free-resize
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def main_window(qapp):
    import ui
    win = ui.MainWindow("nonexistent.png")
    # Offscreen windows need explicit show + resize for real splitter math.
    win.resize(1600, 900)
    win.show()
    qapp.processEvents()
    yield win
    win.close()


def test_activity_log_has_no_upper_width_cap(main_window):
    """Right panel must be freely resizable beyond the old 900px cap."""
    assert main_window._right_panel.maximumWidth() >= 16777215


def test_activity_log_resizes_wide_via_splitter(main_window):
    """Dragging the splitter to ~75% window width must actually apply."""
    total = main_window._chat_splitter.width()
    assert total >= 1200, f"splitter too small to test: {total}px (widget not laid out?)"
    want = int(total * 0.75)
    main_window._chat_splitter.setSizes([total - want, want])
    sizes = main_window._chat_splitter.sizes()
    assert sizes[1] >= want - 40, f"right panel stuck at {sizes[1]}px (cap regression?)"


def test_activity_panel_minimum_width_allows_narrow(main_window):
    assert main_window._right_panel.minimumWidth() <= 260


# ──────────────────────────────────────────────────────────────────────
# 4. Refined-cyber chat colors — roles stay distinct
# ──────────────────────────────────────────────────────────────────────

def _render_all_roles():
    from core.native_chat import NativeRichLogWidget
    chat = NativeRichLogWidget()
    chat.set_logs([
        "You: hello there",
        "Orthos: hi **bold**",
        "SYS: system ready",
        "SYS: \u25b6 search_tools ran",
        "ERR: something exploded",
    ])
    html = chat.document().toHtml()
    return chat, html


def test_all_role_colors_render_and_stay_distinct(qapp):
    chat, _ = _render_all_roles()
    # 2026 flat palette: every role keeps its own accent (bubble tint for
    # you, label color for ai, row accents for ops roles, red for errors).
    cards = [chat._card_html(m) for m in chat._messages]
    by_tag = {m["tag"]: c for m, c in zip(chat._messages, cards)}
    assert "background-color:#0d2f42" in by_tag["you"]
    assert "#4dd8ff" in by_tag["ai"]
    assert "#d6a019" in by_tag["sys"]
    assert "#00a877" in by_tag["tool"]
    assert "#e8496a" in by_tag["err"]


def test_role_texts_render(qapp):
    chat, _ = _render_all_roles()
    plain = chat.toPlainText()
    # Flat 2026: bubbles carry no label; the AI keeps its name label and
    # operational rows keep their source prefixes (SYS/TOOL).
    for expected in ("hello there", "Orthos", "system ready", "search_tools"):
        assert expected in plain, f"role content {expected!r} missing"


def test_old_role_colors_are_gone(qapp):
    _, html = _render_all_roles()
    for legacy in ("#1680a0", "#00bde8", "#c99300", "#00b875", "#ff4262"):
        assert legacy.lower() not in html.lower(), f"legacy color {legacy} still present"
