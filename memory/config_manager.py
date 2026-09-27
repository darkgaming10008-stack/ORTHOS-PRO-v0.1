import json
import sys
from pathlib import Path


def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR    = get_base_dir()
CONFIG_DIR  = BASE_DIR / "config"
CONFIG_FILE = CONFIG_DIR / "api_keys.json"


def ensure_config_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def config_exists() -> bool:
    return CONFIG_FILE.exists()


def _read_json_no_bom(path) -> str:
    """Read text tolerating a UTF-8 BOM (some Windows PowerShell writers add one).

    Also self-heals: when a BOM is found, the file is rewritten BOM-free so
    other, non-tolerant readers (actions/*.py read raw utf-8) never see it.
    """
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
        try:
            path.write_bytes(raw)
            print("[Config] Stripped UTF-8 BOM from", path.name)
        except Exception:
            pass
    return raw.decode("utf-8", errors="replace")


def save_config(cfg: dict) -> None:
    ensure_config_dir()
    existing: dict = {}
    if CONFIG_FILE.exists():
        try:
            existing = json.loads(_read_json_no_bom(CONFIG_FILE))
        except Exception:
            existing = {}
    existing.update(cfg)
    CONFIG_FILE.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def load_api_keys() -> dict:
    if not CONFIG_FILE.exists():
        return {}
    try:
        return json.loads(_read_json_no_bom(CONFIG_FILE))
    except Exception as e:
        print(f"❌ Failed to load api_keys.json: {e}")
        return {}


def is_configured() -> bool:
    cfg = load_api_keys()
    return (
        bool(cfg.get("os_system")) and
        bool(cfg.get("llm_model")) and
        bool(cfg.get("stt_engine")) and
        bool(cfg.get("tts_engine"))
    )
