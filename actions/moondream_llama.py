from __future__ import annotations
import base64
import io
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests as _requests
from PIL import Image

_HTTP = _requests.Session()


def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


_BASE = _base_dir()
_CONFIG_PATH = _BASE / "config" / "api_keys.json"

CLOUD_API_BASE = "https://api.moondream.ai/v1"


def _load_config() -> dict:
    try:
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


class MoondreamDetector:
    def __init__(self):
        self._error: Optional[str] = None
        self._loaded = False
        self._api_key: Optional[str] = None

        cfg = _load_config()
        key = (cfg.get("moondream_api_key") or "").strip()
        if key:
            self._api_key = key
            self._loaded = True
        else:
            self._error = "No moondream_api_key in config/api_keys.json"

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def error(self) -> Optional[str]:
        return self._error

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "User-Agent": "MarkXL/1.0",
            "Authorization": f"Bearer {self._api_key}",
        }

    def _native_headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "User-Agent": "MarkXL/1.0",
            "X-Moondream-Auth": self._api_key,
        }

    @staticmethod
    def _to_pil(image):
        if isinstance(image, str):
            return Image.open(image)
        if isinstance(image, bytes):
            return Image.open(io.BytesIO(image))
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, np.ndarray):
            return Image.fromarray(image)
        raise TypeError(f"Unsupported image type: {type(image)}")

    @staticmethod
    def _pil_to_b64(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def locate(self, image, description: str) -> tuple[int, int] | None:
        if not self._loaded:
            return None
        img = self._to_pil(image)
        b64 = self._pil_to_b64(img)
        w, h = img.size
        try:
            data_url = f"data:image/png;base64,{b64}"
            resp = _HTTP.post(
                f"{CLOUD_API_BASE}/point",
                json={"image_url": data_url, "object": description},
                headers=self._native_headers(),
                timeout=120,
            )
            resp.raise_for_status()
            result = resp.json()
            points = result.get("points", [])
            if points:
                return (int(points[0]["x"] * w), int(points[0]["y"] * h))
        except Exception as e:
            print(f"[Moondream] locate error: {e}")
        return None

    def close(self):
        pass


_detector_instance: MoondreamDetector | None = None


def get_detector() -> MoondreamDetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = MoondreamDetector()
    return _detector_instance
