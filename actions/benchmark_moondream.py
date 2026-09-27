from __future__ import annotations
import io
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from actions.moondream_llama import get_detector


def fmt(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds // 60:.0f}m{seconds % 60:.0f}s"


def make_text_img(text, bg="white", fg="black", size=(800, 400)):
    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)
    try:
        fnt = ImageFont.truetype("arial.ttf", 48)
    except Exception:
        fnt = ImageFont.load_default()
    draw.text((50, 100), text, fill=fg, font=fnt)
    return img


def make_ui_screenshot():
    img = Image.new("RGB", (1920, 1080), (30, 30, 30))
    draw = ImageDraw.Draw(img)
    for color, rect, text, y in [
        ((60, 60, 60), (50, 50, 200, 100), "File", 50),
        ((60, 60, 60), (220, 50, 370, 100), "Edit", 50),
        ((60, 60, 60), (390, 50, 540, 100), "View", 50),
        ((100, 100, 200), (100, 200, 400, 300), "Submit", 230),
        ((200, 100, 100), (500, 200, 800, 300), "Cancel", 230),
        ((255, 255, 255), (200, 400, 600, 500), "Username:", 430),
        ((255, 255, 255), (200, 530, 600, 630), "Password:", 560),
    ]:
        draw.rectangle(rect, fill=color)
        try:
            f = ImageFont.truetype("arial.ttf", 28)
        except Exception:
            f = ImageFont.load_default()
        draw.text((rect[0] + 10, rect[2] - rect[0] + 1), text, fill=(0, 0, 0) if color == (255, 255, 255) else (255, 255, 255), font=f)
    return img


if __name__ == "__main__":
    print("Benchmark: Moondream3 Preview via Cloud API")
    print("=" * 50)

    # ── Test 1: Detector init ──────────────────────
    print("\n1. Detector init")
    t0 = time.time()
    det = get_detector()
    t_load = time.time() - t0
    print(f"   Detector init: {fmt(t_load)}")

    print(f"\n[SKIP] describe/OCR benchmarks — Moondream describe and get_all_ocr_text removed.")
    print(f"       Moondream is now only used for locate() (point API).")
    print(f"       Describe/OCR now go through the main LLM (multimodal).")
