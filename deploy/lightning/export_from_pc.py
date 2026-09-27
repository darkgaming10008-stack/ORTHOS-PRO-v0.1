#!/usr/bin/env python3
"""Orthos → Lightning index migration (PC side).

Packs the existing vector indexes + conversation DB into one zip to upload
to the Lightning Studio. Run from the Orthos project root on the PC:

    python deploy/lightning/export_from_pc.py

Output: orthos-migrate.zip in the project root (~size of your index).
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# Relative paths inside the project (mirrored inside the zip)
INCLUDE_FILES = [
    "memory/conversations.db",          # SQLite: turns, summaries, memory_items
    "memory/.vector_index.npy",         # FAISS vectors
    "memory/.vector_index.faiss",       # FAISS index (if present)
    "memory/.vector_meta.json",         # FAISS metadata
]
INCLUDE_DIRS = [
    "memory/chroma_db",                 # ChromaDB (memories + conversation_turns)
]

OUT = ROOT / "orthos-migrate.zip"


def main() -> int:
    if not OUT.parent.exists():
        print(f"[X] Project root not found: {ROOT}")
        return 1

    packed: list[Path] = []
    missing: list[str] = []

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in INCLUDE_FILES:
            p = ROOT / rel
            if p.exists():
                zf.write(p, rel)
                packed.append(p)
            else:
                missing.append(rel)

        for d in INCLUDE_DIRS:
            base = ROOT / d
            if not base.exists():
                missing.append(d + "/")
                continue
            for p in base.rglob("*"):
                if p.is_file():
                    rel = p.relative_to(ROOT).as_posix()
                    zf.write(p, rel)
                    packed.append(p)

    total_mb = OUT.stat().st_size / 1e6
    print("=" * 56)
    print(f" Packed {len(packed)} files -> {OUT.name} ({total_mb:.1f} MB)")
    print("=" * 56)
    if missing:
        print("Missing/skipped (fine on a fresh setup):")
        for m in missing:
            print(f"  - {m}")
    print("""
NEXT STEPS
  1. Upload orthos-migrate.zip to the Lightning Studio (file panel drag-drop)
  2. In the Studio terminal, from the project root:
       python deploy/lightning/import_index.py ~/orthos-migrate.zip
     (or wherever you uploaded it -- script takes the zip path as argument)
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
