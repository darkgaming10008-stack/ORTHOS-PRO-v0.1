#!/usr/bin/env python3
"""Lightning-side index import.

Unpacks the zip exported on the PC into the right locations, then verifies
the indexes load and answers a probe query. Run on the Studio:

    python deploy/lightning/import_index.py /path/to/orthos-migrate.zip
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python deploy/lightning/import_index.py <path-to-orthos-migrate.zip>")
        return 1
    zpath = Path(sys.argv[1]).expanduser()
    if not zpath.exists():
        print(f"[X] Zip not found: {zpath}")
        return 1

    print(f"→ Unpacking {zpath.name} into {ROOT}…")
    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
        zf.extractall(ROOT)
    print(f"  ✅ {len(names)} files extracted")

    # Sanity checks
    print("→ Verifying stores…")
    ok = True

    db = ROOT / "memory" / "conversations.db"
    if db.exists():
        print(f"  ✅ conversations.db ({db.stat().st_size / 1e6:.1f} MB)")
    else:
        print("  -- conversations.db not in zip (skipped)")
        ok = False

    chroma = ROOT / "memory" / "chroma_db"
    if chroma.exists() and any(chroma.iterdir()):
        n = sum(1 for _ in chroma.rglob("*") if _.is_file())
        print(f"  ✅ chroma_db/ ({n} files)")
    else:
        print("  -- chroma_db/ empty/missing (vector memories start fresh)")
        ok = False

    # Live probe: does the (service-owned) search stack answer?
    print("→ Probe search…")
    try:
        sys.path.insert(0, str(ROOT))
        from memory.chroma_memory import search_turn_embeddings
        hits = search_turn_embeddings("probe query", n_results=3)
        print(f"  ✅ turn index answered with {len(hits)} hit(s)")
    except Exception as e:
        print(f"  -- turn-index probe failed (service may still be loading): {e}")

    print("""
DONE. Restart the service so it picks up the imported index:

    bash deploy/lightning/run_service.sh

Then from the PC re-run the connection test in start_memory_service_cloud.bat.
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
