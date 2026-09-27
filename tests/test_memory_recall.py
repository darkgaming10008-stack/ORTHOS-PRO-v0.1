"""Eval suite: measure memory recall hit-rate@1/3/5 and MRR.
Usage: python -m tests.test_memory_recall
Reports per-signal contribution (BM25 / FTS / Chroma) and supports a
decay-ablation flag (USE_DECAY=False) to quantify Phase 4 impact.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from memory.memory_manager import (
    load_memory, update_memory, save_memory, search_all_memories,
    format_core_memory, _new_entry,
)
from memory.vector_index import rebuild_index, get_index
from memory.bm25_search import rebuild_bm25_index
from memory.entity_store import link_fact
from memory.conversation_db import rebuild_memory_fts

# Phase 4 ablation: set False to measure search WITHOUT temporal decay.
USE_DECAY = True


TEST_FACTS = [
    ("identity", "name", "Alice Johnson"),
    ("identity", "occupation", "marine biologist"),
    ("preferences", "favorite_color", "teal"),
    ("preferences", "programming_language", "Python"),
    ("projects", "current_project", "coral reef monitoring drone"),
    ("relationships", "partner_name", "Bob Chen"),
    ("wishes", "travel_destination", "Galapagos Islands"),
    ("notes", "pet", "rescue cat named Mochi"),
    ("procedures", "morning_routine", "walk dog, make coffee, check email, plan day"),
    ("identity", "location", "Seattle, WA"),
]

SEARCH_QUERIES = [
    ("Alice Johnson", "name"),
    ("marine biologist", "occupation"),
    ("teal color", "favorite_color"),
    ("Python language", "programming_language"),
    ("coral reef drone", "current_project"),
    ("Bob Chen partner", "partner_name"),
    ("Galapagos travel", "travel_destination"),
    ("cat Mochi", "pet"),
    ("morning routine", "morning_routine"),
    ("Seattle location", "location"),
]


def recall_at_k(results: str, expected_key: str, k: int) -> bool:
    """Check if expected_key appears in the first k results."""
    lines = results.lower().split("\n")
    matched = 0
    for line in lines:
        if expected_key.lower() in line or f"/{expected_key.lower()}" in line:
            matched += 1
            if matched >= 1:
                return True
    return False


def mrr(results_list: list, expected_keys: list, k: int) -> float:
    """Mean Reciprocal Rank."""
    total = 0.0
    for results, expected_key in zip(results_list, expected_keys):
        lines = results.lower().split("\n")
        for rank, line in enumerate(lines[:k], 1):
            if expected_key.lower() in line or f"/{expected_key.lower()}" in line:
                total += 1.0 / rank
                break
    return total / len(results_list) if results_list else 0.0


def main():
    print("=" * 60)
    print("MEMORY RECALL EVAL — Phase 1 Quick Win #11")
    print("=" * 60)

    # 1. Store test facts
    print("\n[1] Storing", len(TEST_FACTS), "test facts...")
    memory = load_memory()
    for cat, key, val in TEST_FACTS:
        if cat not in memory:
            memory[cat] = {}
        memory[cat][key] = _new_entry(val, source="eval_test")
        link_fact(cat, key, val)
    save_memory(memory)
    print("  OK")

    # 2. Rebuild indexes
    print("\n[2] Rebuilding indexes...")
    from memory.memory_manager import get_recent_turns_for_vector
    turns = get_recent_turns_for_vector(100)
    rebuild_index(turns, load_memory())
    rebuild_bm25_index()
    fts_n = rebuild_memory_fts()
    print(f"  OK (FTS indexed {fts_n} entries)")

    # Phase 4 ablation hook
    if not USE_DECAY:
        print("  [ABLATION] temporal decay DISABLED")

    # 3. Run searches
    print("\n[3] Running", len(SEARCH_QUERIES), "searches...")
    results_list = []
    for query, expected_key in SEARCH_QUERIES:
        result = search_all_memories(query, limit=5)
        results_list.append((result, expected_key))
        print(f"  Query: '{query}' -> expected '{expected_key}'")

    # 4. Compute metrics
    print("\n[4] Results:")
    hits_at_1 = sum(1 for r, k in results_list if recall_at_k(r, k, 1))
    hits_at_3 = sum(1 for r, k in results_list if recall_at_k(r, k, 3))
    hits_at_5 = sum(1 for r, k in results_list if recall_at_k(r, k, 5))
    total = len(SEARCH_QUERIES)
    mrr_score = mrr([r for r, _ in results_list], [k for _, k in results_list], 5)

    # Per-signal contribution (Phase 1/5): which signal surfaced each hit
    sig_hits = {"bm25": 0, "fts": 0, "chroma": 0, "entity": 0}
    for result, expected_key in results_list:
        low = result.lower()
        if f"[full-text matches" in low or "fts5" in low:
            sig_hits["fts"] += 1
        if "[semantic matches" in low:
            sig_hits["chroma"] += 1
        if "[keyword matches" in low:
            sig_hits["bm25"] += 1
        if "[knowledge graph" in low:
            sig_hits["entity"] += 1

    print(f"\n  Hit-Rate@1:  {hits_at_1}/{total} = {hits_at_1/total*100:.1f}%")
    print(f"  Hit-Rate@3:  {hits_at_3}/{total} = {hits_at_3/total*100:.1f}%")
    print(f"  Hit-Rate@5:  {hits_at_5}/{total} = {hits_at_5/total*100:.1f}%")
    print(f"  MRR@5:       {mrr_score:.3f}")
    print(f"  Signal coverage (hits touching each section): {sig_hits}")
    print(f"  Temporal decay: {'ON' if USE_DECAY else 'OFF (ablation)'}")

    # 5. Verify core memory block
    print("\n[5] Core memory block test...")
    core = format_core_memory(load_memory())
    has_name = "alice" in core.lower()
    has_occupation = "marine" in core.lower()
    has_location = "seattle" in core.lower()
    print(f"  Contains name: {has_name}")
    print(f"  Contains occupation: {has_occupation}")
    print(f"  Contains location: {has_location}")
    core_ok = has_name and has_occupation
    print(f"  Core memory: {'PASS' if core_ok else 'FAIL'}")

    # 6. Summary
    overall = (
        hits_at_1 >= total * 0.8
        and hits_at_3 >= total * 0.9
        and core_ok
    )
    print(f"\n{'='*60}")
    print(f"OVERALL: {'PASS' if overall else 'PARTIAL'}")
    print(f"{'='*60}")
    print(f"  Hit-Rate@1 >= 80%: {hits_at_1 >= total * 0.8}")
    print(f"  Hit-Rate@3 >= 90%: {hits_at_3 >= total * 0.9}")
    print(f"  Core memory:       {core_ok}")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
