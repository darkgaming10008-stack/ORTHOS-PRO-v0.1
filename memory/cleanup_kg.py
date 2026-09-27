"""One-time cleanup of knowledge_graph.json boilerplate pollution.

Removes duplicate 'conversation/recent_response' and 'user_message/input'
fact references/observations (thousands of near-identical entries created
by the old per-response link_fact calls), dedups observations and caps
fact_refs per edge.

Run: python memory/cleanup_kg.py
"""
import json
import shutil
from pathlib import Path

KG = Path(__file__).parent / "knowledge_graph.json"

BOILER = {"conversation/recent_response", "user_message/input"}
KEEP_PER_KEY = {"conversation/recent_response": 1, "user_message/input": 1}


def _clean_refs(refs: list[dict]) -> list[dict]:
    out = []
    for r in refs:
        cat = r.get("category", "")
        key = r.get("key", "")
        sig = f"{cat}/{key}"
        if sig in BOILER:
            continue  # boilerplate: never reference it from edges
        if r not in out:
            out.append(r)
    return out[:20]


def main() -> None:
    graph = json.loads(KG.read_text(encoding="utf-8"))
    ents: dict = graph.get("entities", {})
    edges: list = graph.get("edges", [])

    before_obs = sum(len(e.get("observations", [])) for e in ents.values())
    before_refs = sum(len(e.get("fact_refs", [])) for e in edges)

    for ent in ents.values():
        obs = ent.get("observations", [])
        cleaned = []
        seen = set()
        per_key: dict[str, int] = {}
        for o in obs:
            sig = o.split(":", 1)[0] if ":" in o else o[:80]
            if sig in BOILER:
                kept = per_key.get(sig, 0)
                if kept >= KEEP_PER_KEY.get(sig, 1):
                    continue
                per_key[sig] = kept + 1
            if sig in seen:
                continue
            seen.add(sig)
            cleaned.append(o)
        ent["observations"] = cleaned[-20:]

    for e in edges:
        e["fact_refs"] = _clean_refs(e.get("fact_refs", []))

    backup = KG.with_suffix(".json.bak_pre_cleanup")
    shutil.copy2(KG, backup)

    KG.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")

    after_obs = sum(len(e.get("observations", [])) for e in ents.values())
    after_refs = sum(len(e.get("fact_refs", [])) for e in edges)
    print(f"Entities: {len(ents)} (unchanged)")
    print(f"Edges: {len(edges)} (unchanged)")
    print(f"Observations: {before_obs} -> {after_obs}")
    print(f"fact_refs: {before_refs} -> {after_refs}")
    print(f"Backup written: {backup.name}")


if __name__ == "__main__":
    main()
