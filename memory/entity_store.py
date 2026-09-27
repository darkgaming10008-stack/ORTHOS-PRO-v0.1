"""Knowledge Graph — typed entities + directed edges + NER + multi-hop traversal.
Replaces flat entity-store with industry-standard property graph (Anthropic/KG-style).
Backward-compatible API: extract_entities, link_fact, search_entities, get_entity_graph.
"""
import json
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path
from collections import defaultdict


def re_search_word(term: str, text: str) -> bool:
    """Word-boundary substring match (handles plural-ish forms loosely)."""
    if not term or not text:
        return False
    return re.search(r"(?:\b|(?<=_))" + re.escape(term), text) is not None

KG_PATH = Path(__file__).parent / "knowledge_graph.json"
_lock = threading.Lock()

# ── spaCy NER with graceful fallback ──────────────────────────────────

_NER = None


def _get_ner():
    global _NER
    if _NER is not None:
        return _NER if _NER is not False else None
    try:
        import spacy
        _NER = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer"])
        print("[KG] Loaded spaCy NER model")
    except Exception:
        print("[KG] spaCy NER unavailable, using heuristic fallback")
        _NER = False
    return _NER if _NER is not False else None


def preload_ner():
    try:
        from memory.memory_client import _is_healthy
        if _is_healthy():
            return
    except Exception:
        pass
    _get_ner()


# ── Entity type mapping from spaCy ────────────────────────────────────

SPACY_TYPE_MAP = {
    "PERSON": "Person", "ORG": "Organization", "GPE": "Location",
    "LOC": "Location", "DATE": "Date", "EVENT": "Event",
    "PRODUCT": "Object", "WORK_OF_ART": "Concept", "NORP": "Concept",
}

# ── Knowledge graph schema ────────────────────────────────────────────

EMPTY_GRAPH = {
    "entities": {},   # entity_id -> entity
    "edges": [],      # list of edge dicts
}

RELATION_SYNONYMS = {
    "works at": "WORKS_AT", "works for": "WORKS_AT", "employed by": "WORKS_AT",
    "lives in": "LIVES_IN", "lives at": "LIVES_IN", "located in": "LIVES_IN",
    "knows": "KNOWS", "friends with": "KNOWS", "partner": "KNOWS",
    "likes": "LIKES", "enjoys": "LIKES", "loves": "LIKES", "prefers": "PREFERS",
    "uses": "USES", "codes in": "USES", "writes": "USES",
    "has": "HAS", "owns": "HAS",
    "wants": "WANTS", "wishes": "WANTS", "desires": "WANTS",
    "travels to": "TRAVELS_TO", "visited": "TRAVELS_TO",
}

# ── Coreference / pronoun resolution (Phase 2) ──
# Map ambiguous surface forms to a canonical relation that, when seen,
# links the matched person entity to the user's self-entity ("_user").
CORE_METHOD_KEYWORDS = {
    "wife": "SPOUSE", "husband": "SPOUSE", "spouse": "SPOUSE",
    "partner": "PARTNER", "girlfriend": "PARTNER", "boyfriend": "PARTNER",
    "mom": "MOTHER", "mother": "MOTHER", "dad": "FATHER", "father": "FATHER",
    "son": "CHILD", "daughter": "CHILD", "kid": "CHILD", "child": "CHILD",
    "brother": "SIBLING", "sister": "SIBLING", "sibling": "SIBLING",
    "boss": "EMPLOYER", "manager": "EMPLOYER", "employer": "EMPLOYER",
    "best friend": "FRIEND", "bestfriend": "FRIEND", "bestie": "FRIEND",
}

# Canonical-name normalization: map messy names to stable forms.
_NAME_NORMALIZATION = {
    "orthos": "Orthos",
}


def _canonical_name(name: str) -> str:
    """Normalize a name to its canonical surface form."""
    cleaned = name.strip().title()
    low = cleaned.lower()
    return _NAME_NORMALIZATION.get(low, cleaned)


def _resolve_coref(entities: list[str], value: str) -> list[str]:
    """Expand coreference terms (wife, husband, mom...) to explicit person
    entities so they link into the graph even without a proper name.
    Returns the (possibly expanded) entity name list.
    """
    vlow = value.lower()
    extra = []
    for term, canon in CORE_METHOD_KEYWORDS.items():
        if re_search_word(term, vlow):
            extra.append(f"{canon.title()} ({term})")
    if extra:
        return entities + extra
    return entities



def _load() -> dict:
    if not KG_PATH.exists():
        return {"entities": {}, "edges": []}
    with _lock:
        try:
            return json.loads(KG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {"entities": {}, "edges": []}


def _save(data: dict) -> None:
    with _lock:
        KG_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _empty_entity(name: str, etype: str = "Concept") -> dict:
    return {
        "id": str(uuid.uuid4())[:8],
        "name": name,
        "type": etype,
        "aliases": [name.lower()],
        "observations": [],
        "first_seen": datetime.now().strftime("%Y-%m-%d"),
        "last_seen": datetime.now().strftime("%Y-%m-%d"),
        "mention_count": 1,
        "importance": 0.5,
    }


def _normalize_relation(text: str) -> str:
    """Map natural language relation phrases to canonical edge types."""
    tl = text.lower().strip()
    for phrase, canon in RELATION_SYNONYMS.items():
        if phrase in tl:
            return canon
    return text.upper().replace(" ", "_")[:20]


# ── Public API ────────────────────────────────────────────────────────

def extract_entities(text: str) -> list[str]:
    """Extract named entities using memory service NER + spaCy fallback + heuristic fallback.
    Returns list of entity names.
    """
    try:
        from memory.memory_client import _is_healthy, memory_ner
        if _is_healthy():
            return memory_ner(text)
    except Exception:
        from memory.memory_client import invalidate_cache
        invalidate_cache()

    ner = _get_ner()
    if ner is not None:
        try:
            doc = ner(text)
            entities = set()
            for ent in doc.ents:
                if ent.label_ in SPACY_TYPE_MAP and len(ent.text.strip()) > 1:
                    entities.add(ent.text.strip())
            return list(entities)
        except Exception:
            pass

    # Heuristic fallback: capitalized words + two-word proper names
    entities = set()
    words = text.split()
    for i, w in enumerate(words):
        cleaned = w.strip(".,!?;:'\"()[]{}-")
        if not cleaned:
            continue
        if cleaned[0].isupper() and i > 0 and len(cleaned) > 1:
            entities.add(cleaned)
        if cleaned[0].isupper() and len(cleaned) <= 1 and i < len(words) - 1:
            entities.add(cleaned + " " + words[i+1].strip(".,!?;:'\"()[]{}-"))
    return list(entities)


def _merge_entities(graph: dict, name_a: str, name_b: str) -> None:
    """Merge entity B into entity A if they refer to the same thing."""
    ents = graph["entities"]
    eid_a = next((eid for eid, e in ents.items() if e["name"].lower() == name_a.lower()), None)
    eid_b = next((eid for eid, e in ents.items() if e["name"].lower() == name_b.lower()), None)
    if not eid_a or not eid_b or eid_a == eid_b:
        return

    a = ents[eid_a]
    b = ents[eid_b]

    # Merge aliases
    a["aliases"] = list(set(a["aliases"] + b["aliases"] + [b["name"].lower()]))
    a["observations"].extend(b["observations"])
    a["observations"] = list(set(a["observations"]))
    a["mention_count"] += b["mention_count"]
    a["importance"] = max(a["importance"], b["importance"])
    a["last_seen"] = max(a["last_seen"], b["last_seen"])
    a["type"] = a["type"] if a["type"] != "Concept" else b["type"]  # prefer specific type

    # Redirect edges
    for edge in graph["edges"]:
        if edge["source"] == eid_b:
            edge["source"] = eid_a
        if edge["target"] == eid_b:
            edge["target"] = eid_a

    del ents[eid_b]


def link_fact(category: str, key: str, value: str) -> None:
    """Extract entities from a fact and link them in the knowledge graph.
    Creates typed entities + directed edges + observations.
    """
    # Guard: boilerplate conversation bookkeeping must not flood the graph.
    # (Every assistant response used to be linked, producing thousands of
    #  duplicate 'conversation/recent_response' edges + fact_refs.)
    if category == "conversation" and key == "recent_response":
        if len(value or "") < 80:
            return

    graph = _load()
    ents = graph["entities"]
    today = datetime.now().strftime("%Y-%m-%d")

    # Entity types come from the preloaded memory-service NER (typed); local
    # spaCy is only a fallback when the service is unreachable.
    type_map: dict[str, str] = {}
    try:
        from memory.memory_client import _is_healthy, memory_ner_typed
        if _is_healthy():
            for t in memory_ner_typed(f"{value} {key.replace('_', ' ')}"):
                type_map.setdefault(
                    str(t.get("name", "")).strip().title(),
                    str(t.get("type", "Concept")),
                )
    except Exception:
        from memory.memory_client import invalidate_cache
        invalidate_cache()

    # Extract entities from the value text
    entity_names = extract_entities(value)
    # Also extract from key
    entity_names.extend(extract_entities(key.replace("_", " ")))

    if not entity_names:
        return

    # Phase 2: canonicalize names + coreference expansion
    entity_names = [_canonical_name(n) for n in entity_names]
    entity_names = _resolve_coref(entity_names, value + " " + key)

    # Create/update entity nodes
    eids = []
    for name in entity_names:
        name_clean = name.strip().title()
        name_low = name_clean.lower()
        # Check if exists by name or alias (token-normalized)
        eid = None
        for eid_candidate, ent in ents.items():
            aliases = [a.lower() for a in ent.get("aliases", [])]
            if ent["name"].lower() == name_low or name_low in aliases:
                eid = eid_candidate
                break

        if eid:
            ent = ents[eid]
            ent["mention_count"] += 1
            ent["importance"] = min(1.0, ent["importance"] + 0.05)
            ent["last_seen"] = today
            if name_low not in [a.lower() for a in ent.get("aliases", [])]:
                ent.setdefault("aliases", []).append(name_low)
        else:
            # Determine type: memory service first, local spaCy fallback
            etype = type_map.get(name_clean, "Concept")
            if etype == "Concept":
                ner = _get_ner()
                if ner is not None:
                    try:
                        doc = ner(name_clean)
                        if doc.ents:
                            spacy_label = doc.ents[0].label_
                            etype = SPACY_TYPE_MAP.get(spacy_label, "Concept")
                    except Exception:
                        pass
            eid = str(uuid.uuid4())[:8]
            ents[eid] = _empty_entity(name_clean, etype)
        eids.append(eid)

    # Add observations and create edges between co-occurring entities
    fact_ref = {"category": category, "key": key, "value": value[:200], "date": today}

    for eid in eids:
        ent = ents[eid]
        # Add observation
        obs = f"{category}/{key}: {value[:200]}"
        if obs not in ent.get("observations", []):
            ent.setdefault("observations", []).append(obs)
            ent["observations"] = ent["observations"][-20:]  # cap at 20

    # Create edges between entities that appear together
    if len(eids) >= 2:
        for i in range(len(eids)):
            for j in range(i + 1, len(eids)):
                src, tgt = eids[i], eids[j]
                # Infer relation type from category
                relation = "RELATED_TO"
                cat_lower = category.lower()
                key_lower = key.lower()
                if "works" in cat_lower or "job" in key_lower:
                    relation = "WORKS_AT"
                elif "live" in key_lower or "location" in key_lower:
                    relation = "LIVES_IN"
                elif "friend" in cat_lower or "partner" in key_lower:
                    relation = "KNOWS"
                elif "like" in key_lower or "prefer" in key_lower or "favorite" in key_lower:
                    relation = "LIKES"
                elif "uses" in key_lower or "use" in key_lower or "tool" in key_lower:
                    relation = "USES"
                elif "wants" in key_lower or "goal" in key_lower or "wish" in cat_lower:
                    relation = "WANTS"
                elif "travel" in key_lower or "trip" in key_lower:
                    relation = "TRAVELS_TO"

                # Add edge (avoid duplicates)
                edge_exists = any(
                    e["source"] == src and e["target"] == tgt and e["relation"] == relation
                    for e in graph["edges"]
                )
                if not edge_exists:
                    graph["edges"].append({
                        "source": src,
                        "target": tgt,
                        "relation": relation,
                        "fact_refs": [fact_ref],
                        "weight": 1.0,
                        "valid_at": today,
                        "invalid_at": None,
                    })
                else:
                    # Bump weight on existing edge
                    for e in graph["edges"]:
                        if e["source"] == src and e["target"] == tgt and e["relation"] == relation:
                            e["weight"] = min(5.0, e["weight"] + 0.5)
                            if fact_ref not in e["fact_refs"] and len(e["fact_refs"]) < 20:
                                e["fact_refs"].append(fact_ref)
                            break

    # Entity disambiguation: merge "Alice" and "Alice Johnson" if same context
    name_groups = defaultdict(list)
    for eid, ent in ents.items():
        for alias in ent.get("aliases", [ent["name"].lower()]):
            name_groups[alias.split()[0] if len(alias.split()) > 1 else alias].append(eid)

    for base, group in name_groups.items():
        if len(group) >= 2:
            # Merge all into the first one
            primary = group[0]
            for other in group[1:]:
                if other in ents and primary in ents and primary != other:
                    p_name = ents[primary]["name"]
                    o_name = ents[other]["name"]
                    if p_name.lower() in o_name.lower() or o_name.lower() in p_name.lower():
                        _merge_entities(graph, p_name, o_name)

    _save(graph)


def search_entities(query: str) -> list[dict]:
    """Search knowledge graph for entities and facts matching the query."""
    graph = _load()
    ents = graph["entities"]
    q = query.lower()
    results = []

    # Phase 2: expand coref terms to canonical relation labels
    q_terms = set()
    for term, canon in CORE_METHOD_KEYWORDS.items():
        if re_search_word(term, q):
            q_terms.add(canon.lower())

    for eid, ent in ents.items():
        name = ent.get("name", "")
        aliases = [a.lower() for a in ent.get("aliases", [])]
        obs = ent.get("observations", [])

        # Match on name, aliases, observations, or coref relation label
        matched = (q in name.lower()
                   or any(q in a for a in aliases)
                   or any(q in o.lower() for o in obs)
                   or any(t in aliases for t in q_terms))

        if matched:
            for obs_text in obs:
                # Parse observation back to fact ref format
                parts = obs_text.split(": ", 1)
                if len(parts) == 2:
                    cat_key = parts[0].split("/", 1)
                    if len(cat_key) == 2:
                        results.append({
                            "entity": name,
                            "category": cat_key[0],
                            "key": cat_key[1],
                            "value": parts[1],
                            "type": ent.get("type", "Concept"),
                            "importance": ent.get("importance", 0.5),
                        })

    # Dedup
    seen = set()
    deduped = []
    for r in results:
        sig = (r["entity"], r["category"], r["key"])
        if sig not in seen:
            seen.add(sig)
            deduped.append(r)

    return deduped


def traverse(entity_name: str, max_hops: int = 2) -> list[dict]:
    """Multi-hop graph traversal. Follow edges to find connected entities."""
    graph = _load()
    ents = graph["entities"]

    # Find starting entity
    start_eid = None
    for eid, ent in ents.items():
        if ent["name"].lower() == entity_name.lower():
            start_eid = eid
            break

    if not start_eid:
        return []

    visited = {start_eid}
    frontier = [(start_eid, 0)]
    results = []

    while frontier:
        current, depth = frontier.pop(0)
        if depth >= max_hops:
            continue

        for edge in graph["edges"]:
            next_eid = None
            if edge["source"] == current:
                next_eid = edge["target"]
            elif edge["target"] == current:
                next_eid = edge["source"]

            if next_eid and next_eid not in visited:
                visited.add(next_eid)
                neighbor = ents.get(next_eid, {})
                results.append({
                    "hop": depth + 1,
                    "entity": neighbor.get("name", "?"),
                    "type": neighbor.get("type", "Concept"),
                    "relation": edge["relation"],
                    "observations": neighbor.get("observations", [])[:3],
                    "importance": neighbor.get("importance", 0),
                })
                frontier.append((next_eid, depth + 1))

    return results


def graph_retrieve(query: str, max_hops: int = 2) -> list[dict]:
    """Multi-hop graph retrieval: find entities matching query, traverse relationships,
    return structured results with entity names, types, relations, and observations."""
    direct = search_entities(query)
    if not direct:
        return []
    results = []
    seen_entities = set()
    for r in direct:
        ename = r.get("entity", "")
        etype = r.get("type", "Concept")
        if ename and ename not in seen_entities:
            seen_entities.add(ename)
            results.append({
                "entity": ename,
                "entity_type": etype,
                "relation": "SELF",
                "connected_entity": "",
                "connected_type": "",
                "observations": r.get("value", ""),
                "hop": 0,
            })
        hops = traverse(ename, max_hops=max_hops)
        for hop in hops:
            hop_name = hop.get("entity", "")
            hop_eid = None
            # Get entity type from the store
            graph = _load()
            for eid, ent in graph["entities"].items():
                if ent.get("name", "").lower() == hop_name.lower():
                    hop_eid = eid
                    break
            hop_type = "Concept"
            if hop_eid and hop_eid in graph["entities"]:
                hop_type = graph["entities"][hop_eid].get("type", "Concept")
            sig = (ename, hop.get("relation"), hop_name)
            if sig not in [(r2.get("entity"), r2.get("relation"), r2.get("connected_entity")) for r2 in results]:
                results.append({
                    "entity": ename,
                    "entity_type": etype,
                    "relation": hop.get("relation", "RELATED_TO"),
                    "connected_entity": hop_name,
                    "connected_type": hop_type,
                    "observations": hop.get("observations", [""])[0] if hop.get("observations") else "",
                    "hop": hop.get("hop", 1),
                })
    return results


def get_entity_graph(query: str) -> str:
    """Return multi-hop entity graph for a query. Shows direct + 2-hop connections."""
    direct = search_entities(query)
    if not direct:
        return ""

    lines = ["[KNOWLEDGE GRAPH — typed entities + relationships]"]
    seen_entities = set()

    for r in direct:
        ename = r["entity"]
        if ename not in seen_entities:
            seen_entities.add(ename)
            lines.append(f"  [{r.get('type', 'Concept')}] {ename} (importance: {r.get('importance', 0.5):.2f})")

        lines.append(f"    {r['category']}/{r['key']}: {r['value']}")

        # Show 1-hop neighbors
        hops = traverse(ename, max_hops=1)
        for hop in hops:
            if hop["entity"] not in seen_entities:
                seen_entities.add(hop["entity"])
                lines.append(f"    --[{hop['relation']}]--> [{hop['type']}] {hop['entity']}")

        # Show 2-hop neighbors (one deeper)
        if len(lines) < 30:  # prevent runaway
            hops2 = traverse(ename, max_hops=2)
            for hop in hops2:
                if hop["hop"] == 2 and hop["entity"] not in seen_entities:
                    seen_entities.add(hop["entity"])
                    lines.append(f"      --[{hop['relation']}]--> [{hop['type']}] {hop['entity']}")

    if len(lines) == 1:
        return ""

    return "\n".join(lines)
