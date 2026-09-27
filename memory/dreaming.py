import json
import threading
import time
import hashlib
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "conversations.db"
DREAM_INTERVAL_TURNS = 20

_last_dream_turn = 0
_dream_lock = threading.Lock()
_dreaming_active = False
_dream_thread = None
_idle_last_active = time.time()
IDLE_DEEP_THRESHOLD = 1800  # 30 minutes — only Deep phase after this idle


def mark_active() -> None:
    global _idle_last_active
    _idle_last_active = time.time()


def _is_idle() -> bool:
    return (time.time() - _idle_last_active) >= IDLE_DEEP_THRESHOLD


def _data_path() -> Path:
    return Path(__file__).parent / "dream_data"


def _load_state() -> dict:
    state_file = _data_path() / "state.json"
    if state_file.exists():
        try:
            return json.loads(state_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"last_dream_turn": 0, "phase": "light"}


def _save_state(state: dict) -> None:
    _data_path().mkdir(parents=True, exist_ok=True)
    state_file = _data_path() / "state.json"
    state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _read_phase_log(phase: str) -> list:
    log_file = _data_path() / f"{phase}_log.json"
    if log_file.exists():
        try:
            return json.loads(log_file.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _write_phase_log(phase: str, data: list) -> None:
    _data_path().mkdir(parents=True, exist_ok=True)
    log_file = _data_path() / f"{phase}_log.json"
    log_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def get_total_turn_count() -> int:
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    count = conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
    conn.close()
    return count


def get_turns_since(start: int) -> list[dict]:
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, role, content, timestamp FROM turns WHERE id > ? ORDER BY id ASC",
        (start,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def compute_turn_hash(turns: list[dict]) -> str:
    content = "".join(f"{t['role']}:{t['content']}" for t in turns if t.get("content"))
    return hashlib.md5(content.encode()).hexdigest()


# ── Phase 1: Light ──────────────────────────────────────────────────────

def _phase_light(new_turns: list[dict]) -> dict:
    """Light phase: sort, dedupe, stage material. No durable writes."""
    staged = _read_phase_log("light")

    hashes = {s.get("hash") for s in staged}
    for t in new_turns:
        h = hashlib.md5(f"{t['role']}:{t['content']}".encode()).hexdigest()
        if h not in hashes:
            hashes.add(h)
            staged.append({
                "hash": h,
                "role": t["role"],
                "content": (t.get("content") or "")[:500],
                "timestamp": t.get("timestamp", ""),
                "staged_at": datetime.now().isoformat(),
            })

    # Keep last 100 entries in staging
    staged = staged[-100:]
    _write_phase_log("light", staged)
    return {"staged_count": len(staged), "new_count": len(new_turns)}


# ── Phase 2: REM ────────────────────────────────────────────────────────

def _phase_rem() -> list[dict]:
    """REM phase: extract patterns from staged material. No durable writes."""
    staged = _read_phase_log("light")
    if not staged:
        return []

    candidates = _read_phase_log("rem")
    candidate_hashes = {c.get("pattern_hash") for c in candidates if c.get("pattern_hash")}

    # Group staged by content similarity (simple approach: count repeated keywords)
    from collections import Counter
    all_text = " ".join(s.get("content", "") for s in staged if s.get("role") == "user")
    words = [w.lower() for w in all_text.split() if len(w) > 3]
    word_freq = Counter(words)
    top_topics = [w for w, c in word_freq.most_common(10) if c >= 2]

    # Create pattern entries for frequent topics
    new_candidates = []
    for topic in top_topics:
        h = hashlib.md5(f"topic:{topic}".encode()).hexdigest()
        if h not in candidate_hashes:
            candidate_hashes.add(h)
            new_candidates.append({
                "pattern_hash": h,
                "pattern_type": "topic",
                "content": topic,
                "frequency": word_freq[topic],
                "extracted_at": datetime.now().isoformat(),
            })

    candidates.extend(new_candidates)
    _write_phase_log("rem", candidates)

    return candidates


# ── Phase 1b: Identity self-references ─────────────────────────────────────

_IDENTITY_PATTERNS = [
    r"\bI am\s+([A-Z][a-z]+(?: [A-Z][a-z]+)?)[\.,!;]?(?:\s|$)",
    r"\bI'm\s+([A-Z][a-z]+(?: [A-Z][a-z]+)?)[\.,!;]?(?:\s|$)",
    r"\bmy name is\s+([A-Z][a-z]+(?: [A-Z][a-z]+)?)[\.,!;]?(?:\s|$)",
    r"\bcall me\s+([A-Z][a-z]+(?: [A-Z][a-z]+)?)[\.,!;]?(?:\s|$)",
    r"\b(?:u|you)\s+(?:r|are)\s+wrong\b[^.]*?\b(?:I am|I'm)\s+([A-Z][a-z]+)[\.,!;]?(?:\s|$)",
]

_SKIP_WORDS = {"just", "going", "here", "there", "doing", "trying", "not",
               "actually", "really", "also", "still", "now", "then",
               "being", "having", "getting", "making", "saying"}

_KNOWN_OCCUPATIONS = {"student", "developer", "engineer", "designer", "teacher",
                      "doctor", "lawyer", "manager", "analyst", "consultant",
                      "freelancer", "researcher", "scientist", "writer",
                      "artist", "director", "founder", "ceo", "cto", "coo"}

def _extract_identity_statements(turns: list[dict]) -> list[tuple[str, str, str]]:
    """Extract self-identification statements from user turns.
    Returns list of (category, key, value) for identity facts.
    """
    import re
    facts = []
    for t in turns:
        if t.get("role") != "user":
            continue
        content = (t.get("content") or "").strip()
        for pat in _IDENTITY_PATTERNS:
            m = re.search(pat, content, re.IGNORECASE)
            if m:
                name = m.group(1).strip()
                name = re.sub(r"\s+", " ", name)
                first_word = name.split()[0].lower()
                all_words = [w.lower() for w in name.split()]
                # Reject: too short, skip words, occupations, all-uppercase abbreviations, single letters
                if (len(name) > 2 and len(name) < 50
                        and first_word not in _SKIP_WORDS
                        and not any(w in _KNOWN_OCCUPATIONS for w in all_words)
                        and not name.isupper()
                        and name[0].isupper()):
                    facts.append(("identity", "name", name))
                break
    return facts


# ── Phase 3: Deep ────────────────────────────────────────────────────────

def _score_candidate(cand: dict) -> float:
    """Score a candidate for promotion. Returns 0-1 score."""
    base = 0.0
    freq = cand.get("frequency", 0)
    if freq >= 5:
        base += 0.5
    elif freq >= 3:
        base += 0.3
    elif freq >= 2:
        base += 0.1

    recall_count = cand.get("recall_count", 0)
    if recall_count > 0:
        base += min(0.3, recall_count * 0.1)

    return min(1.0, base)


def _phase_deep(candidates: list[dict]) -> list[tuple[str, str, str]]:
    """Deep phase: score candidates, promote high-scoring to memory."""
    from core.llm_client import call_llm_text

    # First pass: score based on frequency + recall
    high_value = [c for c in candidates if _score_candidate(c) >= 0.3]

    if not high_value:
        return []

    # Second pass: LLM evaluates top candidates
    topics_text = "\n".join(
        f"- {c['content']} (freq: {c.get('frequency', 0)})"
        for c in high_value[:5]
    )
    prompt = (
        "You are a memory curator evaluating candidate facts.\n"
        "Given these frequently mentioned topics, decide which are "
        "important enough to remember permanently.\n\n"
        "Rules:\n"
        "- Only promote if it's a verifiable fact about the user\n"
        "- Skip temporary context, one-time events\n"
        "- Format as: category|key|value\n"
        "- One per line, or NONE if nothing worth keeping\n\n"
        f"Candidates:\n{topics_text}"
    )
    try:
        result = call_llm_text(prompt, system="You score and promote memory candidates.")
        lines = result.strip().split("\n")
        facts = []
        for line in lines:
            line = line.strip()
            if not line or line == "NONE":
                continue
            parts = line.split("|", 2)
            if len(parts) == 3:
                facts.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
        return facts
    except Exception as e:
        print(f"[Dreaming][Deep] LLM error: {e}")
        return []


# ── Orchestration ──────────────────────────────────────────────────────

def load_memory() -> dict:
    """Delegate to memory_manager.load_memory (SQLite-backed source of truth)."""
    from memory.memory_manager import load_memory as _real_load
    return _real_load()


def _empty_memory() -> dict:
    return {
        "identity": {}, "preferences": {}, "projects": {},
        "relationships": {}, "wishes": {}, "notes": {}, "procedures": {},
    }


def save_memory(memory: dict, skip_git: bool = True) -> None:
    """Delegate to memory_manager.save_memory for git-backed versioning."""
    from memory.memory_manager import save_memory as _real_save
    _real_save(memory, skip_git=skip_git)


def merge_facts(memory: dict, new_facts: list[tuple[str, str, str]], source: str = "dreaming") -> bool:
    changed = False
    for cat, key, val in new_facts:
        if cat not in memory:
            memory[cat] = {}
        existing = memory[cat].get(key, {})
        existing_val = existing.get("value") if isinstance(existing, dict) else None
        if existing_val == val:
            continue
        history = existing.get("history", []) if isinstance(existing, dict) else []
        if existing_val is not None:
            history.append({"value": existing_val, "updated": existing.get("updated", "?")})
            history = history[-20:]
        memory[cat][key] = {
            "value": val,
            "updated": datetime.now().strftime("%Y-%m-%d"),
            "history": history,
            "source": source,
            "score": float(existing.get("score", 1.0)) + 0.5 if isinstance(existing, dict) and existing_val else 1.0,
            "access_count": existing.get("access_count", 0) if isinstance(existing, dict) else 0,
        }
        changed = True
    if changed:
        try:
            from memory.memory_manager import _sync_chroma
            for cat, key, val in new_facts:
                _sync_chroma(cat, key, val)
        except Exception:
            pass
    return changed


def _resolve_contradictions() -> int:
    """Dreaming resolve phase: find contradictions, keep the higher-scored fact.
    Returns number of contradictions resolved.
    """
    from memory.memory_manager import load_memory, save_memory, _decay_score
    memory = load_memory()
    resolved = 0
    for cat, entries in memory.items():
        if not isinstance(entries, dict):
            continue
        for key, entry in list(entries.items()):
            if not isinstance(entry, dict):
                continue
            contradictions = entry.get("contradictions", [])
            if not contradictions:
                continue

            current_score = _decay_score(entry)
            # Find the highest-scored historical value in contradictions
            best = entry
            best_score = current_score
            for c in contradictions:
                c_val = c.get("old_value", "")
                c_score = c.get("old_score", 0)
                if c_score > best_score and c_val:
                    # Check if this value is still in history
                    for hist in entry.get("history", []):
                        if hist.get("value") == c_val:
                            # Promote this old value back
                            entry["value"] = c_val
                            entry["score"] = c_score + 0.1
                            entry["contradictions"] = []  # cleared
                            resolved += 1
                            break

            # Keep max 1 contradiction record for reference
            if len(contradictions) > 0:
                entry["contradictions"] = contradictions[-1:]

    if resolved > 0:
        save_memory(memory)
        print(f"[Dreaming][Resolve] Resolved {resolved} contradictions")
    return resolved


def _dream_cycle(force_deep: bool = False) -> None:
    global _last_dream_turn
    # Defer the whole cycle while the user is actively chatting — the Light/
    # REM phases do heavy SQLite + embedding work that stalls prompt building.
    if not force_deep and (time.time() - _idle_last_active) < 60:
        return
    state = _load_state()
    last_turn = state.get("last_dream_turn", 0)

    total = get_total_turn_count()
    if total - last_turn < DREAM_INTERVAL_TURNS:
        return

    new_turns = get_turns_since(last_turn)
    if not new_turns:
        return

    print(f"[Dreaming] Starting 3-phase cycle ({len(new_turns)} new turns)...")

    # Resolve contradictions before dreaming
    try:
        resolved = _resolve_contradictions()
        if resolved:
            print(f"[Dreaming]  Resolve: fixed {resolved} contradictions")
    except Exception as e:
        print(f"[Dreaming]  Resolve error: {e}")

    # Phase 1: Light — always runs
    light_result = _phase_light(new_turns)
    print(f"[Dreaming]  Light: {light_result['staged_count']} staged")

    # Phase 1b: Identity self-references — extract "I am X" patterns immediately
    identity_facts = _extract_identity_statements(new_turns)
    if identity_facts:
        memory = load_memory()
        if merge_facts(memory, identity_facts, source="dreaming_identity"):
            save_memory(memory)
            try:
                from memory.entity_store import link_fact
                for cat, key, val in identity_facts:
                    link_fact(cat, key, val)
            except Exception:
                pass
            print(f"[Dreaming]  Identity: extracted {len(identity_facts)} fact(s) from self-reference")
            # Reload new turns so REM doesn't double-process
            new_turns = get_turns_since(last_turn)

    # Phase 2: REM — always runs (pattern extraction, no writes)
    rem_candidates = _phase_rem()
    print(f"[Dreaming]  REM: {len(rem_candidates)} pattern candidates")

    # Phase 3: Deep — only if idle enough OR forced
    idle_enough = _is_idle()
    if not idle_enough and not force_deep:
        print(f"[Dreaming]  Deep: skipped (user active, need {IDLE_DEEP_THRESHOLD}s idle)")
        _last_dream_turn = total
        _save_state({"last_dream_turn": total, "phase": "light_rem_only"})
        return

    if rem_candidates:
        promoted = _phase_deep(rem_candidates)
        if promoted:
            memory = load_memory()
            if merge_facts(memory, promoted, source="dreaming_deep"):
                save_memory(memory)
                # Enforce memory cap after adding new facts
                try:
                    from memory.memory_manager import enforce_memory_cap
                    enforce_memory_cap()
                except Exception:
                    pass
                try:
                    from memory.entity_store import link_fact
                    for cat, key, val in promoted:
                        link_fact(cat, key, val)
                except Exception:
                    pass
                print(f"[Dreaming]  Deep: ✅ Promoted {len(promoted)} facts")
            else:
                print("[Dreaming]  Deep: All candidates already known")
        else:
            print("[Dreaming]  Deep: Nothing to promote")
    else:
        print("[Dreaming]  REM: No candidates, skipping Deep")

    _last_dream_turn = total
    _save_state({"last_dream_turn": total, "phase": "deep_complete"})


def trigger_dream() -> None:
    global _dreaming_active
    # The loop thread and the chat-save path can both call this; a bare bool
    # is racy (two concurrent _dream_cycle runs double-load/save memory).
    # A lock makes the flag check-and-set atomic and single-owner.
    with _dream_lock:
        if _dreaming_active:
            return
        _dreaming_active = True
    try:
        _dream_cycle()
    finally:
        with _dream_lock:
            _dreaming_active = False


def start_dream_loop(interval_seconds: int = 120) -> None:
    global _dream_thread

    def _loop():
        while True:
            try:
                trigger_dream()
            except Exception as e:
                print(f"[Dreaming] Error: {e}")
            time.sleep(interval_seconds)

    _dream_thread = threading.Thread(target=_loop, daemon=True, name="dreaming")
    _dream_thread.start()
    print(f"[Dreaming] Background 3-phase loop started (interval={interval_seconds}s)")
