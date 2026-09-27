"""Query expansion: synonyms + optional HyDE (Hypothetical Document Embedding).
Integrates into search_all_memories for multi-signal search.
"""
import re

# Simple synonym map for common query terms
SYNONYMS = {
    "name": ["called", "named", "known as", "goes by"],
    "work": ["job", "profession", "occupation", "career", "role", "employer"],
    "live": ["reside", "stay", "located", "based", "address", "hometown"],
    "like": ["enjoy", "prefer", "favorite", "love", "into"],
    "project": ["project", "side project", "work", "building", "creating", "developing"],
    "python": ["python", "programming language"],
    "code": ["code", "program", "script", "software", "app"],
    "learn": ["study", "learn", "pick up", "get into"],
    "plan": ["plan", "planning", "going to", "will", "intend", "goal"],
    "want": ["want", "wish", "desire", "hope", "aim"],
    "travel": ["travel", "trip", "visit", "go to", "vacation"],
    "food": ["food", "cuisine", "eat", "dish", "meal", "cooking"],
    "friend": ["friend", "buddy", "pal", "colleague", "partner", "roommate"],
    "pet": ["pet", "dog", "cat", "animal", "rescue"],
    "color": ["color", "shade", "hue", "tone"],
    "movie": ["movie", "film", "show", "cinema", "documentary"],
    "music": ["music", "song", "band", "genre", "playlist"],
    "book": ["book", "read", "reading", "novel", "author"],
    "game": ["game", "gaming", "play", "esport"],
    "sport": ["sport", "exercise", "workout", "gym", "fitness", "activity"],
    "morning": ["morning", "a.m.", "early", "breakfast", "wake up"],
    "night": ["night", "evening", "p.m.", "sleep", "bedtime"],
    "email": ["email", "mail", "inbox", "gmail", "outlook"],
    "phone": ["phone", "cell", "mobile", "telephone", "number", "call"],
    "car": ["car", "vehicle", "drive", "truck"],
    "house": ["house", "home", "apartment", "condo", "place"],
    "school": ["school", "university", "college", "class", "course", "education"],
    "ai": ["ai", "artificial intelligence", "llm", "machine learning", "model"],
}


def expand_query(query: str) -> list[str]:
    """Generate expanded query variants using synonym map."""
    q_lower = query.lower()
    variants = [query]

    # Generate expanded versions with synonyms
    words = q_lower.split()
    for i, w in enumerate(words):
        if w in SYNONYMS:
            for syn in SYNONYMS[w]:
                new_words = words.copy()
                new_words[i] = syn
                variants.append(" ".join(new_words))

    return variants


def generate_hyde_query(query: str) -> str | None:
    """Generate a hypothetical document for HyDE via simple pattern expansion.
    Returns a longer, more descriptive version of the query.
    """
    q = query.strip()
    if len(q) < 5:
        return None

    # Simple pattern-based expansion
    expansions = [
        f"I remember the user talked about {q}",
        f"The user's information about {q}",
        f"Details related to {q}:",
        f"Fact: the user has a {q}",
    ]
    return " ".join(expansions)


def search_with_expansion(query: str, search_fn, top_k: int = 10) -> list:
    """Run search with query expansion. Deduplicates results.
    search_fn: callable that takes (query, top_k) and returns list of (text, meta, score)
    Returns same 3-tuple format as search_fn.
    """
    variants = expand_query(query)
    seen_scores = {}  # sig -> (text, meta, score)

    for vq in variants[:5]:
        try:
            res = search_fn(vq, top_k)
            for text, meta, score in res:
                if isinstance(meta, dict):
                    sig = (text, meta.get("type", ""), meta.get("category", ""), meta.get("key", ""))
                else:
                    sig = (text, str(meta)[:50])
                if sig not in seen_scores or score > seen_scores[sig][2]:
                    seen_scores[sig] = (text, meta, score)
        except Exception:
            continue

    results = sorted(seen_scores.values(), key=lambda x: x[2], reverse=True)
    return results[:top_k]
