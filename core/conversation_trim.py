"""Conversation trim policy (2026-09-06, user-decided numbers).

When the raw conversation crosses ~50k tokens, summarize ONLY the oldest
~15k into the rolling summary; the newest turns stay RAW in the prompt so
recent exchanges keep full fidelity (the old behaviour summarized the whole
conversation, causing "context amnesia" after every trim).

Kept as a tiny standalone module so BOTH the runtime (main.py) and the test
suite can import it without pulling main.py's argparse side-effects.
"""

# Raw conversation token budget: trim triggers once this is exceeded.
# (LEGACY — superseded 2026-09-25: the trigger is now "estimated total
#  input >= input_budget" in main.py; kept for test/tooling compatibility.)
RAW_CONV_TRIGGER_TOKENS = 50000

# How many tokens of the OLDEST turns are summarized away per trim.
TRIM_CHUNK_TOKENS = 15000


def compact_slice_tokens(raw_tokens: int) -> int:
    """Oldest-slice size (tokens) to summarize when the input budget hits.

    One rule for every provider:
        slice = min(15_000, max(1_000, raw_conv × 0.45))
        summary target = slice // 3

        non-kaggle (raw ≈34k at trigger): slice 15,000 → ≤5,000 summary
            50k − 15k + 5k = 40k   (spec: next turn stays under trigger)
        kaggle     (raw ≈9k at trigger): slice ≈4,140 → ≈1,380 summary
            25k − 4.1k + 1.4k ≈ 22.4k (same ~20% headroom, recent safe)
    """
    return min(TRIM_CHUNK_TOKENS, max(1000, int(raw_tokens * 0.45)))


def oldest_trim_indices(msgs: list[dict], chunk_tokens: int) -> list[int]:
    """Return indices of the OLDEST messages whose combined size ≈ chunk.

    The cut never splits a tool-call chain: a tool result must stay with the
    assistant message that issued its tool_calls.  The split advances PAST any
    message that is immediately followed by a `role:tool` / `tool_call_id`
    continuation, so the boundary always lands on a clean turn edge.
    """
    from memory.conversation_db import _estimate_tokens

    selected: list[int] = []
    total = 0
    n = len(msgs)

    for i, msg in enumerate(msgs):
        # Catalog the tool-relationship on either side of this message.
        is_assistant_tool_call = bool(msg.get("tool_calls"))
        is_tool_result = bool(msg.get("tool_call_id"))

        # Size: content plus a fixed per-tool overhead.
        cost = _estimate_tokens(msg.get("content", "") or "")
        cost += (len(msg.get("tool_calls") or [])) * 40
        total += cost
        selected.append(i)

        # Boundary safety: when we have reached the budget, stop ONLY if the
        # cut lands on a clean turn edge.  Otherwise keep pulling in the rest
        # of the tool chain.
        if total < chunk_tokens:
            continue

        # next message is a tool result following this assistant's call → the
        # assistant must go with it, so do NOT cut here.
        if is_assistant_tool_call and i + 1 < n and msgs[i + 1].get("tool_call_id"):
            continue
        # this message is itself a tool result → it belongs with the previous
        # assistant; keep iterating so it already got included as a pair.
        if is_tool_result:
            continue

        # Clean edge (a user or non-tool-calling assistant turn ending) — stop.
        break

    return selected