"""Cross-stop vibe coherence check via a cheap small model.

Runtime sibling of W6's taste judge — same prompt template, same model
selection idea, but bounded to 1 call per request and gated by env var.
Catches "fancy Italian → dive bar → fancy dessert" mismatches that the
deterministic checks can't see.

The graph wiring (constructing the judge LLM, threading it through
build_agent_graph) is deferred to the W6 PR, where the same judge is
constructed for offline eval. Until then, calling vibe_check with
judge_llm=None or with EVAL_VIBE_CRITIQUE_ENABLED unset returns None.
"""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.messages import HumanMessage

from app.agent.state import ItineraryState

VIBE_THRESHOLD = 3.0  # 0-5 rubric; below this triggers one revision pass.
VIBE_ENV_VAR = "EVAL_VIBE_CRITIQUE_ENABLED"

VIBE_PROMPT = """Rate the vibe coherence of this {n_stops}-stop itinerary on a
0-5 scale where 5 = perfectly matched vibes, 0 = jarring mismatch.

User's request: {user_query}

Stops in order:
{stops_text}

Return JSON only: {{"score": float, "rationale": "one short sentence"}}.
"""


def vibe_check(state: ItineraryState, judge_llm: Any | None) -> float | None:
    """Return a 0-5 score, or None if the check is disabled / inapplicable.

    Returns None when:
    - The env var EVAL_VIBE_CRITIQUE_ENABLED is not "true"
    - judge_llm is None (graph wiring not yet plumbed; W6 will provide)
    - Fewer than 2 stops (vibe coherence undefined for a single stop)
    - The judge response is unparseable (fail open — don't block on parse errors)
    """
    if not is_enabled():
        return None
    if judge_llm is None:
        return None
    if len(state.stops) < 2:
        return None

    user_query = ""
    for m in state.messages:
        if m.__class__.__name__ == "HumanMessage" and isinstance(m.content, str):
            user_query = m.content
            break

    stops_text = "\n".join(
        f"  {i + 1}. {s.name} ({s.primary_type or 'unknown'}) — {s.rationale}"
        for i, s in enumerate(state.stops)
    )
    prompt = VIBE_PROMPT.format(
        n_stops=len(state.stops),
        user_query=user_query,
        stops_text=stops_text,
    )
    raw = judge_llm.invoke([HumanMessage(content=prompt)]).content
    if not isinstance(raw, str):
        return None
    try:
        obj = json.loads(raw)
        return float(obj["score"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def is_enabled() -> bool:
    return os.getenv(VIBE_ENV_VAR, "false").lower() == "true"
