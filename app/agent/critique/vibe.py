"""Cross-stop vibe coherence check via a cheap small model.

Runtime sibling of W6's taste judge — same prompt template, same model
selection. Bounded to 1 call per request and gated by env var.
Catches "fancy Italian → dive bar → fancy dessert" mismatches that the
deterministic checks can't see.

W6 imports `make_judge` to reuse the same model + provider for offline
taste scoring on eval runs.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from app.agent.state import ItineraryState
from app.config import get_settings
from app.llm_factory import build_chat_model

_log = logging.getLogger(__name__)

VIBE_THRESHOLD = 3.0  # 0-5 rubric; below this triggers one revision pass.
VIBE_ENV_VAR = "EVAL_VIBE_CRITIQUE_ENABLED"
JUDGE_MODEL_ENV_VAR = "EVAL_JUDGE_MODEL"
JUDGE_PROVIDER_ENV_VAR = "EVAL_JUDGE_PROVIDER"
DEFAULT_JUDGE_MODEL = "gemini-3.1-flash-lite-preview"
DEFAULT_JUDGE_PROVIDER = "gemini"

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
    # D-10-17 / EVAL-06: this is a SYNC call inside a SYNC LangGraph node
    # (`critique` in app/agent/graph.py). Verified against this repo's pinned
    # LangGraph 1.2.0: sync nodes execute inside a ThreadPoolExecutor thread
    # under `graph.ainvoke` (confirmed via a minimal repro — a sync node
    # returning threading.get_ident() through ainvoke returns a thread id
    # distinct from the main thread's). Therefore this blocking sync
    # judge_llm.invoke does NOT block the asyncio event loop. No async refactor
    # is required; this call stays sync by design. Re-evaluate only if
    # LangGraph is upgraded to a version that changes sync-node dispatch.
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


def make_judge() -> BaseChatModel | None:
    """Construct the cheap LLM used for vibe scoring. Returns None on missing
    credentials so callers can degrade gracefully.

    Provider + model are env-driven so W3 (runtime) and W6 (offline eval)
    pick the same judge without sharing build code.
    """
    provider = os.getenv(JUDGE_PROVIDER_ENV_VAR, DEFAULT_JUDGE_PROVIDER).lower()
    model = os.getenv(JUDGE_MODEL_ENV_VAR, DEFAULT_JUDGE_MODEL)
    s = get_settings()
    try:
        key_attr = {
            "openai": "openai_api_key",
            "gemini": "gemini_api_key",
            "deepseek": "deepseek_api_key",
            "kimi": "moonshot_api_key",
        }.get(provider)
        if key_attr is None:
            _log.warning("unknown vibe judge provider %r; skipping", provider)
            return None
        if not getattr(s, key_attr, ""):
            _log.warning("vibe judge requested but key for provider %r missing; skipping", provider)
            return None
        return build_chat_model(provider, model, temperature=0.0)
    except Exception as e:  # noqa: BLE001
        _log.warning(
            "vibe judge construction failed (provider=%s model=%s): %s", provider, model, e
        )
        return None
