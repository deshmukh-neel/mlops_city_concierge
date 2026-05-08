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
from pydantic import SecretStr

from app.agent.state import ItineraryState
from app.config import get_settings

_log = logging.getLogger(__name__)

VIBE_THRESHOLD = 3.0  # 0-5 rubric; below this triggers one revision pass.
VIBE_ENV_VAR = "EVAL_VIBE_CRITIQUE_ENABLED"
JUDGE_MODEL_ENV_VAR = "EVAL_JUDGE_MODEL"
JUDGE_PROVIDER_ENV_VAR = "EVAL_JUDGE_PROVIDER"
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
DEFAULT_JUDGE_PROVIDER = "openai"

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
        if provider == "openai":
            if not s.openai_api_key:
                _log.warning("vibe judge requested but OPENAI_API_KEY missing; skipping")
                return None
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model=model, api_key=SecretStr(s.openai_api_key), temperature=0.0)
        if provider == "gemini":
            if not s.gemini_api_key:
                _log.warning("vibe judge requested but GEMINI_API_KEY missing; skipping")
                return None
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=SecretStr(s.gemini_api_key),
                temperature=0.0,
            )
    except Exception as e:  # noqa: BLE001
        _log.warning(
            "vibe judge construction failed (provider=%s model=%s): %s", provider, model, e
        )
        return None
    _log.warning("unknown vibe judge provider %r; skipping", provider)
    return None
