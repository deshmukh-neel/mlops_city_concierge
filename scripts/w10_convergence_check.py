"""W10 regression oracle — Gemini 3 itinerary convergence check.

Runs the real agent graph (production path: build_agent_graph + ItineraryState
+ graph.ainvoke) N times on the standard 3-stop Mission query and reports how
many runs converge (commit >=1 stop AND produce a non-empty reply).

Pre-W10 (langchain-google-genai 2.1.x + gemini_compat static bypass) this was
~0/6 on gemini-3.1-pro-preview: the agent looped semantic_search/nearby, never
called commit_itinerary, exhausted max_steps. Post-W10 (lcgg 4.x native
thought-signature round-trip) it should converge consistently. This script is
the empirical confirmation of the root-cause diagnosis.

Needs a real provider API key + a live DB (the agent tools query pgvector).
Defaults match the documented local setup: DATABASE_URL -> cloud-sql-proxy on
127.0.0.1:5433, GEMINI_API_KEY / OPENAI_API_KEY from .env.

Usage:
  poetry run python scripts/w10_convergence_check.py \
      --provider gemini --chat-model gemini-3.1-pro-preview --runs 6
  poetry run python scripts/w10_convergence_check.py \
      --provider openai --chat-model gpt-4o-mini --runs 6
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

from app.agent.graph import build_agent_graph  # noqa: E402
from app.agent.state import ItineraryState  # noqa: E402


def make_llm(
    provider: str, chat_model: str | None, temperature: float
) -> tuple[BaseChatModel, str]:
    """Return (llm, resolved_model_name) via the central factory so the oracle
    drives the exact same provider construction as the app."""
    import os

    from app.llm_factory import build_chat_model

    defaults = {
        "openai": "gpt-4o-mini",
        "gemini": "gemini-3.1-pro-preview",
        "deepseek": os.environ.get("DEEPSEEK_MODEL"),
        "kimi": os.environ.get("MOONSHOT_MODEL"),
    }
    model = chat_model or defaults.get(provider)
    if not model:
        raise SystemExit(f"--chat-model required for provider {provider!r}")
    return build_chat_model(provider, model, temperature), model


# The standard 3-stop Mission itinerary query — the exact failing scenario from
# the W10 root-cause investigation (agent returned stops=0 on Gemini 3).
MISSION_QUERY = (
    "Plan me a 3-stop afternoon in the Mission in San Francisco on Saturday "
    "May 16 2026: a casual lunch spot, a coffee place to follow, and a "
    "vibes-y bar for an early evening drink. Keep them walkable from each other."
)


async def _run_once(graph: object) -> tuple[bool, int, int]:
    """Run the agent once. Returns (converged, n_stops, n_steps)."""
    raw = await graph.ainvoke(  # type: ignore[attr-defined]
        ItineraryState(messages=[HumanMessage(content=MISSION_QUERY)])
    )
    state = raw if isinstance(raw, ItineraryState) else ItineraryState(**raw)
    n_stops = len(state.stops)
    reply = (state.final_reply or "").strip()
    converged = n_stops >= 1 and bool(reply)
    return converged, n_stops, state.step_count


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=["openai", "gemini", "deepseek", "kimi"],
        default="gemini",
    )
    parser.add_argument("--chat-model", default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--runs", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args()

    llm, model_name = make_llm(args.provider, args.chat_model, args.temperature)
    graph = build_agent_graph(llm, max_steps=args.max_steps)

    print(
        f"W10 convergence: {args.provider}/{model_name} temp={args.temperature} runs={args.runs}\n"
    )
    converged_count = 0
    for i in range(1, args.runs + 1):
        start = time.monotonic()
        try:
            converged, n_stops, n_steps = await _run_once(graph)
        except Exception as exc:  # noqa: BLE001
            print(f"  run {i}: ERROR {type(exc).__name__}: {exc}")
            continue
        elapsed = time.monotonic() - start
        converged_count += int(converged)
        mark = "PASS" if converged else "FAIL"
        print(f"  run {i}: {mark}  stops={n_stops} steps={n_steps} ({elapsed:.1f}s)")

    print(f"\nConverged: {converged_count}/{args.runs}")
    return 0 if converged_count == args.runs else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
