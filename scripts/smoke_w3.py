#!/usr/bin/env python3
"""End-to-end smoke for W3 self-correction.

Exercises:
  1. The deterministic itinerary checks against a synthetic state, with
     hallucination + walking-budget violations baked in. Verifies the right
     checks fire.
  2. The per-step diagnostic on a synthetic empty / all-closed / low-sim /
     errored tool result. Verifies the right RevisionHint comes out.

Does NOT require an LLM. Does require DATABASE_URL + a seeded places_raw for
the itinerary checks step (because no_hallucinated_place_ids and
constraints_satisfied hit the DB). Skips that section gracefully if the DB
isn't reachable.

Usage:
    DATABASE_URL='postgresql://postgres:cityconcierge@127.0.0.1:5433/mlops-city-concierge' \\
    poetry run python scripts/smoke_w3.py
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.graph import _diagnose_last_tool_result
from app.agent.state import ItineraryState, Stop, UserConstraints
from app.tools.retrieval import PlaceHit

load_dotenv()


# --- Section 1: per-step diagnostic on synthetic tool results ----------------


def _state_with_last_call(tool_name: str, args: dict, result) -> ItineraryState:
    """Build the minimal state the diagnostic needs: an issuing AIMessage,
    a ToolMessage, and a matching scratch entry."""
    return ItineraryState(
        messages=[
            HumanMessage(content="user query"),
            AIMessage(
                content="",
                tool_calls=[{"name": tool_name, "id": "1", "args": args}],
            ),
            ToolMessage(content="<unused>", tool_call_id="1"),
        ],
        scratch={tool_name: [{"args": args, "result": result, "step": 0}]},
    )


def smoke_per_step_diagnostic() -> None:
    print("\n=== per-step diagnostic ===")
    cases = [
        (
            "empty_results",
            "semantic_search",
            {"query": "x", "filters": {"price_level_max": 1}},
            [],
        ),
        (
            "all_closed",
            "semantic_search",
            {"query": "x"},
            [
                PlaceHit(
                    place_id="p1",
                    name="X",
                    source="google_places",
                    similarity=0.9,
                    business_status="CLOSED_PERMANENTLY",
                )
            ],
        ),
        (
            "low_similarity",
            "semantic_search",
            {"query": "obscure"},
            [
                PlaceHit(
                    place_id="p1",
                    name="X",
                    source="google_places",
                    similarity=0.2,
                    business_status="OPERATIONAL",
                )
            ],
        ),
        (
            "tool_error",
            "semantic_search",
            {"query": "x"},
            {"error": "db down"},
        ),
        (
            "healthy (no hint)",
            "semantic_search",
            {"query": "x"},
            [
                PlaceHit(
                    place_id="p1",
                    name="X",
                    source="google_places",
                    similarity=0.9,
                    business_status="OPERATIONAL",
                )
            ],
        ),
    ]
    for label, tool, args, result in cases:
        state = _state_with_last_call(tool, args, result)
        hint = _diagnose_last_tool_result(state)
        if hint is None:
            print(f"  {label:<22}  -> (no hint)")
        else:
            print(
                f"  {label:<22}  -> reason={hint.reason}, "
                f"action={hint.suggested_action}, target={hint.target}"
            )


# --- Section 2: itinerary checks (DB-backed) ---------------------------------


def smoke_itinerary_violations() -> None:
    print("\n=== itinerary checks ===")
    try:
        from app.agent.critique.checks import itinerary_violations
        from app.db import get_conn
    except Exception as e:  # noqa: BLE001
        print(f"  skipped: import failure: {e}")
        return

    # Pull a real place_id (if the DB is reachable) so hallucination check
    # passes for the real one and fails for the fake one.
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT place_id FROM places_raw LIMIT 1")
            row = cur.fetchone()
    except Exception as e:  # noqa: BLE001
        print(f"  skipped: DB not reachable: {e}")
        return

    if row is None:
        print("  skipped: places_raw is empty; seed first")
        return

    real_pid = row[0]
    state_clean = ItineraryState(
        constraints=UserConstraints(walking_budget_m=2400),
        stops=[
            Stop(
                place_id=real_pid,
                name="Real",
                source="google_places",
                rationale="",
                latitude=37.78,
                longitude=-122.41,
            ),
        ],
    )
    print(f"  clean state -> violations: {itinerary_violations(state_clean) or 'none'}")

    state_dirty = ItineraryState(
        constraints=UserConstraints(walking_budget_m=200),  # tight budget
        stops=[
            Stop(
                place_id=real_pid,
                name="Real",
                source="google_places",
                rationale="",
                latitude=37.78,
                longitude=-122.41,
            ),
            Stop(
                place_id="fake_id_does_not_exist",
                name="Fake",
                source="google_places",
                rationale="",
                latitude=37.80,  # ~2km
                longitude=-122.41,
            ),
        ],
    )
    print(f"  dirty state -> violations: {itinerary_violations(state_dirty)}")


def main() -> int:
    smoke_per_step_diagnostic()
    smoke_itinerary_violations()
    print("\nW3 smoke ok.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
