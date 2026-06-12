"""Tests for the shared viability predicate in app/agent/viability.py.

Plan 13-01 / D-13-03: all_slots_viable and best_viable_candidate_per_slot must
match rule8_met_per_step_from_state semantics exactly (single source of truth).
"""

from __future__ import annotations

import json
from typing import Any

from app.agent.revision import LOW_SIMILARITY_THRESHOLD
from app.agent.state import ItineraryState, UserConstraints

# ── Helpers ──────────────────────────────────────────────────────────────────

THRESHOLD = LOW_SIMILARITY_THRESHOLD  # 0.55


def _hit(
    similarity: float,
    primary_type: str,
    place_id: str,
) -> dict[str, Any]:
    """Build a minimal semantic_search hit dict."""
    return {
        "similarity": similarity,
        "primary_type": primary_type,
        "place_id": place_id,
        "name": f"Place {place_id}",
    }


def _state_with_hits(
    hits: list[dict[str, Any]],
    requested_types: list[str] | None = None,
    num_stops: int = 1,
    step: int = 0,
) -> ItineraryState:
    """Build an ItineraryState with one semantic_search scratch entry."""
    constraints = UserConstraints(
        requested_primary_types=requested_types or [],
        num_stops=num_stops,
    )
    return ItineraryState(
        scratch={
            "semantic_search": [
                {
                    "step": step,
                    "args": {"query": "test"},
                    "result": hits,
                    "id": "tc0",
                }
            ]
        },
        constraints=constraints,
    )


# ── Import guard ──────────────────────────────────────────────────────────────


def test_viability_module_importable() -> None:
    """The viability module exists and exports the expected functions."""
    from app.agent import viability

    assert hasattr(viability, "all_slots_viable")
    assert hasattr(viability, "best_viable_candidate_per_slot")


def test_no_circular_import_from_graph() -> None:
    """Importing viability must not break graph or revision imports."""
    import app.agent.graph  # noqa: F401
    import app.agent.revision  # noqa: F401
    import app.agent.viability  # noqa: F401


# ── all_slots_viable: typed (requested_primary_types set) ────────────────────


def test_all_slots_viable_true_when_each_type_has_viable_hit() -> None:
    """Returns True when every requested type has one distinct viable candidate."""
    from app.agent.viability import all_slots_viable

    state = _state_with_hits(
        hits=[
            _hit(0.8, "Sushi Restaurant", "pid1"),
            _hit(0.7, "Coffee Shop", "pid2"),
        ],
        requested_types=["Sushi Restaurant", "Coffee Shop"],
    )

    assert all_slots_viable(state, THRESHOLD) is True


def test_all_slots_viable_false_when_one_slot_has_no_hit() -> None:
    """Returns False when one requested type has no viable candidate."""
    from app.agent.viability import all_slots_viable

    state = _state_with_hits(
        hits=[
            _hit(0.8, "Sushi Restaurant", "pid1"),
            # No Coffee Shop hit
        ],
        requested_types=["Sushi Restaurant", "Coffee Shop"],
    )

    assert all_slots_viable(state, THRESHOLD) is False


def test_all_slots_viable_false_when_hit_below_threshold() -> None:
    """Returns False when the only hit has cosine < threshold."""
    from app.agent.viability import all_slots_viable

    state = _state_with_hits(
        hits=[_hit(THRESHOLD - 0.01, "Sushi Restaurant", "pid1")],
        requested_types=["Sushi Restaurant"],
    )

    assert all_slots_viable(state, THRESHOLD) is False


def test_all_slots_viable_true_at_exact_threshold() -> None:
    """Returns True when cosine == threshold exactly (boundary inclusive)."""
    from app.agent.viability import all_slots_viable

    state = _state_with_hits(
        hits=[_hit(THRESHOLD, "Sushi Restaurant", "pid1")],
        requested_types=["Sushi Restaurant"],
    )

    assert all_slots_viable(state, THRESHOLD) is True


def test_all_slots_viable_multiset_needs_distinct_ids() -> None:
    """Two slots of same type need two DISTINCT viable place_ids (WR-02)."""
    from app.agent.viability import all_slots_viable

    # Same place_id returned twice — cannot cover two restaurant slots.
    state = _state_with_hits(
        hits=[
            _hit(0.9, "Restaurant", "pid1"),
            _hit(0.8, "Restaurant", "pid1"),  # duplicate!
        ],
        requested_types=["Restaurant", "Restaurant"],
    )

    assert all_slots_viable(state, THRESHOLD) is False


def test_all_slots_viable_multiset_true_with_distinct_ids() -> None:
    """Two slots of same type satisfied by two distinct place_ids."""
    from app.agent.viability import all_slots_viable

    state = _state_with_hits(
        hits=[
            _hit(0.9, "Restaurant", "pid1"),
            _hit(0.8, "Restaurant", "pid2"),
        ],
        requested_types=["Restaurant", "Restaurant"],
    )

    assert all_slots_viable(state, THRESHOLD) is True


# ── all_slots_viable: untyped (no requested_primary_types) ───────────────────


def test_all_slots_viable_untyped_uses_num_stops() -> None:
    """Without requested types, requires num_stops distinct viable place_ids."""
    from app.agent.viability import all_slots_viable

    state = _state_with_hits(
        hits=[
            _hit(0.9, "Restaurant", "pid1"),
            _hit(0.8, "Restaurant", "pid2"),
        ],
        requested_types=[],
        num_stops=2,
    )

    assert all_slots_viable(state, THRESHOLD) is True


def test_all_slots_viable_untyped_false_not_enough_distinct() -> None:
    """Without requested types, requires num_stops distinct; one hit fails."""
    from app.agent.viability import all_slots_viable

    state = _state_with_hits(
        hits=[_hit(0.9, "Restaurant", "pid1")],
        requested_types=[],
        num_stops=2,
    )

    assert all_slots_viable(state, THRESHOLD) is False


# ── all_slots_viable: edge cases ─────────────────────────────────────────────


def test_all_slots_viable_empty_scratch_returns_false() -> None:
    """Empty scratch returns False without raising."""
    from app.agent.viability import all_slots_viable

    state = ItineraryState(
        scratch={},
        constraints=UserConstraints(requested_primary_types=["Sushi Restaurant"]),
    )

    assert all_slots_viable(state, THRESHOLD) is False


def test_all_slots_viable_malformed_scratch_returns_false() -> None:
    """Malformed scratch entries degrade gracefully (no raise)."""
    from app.agent.viability import all_slots_viable

    state = ItineraryState(
        scratch={"semantic_search": "not-a-list"},
        constraints=UserConstraints(requested_primary_types=["Sushi Restaurant"]),
    )

    assert all_slots_viable(state, THRESHOLD) is False


def test_all_slots_viable_nearby_ignored() -> None:
    """Nearby scratch entries are NOT counted (WR-01: similarity=0.0 hardcoded)."""
    from app.agent.viability import all_slots_viable

    state = ItineraryState(
        scratch={
            "nearby": [
                {
                    "step": 0,
                    "args": {},
                    "result": [_hit(0.9, "Sushi Restaurant", "pid1")],
                    "id": "tc1",
                }
            ]
        },
        constraints=UserConstraints(requested_primary_types=["Sushi Restaurant"]),
    )

    assert all_slots_viable(state, THRESHOLD) is False


# ── Drift guard: agrees with rule8_met_per_step_from_state ───────────────────


def test_all_slots_viable_agrees_with_rule8_met_per_step_from_state() -> None:
    """all_slots_viable must agree with the last element of rule8_met_per_step_from_state.

    D-13-03 / PATTERNS.md "Viability definition: single source of truth".
    """
    from app.agent.viability import all_slots_viable
    from scripts.eval_agent import (
        rule8_met_per_step_from_state,
        viable_candidates_per_step_from_state,
    )

    requested_types = ["Sushi Restaurant", "Coffee Shop"]
    hits = [
        _hit(0.9, "Sushi Restaurant", "pid1"),
        _hit(0.8, "Coffee Shop", "pid2"),
    ]
    state = _state_with_hits(hits, requested_types=requested_types)

    # Build the harness result.
    viable_per_step = viable_candidates_per_step_from_state(state, THRESHOLD, requested_types)
    rule8_bools = rule8_met_per_step_from_state(state, viable_per_step, requested_types, THRESHOLD)

    assert all_slots_viable(state, THRESHOLD) == rule8_bools[-1]


def test_all_slots_viable_agrees_rule8_false_case() -> None:
    """Drift guard for the False case."""
    from app.agent.viability import all_slots_viable
    from scripts.eval_agent import (
        rule8_met_per_step_from_state,
        viable_candidates_per_step_from_state,
    )

    requested_types = ["Sushi Restaurant"]
    hits = [_hit(THRESHOLD - 0.01, "Sushi Restaurant", "pid1")]
    state = _state_with_hits(hits, requested_types=requested_types)

    viable_per_step = viable_candidates_per_step_from_state(state, THRESHOLD, requested_types)
    rule8_bools = rule8_met_per_step_from_state(state, viable_per_step, requested_types, THRESHOLD)

    assert all_slots_viable(state, THRESHOLD) == rule8_bools[-1]


# ── best_viable_candidate_per_slot ───────────────────────────────────────────


def test_best_viable_candidate_per_slot_returns_highest_cosine_per_type() -> None:
    """Returns the highest-cosine hit for each requested type."""
    from app.agent.viability import best_viable_candidate_per_slot

    state = _state_with_hits(
        hits=[
            _hit(0.9, "Sushi Restaurant", "pid1"),
            _hit(0.7, "Sushi Restaurant", "pid2"),
            _hit(0.8, "Coffee Shop", "pid3"),
        ],
        requested_types=["Sushi Restaurant", "Coffee Shop"],
    )

    result = best_viable_candidate_per_slot(state, THRESHOLD)

    assert len(result) == 2
    assert result[0] is not None
    assert result[0]["place_id"] == "pid1"  # highest cosine for Sushi
    assert result[1] is not None
    assert result[1]["place_id"] == "pid3"  # only Coffee Shop hit


def test_best_viable_candidate_per_slot_none_for_uncovered_slot() -> None:
    """Returns None for a slot with no viable candidate."""
    from app.agent.viability import best_viable_candidate_per_slot

    state = _state_with_hits(
        hits=[_hit(0.9, "Sushi Restaurant", "pid1")],
        requested_types=["Sushi Restaurant", "Coffee Shop"],
    )

    result = best_viable_candidate_per_slot(state, THRESHOLD)

    assert result[0] is not None
    assert result[1] is None


def test_best_viable_candidate_per_slot_returns_plain_dicts() -> None:
    """Every returned entry must be a plain dict (JSON-safe for forced-commit synthesizer)."""
    from pydantic import BaseModel

    from app.agent.viability import best_viable_candidate_per_slot

    state = _state_with_hits(
        hits=[_hit(0.9, "Sushi Restaurant", "pid1")],
        requested_types=["Sushi Restaurant"],
    )

    result = best_viable_candidate_per_slot(state, THRESHOLD)

    for entry in result:
        if entry is not None:
            assert isinstance(entry, dict), f"Expected dict, got {type(entry)}"
            assert not isinstance(entry, BaseModel), "Entry must not be a Pydantic model"


def test_best_viable_candidate_per_slot_empty_scratch_returns_nones() -> None:
    """Empty scratch returns a list of None for each requested slot."""
    from app.agent.viability import best_viable_candidate_per_slot

    state = ItineraryState(
        scratch={},
        constraints=UserConstraints(requested_primary_types=["Sushi Restaurant", "Coffee Shop"]),
    )

    result = best_viable_candidate_per_slot(state, THRESHOLD)

    assert len(result) == 2
    assert all(r is None for r in result)


def test_best_viable_candidate_per_slot_multiset_two_distinct() -> None:
    """For two slots of the same type, picks the top-2 distinct place_ids."""
    from app.agent.viability import best_viable_candidate_per_slot

    state = _state_with_hits(
        hits=[
            _hit(0.95, "Restaurant", "pid1"),
            _hit(0.85, "Restaurant", "pid2"),
            _hit(0.75, "Restaurant", "pid3"),
        ],
        requested_types=["Restaurant", "Restaurant"],
    )

    result = best_viable_candidate_per_slot(state, THRESHOLD)

    assert len(result) == 2
    assert result[0] is not None and result[1] is not None
    # The two picks must be distinct place_ids
    assert result[0]["place_id"] != result[1]["place_id"]


# ── Task 2: state_fields tests ────────────────────────────────────────────────


def test_itinerary_state_commit_forced_default_false() -> None:
    """A freshly constructed ItineraryState has commit_forced == False."""
    state = ItineraryState()
    assert state.commit_forced is False


def test_itinerary_state_forced_commit_step_default_none() -> None:
    """A freshly constructed ItineraryState has forced_commit_step is None."""
    state = ItineraryState()
    assert state.forced_commit_step is None


def test_itinerary_state_new_fields_json_safe() -> None:
    """Both new fields serialize cleanly through json.dumps(state.model_dump())."""
    state = ItineraryState(commit_forced=True, forced_commit_step=4)
    dumped = state.model_dump(mode="json")
    # Must not raise
    serialized = json.dumps(dumped)
    parsed = json.loads(serialized)
    assert parsed["commit_forced"] is True
    assert parsed["forced_commit_step"] == 4
