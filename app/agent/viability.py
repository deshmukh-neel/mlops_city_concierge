"""Shared viability predicate for the decisiveness experiment arms.

Plan 13-01 / D-13-03: this module is the SINGLE SOURCE OF TRUTH for the
definition "does every requested stop have at least one viable candidate?"
Both the DEC-02 forced-commit gate (app/agent/graph.py) and the DEC-03
critique-scoping logic (app/agent/revision.py) import from here.

CANONICAL SEMANTICS (must match rule8_met_per_step_from_state exactly):
  viable hit = cosine >= threshold AND primary_type in requested_primary_types
               (when requested_primary_types is non-empty)
             = cosine >= threshold (any type) when requested_primary_types is empty
  coverage = MULTISET (WR-02): each requested-type slot needs its own
             DISTINCT viable place_id; two restaurant slots need two different ids.
  scope = semantic_search scratch ONLY (WR-01): nearby hits hardcode 0.0 AS
          similarity in SQL and cannot clear the threshold.

This module lives under app/agent/ (NOT scripts/) so app/agent/graph.py and
app/agent/revision.py can import it without a circular import.  scripts/eval_agent.py
imports app.agent.*, so placing the predicate there would create a cycle.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import BaseModel

from app.agent.revision import LOW_SIMILARITY_THRESHOLD
from app.agent.state import ItineraryState
from app.tools.filters import family_of

__all__ = ["all_slots_viable", "best_viable_candidate_per_slot", "requested_type_for_hit"]


def requested_type_for_hit(hit_primary_type: str, requested_types: list[str]) -> str | None:
    """Map a hit's primary_type to the requested slot type it satisfies.

    2026-06-15 fix: matching is by FAMILY, not exact string equality. The user
    asks for "Dessert Shop" but real venues are typed "Bakery"/"Ice Cream Shop"
    (zero literal "Dessert Shop" near Hayes Valley); exact-match left the slot
    permanently unsatisfiable. A hit satisfies a requested type when:
      - it equals the requested type exactly (fast path / unmapped types), OR
      - it shares the requested type's family (Bakery ∈ dessert family).

    Returns the matched requested type string (so multiset coverage keys stay the
    requested-type names), or None when the hit matches no requested slot. When
    multiple requested types could match (same family requested twice), the first
    exact match wins, else the first same-family requested type.
    """
    if hit_primary_type in requested_types:
        return hit_primary_type
    hit_family = family_of(hit_primary_type)
    if hit_family is None:
        return None
    for req in requested_types:
        if family_of(req) == hit_family:
            return req
    return None


# ── Helpers ───────────────────────────────────────────────────────────────────


def _value_from_hit(hit: Any, field_name: str) -> Any:
    """Read a field from a dict or object hit (mirrors scripts/eval_agent.value_from_hit)."""
    if isinstance(hit, dict):
        return hit.get(field_name)
    return getattr(hit, field_name, None)


def _is_viable_sim(hit: Any, threshold: float) -> bool:
    """True iff the hit's cosine similarity meets the threshold."""
    sim = _value_from_hit(hit, "similarity")
    if not isinstance(sim, (int, float)) or isinstance(sim, bool):
        return False
    return sim >= threshold


def _place_id(hit: Any) -> str | None:
    """Return the hit's place_id string, or None if absent/empty."""
    pid = _value_from_hit(hit, "place_id")
    return pid if isinstance(pid, str) and pid else None


def _collect_step_hits(state: ItineraryState) -> list[Any]:
    """Collect all semantic_search result hits cumulatively (all steps).

    WR-01: only semantic_search scratch is scanned.  nearby hardcodes
    similarity=0.0 in SQL and can never clear the threshold.
    """
    hits: list[Any] = []
    entries = state.scratch.get("semantic_search")
    if not isinstance(entries, list):
        return hits
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        result = entry.get("result")
        if not isinstance(result, list):
            continue
        hits.extend(result)
    return hits


# ── Public API ────────────────────────────────────────────────────────────────


def all_slots_viable(
    state: ItineraryState,
    threshold: float = LOW_SIMILARITY_THRESHOLD,
) -> bool:
    """Return True iff every requested stop has at least one viable candidate.

    Viable candidate: cosine >= threshold AND (primary_type in
    requested_primary_types when set — else any type counts).

    MULTISET COVERAGE (WR-02): requested_primary_types is treated as a multiset.
    Two restaurant slots require two DISTINCT viable place_ids; the same venue
    returned multiple times counts as one.

    When requested_primary_types is empty, the target is constraints.num_stops
    (or 1 when unset) distinct viable place_ids.

    Malformed/empty scratch returns False without raising.

    This predicate agrees with the LAST element of
    ``scripts.eval_agent.rule8_met_per_step_from_state`` for the same inputs
    (single source of truth — D-13-03).
    """
    hits = _collect_step_hits(state)
    if not hits:
        return False

    requested_types = list(state.constraints.requested_primary_types)

    if not requested_types:
        # Untyped path: count DISTINCT viable place_ids cumulatively (WR-02).
        target = state.constraints.num_stops if state.constraints.num_stops is not None else 1
        seen_ids: set[str] = set()
        anon_count = 0
        for hit in hits:
            if not _is_viable_sim(hit, threshold):
                continue
            pid = _place_id(hit)
            if pid is not None:
                seen_ids.add(pid)
            else:
                anon_count += 1
        return (len(seen_ids) + anon_count) >= target

    # Typed path: multiset coverage (WR-02).
    required = Counter(requested_types)
    per_type_ids: dict[str, set[str]] = {t: set() for t in required}
    per_type_anon: dict[str, int] = dict.fromkeys(required, 0)

    for hit in hits:
        if not _is_viable_sim(hit, threshold):
            continue
        ptype = _value_from_hit(hit, "primary_type")
        if not isinstance(ptype, str):
            continue
        # Family-aware matching (2026-06-15): a "Bakery" hit satisfies a
        # "Dessert Shop" slot. Returns the requested-type key (or None).
        matched = requested_type_for_hit(ptype, requested_types)
        if matched is None:
            continue
        pid = _place_id(hit)
        if pid is not None:
            per_type_ids[matched].add(pid)
        else:
            per_type_anon[matched] += 1

    return all(len(per_type_ids[t]) + per_type_anon[t] >= count for t, count in required.items())


def best_viable_candidate_per_slot(
    state: ItineraryState,
    threshold: float = LOW_SIMILARITY_THRESHOLD,
) -> list[dict[str, Any] | None]:
    """Return the highest-cosine viable hit for each requested slot.

    Returns a list with one entry per requested slot (len ==
    len(requested_primary_types) when set, else len == num_stops).  Each entry
    is the best viable hit dict for that slot, or None when no viable candidate
    exists.

    MULTISET COVERAGE (WR-02): for two slots of the same type, picks the top-2
    distinct place_ids (highest cosine first).  Each slot gets an exclusive
    assignment — the second slot cannot reuse the first slot's place_id.

    All returned entries are plain dicts (JSON-safe), required by the
    DEC-02 forced-commit synthesizer (D-13-03).

    When requested_primary_types is empty, returns a list of num_stops entries
    keyed on the top-N distinct viable place_ids by cosine.

    Malformed/empty scratch returns a list of None entries.
    """
    hits = _collect_step_hits(state)
    requested_types = list(state.constraints.requested_primary_types)

    if not requested_types:
        target = state.constraints.num_stops if state.constraints.num_stops is not None else 1
        # Collect all viable hits with their cosine score.
        viable: list[tuple[float, str, dict[str, Any]]] = []
        for hit in hits:
            if not _is_viable_sim(hit, threshold):
                continue
            sim = _value_from_hit(hit, "similarity")
            pid = _place_id(hit)
            if pid is None:
                continue  # skip anonymous hits in untyped path (cannot deduplicate)
            if isinstance(hit, dict):
                hit_dict: dict[str, Any] = dict(hit)
            elif isinstance(hit, BaseModel):
                hit_dict = hit.model_dump(mode="json")
            else:
                continue  # skip unusable shapes — no {} placeholder
            viable.append((float(sim), pid, hit_dict))

        # Sort by descending cosine, deduplicate on place_id, pick top-target.
        viable.sort(key=lambda x: -x[0])
        seen: set[str] = set()
        top: list[dict[str, Any] | None] = []
        for _, pid, hit_dict in viable:
            if pid in seen:
                continue
            seen.add(pid)
            top.append(hit_dict)
            if len(top) == target:
                break
        # Pad with None if fewer viable candidates than target.
        while len(top) < target:
            top.append(None)
        return top

    # Typed path: one entry per slot in requested_types order; multiset-aware.
    # Group viable hits by type (sorted by descending cosine), deduplicate on place_id.
    type_candidates: dict[str, list[tuple[float, str, dict[str, Any]]]] = {}
    for hit in hits:
        if not _is_viable_sim(hit, threshold):
            continue
        ptype = _value_from_hit(hit, "primary_type")
        if not isinstance(ptype, str):
            continue
        # Family-aware matching (2026-06-15): bucket the hit under the requested
        # slot type it satisfies (Bakery -> "Dessert Shop"), not its raw ptype.
        matched = requested_type_for_hit(ptype, requested_types)
        if matched is None:
            continue
        sim = _value_from_hit(hit, "similarity")
        pid = _place_id(hit)
        if pid is None:
            continue
        if isinstance(hit, dict):
            hit_dict = dict(hit)
        elif isinstance(hit, BaseModel):
            hit_dict = hit.model_dump(mode="json")
        else:
            continue  # skip unusable shapes — no {} placeholder
        type_candidates.setdefault(matched, []).append((float(sim), pid, hit_dict))

    # Sort each type's candidates by descending cosine.
    for cands in type_candidates.values():
        cands.sort(key=lambda x: -x[0])

    # Assign one candidate per slot; multiset-aware: track used ids per type.
    used_per_type: dict[str, set[str]] = {}
    result: list[dict[str, Any] | None] = []
    for slot_type in requested_types:
        cands = type_candidates.get(slot_type, [])
        used = used_per_type.setdefault(slot_type, set())
        picked: dict[str, Any] | None = None
        for _, pid, hit_dict in cands:
            if pid not in used:
                used.add(pid)
                picked = hit_dict
                break
        result.append(picked)

    return result
