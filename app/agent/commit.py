"""Commit LLM-proposed stops into validated Stops + enrich with booking/card data.

Split out of graph.py (FUTURE_WATCH: app/agent/ directory size). `commit_stops`
is the cross-module entry point called by the graph's `act` node;
`enrich_stops_with_booking` is also public so a future constraint-edit path can
refresh booking URLs in place without re-committing through the LLM.
"""

from __future__ import annotations

import logging
from typing import Any

import psycopg2

from app.agent.state import ItineraryState, Stop, price_level_to_rank
from app.tools.booking import propose_booking_from_details
from app.tools.retrieval import get_details_many

logger = logging.getLogger(__name__)


def _grounded_place_ids(scratch: dict[str, Any]) -> set[str]:
    """All place_ids the agent has actually seen via prior tool results."""
    grounded: set[str] = set()
    for entries in scratch.values():
        for entry in entries:
            result = entry.get("result")
            if isinstance(result, list):
                for hit in result:
                    pid = getattr(hit, "place_id", None) or (
                        hit.get("place_id") if isinstance(hit, dict) else None
                    )
                    if pid:
                        grounded.add(pid)
            elif result is not None:
                pid = getattr(result, "place_id", None) or (
                    result.get("place_id") if isinstance(result, dict) else None
                )
                if pid:
                    grounded.add(pid)
    return grounded


def commit_stops(state: ItineraryState, raw_stops: Any) -> tuple[list[Stop], dict[str, Any]]:
    """Validate and coerce LLM-supplied stops into Stop models.

    Returns (committed_stops, tool_result_payload). The payload is what the
    LLM sees back as the tool result; rejected place_ids surface there so the
    model can self-correct in W3.
    """
    if not isinstance(raw_stops, list):
        return [], {"error": "stops must be a list"}
    grounded = _grounded_place_ids(state.scratch)
    committed: list[Stop] = []
    rejected: list[dict[str, Any]] = []
    for raw in raw_stops:
        if not isinstance(raw, dict):
            rejected.append({"reason": "stop must be an object", "value": str(raw)})
            continue
        pid = raw.get("place_id")
        if not pid or pid not in grounded:
            rejected.append({"place_id": pid, "reason": "place_id not seen via prior tool result"})
            continue
        try:
            committed.append(Stop(**raw))
        except Exception as e:  # noqa: BLE001
            rejected.append({"place_id": pid, "reason": f"invalid stop: {e}"})
    enrich_stops_with_booking(committed, state)
    return committed, {
        "committed": [s.place_id for s in committed],
        "rejected": rejected,
    }


def enrich_stops_with_booking(stops: list[Stop], state: ItineraryState) -> None:
    """Stamp booking_url + booking_provider on each committed stop in-place.

    Deterministic — URL construction is a pure transform of (PlaceDetails,
    when, party_size), so the LLM is not involved.

    One batched DB read (get_details_many) covers all stops; previously this
    was an N+1 over commit_itinerary. Per-stop URL building is pure and
    cannot raise ValueError/psycopg2.Error, so the error-handling moved to
    the single batched read: a DB blip skips enrichment for this commit
    (cards ship without booking links), bugs propagate.

    Called by commit_stops on initial commit. Public (no leading underscore)
    so a future constraint-edit path — "make it 4 people instead of 2" or
    "shift dinner to 8pm" — can refresh URLs in place without round-tripping
    through the LLM and re-committing the same place_ids.
    """
    party_size = state.constraints.party_size or 2
    place_ids = [stop.place_id for stop in stops]
    try:
        details_by_id = get_details_many(place_ids)
    except psycopg2.Error:
        # Single point of DB failure for the whole enrichment. Skip enrichment
        # for the entire commit; cards still ship without booking links.
        logger.warning(
            "booking enrichment DB read failed for %d stops",
            len(place_ids),
            exc_info=True,
        )
        return

    for stop in stops:
        details = details_by_id.get(stop.place_id)
        if details is None:
            # place_id grounded in scratch but missing from DB at enrichment
            # time — race condition on the deletion side, or a stale id. Same
            # recoverable case as the old ValueError("unknown place_id"): skip
            # both card-field and booking enrichment for this stop.
            logger.warning("enrichment skipped: place_id=%s not in DB", stop.place_id)
            continue

        # Card fields do NOT depend on a booking time — stamp them before the
        # `when is None` skip so a timeless stop still renders a full card.
        stop.address = details.formatted_address
        stop.rating = details.rating
        stop.price_level = price_level_to_rank(details.price_level)
        # The LLM commits without coordinates (optional in the prompt), so
        # backfill from the DB — otherwise every stop is lat=lng=None and the
        # frontend's `routable` filter drops them all (no pins, no route).
        # Only fill a missing coord: a model-grounded coordinate still wins.
        if stop.latitude is None:
            stop.latitude = details.latitude
        if stop.longitude is None:
            stop.longitude = details.longitude

        when = stop.arrival_time or state.constraints.when
        if when is None:
            # No time → no booking link. Falling back to datetime.now() would
            # embed wall-clock time in the URL, breaking re-commit idempotency
            # (same inputs, different URL each call) and meaning nothing to
            # the user. The card ships without a booking link; downstream can
            # re-enrich once the user supplies a time.
            continue
        proposal = propose_booking_from_details(details, when, party_size)
        stop.booking_url = proposal.booking_url
        stop.booking_provider = proposal.provider
