# PlaceCard address / rating / price_level — design

**Date:** 2026-05-16
**Branch:** `fix/placecard-address-rating-price`
**Source:** `implementation_plan/james/FUTURE_WATCH.md` → "`PlaceCard` is missing address / rating / price_level"

## Problem

`state_to_cards` (`app/agent/io.py:38`) projects `Stop` → `PlaceCard`, but `Stop`
never carried `address`, `rating`, or `price_level`. Every card the frontend
renders ships `address: null, rating: null, price_level: null` even though
`places_raw` (and the already-fetched `PlaceDetails`) has all three. `PlaceCard`
declares the fields optional, so the gap is silent. The frontend already reads
`address` and `rating`; `price_level` is in the contract for future use.

## Type mismatch (not flagged in FUTURE_WATCH)

`PlaceDetails.price_level` is the Google v1 enum **string**
(`'PRICE_LEVEL_MODERATE'`), but `PlaceCard.price_level` is typed `int | None`.
The fix must convert string → int 0..4, mirroring the existing SQL function
`price_level_rank()` (alembic `c428add573d7`):

| enum | rank |
|---|---|
| `PRICE_LEVEL_FREE` | 0 |
| `PRICE_LEVEL_INEXPENSIVE` | 1 |
| `PRICE_LEVEL_MODERATE` | 2 |
| `PRICE_LEVEL_EXPENSIVE` | 3 |
| `PRICE_LEVEL_VERY_EXPENSIVE` | 4 |
| anything else / null | None |

## Approach (chosen: A)

Extend `Stop` and populate the fields inside the **existing** batched read in
`enrich_stops_with_booking` (`app/agent/graph.py:98`). That function already
calls `get_details_many(place_ids)` once per commit and iterates each stop's
`PlaceDetails` — the address/rating/price data is already in hand. **Zero new
DB calls.**

Rejected alternatives:
- **B** — re-fetch via `get_details` in `state_to_cards`: N reads per render,
  and `state_to_cards` is a pure projection today.
- **C** — carry raw `PlaceDetails` on `Stop`: bloats state with unused fields.

## Changes

1. **`app/agent/state.py`**
   - `Stop` gains: `address: str | None = None`, `rating: float | None = None`,
     `price_level: int | None = None`.
   - Add module-level helper `price_level_to_rank(value: str | None) -> int | None`
     mirroring the `price_level_rank()` WHEN-table. Lives in `state.py` next to
     `Stop` (the only consumer), keeping the projection in `io.py` pure.

2. **`app/agent/graph.py`** — in `enrich_stops_with_booking`, inside the existing
   `for stop in stops:` loop, after `details = details_by_id.get(stop.place_id)`
   resolves non-None, stamp:
   - `stop.address = details.formatted_address`
   - `stop.rating = details.rating`
   - `stop.price_level = price_level_to_rank(details.price_level)`

   Important ordering nuance: today the loop `continue`s early when `when is
   None` (no booking time) — but address/rating/price do **not** depend on
   `when`. The enrichment of these three fields must happen **before** the
   `when is None` continue, so a stop with no arrival time still gets its
   card data (it just won't get a booking link). The `details is None` /
   DB-failure skips still apply (same degradation as booking links today).

3. **`app/agent/io.py`** — `state_to_cards` passes the three new fields through
   to `PlaceCard`.

## Error handling / degradation

Unchanged failure model: if `get_details_many` raises `psycopg2.Error` or a
`place_id` is missing from the DB at enrichment time, those stops keep
`address/rating/price_level = None`. This is the exact same degradation that
already applies to booking links and is acceptable per the existing design.

## Testing

Per project test-layering convention (unit/mock + smoke + functional):

- **Unit** — new `tests/unit/test_io.py`: `state_to_cards` carries the three
  fields end-to-end from `Stop`. Unit test for `price_level_to_rank` covering
  every enum, unknown string, and `None`.
- **Unit** — extend graph enrichment tests: a stop with `when=None` still gets
  address/rating/price (regression guard for the ordering nuance); DB-failure
  path leaves them `None`.
- **Functional** — existing `test_chat_functional.py` style: a committed
  itinerary's cards expose non-null address/rating where the fixture place has
  them.

## Out of scope

- Frontend changes to display `price_level` (contract-only today).
- Backfilling `Stop` fields on a constraint-edit re-enrich path (no such path
  exists yet; `enrich_stops_with_booking` is already structured to support it
  when it does).
