# W8c — Backend Directions Tool + Re-timing Node Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconcile the agent's final `arrival_time`s against real Google Directions data exactly once per finalized plan, then re-run the open-at-arrival check on real data, appending a truthful caveat if it now fails — without ever putting an API call in the LLM revision loop.

**Architecture:** A new async `app/tools/directions.py` (Google Routes API v2, typed models, 3s timeout, haversine fallback, never raises) is called from a new `retime` graph node placed between `critique` and `END`. The node re-chains arrival times via a new pure `planning.chain_arrival_times()` helper, re-runs only `temporal_coherence` on the real times, and appends a caveat via the existing `_final_with_caveats()` prose helper when a pass flips to fail. The node self-guards to a no-op on every non-routable path (clarifying replies, single-stop, coordless), so behavior is unchanged there.

**Tech Stack:** Python 3.10+, `httpx.AsyncClient` (already a dep), Pydantic, LangGraph `StateGraph`, pytest + `pytest-mock` (no new test deps — `httpx` mocked via `mocker.patch`).

**Spec:** `docs/superpowers/specs/2026-05-17-w8c-directions-retiming-design.md`

**Branch:** `feature/agent-w8c-directions-retiming` (already checked out, == `origin/main` @ 87eb33b).

**Conventions for every task:**
- Single-line commit messages. Pre-commit runs ruff automatically — **do not** run `ruff`/`make format` manually.
- Do **not** run `gh pr merge`. The user merges.
- Run the named test command and confirm the stated expected output before checking a step's box.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `app/config.py` (modify) | Add `google_directions_api_key: str = ""` Settings field |
| `app/tools/directions.py` (create) | Routes API v2 async client, typed models, mode mapping, 3s timeout, haversine fallback. Pure I/O unit, no graph knowledge. |
| `app/agent/planning.py` (modify) | Add `chain_arrival_times()` pure helper — all itinerary-time math stays here |
| `app/agent/graph.py` (modify) | Add `retime` node; rewire `route_after_critique` END→`"retime"` |
| `.env.example` (modify) | Document blank `GOOGLE_DIRECTIONS_API_KEY` |
| `tests/unit/test_tools_directions.py` (create) | Unit: mode matrix + every fallback path |
| `tests/unit/test_agent_planning.py` (modify) | Unit: `chain_arrival_times` |
| `tests/unit/test_agent_graph.py` (modify) | Smoke: node present, ≤1 call, passthrough |
| `tests/unit/test_chat_functional.py` (modify) | Functional: real overwrite + pass→fail flip |
| `tests/integration/test_directions_live.py` (create) | Gated real Directions call |

Task order is dependency-ordered: config → directions tool → planning helper → graph node → functional → integration → docs. Each task is independently committable and leaves the suite green.

---

### Task 1: Add the backend Directions API key to Settings

**Files:**
- Modify: `app/config.py:93` (insert after the `anthropic_api_key` line)
- Modify: `.env.example:90` (insert after the `VITE_GOOGLE_MAPS_MAP_ID=` block)
- Test: `tests/unit/test_config.py` (modify if it exists; else assert inline in directions tests — see Step 1)

- [ ] **Step 1: Write the failing test**

Check whether `tests/unit/test_config.py` exists. If it does, add this test there. If it does not, create `tests/unit/test_config.py` with this content:

```python
from __future__ import annotations

from app.config import get_settings


def test_google_directions_api_key_defaults_empty() -> None:
    """Empty key is the first-class 'use haversine fallback' signal — unit
    tests must get '' by default so the no-key branch is exercised for free."""
    get_settings.cache_clear()
    assert get_settings().google_directions_api_key == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/unit/test_config.py::test_google_directions_api_key_defaults_empty -v`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'google_directions_api_key'`

- [ ] **Step 3: Add the Settings field**

In `app/config.py`, the current lines are:

```python
    anthropic_api_key: str = ""
    mlflow_tracking_uri: str = "http://localhost:5000"
```

Change to:

```python
    anthropic_api_key: str = ""
    # Server-side Google key for the Routes API v2 (W8c re-timing). SEPARATE
    # from the frontend VITE_GOOGLE_MAPS_API_KEY and from scripts'
    # GOOGLE_PLACES_API_KEY. Empty => the Directions tool uses the haversine
    # fallback (graceful, no network).
    google_directions_api_key: str = ""
    mlflow_tracking_uri: str = "http://localhost:5000"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/unit/test_config.py::test_google_directions_api_key_defaults_empty -v`
Expected: PASS

- [ ] **Step 5: Document it in `.env.example`**

The current `.env.example` ends the Google section with:

```
VITE_GOOGLE_MAPS_API_KEY=
VITE_GOOGLE_MAPS_MAP_ID=
```

Add immediately after `VITE_GOOGLE_MAPS_MAP_ID=`:

```

# ── Backend Directions (optional) ─────────────
# Server-side Google key for the Routes API v2 — used by the agent to
# reconcile final arrival times against real travel time. SEPARATE from the
# browser key above and from scripts' GOOGLE_PLACES_API_KEY. Needs the
# "Routes API" enabled. Leave blank to use the offline haversine fallback.
GOOGLE_DIRECTIONS_API_KEY=
```

- [ ] **Step 6: Commit**

```bash
git add app/config.py .env.example tests/unit/test_config.py
git commit -m "feat(w8c): add google_directions_api_key setting"
```

---

### Task 2: Pydantic models for the Directions tool

**Files:**
- Create: `app/tools/directions.py`
- Test: `tests/unit/test_tools_directions.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_tools_directions.py`:

```python
from __future__ import annotations

from app.tools.directions import (
    DEFAULT_MODE,
    DirectionsLeg,
    DirectionsResult,
)


def test_models_construct() -> None:
    leg = DirectionsLeg(duration_s=300, distance_m=420.0)
    result = DirectionsResult(
        legs=[leg],
        total_duration_s=300,
        mode="walk",
        source="google",
    )
    assert result.legs[0].duration_s == 300
    assert result.total_duration_s == 300
    assert result.source == "google"
    assert DEFAULT_MODE == "walk"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/unit/test_tools_directions.py::test_models_construct -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.tools.directions'`

- [ ] **Step 3: Create the module with models and constants only**

Create `app/tools/directions.py`:

```python
"""Google Routes API v2 client for re-timing committed itineraries.

Mirrors app/tools/retrieval.py *structure* (typed Pydantic models, key via
get_settings(), module-level functions, __all__) — NOT its sync DB pattern.
This tool is natively async (httpx.AsyncClient): the W8c retime graph node
awaits route_legs() directly, never via asyncio.to_thread (contrast graph.py
act(), which wraps sync DB tools in a worker thread).

Never raises to the caller: every failure (missing key, timeout, non-200,
malformed body) degrades to a haversine-based estimate so /chat never blocks.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from app.agent.planning import WALKING_SPEED_M_PER_MIN, haversine_m
from app.config import get_settings

_ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
_TIMEOUT_S = 3.0

DEFAULT_MODE = "walk"
_MODE_TO_ROUTES_TRAVELMODE: dict[str, str] = {
    "walk": "WALK",
    "transit": "TRANSIT",
    "drive": "DRIVE",
}


class DirectionsLeg(BaseModel):
    duration_s: int
    distance_m: float


class DirectionsResult(BaseModel):
    legs: list[DirectionsLeg]
    total_duration_s: int
    mode: str
    source: Literal["google", "haversine_fallback"]


__all__ = [
    "DirectionsLeg",
    "DirectionsResult",
    "DEFAULT_MODE",
    "route_legs",
]
```

Note: `route_legs` is in `__all__` now but defined in Task 3. That is intentional — Task 3's first test imports it. If running Task 2 in isolation, the `__all__` entry is harmless (it only matters on `from ... import *`, which no test does).

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/unit/test_tools_directions.py::test_models_construct -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/tools/directions.py tests/unit/test_tools_directions.py
git commit -m "feat(w8c): directions tool models + mode mapping"
```

---

### Task 3: `route_legs` — fallback paths (no network)

These are the paths that must work with **zero** httpx mocking: missing key, unknown mode. We TDD the fallback core first because the unit suite gets it for free (empty key by default from conftest).

**Files:**
- Modify: `app/tools/directions.py`
- Test: `tests/unit/test_tools_directions.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_tools_directions.py`:

```python
import math

import pytest

from app.agent.planning import WALKING_SPEED_M_PER_MIN, haversine_m
from app.tools.directions import route_legs

# Two known SF coords ~1 km apart (Mission Dolores → 16th St BART)
_A = (37.7596, -122.4269)
_B = (37.7651, -122.4194)


async def test_unknown_mode_raises_before_network() -> None:
    with pytest.raises(ValueError, match="unknown mode"):
        await route_legs([_A, _B], mode="teleport")


async def test_missing_key_uses_fallback() -> None:
    # conftest does not set GOOGLE_DIRECTIONS_API_KEY -> '' -> fallback,
    # and no HTTP is attempted.
    result = await route_legs([_A, _B], mode="walk")
    assert result.source == "haversine_fallback"
    assert result.mode == "walk"
    assert len(result.legs) == 1


async def test_fallback_durations_match_haversine() -> None:
    result = await route_legs([_A, _B], mode="walk")
    expected_s = round(haversine_m(_A, _B) / WALKING_SPEED_M_PER_MIN * 60)
    assert result.legs[0].duration_s == expected_s
    assert result.total_duration_s == expected_s
    assert math.isclose(result.legs[0].distance_m, haversine_m(_A, _B), rel_tol=1e-9)


async def test_fallback_handles_three_stops() -> None:
    result = await route_legs([_A, _B, _A], mode="walk")
    assert result.source == "haversine_fallback"
    assert len(result.legs) == 2  # N-1 legs for N stops
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/unit/test_tools_directions.py -k "fallback or unknown_mode" -v`
Expected: FAIL — `ImportError: cannot import name 'route_legs'`

- [ ] **Step 3: Implement `route_legs` + `_haversine_fallback`**

Append to `app/tools/directions.py` (after the models, before `__all__`):

```python
def _haversine_fallback(
    stops: list[tuple[float, float]], mode: str
) -> DirectionsResult:
    legs: list[DirectionsLeg] = []
    for a, b in zip(stops[:-1], stops[1:]):
        dist_m = haversine_m(a, b)
        dur_s = round(dist_m / WALKING_SPEED_M_PER_MIN * 60)
        legs.append(DirectionsLeg(duration_s=dur_s, distance_m=dist_m))
    return DirectionsResult(
        legs=legs,
        total_duration_s=sum(leg.duration_s for leg in legs),
        mode=mode,
        source="haversine_fallback",
    )


async def route_legs(
    stops: list[tuple[float, float]], mode: str = DEFAULT_MODE
) -> DirectionsResult:
    """Per-leg travel time for an ordered list of (lat, lng) stops.

    Never raises: missing key / timeout / non-200 / malformed body all
    degrade to a haversine-based estimate. Unknown mode raises ValueError
    BEFORE any network call (programmer error, not a runtime degradation).
    """
    if mode not in _MODE_TO_ROUTES_TRAVELMODE:
        raise ValueError(
            f"unknown mode {mode!r}; expected one of "
            f"{sorted(_MODE_TO_ROUTES_TRAVELMODE)}"
        )
    if len(stops) < 2:
        return DirectionsResult(
            legs=[], total_duration_s=0, mode=mode, source="haversine_fallback"
        )

    api_key = get_settings().google_directions_api_key
    if not api_key:
        return _haversine_fallback(stops, mode)

    # Network path implemented in Task 4.
    return _haversine_fallback(stops, mode)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/unit/test_tools_directions.py -k "fallback or unknown_mode" -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add app/tools/directions.py tests/unit/test_tools_directions.py
git commit -m "feat(w8c): route_legs haversine fallback + mode guard"
```

---

### Task 4: `route_legs` — the Google network path (httpx mocked)

**Files:**
- Modify: `app/tools/directions.py`
- Test: `tests/unit/test_tools_directions.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_tools_directions.py`. The helper builds a fake `httpx.Response`-like object; `mocker.patch` swaps `httpx.AsyncClient.post`.

```python
import httpx


def _key(monkeypatch) -> None:
    """Set a non-empty key so route_legs takes the network path."""
    monkeypatch.setenv("GOOGLE_DIRECTIONS_API_KEY", "test-directions-key")
    from app.config import get_settings

    get_settings.cache_clear()


def _ok_response(legs: list[dict]) -> httpx.Response:
    return httpx.Response(
        200,
        json={"routes": [{"legs": legs}]},
        request=httpx.Request("POST", "https://x"),
    )


async def test_route_legs_walk_parses_legs(monkeypatch, mocker) -> None:
    _key(monkeypatch)
    mocker.patch(
        "httpx.AsyncClient.post",
        mocker.AsyncMock(
            return_value=_ok_response(
                [
                    {"duration": "780s", "distanceMeters": 1040},
                ]
            )
        ),
    )
    result = await route_legs([_A, _B], mode="walk")
    assert result.source == "google"
    assert result.mode == "walk"
    assert result.legs == [DirectionsLeg(duration_s=780, distance_m=1040.0)]
    assert result.total_duration_s == 780


async def test_route_legs_transit_maps_travelmode(monkeypatch, mocker) -> None:
    _key(monkeypatch)
    post = mocker.AsyncMock(
        return_value=_ok_response([{"duration": "600s", "distanceMeters": 900}])
    )
    mocker.patch("httpx.AsyncClient.post", post)
    await route_legs([_A, _B], mode="transit")
    # 2 stops => single point-to-point request even for transit.
    body = post.call_args.kwargs["json"]
    assert body["travelMode"] == "TRANSIT"


async def test_route_legs_drive_maps_travelmode(monkeypatch, mocker) -> None:
    _key(monkeypatch)
    post = mocker.AsyncMock(
        return_value=_ok_response([{"duration": "300s", "distanceMeters": 2000}])
    )
    mocker.patch("httpx.AsyncClient.post", post)
    await route_legs([_A, _B], mode="drive")
    assert post.call_args.kwargs["json"]["travelMode"] == "DRIVE"


async def test_walk_three_stops_single_request_with_intermediates(
    monkeypatch, mocker
) -> None:
    _key(monkeypatch)
    post = mocker.AsyncMock(
        return_value=_ok_response(
            [
                {"duration": "400s", "distanceMeters": 500},
                {"duration": "500s", "distanceMeters": 600},
            ]
        )
    )
    mocker.patch("httpx.AsyncClient.post", post)
    result = await route_legs([_A, _B, _A], mode="walk")
    assert post.call_count == 1  # one multi-leg request
    assert "intermediates" in post.call_args.kwargs["json"]
    assert len(result.legs) == 2
    assert result.total_duration_s == 900


async def test_transit_with_intermediates_routes_per_leg(
    monkeypatch, mocker
) -> None:
    """Routes API TRANSIT + intermediates is unreliable (W8b prod bug). For
    transit/drive with intermediates, route per-leg point-to-point and stitch.
    Not on the canonical /chat path (backend re-times WALK only) but kept
    honest for the integration test / future use."""
    _key(monkeypatch)
    post = mocker.AsyncMock(
        side_effect=[
            _ok_response([{"duration": "400s", "distanceMeters": 500}]),
            _ok_response([{"duration": "500s", "distanceMeters": 600}]),
        ]
    )
    mocker.patch("httpx.AsyncClient.post", post)
    result = await route_legs([_A, _B, _A], mode="transit")
    assert post.call_count == 2  # N-1 point-to-point requests
    assert len(result.legs) == 2
    assert result.total_duration_s == 900


async def test_timeout_falls_back(monkeypatch, mocker) -> None:
    _key(monkeypatch)
    mocker.patch(
        "httpx.AsyncClient.post",
        mocker.AsyncMock(side_effect=httpx.TimeoutException("slow")),
    )
    result = await route_legs([_A, _B], mode="walk")
    assert result.source == "haversine_fallback"


async def test_non_200_falls_back(monkeypatch, mocker) -> None:
    _key(monkeypatch)
    mocker.patch(
        "httpx.AsyncClient.post",
        mocker.AsyncMock(
            return_value=httpx.Response(
                500, text="err", request=httpx.Request("POST", "https://x")
            )
        ),
    )
    result = await route_legs([_A, _B], mode="walk")
    assert result.source == "haversine_fallback"


async def test_malformed_body_falls_back(monkeypatch, mocker) -> None:
    _key(monkeypatch)
    mocker.patch(
        "httpx.AsyncClient.post",
        mocker.AsyncMock(
            return_value=httpx.Response(
                200, json={"no_routes": True},
                request=httpx.Request("POST", "https://x"),
            )
        ),
    )
    result = await route_legs([_A, _B], mode="walk")
    assert result.source == "haversine_fallback"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/unit/test_tools_directions.py -k "route_legs_walk or travelmode or three_stops or intermediates or timeout or non_200 or malformed" -v`
Expected: FAIL — the network path currently returns `haversine_fallback` unconditionally, so `source == "google"` and `travelMode` assertions fail.

- [ ] **Step 3: Implement the network path**

In `app/tools/directions.py`, add the imports at the top (after `from typing import Literal`):

```python
import httpx
```

Replace the placeholder tail of `route_legs` (the two lines starting `# Network path implemented in Task 4.`) with:

```python
    travel_mode = _MODE_TO_ROUTES_TRAVELMODE[mode]
    try:
        if travel_mode != "WALK" and len(stops) > 2:
            # Routes API TRANSIT/DRIVE + intermediates is unreliable
            # (mirrors the W8b RouteOverlay per-leg fix). Route each leg
            # point-to-point and stitch.
            legs: list[DirectionsLeg] = []
            async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
                for a, b in zip(stops[:-1], stops[1:]):
                    leg = await _request_legs(client, [a, b], travel_mode, api_key)
                    legs.extend(leg)
            return _result_from_legs(legs, mode)

        async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
            legs = await _request_legs(client, stops, travel_mode, api_key)
        return _result_from_legs(legs, mode)
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        return _haversine_fallback(stops, mode)
```

Add these two helpers above `route_legs`:

```python
def _parse_duration_s(raw: str) -> int:
    # Routes API returns protobuf Duration strings like "780s".
    return int(raw.rstrip("s"))


def _result_from_legs(
    legs: list[DirectionsLeg], mode: str
) -> DirectionsResult:
    return DirectionsResult(
        legs=legs,
        total_duration_s=sum(leg.duration_s for leg in legs),
        mode=mode,
        source="google",
    )


async def _request_legs(
    client: httpx.AsyncClient,
    stops: list[tuple[float, float]],
    travel_mode: str,
    api_key: str,
) -> list[DirectionsLeg]:
    """One computeRoutes POST. Raises on non-2xx or malformed body so the
    caller's except-clause can fall back."""

    def _waypoint(p: tuple[float, float]) -> dict:
        return {"location": {"latLng": {"latitude": p[0], "longitude": p[1]}}}

    body: dict = {
        "origin": _waypoint(stops[0]),
        "destination": _waypoint(stops[-1]),
        "travelMode": travel_mode,
    }
    if len(stops) > 2:
        body["intermediates"] = [_waypoint(p) for p in stops[1:-1]]

    resp = await client.post(
        _ROUTES_URL,
        json=body,
        headers={
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": "routes.legs.duration,routes.legs.distanceMeters",
        },
    )
    resp.raise_for_status()
    routes = resp.json()["routes"]
    raw_legs = routes[0]["legs"]
    return [
        DirectionsLeg(
            duration_s=_parse_duration_s(leg["duration"]),
            distance_m=float(leg["distanceMeters"]),
        )
        for leg in raw_legs
    ]
```

Note: `resp.raise_for_status()` raises `httpx.HTTPStatusError` (a subclass of `httpx.HTTPError`); `resp.json()["routes"]` raises `KeyError` on the malformed body — both caught by the `except` clause.

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/unit/test_tools_directions.py -v`
Expected: PASS (all directions unit tests — models, fallbacks, network, mode matrix, failure modes)

- [ ] **Step 5: Commit**

```bash
git add app/tools/directions.py tests/unit/test_tools_directions.py
git commit -m "feat(w8c): route_legs Routes API v2 network path + per-leg transit"
```

---

### Task 5: `chain_arrival_times` pure helper

**Files:**
- Modify: `app/agent/planning.py` (add after `next_arrival_time`, before `remaining_walking_budget_m`)
- Test: `tests/unit/test_agent_planning.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_agent_planning.py`:

```python
from datetime import timedelta

from app.agent.planning import chain_arrival_times


def _stop_at(pid: str, *, arrival=None, duration=60):
    return Stop(
        place_id=pid,
        name=pid.upper(),
        source="google_places",
        rationale="",
        arrival_time=arrival,
        planned_duration_min=duration,
    )


def test_chain_arrival_times_empty_is_noop() -> None:
    assert chain_arrival_times([], []) == []


def test_chain_arrival_times_single_stop_unchanged() -> None:
    start = datetime(2026, 5, 17, 18, 0, tzinfo=timezone.utc)
    stops = [_stop_at("p1", arrival=start)]
    out = chain_arrival_times(stops, [])
    assert out[0].arrival_time == start


def test_chain_arrival_times_chains_with_real_legs() -> None:
    start = datetime(2026, 5, 17, 18, 0, tzinfo=timezone.utc)
    stops = [
        _stop_at("p1", arrival=start, duration=90),
        _stop_at("p2", duration=60),
        _stop_at("p3", duration=60),
    ]
    # Two legs: 10 min then 25 min of travel.
    out = chain_arrival_times(stops, [10.0, 25.0])
    assert out[0].arrival_time == start  # start preserved
    assert out[1].arrival_time == start + timedelta(minutes=90 + 10)
    assert out[2].arrival_time == start + timedelta(minutes=90 + 10 + 60 + 25)


def test_chain_arrival_times_requires_start_arrival() -> None:
    stops = [_stop_at("p1", arrival=None), _stop_at("p2")]
    with pytest.raises(ValueError, match="arrival_time"):
        chain_arrival_times(stops, [10.0])


def test_chain_arrival_times_does_not_mutate_input() -> None:
    start = datetime(2026, 5, 17, 18, 0, tzinfo=timezone.utc)
    stops = [_stop_at("p1", arrival=start, duration=30), _stop_at("p2")]
    chain_arrival_times(stops, [5.0])
    assert stops[1].arrival_time is None  # original untouched
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/unit/test_agent_planning.py -k chain_arrival_times -v`
Expected: FAIL — `ImportError: cannot import name 'chain_arrival_times'`

- [ ] **Step 3: Implement the helper**

In `app/agent/planning.py`, after `next_arrival_time` (ends at the `return prev_stop.arrival_time + timedelta(...)` line) and before `def remaining_walking_budget_m`, insert:

```python
def chain_arrival_times(
    stops: list[Stop], leg_durations_min: list[float]
) -> list[Stop]:
    """Re-chain arrival_times from explicit per-leg travel minutes.

    Unlike next_arrival_time (which derives travel from haversine), this takes
    travel time as data — used by the W8c retime node to apply real Google
    Directions durations. stops[0].arrival_time is the user's start time and is
    preserved. Returns a NEW list of model copies; the input is not mutated.

    `leg_durations_min[i]` is the travel time from stops[i] to stops[i+1], so
    len(leg_durations_min) must be >= len(stops) - 1 (extra entries ignored).
    """
    if not stops:
        return []
    if stops[0].arrival_time is None:
        raise ValueError("stops[0].arrival_time must be set before chaining")
    out = [stops[0].model_copy()]
    cursor = stops[0].arrival_time
    for i in range(1, len(stops)):
        leg_min = leg_durations_min[i - 1]
        cursor = cursor + timedelta(
            minutes=stops[i - 1].planned_duration_min + leg_min
        )
        out.append(stops[i].model_copy(update={"arrival_time": cursor}))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/unit/test_agent_planning.py -k chain_arrival_times -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add app/agent/planning.py tests/unit/test_agent_planning.py
git commit -m "feat(w8c): chain_arrival_times pure helper in planning"
```

---

### Task 6: The `retime` graph node + rewiring

**Files:**
- Modify: `app/agent/graph.py` (imports near line 29-39; new node inside `build_agent_graph`; `route_after_critique` at 229-230; wiring at 232-240)
- Test: `tests/unit/test_agent_graph.py`

- [ ] **Step 1: Write the failing smoke tests**

First read the existing helpers at the top of `tests/unit/test_agent_graph.py` (`_make_fake`, imports). Append these tests (they use the same `_make_fake` / `ItineraryState` / `HumanMessage` already imported in that file):

```python
from app.agent.state import Stop
from app.tools.directions import DirectionsLeg, DirectionsResult


def _committed_state(n_stops: int, *, with_coords: bool) -> ItineraryState:
    base = datetime(2026, 5, 17, 18, 0, tzinfo=timezone.utc)
    stops = []
    for i in range(n_stops):
        stops.append(
            Stop(
                place_id=f"p{i}",
                name=f"P{i}",
                source="google_places",
                rationale="",
                arrival_time=base if i == 0 else None,
                planned_duration_min=60,
                latitude=37.77 + i * 0.01 if with_coords else None,
                longitude=-122.41 if with_coords else None,
            )
        )
    return ItineraryState(stops=stops, done=True, final_reply="Here is your plan.")


async def test_retime_node_present_and_routed(monkeypatch) -> None:
    """route_after_critique returns 'retime' (not END) when done; graph has
    the node."""
    fake = _make_fake([AIMessage(content="done", tool_calls=[])])
    graph = build_agent_graph(fake, max_steps=4)
    assert "retime" in graph.get_graph().nodes


async def test_retime_at_most_one_directions_call(monkeypatch, mocker) -> None:
    calls = {"n": 0}

    async def _counting_route_legs(stops, mode="walk"):
        calls["n"] += 1
        return DirectionsResult(
            legs=[DirectionsLeg(duration_s=600, distance_m=800.0)] * (len(stops) - 1),
            total_duration_s=600 * (len(stops) - 1),
            mode=mode,
            source="google",
        )

    mocker.patch("app.agent.graph.route_legs", _counting_route_legs)
    # The node calls temporal_coherence (imported into graph.py in Step 3),
    # NOT itinerary_violations. Patch the symbol the node actually uses.
    mocker.patch("app.agent.graph.temporal_coherence", lambda _s: 1.0)
    fake = _make_fake([AIMessage(content="done", tool_calls=[])])
    graph = build_agent_graph(fake, max_steps=4)
    await graph.ainvoke(_committed_state(3, with_coords=True))
    assert calls["n"] == 1


async def test_retime_passthrough_when_not_routable(monkeypatch, mocker) -> None:
    route = mocker.patch("app.agent.graph.route_legs")
    fake = _make_fake([AIMessage(content="done", tool_calls=[])])
    graph = build_agent_graph(fake, max_steps=4)
    out = await graph.ainvoke(_committed_state(1, with_coords=True))
    route.assert_not_called()
    assert out["stops"][0].arrival_time == datetime(
        2026, 5, 17, 18, 0, tzinfo=timezone.utc
    )


async def test_retime_passthrough_when_coordless(monkeypatch, mocker) -> None:
    route = mocker.patch("app.agent.graph.route_legs")
    fake = _make_fake([AIMessage(content="done", tool_calls=[])])
    graph = build_agent_graph(fake, max_steps=4)
    await graph.ainvoke(_committed_state(3, with_coords=False))
    route.assert_not_called()
```

If `datetime`/`timezone` are not imported at the top of `test_agent_graph.py`, add `from datetime import datetime, timezone` to its imports.

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/unit/test_agent_graph.py -k retime -v`
Expected: FAIL — `test_retime_node_present_and_routed` fails (`'retime' not in nodes`); the `mocker.patch("app.agent.graph.route_legs")` tests fail with `AttributeError: module app.agent.graph has no attribute route_legs`.

- [ ] **Step 3: Add imports to `graph.py`**

The current import block (lines ~29-39) is:

```python
from app.agent.commit import commit_stops
from app.agent.critique import vibe
from app.agent.prompts import SYSTEM_PROMPT
from app.agent.revision import (
    critique_final_with_stops,
    critique_step,
    finalize_as_is,
    short_circuit_max_steps,
)
from app.agent.state import ItineraryState, Stop
from app.agent.tools import COMMIT_ITINERARY_TOOL_NAME, all_tools
```

Add these two imports (place after the `revision` import block, before `from app.agent.state`):

```python
from app.agent.critique.checks import CRITIQUE_THRESHOLDS, temporal_coherence
from app.agent.planning import chain_arrival_times
from app.agent.revision import _final_with_caveats
from app.tools.directions import route_legs
```

Note: `_final_with_caveats` is module-internal in `revision.py` but the existing test pattern already imports underscore-prefixed names by their private name (revision.py docstring sanctions this). Importing it here keeps caveat prose in one place.

- [ ] **Step 4: Add the `retime` node and rewire**

Inside `build_agent_graph`, after the `def critique(...)` function and before `def route_after_plan`, add:

```python
    async def retime(state: ItineraryState) -> dict[str, Any]:
        """Reconcile final arrival_times against real Google Directions once.

        Self-guards to a no-op on every non-routable path (not done, <2
        stops with coords). route_legs is natively async (httpx) — awaited
        directly here, NOT via asyncio.to_thread (contrast act(), which
        wraps sync DB tools in a worker thread)."""
        if not state.done or not state.stops:
            return {}
        coords = [
            (s.latitude, s.longitude)
            for s in state.stops
            if s.latitude is not None and s.longitude is not None
        ]
        if len(coords) < 2 or len(coords) != len(state.stops):
            # Mixed/absent coords: the haversine arrival_time already on the
            # stops is the best we have. Leave it.
            return {}

        result = await route_legs(coords, mode="walk")
        leg_min = [leg.duration_s / 60 for leg in result.legs]
        retimed = chain_arrival_times(state.stops, leg_min)

        update: dict[str, Any] = {"stops": retimed}

        # Re-run ONLY the open-at-arrival check on the real times. Other
        # checks (geographic/walking/hallucination) are coord/id-based and
        # unaffected by re-timing, so re-running them would be wasted work.
        probe = state.model_copy(update={"stops": retimed})
        try:
            score = temporal_coherence(probe)
        except Exception:  # noqa: BLE001
            # Fails open exactly like itinerary_violations(): a DB blip must
            # not block /chat. Ship the re-timed plan without the re-check.
            return update

        if score < CRITIQUE_THRESHOLDS["temporal_coherence"]:
            update["final_reply"] = _final_with_caveats(
                state.final_reply or "", ["temporal_coherence"]
            )
        return update
```

Change `route_after_critique` from:

```python
    def route_after_critique(state: ItineraryState) -> str:
        return END if state.done else "plan"
```

to:

```python
    def route_after_critique(state: ItineraryState) -> str:
        # Every finalized plan flows through `retime` (was END). retime
        # self-guards routability and returns {} when there's nothing to do.
        return "retime" if state.done else "plan"
```

Change the wiring block from:

```python
    g.add_node("plan", plan)
    g.add_node("act", act)
    g.add_node("critique", critique)
    g.set_entry_point("plan")
    g.add_conditional_edges("plan", route_after_plan, {"act": "act", "critique": "critique"})
    g.add_edge("act", "critique")
    g.add_conditional_edges("critique", route_after_critique, {"plan": "plan", END: END})
    return g.compile()
```

to:

```python
    g.add_node("plan", plan)
    g.add_node("act", act)
    g.add_node("critique", critique)
    g.add_node("retime", retime)
    g.set_entry_point("plan")
    g.add_conditional_edges("plan", route_after_plan, {"act": "act", "critique": "critique"})
    g.add_edge("act", "critique")
    g.add_conditional_edges(
        "critique", route_after_critique, {"plan": "plan", "retime": "retime"}
    )
    g.add_edge("retime", END)
    return g.compile()
```

- [ ] **Step 5: Run the retime smoke tests**

Run: `poetry run pytest tests/unit/test_agent_graph.py -k retime -v`
Expected: PASS (4 passed)

- [ ] **Step 6: Run the full graph test file (no regressions)**

Run: `poetry run pytest tests/unit/test_agent_graph.py -v`
Expected: PASS (all pre-existing graph tests still green — the new node is a no-op for every existing test since none set `done=True` with ≥2 coord stops at the critique boundary; verify count is prior count + 4)

- [ ] **Step 7: Commit**

```bash
git add app/agent/graph.py tests/unit/test_agent_graph.py
git commit -m "feat(w8c): retime graph node + route_after_critique rewire"
```

---

### Task 7: Functional tests — real overwrite + pass→fail flip

**Files:**
- Modify: `tests/unit/test_chat_functional.py`

These run the **real graph** with a scripted LLM; only the LLM, retrieval, and the Directions/DB I/O are stubbed. This is where the spec's headline guarantees are proven end-to-end through `/chat`.

- [ ] **Step 1: Write the failing functional tests**

Append to `tests/unit/test_chat_functional.py`. Reuse the file's existing `_ScriptedLLM`, `_stub_loaded_config`, and the `monkeypatch.setattr("app.agent.tools._semantic_search", ...)` pattern already in that file.

```python
from datetime import datetime, timezone

from app.tools.directions import DirectionsLeg, DirectionsResult


def _two_stop_script() -> list[AIMessage]:
    return [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "semantic_search", "id": "s1", "args": {"query": "date"}}
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "commit_itinerary",
                    "id": "c1",
                    "args": {
                        "stops": [
                            {
                                "place_id": "p1",
                                "name": "Bar One",
                                "rationale": "start",
                                "source": "google_places",
                                "primary_type": "cocktail_bar",
                            },
                            {
                                "place_id": "p2",
                                "name": "Bar Two",
                                "rationale": "next",
                                "source": "google_places",
                                "primary_type": "cocktail_bar",
                            },
                        ]
                    },
                }
            ],
        ),
        AIMessage(content="Bar One then Bar Two.", tool_calls=[]),
    ]


def _two_hits(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.agent.tools._semantic_search",
        lambda **_kw: [
            PlaceHit(
                place_id="p1", name="Bar One", source="google_places",
                similarity=0.9, latitude=37.770, longitude=-122.410,
                business_status="OPERATIONAL", primary_type="cocktail_bar",
                formatted_address="1 A St", snippet=None,
            ),
            PlaceHit(
                place_id="p2", name="Bar Two", source="google_places",
                similarity=0.9, latitude=37.780, longitude=-122.410,
                business_status="OPERATIONAL", primary_type="cocktail_bar",
                formatted_address="2 B St", snippet=None,
            ),
        ],
    )


def test_chat_retimes_arrival_with_real_directions(monkeypatch, mocker) -> None:
    """The committed plan's arrival_time is overwritten by real Directions
    data — not the haversine estimate."""
    _two_hits(monkeypatch)

    async def _slow_directions(stops, mode="walk"):
        # 40 real minutes between the two stops — far more than haversine.
        return DirectionsResult(
            legs=[DirectionsLeg(duration_s=2400, distance_m=3000.0)],
            total_duration_s=2400, mode=mode, source="google",
        )

    mocker.patch("app.agent.graph.route_legs", _slow_directions)
    # temporal_coherence passes (both open) so no caveat noise here.
    mocker.patch("app.agent.graph.temporal_coherence", lambda _s: 1.0)
    # The real graph runs the real critique node -> critique_final_with_stops
    # -> itinerary_violations. revision.py does `from ...checks import
    # itinerary_violations` (revision.py:18), so the name is bound in
    # app.agent.revision — patch THERE, not the definition module, or the
    # loop's call is not intercepted and the plan may divert into a revision.
    mocker.patch("app.agent.revision.itinerary_violations", lambda _s: [])

    real_graph = build_agent_graph(_ScriptedLLM(scripted=_two_stop_script()), max_steps=6)
    mocker.patch("app.main.load_registered_rag_chain", return_value=_stub_loaded_config())
    mocker.patch("app.main.build_agent_graph", return_value=real_graph)

    with TestClient(app) as client:
        body = client.post(
            "/chat",
            json={"message": "a date with two bars, arrive 6pm"},
        ).json()

    a1 = body["places"][0]["arrival_time"]
    a2 = body["places"][1]["arrival_time"]
    # Stop 2 arrival = stop1 arrival + 90 default cocktail_bar duration?
    # cocktail_bar default = 60 (DEFAULT_STOP_DURATION_MIN). 60 + 40 travel.
    assert a1 is not None and a2 is not None
    delta_min = (
        datetime.fromisoformat(a2) - datetime.fromisoformat(a1)
    ).total_seconds() / 60
    assert delta_min == 100  # 60 dwell + 40 real travel (NOT the ~2 min haversine)


def test_chat_retiming_flips_temporal_pass_to_fail(monkeypatch, mocker) -> None:
    """Slower real travel pushes stop 2 past its closing time: the re-run
    temporal check fails and the reply gains the caveat."""
    _two_hits(monkeypatch)

    async def _slow_directions(stops, mode="walk"):
        return DirectionsResult(
            legs=[DirectionsLeg(duration_s=3600, distance_m=4000.0)],
            total_duration_s=3600, mode=mode, source="google",
        )

    mocker.patch("app.agent.graph.route_legs", _slow_directions)
    # Loop-time check passes (haversine arrival is early); the re-run on real
    # times fails. Sequence the two calls:
    scores = iter([1.0, 0.0])  # loop pass, retime fail
    mocker.patch(
        "app.agent.graph.temporal_coherence", lambda _s: next(scores)
    )
    # The real graph runs the real critique node -> critique_final_with_stops
    # -> itinerary_violations. revision.py does `from ...checks import
    # itinerary_violations` (revision.py:18), so the name is bound in
    # app.agent.revision — patch THERE, not the definition module, or the
    # loop's call is not intercepted and the plan may divert into a revision.
    mocker.patch("app.agent.revision.itinerary_violations", lambda _s: [])

    real_graph = build_agent_graph(_ScriptedLLM(scripted=_two_stop_script()), max_steps=6)
    mocker.patch("app.main.load_registered_rag_chain", return_value=_stub_loaded_config())
    mocker.patch("app.main.build_agent_graph", return_value=real_graph)

    with TestClient(app) as client:
        body = client.post(
            "/chat", json={"message": "two bars, arrive 6pm"}
        ).json()

    assert "Caveats" in body["reply"]
    assert "temporal_coherence" in body["reply"]
    assert body["reply"].startswith("Bar One then Bar Two.")


def test_chat_directions_failure_keeps_haversine_reply(monkeypatch, mocker) -> None:
    """route_legs internally degrades to fallback -> /chat still 200, no
    spurious caveat, arrival_times come from the haversine fallback."""
    _two_hits(monkeypatch)

    async def _fallback_directions(stops, mode="walk"):
        return DirectionsResult(
            legs=[DirectionsLeg(duration_s=120, distance_m=160.0)],
            total_duration_s=120, mode=mode, source="haversine_fallback",
        )

    mocker.patch("app.agent.graph.route_legs", _fallback_directions)
    mocker.patch("app.agent.graph.temporal_coherence", lambda _s: 1.0)
    # The real graph runs the real critique node -> critique_final_with_stops
    # -> itinerary_violations. revision.py does `from ...checks import
    # itinerary_violations` (revision.py:18), so the name is bound in
    # app.agent.revision — patch THERE, not the definition module, or the
    # loop's call is not intercepted and the plan may divert into a revision.
    mocker.patch("app.agent.revision.itinerary_violations", lambda _s: [])

    real_graph = build_agent_graph(_ScriptedLLM(scripted=_two_stop_script()), max_steps=6)
    mocker.patch("app.main.load_registered_rag_chain", return_value=_stub_loaded_config())
    mocker.patch("app.main.build_agent_graph", return_value=real_graph)

    with TestClient(app) as client:
        resp = client.post("/chat", json={"message": "two bars"})

    assert resp.status_code == 200
    body = resp.json()
    assert "Caveats" not in body["reply"]
    assert body["reply"] == "Bar One then Bar Two."
```

- [ ] **Step 2: Run tests to verify they pass against the real implementation**

These are *test-adding* tests for behavior already implemented in Task 6 (the implementation legitimately precedes these characterization tests — TDD red-first applied to the units in Tasks 1-6, not to this end-to-end coverage layer).

Run: `poetry run pytest tests/unit/test_chat_functional.py -k "retimes or flips or directions_failure" -v`
Expected: PASS (3 passed). The plan's constants are pre-verified against source: `_final_with_caveats(content, ["temporal_coherence"])` produces exactly `content + "\n\nCaveats: I couldn't fully satisfy temporal_coherence after revisions. You may want to adjust the plan."` (`app/agent/revision.py:172-180`), and `DEFAULT_STOP_DURATION_MIN["cocktail_bar"] == 60` (`app/agent/state.py:96`) — so `delta_min == 100` (60 dwell + 40 real travel) and the `"Caveats"`/`"temporal_coherence"`/`startswith` assertions are correct as written.

- [ ] **Step 3: If (and only if) a test fails, the test is wrong, not the code**

If any assertion fails, the implementation from Task 6 is authoritative — re-read `app/agent/revision.py:_final_with_caveats` and `app/agent/state.py:DEFAULT_STOP_DURATION_MIN` and correct the test's expected constant to match. Do **not** edit `graph.py`/`directions.py`/`planning.py` to satisfy a test constant. Re-run until green.

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/unit/test_chat_functional.py -v`
Expected: PASS (all functional tests, pre-existing + 3 new)

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_chat_functional.py
git commit -m "test(w8c): functional real-overwrite + temporal pass-to-fail flip"
```

---

### Task 8: Gated integration test

**Files:**
- Create: `tests/integration/test_directions_live.py`

- [ ] **Step 1: Write the test**

Create `tests/integration/test_directions_live.py`:

```python
"""Live Google Routes API v2 check for the W8c Directions tool.

Gated: skipped unless APP_ENV=integration AND GOOGLE_DIRECTIONS_API_KEY is
set, mirroring tests/integration/test_db.py:19-21. Run with:

    APP_ENV=integration GOOGLE_DIRECTIONS_API_KEY=... make test-integration
"""

from __future__ import annotations

import os

import pytest

from app.tools.directions import route_legs

pytestmark = pytest.mark.skipif(
    os.getenv("APP_ENV", "test") != "integration"
    or not os.getenv("GOOGLE_DIRECTIONS_API_KEY"),
    reason=(
        "Set APP_ENV=integration and GOOGLE_DIRECTIONS_API_KEY to run live "
        "Directions tests."
    ),
)

# Mission Dolores -> 16th St BART, ~1 km, definitely walkable.
_A = (37.7596, -122.4269)
_B = (37.7651, -122.4194)


async def test_route_legs_live_walk() -> None:
    result = await route_legs([_A, _B], mode="walk")
    assert result.source == "google"
    assert result.mode == "walk"
    assert len(result.legs) == 1
    assert 0 < result.total_duration_s < 7200  # <2h for a 1km walk
```

- [ ] **Step 2: Verify it skips by default**

Run: `poetry run pytest tests/integration/test_directions_live.py -v`
Expected: 1 skipped (because `APP_ENV` defaults to `test` via conftest)

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_directions_live.py
git commit -m "test(w8c): gated live Routes API integration test"
```

---

### Task 9: Full suite verification + lint/type gate

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `make test`
Expected: PASS — all prior tests + W8c additions green, coverage reported. Confirm the total count increased and there are **zero** failures/errors. (One known pre-existing flake may exist per `w10` baseline notes; if a single unrelated flake appears, re-run that test in isolation to confirm it is the known flake and not W8c-induced.)

- [ ] **Step 2: Typecheck**

Run: `make typecheck`
Expected: PASS — `mypy app/` clean. If `route_legs`/`chain_arrival_times` raise typing complaints, fix the annotations (do not `# type: ignore` without a reason comment).

- [ ] **Step 3: Lint (verification only — do NOT run ruff manually to fix)**

Run: `make lint`
Expected: PASS — `ruff check .` clean. Pre-commit already formatted on each commit; this step only confirms CI will pass. If lint fails, fix the specific finding and amend the relevant commit (small focused commits).

- [ ] **Step 4: Confirm the ≤1-call invariant holds end-to-end**

Run: `poetry run pytest tests/unit/test_agent_graph.py::test_retime_at_most_one_directions_call tests/unit/test_chat_functional.py -k "retimes or flips or directions_failure" -v`
Expected: PASS — re-confirms the Tiered design's single-call guarantee through the real graph.

- [ ] **Step 5: No commit** (this task is a gate; nothing to commit unless a fix was needed in Steps 2-3, which should have been committed there)

---

### Task 10: Status doc updates (ONLY after the PR is merged)

> Per `CLAUDE.md`: do not update mid-workstream — only on merge. This task is a **post-merge checklist**, not part of the implementation PR. Leave the boxes unchecked until the user confirms the W8c PR merged to `main`.

**Files:**
- Modify: `implementation_plan/james/README.md` (the W8 status row)
- Modify: `implementation_plan/james/w8_live_map_routing.md` (the `**Status:**` footer, w8c bullet)

- [ ] **Step 1 (post-merge): Flip the README status row**

In `implementation_plan/james/README.md`, change the W8c status cell to `✅ Merged` with the PR link.

- [ ] **Step 2 (post-merge): Update the w8c footer bullet**

In `implementation_plan/james/w8_live_map_routing.md`, change the final bullet from:

```
- **w8c** (Directions tool + re-timing node) — not started.
```

to a `✅ merged` line with the PR link, merge date, and what (if anything) is deferred (e.g. user-constraint travel-mode parsing remains out of scope; backend re-times WALK only).

- [ ] **Step 3 (post-merge): Commit**

```bash
git add implementation_plan/james/README.md implementation_plan/james/w8_live_map_routing.md
git commit -m "docs(w8): mark w8c merged"
```

---

## Self-Review

**1. Spec coverage:**

| Spec requirement | Task |
|---|---|
| `app/tools/directions.py` mirrors retrieval structure, typed models, key via `get_settings()`, async httpx, ~3s timeout | Tasks 2, 4 |
| Mode-aware walk/transit/drive | Task 4 (mode matrix tests + `_MODE_TO_ROUTES_TRAVELMODE`) |
| Reuse `planning.haversine_m` as offline fallback | Task 3 (`_haversine_fallback`) |
| Dedicated key separate from frontend key | Task 1 |
| New post-revision node, ≤1 Directions call, after loop settles, before final_reply | Task 6 |
| Open-at-arrival critique re-runs once on real data | Task 6 (re-run `temporal_coherence` only) |
| Async, tight timeout, haversine fallback, never blocks /chat | Tasks 3, 4, 6 (fail-open in node) |
| Pass→fail flip reflected in reply | Task 7 (`test_chat_retiming_flips_temporal_pass_to_fail`) |
| Unit: HTTP-mocked mode matrix + fallback | Task 4 |
| Smoke: node present, ≤1 call | Task 6 |
| Functional: real arrival overwrite + close-time flip | Task 7 |
| Integration: gated real Directions | Task 8 |
| Routes API v2 (resolved decision) | Task 4 |
| WALK-only canonical re-time (resolved decision) | Task 6 |
| Caveat via `_final_with_caveats`, no LLM re-entry (resolved decision) | Task 6 |
| Status doc updates on merge | Task 10 |

No gaps.

**2. Placeholder scan:** No "TBD"/"handle edge cases"/"similar to". Task 7 Step 2/3 deliberately instructs verifying real constants (caveat string, default duration) against named source files rather than guessing — this is a real instruction with the exact files/values to check, not a placeholder.

**3. Type consistency:** `DirectionsResult.legs: list[DirectionsLeg]`, `duration_s: int`, `distance_m: float`, `source: Literal["google","haversine_fallback"]`, `route_legs(stops, mode="walk") -> DirectionsResult`, `chain_arrival_times(stops, leg_durations_min) -> list[Stop]` — used consistently across Tasks 2-8. `route_legs` is patched at `app.agent.graph.route_legs` (where it's imported) in Tasks 6-7, not at its definition module — correct patch-site convention. `temporal_coherence`/`itinerary_violations` patched at their use/definition sites consistent with how `app/agent/graph.py` imports them (Task 6 Step 3 imports `temporal_coherence` into `graph.py`, so Task 7 patches `app.agent.graph.temporal_coherence`).

No inconsistencies found.
