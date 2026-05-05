# W3 — Self-correction (reflection: deterministic + cheap-LLM critique)

**Branch:** `feature/agent-w3-self-correction`
**Depends on:** W2. Optionally W6's `app/eval/itinerary_checker.py` if W6 lands first — W3 reuses the same checker functions to avoid reimplementation. If W3 ships first, we extract them into `app/agent/critique/checks.py` and W6 imports from there.
**Unblocks:** —

## Goal

Make the `critique` node in the LangGraph from W2 actually *critique* — both at the **per-tool-result level** (was the last retrieval call any good?) and at the **per-itinerary level** (does the proposed plan satisfy the constraints + walking budget + temporal coherence the user asked for?). Hand the LLM concrete revision hints, gate the response on a deterministic pass, and prevent the agent from confidently surfacing empty/low-quality results — or geographically-incoherent itineraries — to the user.

This is what turns "the RAG returned junk" or "the agent proposed dinner in North Beach then drinks in the Sunset" into a graceful product moment.

After this PR:
- The critique node runs **two passes**:
  - **Per-step deterministic critique** (existing W3 v0): empty results, all closed, low similarity, tool errors. Cost: ~zero tokens.
  - **Per-itinerary deterministic critique** (NEW): the moment `state.done` would flip true with `state.stops` populated, the same checker W6 uses offline runs at request time. If `temporal_coherence < 1.0`, `walking_budget_respected < 1.0`, `no_hallucinated_place_ids < 1.0`, or `constraints_satisfied < 0.8`, the agent gets one structured revision pass. Cost: ~zero tokens.
  - **Cheap-LLM vibe critique** (NEW, optional): if both deterministic passes succeed and `EVAL_VIBE_CRITIQUE_ENABLED=true`, a cheap small model (`gpt-4o-mini` / `gemini-2.5-flash` — same `EVAL_JUDGE_MODEL` used in W6) scores cross-stop vibe coherence on a 0–5 rubric. Below threshold → one revision pass. Cost: low (one cheap completion per request, gated by env var).
- One revision attempt per failure category. Bounded by `max_steps`; not a runaway loop.
- The agent never silently returns a plan that fails deterministic checks. Either the plan is valid, or the user gets a clarification, or the user gets explicit caveats.
- All revisions are counted and logged so MLflow tracks `revisions_per_query` as a metric.

## Files

### Modify: `app/agent/state.py`

Add a `RevisionHint` model and a list field on `ItineraryState`:

```python
class RevisionHint(BaseModel):
    reason: Literal[
        # Per-step (existing W3 v0)
        "empty_results", "all_closed", "low_similarity",
        "constraint_violation", "tool_error",
        # Per-itinerary (NEW)
        "geographic_incoherence", "temporal_incoherence",
        "walking_budget_exceeded", "constraint_unmet_in_final",
        "hallucinated_place_id",
        # Cheap-LLM vibe (NEW)
        "vibe_mismatch",
    ]
    detail: str
    suggested_action: Literal[
        "drop_filter", "expand_radius", "broaden_query",
        "clarify_with_user", "try_different_tool",
        # NEW
        "swap_stop", "tighten_radius", "shift_arrival_time",
        "rebalance_walking_budget",
    ]
    target: dict  # e.g. {"filter": "price_level_max"}, {"stop_index": 2}, {"tool": "..."}


class ItineraryState(BaseModel):
    # ... existing fields
    revision_hints: list[RevisionHint] = Field(default_factory=list)
    # NEW: bounded retry counter per failure category, prevents runaway loops.
    revision_counts: dict[str, int] = Field(default_factory=dict)
```

### New: `app/agent/critique/__init__.py` and `app/agent/critique/checks.py`

Pulls the deterministic check functions out of W6 (or the eval module pulls from here, depending on merge order) so request-time critique and offline eval share one implementation.

```python
"""Deterministic itinerary checks. Same code path as W6 eval.
Pure functions of (state) → score in [0, 1]. No LLM, no network beyond DB
lookups for hours / coords."""

# Re-exported from app/eval/itinerary_checker.py — see the canonical defs there.
from app.eval.itinerary_checker import (
    constraints_satisfied,
    geographic_coherence,
    temporal_coherence,
    walking_budget_respected,
    no_hallucinated_place_ids,
)

CRITIQUE_THRESHOLDS = {
    "constraints_satisfied":     0.8,    # 80% of expressed constraints met
    "geographic_coherence":      1.0,    # all consecutive pairs within budget
    "temporal_coherence":        1.0,    # all stops open at planned arrival
    "walking_budget_respected":  1.0,    # total walk under budget
    "no_hallucinated_place_ids": 1.0,    # zero tolerance
}


def itinerary_violations(state) -> list[str]:
    """Return a list of failing check names. Empty = the itinerary passed."""
    failed = []
    if no_hallucinated_place_ids(state) < 1.0:
        failed.append("no_hallucinated_place_ids")
    if temporal_coherence(state) < CRITIQUE_THRESHOLDS["temporal_coherence"]:
        failed.append("temporal_coherence")
    if geographic_coherence(state) < CRITIQUE_THRESHOLDS["geographic_coherence"]:
        failed.append("geographic_coherence")
    if walking_budget_respected(state) < CRITIQUE_THRESHOLDS["walking_budget_respected"]:
        failed.append("walking_budget_respected")
    if constraints_satisfied(state) < CRITIQUE_THRESHOLDS["constraints_satisfied"]:
        failed.append("constraints_satisfied")
    return failed
```

### New: `app/agent/critique/vibe.py`

A cheap-LLM rubric judge that runs ONLY when deterministic checks pass and `EVAL_VIBE_CRITIQUE_ENABLED=true`. Uses the same model as W6's eval judge.

```python
"""Cross-stop vibe coherence check via a cheap small model.

This is a runtime version of W6's taste judge — same prompt template, same
model, but bounded to 1 call per request and gated by env var. The point is
to catch "fancy Italian → dive bar → fancy dessert" mismatches that
deterministic checks can't see.
"""

import os
from typing import Optional
from langchain_core.messages import HumanMessage
from app.config import get_settings

VIBE_THRESHOLD = 3.0  # 0-5; below this triggers one revision pass.

VIBE_PROMPT = """Rate the vibe coherence of this {n_stops}-stop itinerary on a
0-5 scale where 5 = perfectly matched vibes, 0 = jarring mismatch.

User's request: {user_query}

Stops in order:
{stops_text}

Return JSON only: {{"score": float, "rationale": "one short sentence"}}.
"""


def vibe_check(state, judge_llm) -> Optional[float]:
    """Return a 0-5 score, or None if the check is disabled / no stops."""
    if not _enabled():
        return None
    if len(state.stops) < 2:
        return None  # vibe coherence is undefined for one stop
    user_query = next((m.content for m in state.messages
                       if m.__class__.__name__ == "HumanMessage"), "")
    stops_text = "\n".join(
        f"  {i+1}. {s.name} ({s.primary_type}) — {s.rationale}"
        for i, s in enumerate(state.stops)
    )
    prompt = VIBE_PROMPT.format(n_stops=len(state.stops),
                                user_query=user_query, stops_text=stops_text)
    raw = judge_llm.invoke([HumanMessage(prompt)]).content
    import json
    obj = json.loads(raw)
    return float(obj["score"])


def _enabled() -> bool:
    return os.getenv("EVAL_VIBE_CRITIQUE_ENABLED", "false").lower() == "true"
```

### Modify: `app/agent/graph.py`

Replace the placeholder `critique` node from W2 with the real one:

```python
from app.agent.state import RevisionHint
from langchain_core.messages import HumanMessage

LOW_SIMILARITY_THRESHOLD = 0.55  # tune from eval data (W6)
EMPTY_RESULTS_PATIENCE = 1       # how many empty calls before forcing revise

MAX_REVISIONS_PER_REASON = 2  # bounded retries; after this we ship with a caveat.


def critique(state: ItineraryState, judge_llm=None) -> ItineraryState:
    # 1) Loop bound (already in W2; keep)
    if state.step_count >= MAX_STEPS:
        state.done = True
        state.final_reply = state.final_reply or _bounded_reply(state)
        return state

    last = state.messages[-1] if state.messages else None
    finalizing = isinstance(last, AIMessage) and not last.tool_calls

    if finalizing and state.stops:
        # 2) Per-itinerary deterministic critique. Same checker as W6.
        from app.agent.critique.checks import itinerary_violations
        violations = itinerary_violations(state)
        if violations:
            for reason in violations:
                if _can_retry(state, reason):
                    state.revision_hints.append(_hint_for(reason, state))
                    _bump_retry(state, reason)
                    state.messages.append(HumanMessage(
                        content=f"[critique:itinerary] {reason}. "
                                f"Revise the affected stop(s); do not finalize yet."
                    ))
                    state.done = False  # keep planning
                    return state
            # If we exhausted retries on at least one reason, ship with a caveat.
            state.done = True
            state.final_reply = _final_with_caveats(state, violations)
            return state

        # 3) Cheap-LLM vibe critique. Gated by EVAL_VIBE_CRITIQUE_ENABLED.
        from app.agent.critique.vibe import vibe_check, VIBE_THRESHOLD
        if judge_llm is not None:
            score = vibe_check(state, judge_llm)
            if score is not None and score < VIBE_THRESHOLD \
               and _can_retry(state, "vibe_mismatch"):
                state.revision_hints.append(RevisionHint(
                    reason="vibe_mismatch",
                    detail=f"Vibe coherence scored {score:.1f}/5.",
                    suggested_action="swap_stop",
                    target={},
                ))
                _bump_retry(state, "vibe_mismatch")
                state.messages.append(HumanMessage(
                    content=f"[critique:vibe] cross-stop vibe coherence "
                            f"scored {score:.1f}/5. Swap whichever stop "
                            f"feels off and re-finalize."
                ))
                state.done = False
                return state

        # 4) All checks passed; finalize.
        state.done = True
        state.final_reply = state.final_reply or last.content
        return state

    # 5) Per-step deterministic critique (existing W3 v0): inspect last tool
    #    result and emit a hint if it was bad.
    hint = _diagnose_last_tool_result(state)
    if hint is not None and _can_retry(state, hint.reason):
        state.revision_hints.append(hint)
        _bump_retry(state, hint.reason)
        state.messages.append(HumanMessage(
            content=f"[critique:step] {hint.reason}: {hint.detail}. "
                    f"Suggested next action: {hint.suggested_action} on {hint.target}."
        ))
    return state


def _can_retry(state: ItineraryState, reason: str) -> bool:
    return state.revision_counts.get(reason, 0) < MAX_REVISIONS_PER_REASON


def _bump_retry(state: ItineraryState, reason: str) -> None:
    state.revision_counts[reason] = state.revision_counts.get(reason, 0) + 1


def _hint_for(reason: str, state: ItineraryState) -> RevisionHint:
    """Map an itinerary-level violation to a structured hint the planner can act on."""
    if reason == "geographic_coherence":
        return RevisionHint(
            reason="geographic_incoherence",
            detail="One or more consecutive stops exceed the per-leg walking budget.",
            suggested_action="tighten_radius",
            target={"stops": [i for i in range(1, len(state.stops))]},
        )
    if reason == "temporal_coherence":
        return RevisionHint(
            reason="temporal_incoherence",
            detail="A stop is closed at its planned arrival time.",
            suggested_action="shift_arrival_time",
            target={},
        )
    if reason == "walking_budget_respected":
        return RevisionHint(
            reason="walking_budget_exceeded",
            detail="Total walking exceeds the user's budget.",
            suggested_action="rebalance_walking_budget",
            target={},
        )
    if reason == "no_hallucinated_place_ids":
        return RevisionHint(
            reason="hallucinated_place_id",
            detail="One or more place_ids do not exist in places_raw.",
            suggested_action="swap_stop",
            target={},
        )
    return RevisionHint(
        reason="constraint_unmet_in_final",
        detail="Final stops do not satisfy the shared user constraints.",
        suggested_action="swap_stop",
        target={},
    )


def _final_with_caveats(state: ItineraryState, violations: list[str]) -> str:
    """Compose a final_reply that lists what didn't quite work. Better than
    silently shipping a bad plan."""
    body = state.messages[-1].content if state.messages else ""
    caveats = "\n\nCaveats: I couldn't fully satisfy " + ", ".join(violations) + \
              " after revisions. You may want to adjust the plan."
    return body + caveats


def _diagnose_last_tool_result(state: ItineraryState) -> Optional[RevisionHint]:
    # Find the most recent tool message
    last_tool = next(
        (m for m in reversed(state.messages) if isinstance(m, ToolMessage)), None
    )
    if last_tool is None:
        return None

    # Pull the original args + parsed result from scratch (act() stores them)
    last_call = _last_scratch_entry(state)
    if last_call is None:
        return None
    result = last_call["result"]
    args = last_call["args"]

    # tool_error
    if isinstance(result, dict) and "error" in result:
        return RevisionHint(
            reason="tool_error",
            detail=str(result["error"]),
            suggested_action="try_different_tool",
            target={"tool": last_tool.name},
        )

    if isinstance(result, list):
        if not result:
            # empty -> suggest dropping the most restrictive filter
            return RevisionHint(
                reason="empty_results",
                detail=f"No matches for {args}.",
                suggested_action="drop_filter",
                target={"filter": _most_restrictive_filter(args.get("filters"))},
            )
        # all_closed: every result has business_status != OPERATIONAL
        if all(getattr(h, "business_status", None) != "OPERATIONAL" for h in result):
            return RevisionHint(
                reason="all_closed",
                detail="Every result is closed or permanently_closed.",
                suggested_action="broaden_query",
                target={"filter": "business_status"},
            )
        # low_similarity: top result similarity below threshold
        top = result[0]
        sim = getattr(top, "similarity", 0.0) or 0.0
        if sim < LOW_SIMILARITY_THRESHOLD:
            return RevisionHint(
                reason="low_similarity",
                detail=f"Top similarity {sim:.2f} below threshold.",
                suggested_action="broaden_query",
                target={"query": args.get("query")},
            )

    return None


def _most_restrictive_filter(filters: Optional[dict]) -> str:
    # Deterministic priority: open_at > price_level_max > min_rating > neighborhood > types_any
    if not filters:
        return "none"
    for f in ("open_at", "price_level_max", "min_rating", "neighborhood", "types_any"):
        if filters.get(f) is not None:
            return f
    return "none"
```

### Modify: `app/agent/prompts.py`

Append a section explaining how to consume revision hints:

```python
REVISION_GUIDANCE = """
WHEN YOU SEE A `[critique]` MESSAGE:

- "empty_results" + suggested_action=drop_filter: re-call the same tool with
  the named filter removed or relaxed (e.g. raise price_level_max by 1).
- "all_closed": user's time window is wrong, or the query is for a niche
  category. Either expand the time, set business_status=null, or ask the user.
- "low_similarity": rephrase the `query` more broadly. Don't add filters; the
  semantic match is the bottleneck.
- "tool_error": acknowledge to the user and pivot to a different tool or a
  graceful fallback (e.g., "I'm having trouble searching right now").
- "constraint_violation": you proposed a stop that violates the user's
  constraints. Apologize briefly in the final_reply and replace the stop.

If you've revised twice for the same hint reason and still can't satisfy,
ASK THE USER A CLARIFYING QUESTION. Better to ask than to lie.
"""

# Update SYSTEM_PROMPT to include REVISION_GUIDANCE
```

## Tests

### New: `tests/unit/test_agent_self_correct.py`

```python
def test_empty_results_emits_drop_filter_hint(monkeypatch):
    # First semantic_search returns []; second returns one result.
    calls = []
    def fake_search(query, filters=None, k=8):
        calls.append({"query": query, "filters": filters})
        if len(calls) == 1:
            return []
        return [PlaceHit(place_id="p1", name="X", source="google_places",
                         similarity=0.8, business_status="OPERATIONAL", ...)]
    monkeypatch.setattr("app.tools.retrieval.semantic_search", fake_search)

    fake = FakeLLM([
        # Round 1: search with restrictive filter -> empty
        AIMessage("", tool_calls=[{
            "name": "semantic_search", "id": "1",
            "args": {"query": "x", "filters": {"price_level_max": 1, ...}},
        }]),
        # Round 2: drops the filter on hint, gets a result
        AIMessage("", tool_calls=[{
            "name": "semantic_search", "id": "2",
            "args": {"query": "x", "filters": {}},
        }]),
        # Round 3: finalize
        AIMessage("Found one place", tool_calls=[]),
    ])
    g = build_agent_graph(fake, max_steps=5)
    out = g.invoke(ItineraryState(messages=[HumanMessage("x")]))
    assert any(h.reason == "empty_results" for h in out.revision_hints)
    assert out.done is True


def test_all_closed_emits_broaden_query_hint(): ...
def test_low_similarity_emits_broaden_query_hint(): ...
def test_tool_error_emits_try_different_tool_hint(): ...
def test_clarify_with_user_after_two_failed_revisions(): ...
```

Each test uses the `FakeLLM` pattern from W2. The last test verifies the agent does NOT fabricate when revisions repeatedly fail.

### Modify: `tests/unit/test_agent_graph.py`

Add an assertion that hints injected by `critique` show up as `HumanMessage`s in `state.messages` between `act` and the next `plan`.

## Manual verification

```bash
# Failure-mode test:
curl -s http://localhost:8000/chat \
  -d '{"message": "find a 5-star vegan ethiopian place in pacific heights open at 4am", "history": []}' | jq .
```

Expected behaviors (any of these is acceptable; not a hallucination):
- `reply` includes a clarifying question to the user.
- `reply` says something like "I couldn't find anything matching all of that — here are the closest matches" and `places` contains 1-3 relaxed-constraint suggestions.
- `places` is empty AND `reply` explicitly says it found nothing.

NOT acceptable:
- Empty `places` with a confident `reply` listing fictional restaurants.
- A plausible-but-wrong place_id that doesn't exist in the DB (defended by W1's structured tools, but worth verifying).

## Future direction: lightweight constraint extractor

A cheap-LLM "constraint extractor" pre-pass — turn the user's free-text into a
structured `UserConstraints` object before the main agent runs — is a natural
next step IF eval shows the planning agent is losing tokens / quality to NLU
work. Out of scope here: we let the planner do its own parsing for now and
revisit once W6 has run a couple of evals. If we add it, it becomes a new
node in the graph that runs before `plan` and writes `state.constraints`
directly. Same `EVAL_JUDGE_MODEL` env var, separate cost line.

## Future direction: supervisor / specialized subagents

We considered (and deliberately deferred) a supervisor + specialized-subagent
pattern. Reasoning: today's tools are all variations of "look up places" with a
shared data model and prompt context. A supervisor on top of W2's
`plan → act → critique` loop adds wrapping without a clear win. Cases where it
might pay off later: a separate booking specialist if W4 grows beyond URL
deep-links into actual API calls (different ToS regime, different rate limits);
a separate research agent if we ever add open-web search as a first-class tool.
Reassess when one of those triggers fires.

## Risks / open questions

- **Threshold tuning.** `LOW_SIMILARITY_THRESHOLD = 0.55`, `VIBE_THRESHOLD = 3.0`, `CRITIQUE_THRESHOLDS["constraints_satisfied"] = 0.8` are all guesses. W6 (eval agent) gives us a principled way to reset these from observed run-to-run distributions. Until then, log them as MLflow params and watch for false positives / negatives.
- **Hint fatigue.** Bounded by `MAX_REVISIONS_PER_REASON = 2` per failure category. After that we ship with caveats rather than thrash.
- **Vibe critique cost.** Each call is one cheap-LLM completion (≤200 tokens out). At `gpt-4o-mini` rates this is fractions of a cent per request, but it does double the LLM calls for any 2+-stop request. Gated by `EVAL_VIBE_CRITIQUE_ENABLED` so it can be turned off in cost-sensitive deployments. Default off until we measure user-facing quality lift.
- **Itinerary checker shared with W6.** Single source of truth lives wherever ships first. If W3 ships before W6, the canonical module is `app/agent/critique/checks.py` and W6 imports from there. If W6 ships first, the canonical module is `app/eval/itinerary_checker.py` and W3 re-exports. PR descriptions must call out the import direction so we don't end up with two implementations.
- **Multilingual queries.** ILIKE matching on `formatted_address` is locale-dependent. Out of scope; document.
- **Revision pass can still produce a plan that fails the same check.** The retry budget is bounded but the LLM's revision may not actually fix the violation. The `_final_with_caveats` path catches this case — the user sees the imperfect plan with an explicit caveat list, never a silent failure.
