# W3 — Self-correction

**Branch:** `feature/agent-w3-self-correction`
**Depends on:** W2
**Unblocks:** —

## Goal

Make the `critique` node in the LangGraph from W2 actually *critique*: detect bad retrieval outcomes, hand the LLM concrete revision hints, and prevent the agent from confidently surfacing empty/low-quality results to the user. This is what turns "the RAG returned junk" into a graceful product moment — and it's why the `/chat` endpoint won't embarrass us when audiences ask edge-case questions on the fly.

After this PR:
- The critique node enforces deterministic checks (empty results, all closed, low similarity, constraint violations).
- On failure, the agent receives a `RevisionHint` in the scratch and the system prompt instructs it how to use those hints.
- A small set of integration tests exercise the most common failure modes end-to-end.

## Files

### Modify: `app/agent/state.py`

Add a `RevisionHint` model and a list field on `ItineraryState`:

```python
class RevisionHint(BaseModel):
    reason: Literal[
        "empty_results", "all_closed", "low_similarity",
        "constraint_violation", "tool_error",
    ]
    detail: str
    suggested_action: Literal[
        "drop_filter", "expand_radius", "broaden_query",
        "clarify_with_user", "try_different_tool",
    ]
    target: dict  # e.g. {"filter": "price_level_max"} or {"tool": "semantic_search"}


class ItineraryState(BaseModel):
    # ... existing fields
    revision_hints: list[RevisionHint] = Field(default_factory=list)
```

### Modify: `app/agent/graph.py`

Replace the placeholder `critique` node from W2 with the real one:

```python
from app.agent.state import RevisionHint
from langchain_core.messages import HumanMessage

LOW_SIMILARITY_THRESHOLD = 0.55  # tune from eval data (W6)
EMPTY_RESULTS_PATIENCE = 1       # how many empty calls before forcing revise

def critique(state: ItineraryState) -> ItineraryState:
    # 1) Loop bound (already in W2; keep)
    if state.step_count >= MAX_STEPS:
        state.done = True
        state.final_reply = state.final_reply or _bounded_reply(state)
        return state

    # 2) If the last AIMessage has no tool calls AND the agent produced stops,
    #    consider it final.
    last = state.messages[-1] if state.messages else None
    if isinstance(last, AIMessage) and not last.tool_calls:
        state.done = True
        state.final_reply = state.final_reply or last.content
        return state

    # 3) Inspect the most recent tool result and emit a hint if it was bad.
    hint = _diagnose_last_tool_result(state)
    if hint is not None:
        state.revision_hints.append(hint)
        # Inject a HumanMessage carrying the hint so the LLM sees it on next plan().
        state.messages.append(HumanMessage(
            content=f"[critique] {hint.reason}: {hint.detail}. "
                    f"Suggested next action: {hint.suggested_action} on {hint.target}."
        ))
    return state


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

## Risks / open questions

- **Threshold tuning.** `LOW_SIMILARITY_THRESHOLD = 0.55` is a guess. W6 (eval agent) gives us a principled way to set it from data. Until then, log similarity distributions and adjust if false positives/negatives are noticeable.
- **Hint fatigue.** If every step emits a hint, the agent may thrash. Cap at 3 revision hints per request; after that, force `done=True` with a transparent explanation.
- **Multilingual queries.** ILIKE matching on `formatted_address` is locale-dependent. Out of scope; document.
