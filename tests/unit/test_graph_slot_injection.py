"""Unit + functional tests for the graph-layer primary_type_family injection.

Phase 4 Plan 03 (D-04-04): the graph injects `filters.primary_type_family`
into retrieval tool_call args when the user named per-slot categories AND
the model cooperated by emitting `slot_index`. This file mirrors
`tests/unit/test_swap.py:498-602`'s shape for `_inject_closure_exclusions`,
plus functional tests that drive `act()` through a scripted LLM.

The JSON-safety invariant (project memory `aimessage_tool_call_args_json_safe.md`)
is the load-bearing contract: every helper output MUST be `json.dumps`-safe,
and `tc["args"]` on the AIMessage MUST NEVER be mutated.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.agent.graph import _inject_primary_type_family, build_agent_graph
from app.agent.state import ItineraryState, UserConstraints
from app.tools.filters import SearchFilters

# ---------------------------------------------------------------------------
# Pure-function tests (mirror tests/unit/test_swap.py:498-602)
# ---------------------------------------------------------------------------


def test_inject_primary_type_family_writes_family_into_filters() -> None:
    """Happy path: slot_index resolves a requested primary_type to its family
    and writes it into filters as a plain dict."""
    out = _inject_primary_type_family(
        "semantic_search",
        {"query": "omakase", "slot_index": 0, "filters": {"min_rating": 4.0}},
        ["Sushi Restaurant", "Cocktail Bar", "Dessert Shop"],
    )
    assert isinstance(out["filters"], dict)
    assert out["filters"]["primary_type_family"] == "restaurant"
    # Existing filter fields are preserved.
    assert out["filters"]["min_rating"] == 4.0


def test_inject_primary_type_family_creates_filters_when_absent() -> None:
    """No `filters` key in args: helper creates one and writes the family in."""
    out = _inject_primary_type_family(
        "semantic_search",
        {"query": "omakase", "slot_index": 0},
        ["Sushi Restaurant"],
    )
    assert "filters" in out
    assert isinstance(out["filters"], dict)
    assert out["filters"]["primary_type_family"] == "restaurant"


def test_inject_primary_type_family_noop_when_requested_empty() -> None:
    """Empty requested_primary_types: helper returns a copy of args unchanged."""
    args = {"query": "ramen", "slot_index": 0}
    out = _inject_primary_type_family("semantic_search", args, [])
    assert out == args
    assert "filters" not in out
    # New dict (defensive copy), not the same reference.
    assert out is not args


def test_inject_primary_type_family_noop_when_slot_index_none_and_query_ambiguous() -> None:
    """No slot_index AND no family keyword in the query → graph does NOT inject.

    The 2026-06-15 fallback (family_from_query) only fires when the query text
    carries a keyword for a requested family; "ramen" matches none of the
    restaurant keywords, so the fallback no-ops and the historical D-04-06
    behavior (no injection) is preserved for keyword-free queries.
    """
    # slot_index missing entirely
    args1 = {"query": "ramen"}
    out1 = _inject_primary_type_family("semantic_search", args1, ["Sushi Restaurant"])
    assert out1 == args1
    assert "filters" not in out1
    # slot_index explicitly None
    args2 = {"query": "ramen", "slot_index": None}
    out2 = _inject_primary_type_family("semantic_search", args2, ["Sushi Restaurant"])
    assert out2 == args2
    assert "filters" not in out2


def test_inject_primary_type_family_fallback_infers_from_query_without_slot_index() -> None:
    """2026-06-15 refinement_cheaper fix: when the model omits slot_index but the
    query carries a keyword for a REQUESTED family, the graph infers and injects
    that family (the live failure mode: models emit "drinks in Hayes Valley"
    with no slot_index, leaving every slot below the viability threshold)."""
    requested = ["Restaurant", "Cocktail Bar", "Dessert Shop"]
    out = _inject_primary_type_family(
        "semantic_search", {"query": "drinks in Hayes Valley"}, requested
    )
    assert out["filters"]["primary_type_family"] == "bar"
    out2 = _inject_primary_type_family(
        "semantic_search", {"query": "dessert in Hayes Valley"}, requested
    )
    assert out2["filters"]["primary_type_family"] == "dessert"


def test_inject_primary_type_family_fallback_only_uses_requested_families() -> None:
    """The query-text fallback never injects a family the user did not request:
    a "coffee" query with only Restaurant+Bar requested stays uninjected."""
    out = _inject_primary_type_family(
        "semantic_search", {"query": "coffee shop nearby"}, ["Restaurant", "Cocktail Bar"]
    )
    assert "filters" not in out or "primary_type_family" not in (out.get("filters") or {})


def test_inject_primary_type_family_noop_when_slot_index_out_of_range() -> None:
    """Defensive against bad model output: out-of-range slot_index → noop."""
    args = {"query": "x", "slot_index": 5}
    out = _inject_primary_type_family(
        "semantic_search",
        args,
        ["Sushi Restaurant", "Cocktail Bar"],
    )
    assert out == args
    assert "filters" not in out


def test_inject_primary_type_family_noop_when_slot_index_negative() -> None:
    """Negative slot_index is treated like out-of-range (defensive)."""
    args = {"query": "x", "slot_index": -1}
    out = _inject_primary_type_family(
        "semantic_search",
        args,
        ["Sushi Restaurant"],
    )
    assert out == args


def test_inject_primary_type_family_noop_when_slot_index_not_int() -> None:
    """Non-int slot_index (e.g., a string) → noop. The retrieval tools declare
    `int | None`, but the model could still emit nonsense."""
    args = {"query": "x", "slot_index": "0"}
    out = _inject_primary_type_family(
        "semantic_search",
        args,
        ["Sushi Restaurant"],
    )
    assert out == args


def test_inject_primary_type_family_noop_when_family_unmappable() -> None:
    """Fail-open: when family_of returns None for the requested type, the
    helper does NOT inject. unknown keyword → no enforcement (same philosophy
    as closure-exclusions returning dict(args) unchanged)."""
    out = _inject_primary_type_family(
        "semantic_search",
        {"query": "x", "slot_index": 0},
        ["unknown_thing"],
    )
    assert "filters" not in out


def test_inject_primary_type_family_noop_when_tool_name_not_retrieval() -> None:
    """commit_itinerary and kg_traverse don't take primary_type_family filters
    (kg_traverse has no `filters` arg at all). Helper is a noop for
    non-(semantic_search|nearby) tool names."""
    for name in ("commit_itinerary", "kg_traverse", "get_details"):
        args = {"query": "x", "slot_index": 0}
        out = _inject_primary_type_family(name, args, ["Sushi Restaurant"])
        assert out == args, f"helper must be noop for tool {name!r}"


def test_inject_primary_type_family_applies_to_nearby() -> None:
    """Both retrieval tools (semantic_search AND nearby) get the injection."""
    out = _inject_primary_type_family(
        "nearby",
        {"place_id": "ChIJtest_p1_aaaaaaaa", "slot_index": 1},
        ["Restaurant", "Cocktail Bar"],
    )
    assert out["filters"]["primary_type_family"] == "bar"


def test_inject_primary_type_family_accepts_dict_filters_from_llm() -> None:
    """LangChain delivers `filters` as a dict in tool_call args. Helper must
    accept dict input, round-trip through SearchFilters for validation, and
    output a dict (NOT a Pydantic instance)."""
    out = _inject_primary_type_family(
        "semantic_search",
        {
            "query": "omakase",
            "slot_index": 0,
            "filters": {"price_level_max": 3, "min_rating": 4.0},
        },
        ["Sushi Restaurant"],
    )
    assert isinstance(out["filters"], dict)
    assert not isinstance(out["filters"], SearchFilters)
    assert out["filters"]["primary_type_family"] == "restaurant"
    # Existing dict fields preserved through the round-trip.
    assert out["filters"]["price_level_max"] == 3
    assert out["filters"]["min_rating"] == 4.0


def test_inject_primary_type_family_normalizes_searchfilters_instance() -> None:
    """When `filters` is a SearchFilters Pydantic instance (legacy callers /
    direct test usage), helper normalizes via model_dump and emits a plain
    dict, NOT a Pydantic instance."""
    out = _inject_primary_type_family(
        "semantic_search",
        {
            "query": "omakase",
            "slot_index": 0,
            "filters": SearchFilters(price_level_max=3, min_rating=4.0),
        },
        ["Sushi Restaurant"],
    )
    assert isinstance(out["filters"], dict)
    assert not isinstance(out["filters"], SearchFilters)
    assert out["filters"]["primary_type_family"] == "restaurant"
    assert out["filters"]["price_level_max"] == 3


def test_inject_primary_type_family_overwrites_model_emitted_family() -> None:
    """When the model already set primary_type_family in filters, the helper
    OVERWRITES it with the slot-derived family — explicit graph enforcement
    per D-04-04 (T-04-03-05)."""
    out = _inject_primary_type_family(
        "semantic_search",
        {
            "query": "omakase",
            "slot_index": 0,
            # Model said "bar" but slot 0 is sushi — graph wins.
            "filters": {"primary_type_family": "bar"},
        },
        ["Sushi Restaurant"],
    )
    assert out["filters"]["primary_type_family"] == "restaurant"


def test_inject_primary_type_family_result_is_json_dumps_safe() -> None:
    """LOAD-BEARING: per memory `aimessage_tool_call_args_json_safe.md`, the
    helper output is what gets recorded in scratch (and indirectly serialized
    through MLflow tracing). Every output path must be json.dumps-safe.

    Mirrors `test_inject_closure_exclusions_output_is_json_serializable`
    (tests/unit/test_swap.py:578-602)."""
    # Every shape of `filters` input the helper might see.
    filters_shapes: list[Any] = [
        None,  # absent
        {"min_rating": 4.0},  # dict (LangChain wire shape)
        SearchFilters(min_rating=4.0),  # Pydantic instance (direct callers)
        {},  # empty dict
        {"primary_type_family": "cafe"},  # pre-set by model, will be overwritten
    ]
    requested = ["Sushi Restaurant"]
    for filters_in in filters_shapes:
        base_args: dict[str, Any] = {"query": "omakase", "slot_index": 0}
        if filters_in is not None:
            base_args["filters"] = filters_in
        out = _inject_primary_type_family("semantic_search", base_args, requested)
        # If json.dumps doesn't raise, the next plan()'s OpenAI serialization
        # won't crash on this args dict either.
        json.dumps(out)


def test_inject_primary_type_family_does_not_mutate_input_args() -> None:
    """The helper must NEVER mutate its input args dict. Any mutation of the
    returned dict (or its nested `filters`) must not propagate to args."""
    args: dict[str, Any] = {
        "query": "omakase",
        "slot_index": 0,
        "filters": {"min_rating": 4.0},
    }
    args_filters_before = dict(args["filters"])
    out = _inject_primary_type_family("semantic_search", args, ["Sushi Restaurant"])
    # Return value is a fresh dict.
    assert out is not args
    # Mutating the result must not bleed back to the input.
    out["filters"]["primary_type_family"] = "TAMPER"
    out["new_key"] = "TAMPER"
    assert args["filters"] == args_filters_before
    assert "new_key" not in args
    assert "primary_type_family" not in args["filters"]


# ---------------------------------------------------------------------------
# Functional tests on act() (Task 2 — see test_act_* below)
# ---------------------------------------------------------------------------


class _OneShotLLM(BaseChatModel):
    """LLM that emits exactly one scripted AIMessage on the first call, then
    closes with a content-only AIMessage on every subsequent call. Useful for
    forcing `act()` to run on a single tool_call without running a full graph
    revision loop."""

    scripted: AIMessage

    @property
    def _llm_type(self) -> str:
        return "oneshot"

    def bind_tools(self, tools: Any, **kwargs: Any) -> _OneShotLLM:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        already_issued = any(isinstance(m, AIMessage) and m.tool_calls for m in messages)
        if already_issued:
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="done", tool_calls=[]))]
            )
        return ChatResult(generations=[ChatGeneration(message=self.scripted)])


async def _drive_act_once(
    monkeypatch: pytest.MonkeyPatch,
    tool_call: dict[str, Any],
    constraints: UserConstraints,
    capture_invokes: list[dict[str, Any]] | None = None,
) -> ItineraryState:
    """Build a graph with _OneShotLLM that emits `tool_call`, run it once, and
    return the resulting state. If `capture_invokes` is supplied, every call
    to the retrieval tool's invoke() records its args there."""

    # Stub the underlying retrieval functions so no real DB hits occur AND
    # capture-by-side-effect when requested.
    def _record(name: str, args: dict[str, Any]) -> list[Any]:
        if capture_invokes is not None:
            capture_invokes.append({"tool": name, "args": dict(args)})
        return []

    # The underlying retrieval functions in app.tools.retrieval don't accept
    # slot_index — capturing here proves the strip happened. We monkeypatch
    # the underlying retrieval functions so the wrapper passes through.
    monkeypatch.setattr(
        "app.agent.tools._semantic_search",
        lambda **kw: _record("semantic_search", kw),
    )
    monkeypatch.setattr(
        "app.agent.tools._nearby",
        lambda **kw: _record("nearby", kw),
    )

    state = ItineraryState(
        messages=[HumanMessage(content="omakase, drinks, dessert")],
        constraints=constraints,
    )
    ai = AIMessage(content="", tool_calls=[tool_call])
    llm = _OneShotLLM(scripted=ai)
    graph = build_agent_graph(llm, max_steps=4)
    out = await graph.ainvoke(state)
    # The graph returns a dict-shape state from ainvoke — wrap it in a fresh
    # ItineraryState so the caller can read .scratch ergonomically. The graph
    # is built on Pydantic states so it round-trips.
    return ItineraryState.model_validate(out)


async def test_act_injects_primary_type_family_when_model_passes_slot_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Functional integration: when the model emits a slot_index, the graph
    injects primary_type_family into the recorded scratch args."""
    constraints = UserConstraints(
        requested_primary_types=["Sushi Restaurant", "Cocktail Bar", "Dessert Shop"]
    )
    state = await _drive_act_once(
        monkeypatch,
        {
            "name": "semantic_search",
            "id": "call_1",
            "args": {"query": "omakase", "slot_index": 0},
            "type": "tool_call",
        },
        constraints,
    )
    entries = state.scratch.get("semantic_search") or []
    assert entries, "act() never recorded a semantic_search entry"
    recorded = entries[0]["args"]
    assert isinstance(recorded["filters"], dict)
    assert recorded["filters"]["primary_type_family"] == "restaurant"
    # The strip step removed slot_index from the recorded args.
    assert "slot_index" not in recorded


async def test_act_does_not_inject_when_model_omits_slot_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No slot_index AND a keyword-free query ("omakase") → graph does NOT
    inject. The 2026-06-15 family_from_query fallback only fires on queries
    that carry a requested-family keyword; "omakase" matches none, so the
    historical D-04-06 non-cooperation behavior is preserved here."""
    constraints = UserConstraints(requested_primary_types=["Sushi Restaurant", "Cocktail Bar"])
    state = await _drive_act_once(
        monkeypatch,
        {
            "name": "semantic_search",
            "id": "call_2",
            "args": {"query": "omakase"},  # no slot_index, no family keyword
            "type": "tool_call",
        },
        constraints,
    )
    entries = state.scratch.get("semantic_search") or []
    assert entries, "act() never recorded a semantic_search entry"
    recorded = entries[0]["args"]
    # No filters injected because the model didn't cooperate.
    if "filters" in recorded:
        assert "primary_type_family" not in (recorded["filters"] or {})


async def test_act_strips_slot_index_before_tool_invoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The underlying retrieval function in app.tools.retrieval doesn't accept
    slot_index. The graph's strip step must remove it from effective_args
    before tool.invoke. Capture the kwargs the underlying retrieval sees."""
    invokes: list[dict[str, Any]] = []
    constraints = UserConstraints(requested_primary_types=["Sushi Restaurant"])
    await _drive_act_once(
        monkeypatch,
        {
            "name": "semantic_search",
            "id": "call_3",
            "args": {"query": "omakase", "slot_index": 0},
            "type": "tool_call",
        },
        constraints,
        capture_invokes=invokes,
    )
    assert invokes, "underlying _semantic_search was never invoked"
    # `_semantic_search` is called with keyword args from the tool wrapper
    # AFTER the wrapper strips slot_index, so its kwargs must NOT include it.
    underlying = invokes[0]["args"]
    assert "slot_index" not in underlying, (
        f"underlying retrieval must NOT receive slot_index, got {underlying!r}"
    )


async def test_act_does_not_alter_kg_traverse_args_when_strip_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ADVISORY 4: kg_traverse args have no `slot_index` key. The strip block
    is a NO-OP in that case — effective_args byte-equals the
    closure_exclusions output. This guards against future refactors
    accidentally adding side effects to kg_traverse handling."""
    from app.agent.swap import _inject_closure_exclusions

    captured: list[dict[str, Any]] = []

    def _kg(**kw: Any) -> list[Any]:
        captured.append(dict(kw))
        return []

    monkeypatch.setattr("app.tools.graph.kg_traverse", _kg)

    # Drive kg_traverse via the same _OneShotLLM. closure_context is empty so
    # _inject_closure_exclusions is also a noop — effective_args must equal
    # the original args minus any (absent) slot_index, i.e., unchanged.
    constraints = UserConstraints(requested_primary_types=["Sushi Restaurant", "Cocktail Bar"])
    state = ItineraryState(
        messages=[HumanMessage(content="similar to Y")],
        constraints=constraints,
    )
    ai = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "kg_traverse",
                "id": "call_kg",
                "args": {"place_id": "ChIJtest_anchor_aaaa", "relation_type": "SIMILAR_VECTOR"},
                "type": "tool_call",
            }
        ],
    )
    graph = build_agent_graph(_OneShotLLM(scripted=ai), max_steps=4)
    out = await graph.ainvoke(state)
    state_out = ItineraryState.model_validate(out)

    # The scratch records the effective_args (post-chain). Verify it's
    # byte-equal to what closure-exclusions alone would have produced — i.e.,
    # the primary_type_family helper and the slot_index strip were both
    # NO-OPS for kg_traverse.
    entries = state_out.scratch.get("kg_traverse") or []
    assert entries, "act() never recorded a kg_traverse entry"
    recorded = entries[0]["args"]
    closure_only = _inject_closure_exclusions(
        "kg_traverse",
        {"place_id": "ChIJtest_anchor_aaaa", "relation_type": "SIMILAR_VECTOR"},
        list(state_out.closure_context),
    )
    assert recorded == closure_only, (
        f"kg_traverse args drifted in the chain: recorded={recorded!r}, "
        f"closure_only={closure_only!r}"
    )


async def test_act_does_not_mutate_tc_args_under_slot_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LOAD-BEARING: per memory `aimessage_tool_call_args_json_safe.md`, the
    AIMessage's tc['args'] must stay byte-equal across act() — the next
    plan() step re-serializes it via json.dumps. Specifically check that the
    chained primary_type_family injection does NOT bleed into tc['args']."""
    constraints = UserConstraints(requested_primary_types=["Sushi Restaurant"])
    state = ItineraryState(
        messages=[HumanMessage(content="omakase")],
        constraints=constraints,
    )
    original_args: dict[str, Any] = {"query": "omakase", "slot_index": 0}
    ai = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "semantic_search",
                "id": "call_4",
                "args": dict(original_args),  # defensive copy so we have a snapshot
                "type": "tool_call",
            }
        ],
    )
    # Snapshot what we put into the AIMessage by value.
    snapshot_args = dict(ai.tool_calls[0]["args"])

    monkeypatch.setattr("app.agent.tools._semantic_search", lambda **kw: [])
    graph = build_agent_graph(_OneShotLLM(scripted=ai), max_steps=4)
    out = await graph.ainvoke(state)

    # Walk the messages on output and find the same AIMessage. Its tool_calls
    # must still match the snapshot byte-for-byte.
    found_ai = next(
        (m for m in out["messages"] if isinstance(m, AIMessage) and m.tool_calls),
        None,
    )
    assert found_ai is not None, "AIMessage with tool_calls missing from output"
    assert found_ai.tool_calls[0]["args"] == snapshot_args, (
        "tc['args'] on the AIMessage was mutated by act() — "
        "this breaks the next plan()'s json.dumps invariant"
    )
    # And the mutation snapshot must still be json.dumps-safe.
    json.dumps(found_ai.tool_calls[0]["args"])
