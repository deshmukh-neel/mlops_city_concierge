# W2 — Agent loop + ItineraryState + `/chat` endpoint

**Branch:** `feature/agent-w2-agent-graph`
**Depends on:** W1
**Unblocks:** W3, W4, W6

## Goal

Replace the single-pass `RetrievalQA` chain with a **LangGraph tool-calling agent** that drives multiple retrieval calls, accumulates a structured `ItineraryState`, and returns a response that matches the contract the frontend already expects (`POST /chat` → `{reply, places, ragLabel}`).

After this PR:
- `POST /chat` is the primary serving endpoint.
- `POST /predict` becomes a thin shim that wraps the agent for backwards compat.
- The agent driver is selected via the existing MLflow registry path — no new model selection.
- `ItineraryState` is the single source of truth for what the agent is building.

## Files

### Modify: `pyproject.toml`

Add `langgraph` to `[tool.poetry.dependencies]`:

```toml
langgraph = ">=0.2.0,<1.0.0"
```

(LangChain 0.2 is already pinned at `pyproject.toml:25`, so this is the natural fit.)

### New: `app/agent/__init__.py`

```python
"""Agentic core: state, tools registration, graph, and prompt."""
```

### New: `app/agent/state.py`

```python
"""Structured state passed through every node of the agent graph.

Keeping state as a Pydantic model (not just messages) lets the LLM revise
specific stops or constraints without regenerating prose, and lets the
critique node do deterministic checks (geographic coherence, hours).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class UserConstraints(BaseModel):
    """Parsed/inferred constraints from the user message."""
    party_size: Optional[int] = None
    budget_per_person_max: Optional[int] = None  # USD
    price_level_max: Optional[int] = None        # 0-4 (Google)
    min_rating: Optional[float] = None
    when: Optional[datetime] = None              # primary anchor time
    neighborhood: Optional[str] = None
    vibes: list[str] = Field(default_factory=list)  # free-text tags
    must_be_open: bool = True


class Stop(BaseModel):
    place_id: str
    name: str
    arrival_time: Optional[datetime] = None
    rationale: str
    source: str  # 'google_places' | 'editorial'


class ItineraryState(BaseModel):
    """The single piece of state passed through every graph node."""
    messages: list[BaseMessage] = Field(default_factory=list)
    constraints: UserConstraints = Field(default_factory=UserConstraints)
    stops: list[Stop] = Field(default_factory=list)
    scratch: dict[str, Any] = Field(default_factory=dict)
    step_count: int = 0
    done: bool = False
    final_reply: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class PlaceCard(BaseModel):
    """Frontend-facing place shape. Matches what frontend/src/api/chat.js
    renders. Derived from ItineraryState.stops at response time."""
    place_id: str
    name: str
    address: Optional[str] = None
    rating: Optional[float] = None
    price_level: Optional[int] = None
    primary_type: Optional[str] = None
    arrival_time: Optional[datetime] = None
    rationale: str
    booking_url: Optional[str] = None  # populated by W4
```

### New: `app/agent/tools.py`

```python
"""Wrap W1's plain Python functions as LangChain @tool callables so the
LLM can invoke them. We keep the underlying functions importable from
app.tools.* because the eval agent (W6) and tests use them directly."""

from langchain_core.tools import tool

from app.tools.filters import SearchFilters
from app.tools.retrieval import (
    PlaceDetails,
    PlaceHit,
    get_details as _get_details,
    nearby as _nearby,
    semantic_search as _semantic_search,
)


@tool("semantic_search", args_schema=None)  # args_schema set explicitly below
def semantic_search_tool(query: str, filters: SearchFilters | None = None,
                         k: int = 8) -> list[PlaceHit]:
    """Search for places by meaning + structured filters.

    Use for: "find me a romantic italian spot in north beach under $$$ open Sunday at 7pm".
    The `filters` argument is preferred over packing constraints into `query`.
    """
    return _semantic_search(query=query, filters=filters, k=k)


@tool("nearby")
def nearby_tool(place_id: str, radius_m: int = 800,
                filters: SearchFilters | None = None, k: int = 8) -> list[PlaceHit]:
    """Find places within radius_m meters of an anchor place. Use this AFTER
    you've picked a first stop and need a second stop within walking distance."""
    return _nearby(place_id=place_id, radius_m=radius_m, filters=filters, k=k)


@tool("get_details")
def get_details_tool(place_id: str) -> PlaceDetails | None:
    """Fetch the full record for a place: hours, website, ratings count, types."""
    return _get_details(place_id=place_id)


@tool("kg_traverse")
def kg_traverse_tool(place_id: str, relation: str = "co_mentioned") -> dict:
    """Traverse the editorial knowledge graph from `place_id`. NOT YET AVAILABLE.

    This is a stub: the KG lands in a future PR after the editorial scrape is done.
    The tool exists now so the agent's tool surface is stable.
    """
    return {"available": False,
            "reason": "knowledge graph not yet built; use semantic_search instead"}


def all_tools():
    return [semantic_search_tool, nearby_tool, get_details_tool, kg_traverse_tool]
```

W4 will append `propose_booking_tool` to `all_tools()`.

### New: `app/agent/prompts.py`

```python
SYSTEM_PROMPT = """You are City Concierge, an AI agent that plans dining and
nightlife itineraries grounded in a structured database of real places.

You have tools for retrieval; do not invent places, addresses, or hours. Every
recommendation must come from a tool call.

CRITICAL BEHAVIORS:

1. PARSE constraints from the user message into structured filters before
   searching. Time of day → `open_at`. "Affordable" / "fancy" → `price_level_max`.
   "Walking distance" → use the `nearby` tool, not text matching.

2. PREFER structured filters over keyword stuffing. Don't search for
   "italian under $$$ in north beach"; instead call:
       semantic_search(query="romantic italian dinner",
                       filters={price_level_max: 3, neighborhood: "North Beach",
                                open_at: <user time>})

3. PLAN MULTI-STOP itineraries when the user asks for one ("dinner then
   drinks"). Pick stop 1 from semantic_search, then call `nearby` from stop 1
   for stop 2 with appropriate filters (e.g. type=bar, open after dinner ends).

4. JUSTIFY every stop in 1-2 sentences referencing concrete attributes
   (rating, price level, vibe from editorial_summary if present).

5. If a tool returns empty or low-quality results, REVISE: drop the most
   restrictive filter, expand the radius, or ask the user a clarifying
   question. Do NOT pretend you found something you didn't. (Self-correction
   logic in W3.)

6. STOP after at most {max_steps} tool calls. If you don't have a confident
   answer by then, return what you have with an explicit caveat.

OUTPUT FORMAT (when finalizing):
- Set `done=True` and `final_reply` to a 2-4 sentence summary the user reads.
- The structured `stops` list is rendered as cards in the UI; the user sees
  both your prose and the cards.

You are the reasoning model. Use your judgement. Tools are for grounding,
not for thinking on your behalf.
"""
```

### New: `app/agent/graph.py`

```python
"""LangGraph StateGraph for the agent. Three nodes: plan, act, critique.

plan      LLM proposes the next action: a tool call or `final`.
act       Executes the tool. Tool result → state.scratch[<tool>][<step>].
critique  Deterministic + LLM check; can request `revise` (W3 expands this).

Edge logic:
  plan -> act (if tool call) | END (if final)
  act  -> critique
  critique -> plan (if revise or more work) | END (if good)
"""

from __future__ import annotations

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

from app.agent.prompts import SYSTEM_PROMPT
from app.agent.state import ItineraryState, PlaceCard, Stop
from app.agent.tools import all_tools


def build_agent_graph(llm: BaseChatModel, max_steps: int = 8):
    tools = all_tools()
    llm_with_tools = llm.bind_tools(tools)
    tool_by_name = {t.name: t for t in tools}

    # ---- nodes ----------------------------------------------------------------

    def plan(state: ItineraryState) -> ItineraryState:
        if state.step_count == 0:
            state.messages.insert(
                0, SystemMessage(SYSTEM_PROMPT.format(max_steps=max_steps))
            )
        ai: AIMessage = llm_with_tools.invoke(state.messages)
        state.messages.append(ai)
        return state

    def act(state: ItineraryState) -> ItineraryState:
        ai = state.messages[-1]
        if not isinstance(ai, AIMessage) or not ai.tool_calls:
            return state
        for tc in ai.tool_calls:
            tool = tool_by_name.get(tc["name"])
            if tool is None:
                state.messages.append(
                    ToolMessage(content=f"unknown tool {tc['name']}", tool_call_id=tc["id"])
                )
                continue
            try:
                result = tool.invoke(tc["args"])
            except Exception as e:                                 # noqa: BLE001
                result = {"error": str(e)}
            state.messages.append(
                ToolMessage(content=str(result), tool_call_id=tc["id"])
            )
            state.scratch.setdefault(tc["name"], []).append(
                {"args": tc["args"], "result": result, "step": state.step_count}
            )
        state.step_count += 1
        return state

    def critique(state: ItineraryState) -> ItineraryState:
        # W3 fills this in. For W2, do the bare minimum: check loop bound.
        if state.step_count >= max_steps:
            state.done = True
            if not state.final_reply:
                state.final_reply = (
                    "I hit the planning step limit. Here is the best plan I had so far."
                )
        # If the last AI message had no tool calls, treat as final.
        last = state.messages[-1] if state.messages else None
        if isinstance(last, AIMessage) and not last.tool_calls:
            state.done = True
            state.final_reply = state.final_reply or last.content
        return state

    # ---- edges ----------------------------------------------------------------

    def route_after_plan(state: ItineraryState) -> Literal["act", "critique"]:
        last = state.messages[-1]
        return "act" if isinstance(last, AIMessage) and last.tool_calls else "critique"

    def route_after_critique(state: ItineraryState) -> Literal["plan", "__end__"]:
        return END if state.done else "plan"

    g = StateGraph(ItineraryState)
    g.add_node("plan", plan)
    g.add_node("act", act)
    g.add_node("critique", critique)
    g.set_entry_point("plan")
    g.add_conditional_edges("plan", route_after_plan, {"act": "act", "critique": "critique"})
    g.add_edge("act", "critique")
    g.add_conditional_edges("critique", route_after_critique, {"plan": "plan", END: END})
    return g.compile()


def state_to_response(state: ItineraryState, rag_label: str) -> dict:
    """Convert ItineraryState into the {reply, places, ragLabel} contract that
    frontend/src/api/chat.js expects."""
    cards = [
        PlaceCard(
            place_id=s.place_id,
            name=s.name,
            arrival_time=s.arrival_time,
            rationale=s.rationale,
            # other fields filled from get_details cache in scratch if available
        ).model_dump(mode="json")
        for s in state.stops
    ]
    return {
        "reply": state.final_reply or "",
        "places": cards,
        "ragLabel": rag_label,
    }
```

### Modify: `app/main.py`

```python
# Add near existing chain construction:
from app.agent.graph import build_agent_graph, state_to_response
from app.agent.state import ItineraryState
from langchain_core.messages import HumanMessage

# In lifespan() — currently lines 128-134:
async def lifespan(app: FastAPI):
    config = load_registered_rag_chain(...)  # existing
    app.state.rag_chain = config.chain        # existing, kept for /predict
    app.state.active_model_config = config.params

    # NEW: build the agent using the SAME LLM instance/provider the chain uses.
    # build_rag_chain returns (chain, llm) after a small refactor — see note.
    app.state.agent_graph = build_agent_graph(config.llm)
    app.state.rag_label = f"{config.params.llm_provider}:{config.params.chat_model}"
    yield

# NEW Pydantic models matching frontend/src/api/chat.js contract:
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []

class ChatResponse(BaseModel):
    reply: str
    places: list[dict]      # PlaceCard-shaped, validated downstream
    ragLabel: str

# NEW endpoint:
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    graph = request.app.state.agent_graph
    state = ItineraryState(messages=[
        *(_history_to_messages(req.history)),
        HumanMessage(content=req.message),
    ])
    final_state = await graph.ainvoke(state)
    return state_to_response(final_state, request.app.state.rag_label)


def _history_to_messages(history: list[ChatMessage]):
    out = []
    for m in history:
        if m.role == "user":
            out.append(HumanMessage(m.content))
        else:
            out.append(AIMessage(m.content))
    return out


# /predict (existing lines 183-196): keep, but rewrite as adapter:
@app.post("/predict", response_model=RecommendationResponse)
async def predict(body: RecommendationRequest, request: Request):
    chat_resp = await chat(
        ChatRequest(message=body.query, history=[]), request,
    )
    return RecommendationResponse(
        response=chat_resp["reply"],
        sources=[
            RecommendationSource(
                place_id=p["place_id"], name=p["name"],
                rating=p.get("rating"), address=p.get("address"),
                primary_type=p.get("primary_type"),
                similarity=0.0,
            )
            for p in chat_resp["places"][: body.limit]
        ],
    )
```

### Modify: `app/chain.py`

Tiny refactor: `build_rag_chain` should return both the chain and the LLM instance so `lifespan` can hand the LLM to `build_agent_graph` without re-instantiating it.

```python
@dataclass
class BuiltChain:
    chain: RetrievalQA
    llm: BaseChatModel

def build_rag_chain(...) -> BuiltChain:
    ...  # existing logic
    return BuiltChain(chain=chain, llm=llm)
```

Update `load_registered_rag_chain` (`app/main.py:75-110`) to return both `chain` and `llm` (small dataclass `LoadedConfig`). Existing tests will need shape updates — covered below.

## Tests

### New: `tests/unit/test_agent_state.py`

- Construct `ItineraryState`, append a `Stop`, serialize via `model_dump()`, assert round-trip.
- Confirm empty `stops` produces empty `places` in `state_to_response`.

### New: `tests/unit/test_agent_graph.py`

```python
"""Integration of the graph with a fake LLM that emits a scripted sequence
of tool calls. Verifies plan->act->critique loops correctly and terminates."""

class FakeLLM(BaseChatModel):
    """Configured with a list of responses to return in sequence."""
    def __init__(self, scripted: list[AIMessage]):
        self._scripted = list(scripted)
    def bind_tools(self, tools):
        return self
    def invoke(self, messages, **_):
        return self._scripted.pop(0)


def test_graph_terminates_on_no_tool_call():
    fake = FakeLLM([AIMessage(content="here is your plan", tool_calls=[])])
    g = build_agent_graph(fake, max_steps=4)
    out = g.invoke(ItineraryState(messages=[HumanMessage("hi")]))
    assert out.done is True
    assert out.final_reply == "here is your plan"


def test_graph_executes_tool_and_continues(monkeypatch):
    monkeypatch.setattr(
        "app.tools.retrieval.semantic_search",
        lambda **kw: [PlaceHit(place_id="p1", name="X", source="google_places",
                               similarity=0.9, latitude=None, longitude=None,
                               rating=4.5, price_level=2,
                               business_status="OPERATIONAL",
                               primary_type="restaurant",
                               formatted_address="...", snippet=None)],
    )
    fake = FakeLLM([
        AIMessage(content="", tool_calls=[{
            "name": "semantic_search", "id": "1",
            "args": {"query": "italian", "k": 3},
        }]),
        AIMessage(content="found one place", tool_calls=[]),
    ])
    g = build_agent_graph(fake, max_steps=4)
    out = g.invoke(ItineraryState(messages=[HumanMessage("italian please")]))
    assert out.step_count == 1
    assert "semantic_search" in out.scratch
    assert out.done is True


def test_graph_respects_max_steps():
    # Fake that always emits a tool call -> runaway -> bounded
    looping = [
        AIMessage(content="", tool_calls=[{
            "name": "semantic_search", "id": str(i),
            "args": {"query": "x"},
        }])
        for i in range(20)
    ]
    fake = FakeLLM(looping)
    g = build_agent_graph(fake, max_steps=3)
    out = g.invoke(ItineraryState(messages=[HumanMessage("x")]))
    assert out.step_count == 3
    assert out.done is True
```

### New: `tests/unit/test_chat_endpoint.py`

Mirror `tests/unit/test_predict_endpoint.py:1-60`. Mock `app.state.agent_graph.ainvoke` to return a known `ItineraryState`; assert response shape exactly matches `{reply, places, ragLabel}` and `places` items have the expected keys.

### Modify: `tests/unit/test_chain.py`

Update for the `BuiltChain` return type. Three existing tests at `tests/unit/test_chain.py:1-88` need shape updates only.

### Modify: `tests/unit/test_predict_endpoint.py`

Update fixtures to construct an agent graph mock; the endpoint now goes through the agent.

## Manual verification

```bash
make dev
make migrate
python scripts/seed.py

# Smoke test the new endpoint:
curl -s http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "italian dinner around 7pm tonight in north beach", "history": []}' | jq .

# Confirm /predict still works (compat):
curl -s http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"query": "italian dinner around 7pm tonight in north beach", "limit": 5}' | jq .

# Confirm the frontend can call /chat without code changes:
( cd frontend && npm run dev )  # then load the UI and send a message
```

Expected: at least one stop, all rows real (cross-check `place_id` against DB), `ragLabel` reflects the current MLflow-selected provider:model.

## Risks / open questions

- **Streaming:** the frontend may want token streaming. Out of scope for this PR; LangGraph supports `astream_events()` so we can layer it on later without changing the contract.
- **Session memory:** `history` is taken from the request, so no server-side state is needed. If we add server-side sessions later, it's an `app.state.sessions` cache, not a graph change.
- **Cost ceiling:** `max_steps=8` with Opus 4.7 could be expensive per request. Add a `AGENT_MAX_STEPS` env var (default 8) and surface it in `app/config.py` so we can tune without redeploying. Add a per-request token-counter log line for visibility.
- **Tool argument coercion:** LangChain's `@tool` parses Pydantic args from the LLM's JSON. Validate that `SearchFilters` (a nested model) deserializes correctly — add an explicit `args_schema` if not. Cover this in `test_agent_graph.py` with a tool call that includes filters.
- **Provider parity:** Anthropic + OpenAI + Gemini all support tool-calling but with subtle differences (Gemini occasionally emits malformed tool args). LangChain abstracts this; if Gemini misbehaves in practice, downgrade Gemini to "tool-light" mode (single search call only) via a config flag.
