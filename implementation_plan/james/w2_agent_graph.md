# W2 — Agent loop + ItineraryState + `/chat` endpoint

**Branch:** `feature/agent-w2-agent-graph`
**Depends on:** W0a (the `EMBEDDING_TABLE` env var that determines which embeddings retrieval reads), W1 (filter columns and `nearby`)
**Unblocks:** W3, W4, W6, W7

## Goal

Replace the single-pass `RetrievalQA` chain with a **LangGraph tool-calling agent** that drives multiple retrieval calls, accumulates a structured `ItineraryState`, and returns a response that matches the contract the frontend already expects (`POST /chat` → `{reply, places, ragLabel}`).

After this PR:
- `POST /chat` is the primary serving endpoint.
- `POST /predict` becomes a thin shim that wraps the agent for backwards compat.
- The agent driver is selected via the existing MLflow registry path — no new model selection.
- `ItineraryState` is the single source of truth for what the agent is building.
- The agent plans **N anchored stops** (default 3, asked from the user when ambiguous), with shared constraints (price, rating, rating-count floor) carried across all stops.
- Each stop has a **planned arrival time** and **planned duration**; subsequent stops are filtered by `open_at = arrival_K`, not by the user's prompt time.
- A total **walking-distance budget** caps the itinerary so "plan a date" doesn't return three places spread across SF.
- The agent surfaces planned durations in the reply so the user can override ("actually, only 30 min for coffee").

## Files

### Modify: `pyproject.toml`

Add LangGraph for orchestration and Pydantic AI for tool definitions:

```toml
langgraph = ">=0.2.0,<1.0.0"
pydantic-ai = ">=0.0.40,<1.0.0"   # type-checked tool args, model-agnostic
```

**Why both, not one:** LangGraph is the orchestration framework — `StateGraph`, durable execution, LangSmith integration, lowest token usage in benchmarks. Pydantic AI is the tool-definition framework — by the Pydantic team, gives compile-time type-checked tool args across 25+ providers, catches "the LLM passed a malformed filter dict" bugs at definition time. Pydantic AI tools are wrappable for LangGraph; we use each library where it's strongest.

(LangChain 0.2 is already pinned at `pyproject.toml:25`.)

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
    """Parsed/inferred constraints from the user message.

    These are SHARED across all stops: a "$$$" budget applies to dinner AND
    drinks AND dessert. Per-stop constraints (cuisine, vibe) live on `Stop`.
    """
    party_size: Optional[int] = None
    budget_per_person_max: Optional[int] = None  # USD
    price_level_max: Optional[int] = None        # 0-4 (Google)
    min_rating: Optional[float] = None
    min_user_rating_count: int = 50              # quality floor (W1 default)
    when: Optional[datetime] = None              # arrival time for stop 1
    neighborhood: Optional[str] = None
    vibes: list[str] = Field(default_factory=list)  # free-text tags
    must_be_open: bool = True
    # --- itinerary-shape constraints (NEW in this revision) ---
    num_stops: Optional[int] = Field(
        default=None,
        description="Total stops the user wants. None = ask. Default 3 if user "
                    "says 'plan a date/evening' without a count.",
    )
    walking_budget_m: int = Field(
        default=2400,
        description="Total walking distance budget across the whole itinerary, "
                    "in meters. Default ~30 min total walking at 80 m/min.",
    )


# Hard-coded duration defaults per primary_type. The agent surfaces these in
# the reply so the user can override. NOT a hidden assumption.
DEFAULT_STOP_DURATION_MIN: dict[str, int] = {
    "restaurant":     90,
    "fine_dining_restaurant": 120,
    "cafe":           45,
    "coffee_shop":    45,
    "bar":            60,
    "wine_bar":       60,
    "cocktail_bar":   60,
    "bakery":         30,
    "ice_cream_shop": 30,
    "museum":         120,
    "art_gallery":    60,
    "park":           30,
    "tourist_attraction": 60,
}
DEFAULT_STOP_DURATION_MIN_FALLBACK = 60


def default_duration_for(primary_type: Optional[str]) -> int:
    if not primary_type:
        return DEFAULT_STOP_DURATION_MIN_FALLBACK
    return DEFAULT_STOP_DURATION_MIN.get(primary_type.lower(),
                                         DEFAULT_STOP_DURATION_MIN_FALLBACK)


class Stop(BaseModel):
    place_id: str
    name: str
    arrival_time: Optional[datetime] = None
    planned_duration_min: int = DEFAULT_STOP_DURATION_MIN_FALLBACK
    rationale: str
    source: str  # 'google_places' | 'editorial'
    # Geo info copied at planning time so the critique can compute walking
    # distance without a second lookup.
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    primary_type: Optional[str] = None


class ItineraryState(BaseModel):
    """The single piece of state passed through every graph node."""
    messages: list[BaseMessage] = Field(default_factory=list)
    constraints: UserConstraints = Field(default_factory=UserConstraints)
    stops: list[Stop] = Field(default_factory=list)
    scratch: dict[str, Any] = Field(default_factory=dict)
    step_count: int = 0
    done: bool = False
    final_reply: Optional[str] = None
    # --- planning bookkeeping (NEW) ---
    awaiting_stops_count: bool = Field(
        default=False,
        description="True after the agent asked the user how many stops they "
                    "want. The /chat handler echoes the question and the next "
                    "turn parses the answer back into constraints.num_stops.",
    )
    walked_meters_so_far: float = 0.0

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

We use **Pydantic AI** to define tools because it gives us type-checked tool args at definition time and a clean function-tool decorator that's portable across providers. We then wrap those tools as LangChain `Tool` instances so the LangGraph node can call them via `llm.bind_tools(...)`. This is the smallest possible adapter — no architectural lock-in either way.

```python
"""Tool definitions. We author tools with Pydantic AI for type safety, and
expose them as LangChain Tool instances so LangGraph's plan() node can bind
them to the LLM. Underlying Python functions remain importable from
app.tools.* for eval (W6) and tests."""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import Tool
from pydantic import BaseModel, Field
from pydantic_ai import RunContext, Tool as PaiTool

from app.tools.filters import SearchFilters
from app.tools.retrieval import (
    PlaceDetails,
    PlaceHit,
    get_details as _get_details,
    nearby as _nearby,
    semantic_search as _semantic_search,
)


# ---- Pydantic AI tool functions -------------------------------------------
# These are the canonical definitions. Pydantic AI validates args via the
# function's type hints. Keep tool docstrings tight — the LLM reads them.

def semantic_search(
    ctx: RunContext[None],
    query: str,
    filters: Optional[SearchFilters] = None,
    k: int = 8,
) -> list[PlaceHit]:
    """Search for places by meaning + structured filters.

    Use this for queries like "romantic italian in north beach under $$$ open
    Sunday at 7pm". Prefer the structured `filters` argument over packing
    constraints into `query`.
    """
    return _semantic_search(query=query, filters=filters, k=k)


def nearby(
    ctx: RunContext[None],
    place_id: str,
    radius_m: int = 800,
    filters: Optional[SearchFilters] = None,
    k: int = 8,
) -> list[PlaceHit]:
    """Find places within radius_m meters of an anchor place. Call this AFTER
    you've picked a first stop and need a second stop within walking distance."""
    return _nearby(place_id=place_id, radius_m=radius_m, filters=filters, k=k)


def get_details(ctx: RunContext[None], place_id: str) -> Optional[PlaceDetails]:
    """Fetch the full record for a place: hours, website, ratings count, types."""
    return _get_details(place_id=place_id)


def kg_traverse(ctx: RunContext[None], place_id: str,
                relation: str = "co_mentioned") -> dict:
    """Traverse the editorial knowledge graph from `place_id`. NOT YET AVAILABLE.

    Stub: the KG lands in a future PR after the editorial scrape is done.
    The tool exists now so the agent's tool surface is stable.
    """
    return {"available": False,
            "reason": "knowledge graph not yet built; use semantic_search instead"}


# ---- LangGraph adapter -----------------------------------------------------
# Convert a Pydantic AI tool function into a LangChain Tool that LangGraph's
# plan() node can pass into llm.bind_tools(). The wrapper is one-line per tool.

def _to_lc_tool(name: str, description: str, fn) -> Tool:
    def _runner(**kwargs):
        # ctx=None because we don't use RunContext deps in v1.
        return fn(None, **kwargs)
    return Tool.from_function(
        name=name,
        description=description,
        func=_runner,
        args_schema=_args_schema_for(fn),
    )


def _args_schema_for(fn):
    """Reuse the tool function's annotations as a Pydantic args schema. Keeps
    a single source of truth for arg validation."""
    import inspect
    from pydantic import create_model
    sig = inspect.signature(fn)
    fields = {}
    for pname, param in sig.parameters.items():
        if pname == "ctx":
            continue
        ann = param.annotation if param.annotation is not inspect._empty else str
        default = param.default if param.default is not inspect._empty else ...
        fields[pname] = (ann, default)
    return create_model(f"{fn.__name__}_args", **fields)


def all_tools() -> list[Tool]:
    return [
        _to_lc_tool("semantic_search", semantic_search.__doc__, semantic_search),
        _to_lc_tool("nearby",          nearby.__doc__,          nearby),
        _to_lc_tool("get_details",     get_details.__doc__,     get_details),
        _to_lc_tool("kg_traverse",     kg_traverse.__doc__,     kg_traverse),
    ]
```

W4 will append `propose_booking` (Pydantic AI tool function) and a corresponding `_to_lc_tool` entry to `all_tools()`.

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
                                open_at: <stop's arrival time>})

3. ITINERARY SHAPE — number of stops:
   - If the user explicitly says "2 stops" / "dinner then drinks" / "3 spots",
     respect it.
   - If ambiguous ("plan me a date", "evening in the mission"), ASK the user
     how many stops they want. Use a single short clarifying question and set
     `awaiting_stops_count=True` in your response. Do not plan stops yet.
   - If the user pushes back ("you decide"), default to 3 stops.

4. PLAN MULTI-STOP itineraries as anchored search:
   - Stop 1 = `semantic_search(...)` with shared constraints +
     `open_at = constraints.when`.
   - Stop K (K > 1) = `nearby(stop_{K-1}.place_id, radius_m, ...)` with shared
     constraints + `open_at = arrival_K`, where
        arrival_K = arrival_{K-1} + planned_duration_{K-1} + walking_time(K-1→K)
        walking_time(a→b) = haversine_meters(a, b) / 80   # m/min
   - SHARED constraints (price_level_max, min_rating, min_user_rating_count,
     overall vibes) carry across every stop. Don't relax them per stop unless
     the user says so or self-correction (W3) requires it.
   - Use a default per-stop duration based on `primary_type`. The planning code
     fills these in from DEFAULT_STOP_DURATION_MIN; surface them in your reply
     so the user can override.

5. WALKING BUDGET:
   - Total walking across all stops should fit `constraints.walking_budget_m`
     (default 2400m ≈ 30 min). For each stop after the first, prefer
     `radius_m` ≤ remaining budget / (remaining stops).
   - You may use `kg_traverse(stop_K, relation_type='NEAR')` after W7 ships as
     a cheaper substitute for `nearby` when you only need geographic neighbors.

6. JUSTIFY every stop in 1-2 sentences referencing concrete attributes
   (rating, price level, vibe from editorial_summary if present).

7. If a tool returns empty or low-quality results, REVISE: drop the most
   restrictive filter, expand the radius, or ask the user a clarifying
   question. Do NOT pretend you found something you didn't. (Self-correction
   logic in W3.)

8. STOP after at most {max_steps} tool calls. If you don't have a confident
   answer by then, return what you have with an explicit caveat.

OUTPUT FORMAT (when finalizing):
- Set `done=True` and `final_reply` to a 2-4 sentence summary the user reads.
- Mention the planned arrival + duration for each stop ("Dinner at 7:00,
  ~90 min") so the user can override if they want.
- The structured `stops` list is rendered as cards in the UI; the user sees
  both your prose and the cards.

You are the reasoning model. Use your judgement. Tools are for grounding,
not for thinking on your behalf.
"""


CLARIFYING_STOPS_COUNT_TEMPLATE = (
    "How many stops would you like me to plan? (e.g. 'just dinner', "
    "'dinner + drinks', or '3 spots'). I default to 3 if you'd rather I pick."
)
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
from app.observability import langgraph_callbacks, trace_request  # from W0

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    graph = request.app.state.agent_graph
    with trace_request("chat", message=req.message[:200]) as trace_id:
        state = ItineraryState(messages=[
            *(_history_to_messages(req.history)),
            HumanMessage(content=req.message),
        ])
        final_state = await graph.ainvoke(
            state,
            config={"callbacks": langgraph_callbacks(),
                    "metadata": {"trace_id": trace_id}},
        )
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

### New: `app/agent/planning.py`

Pure helpers the planning loop relies on. Not an LLM-facing tool — these are
deterministic Python the agent or critique node calls.

```python
"""Pure helpers for itinerary planning math. No LLM, no DB."""

from __future__ import annotations
from datetime import datetime, timedelta
from math import asin, cos, radians, sin, sqrt
from typing import Optional

from app.agent.state import (
    DEFAULT_STOP_DURATION_MIN, DEFAULT_STOP_DURATION_MIN_FALLBACK,
    Stop, default_duration_for,
)


WALKING_SPEED_M_PER_MIN = 80.0  # ~3 mph, casual pace


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = map(radians, a)
    lat2, lon2 = map(radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6371000 * 2 * asin(sqrt(h))


def walking_time_min(a_lat: float, a_lng: float,
                     b_lat: float, b_lng: float) -> float:
    return haversine_m((a_lat, a_lng), (b_lat, b_lng)) / WALKING_SPEED_M_PER_MIN


def next_arrival_time(prev_stop: Stop, next_lat: float, next_lng: float) -> datetime:
    if prev_stop.arrival_time is None:
        raise ValueError("prev_stop.arrival_time must be set before chaining")
    walk = walking_time_min(prev_stop.latitude, prev_stop.longitude,
                            next_lat, next_lng)
    return prev_stop.arrival_time + timedelta(
        minutes=prev_stop.planned_duration_min + walk
    )


def remaining_walking_budget_m(state) -> float:
    """How many meters of walking are left in the user's budget."""
    return max(0.0, state.constraints.walking_budget_m - state.walked_meters_so_far)


def suggested_radius_m(state, remaining_stops: int) -> int:
    """Per-stop radius suggestion given how much walking budget is left."""
    if remaining_stops <= 0:
        return 0
    budget = remaining_walking_budget_m(state)
    return int(min(1500, max(300, budget / max(remaining_stops, 1))))


def parse_stops_count(user_text: str, default: int = 3) -> int:
    """Parse '2', 'just dinner', 'dinner + drinks', '3 spots' → integer."""
    text = user_text.lower().strip()
    if "just dinner" in text or "only dinner" in text:
        return 1
    if "dinner and drinks" in text or "dinner + drinks" in text or "dinner then drinks" in text:
        return 2
    for token in text.split():
        if token.isdigit():
            n = int(token)
            if 1 <= n <= 5:
                return n
    return default
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
- **Provider parity:** Anthropic + OpenAI + Gemini all support tool-calling but with subtle differences (Gemini occasionally emits malformed tool args). Pydantic AI's strict type validation catches malformed args at the boundary instead of letting them propagate; if Gemini misbehaves in practice, the failure surface is "tool call rejected with validation error" (recoverable in W3's critique node) rather than "agent crashes downstream".
- **Stops-count clarifier turn.** The clarifier branch adds one extra round-trip when the user is ambiguous, which slightly hurts latency. Worth it for plan quality; if eval (W6) shows users bounce on the question, swap to "default to 3, ask only on truly ambiguous queries."
- **Walking-time model is naive.** `haversine / 80 m·min⁻¹` ignores hills, traffic lights, and one-way streets. SF has all three. For "is this walkable?" decisions it's good enough; for actual routing the W4 booking-handoff flow uses Google Maps directly. We deliberately don't call the Maps API mid-planning to keep latency / cost predictable.
- **Per-type duration defaults will need tuning.** The numbers in `DEFAULT_STOP_DURATION_MIN` are reasonable guesses (90 min for dinner, 60 min for drinks). After the first eval pass we should reset them from any logged user overrides, not from intuition.
- **Vibe-consistency across stops is a soft prompt instruction, not a hard filter.** Hard-coding "no two consecutive bars" would over-constrain bar crawls. We rely on the LLM + the user's prompt for vibe sequencing; if eval shows incoherent itineraries (fancy → dive bar), revisit by adding a `vibes` carryover that compiles to a soft re-rank.

- **Why not pure Pydantic AI for the orchestration?** Pydantic AI has its own `Agent` class and could replace LangGraph entirely. We don't because LangGraph's `StateGraph` gives us durable execution (resume after crash mid-tool-call), explicit critique node for W3, and direct LangSmith / Langfuse tracing. Using Pydantic AI for tool definitions only is the smallest commitment that gets us the type-safety win without locking us out of LangGraph's orchestration features. If LangGraph proves to be the wrong call later, swapping in Pydantic AI's `Agent` is a localized change in `app/agent/graph.py`.
- **Adapter may be obsolete by implementation time.** The `_to_lc_tool` helper that wraps Pydantic AI tools as LangChain Tool instances exists because (as of plan-author training data, January 2026) LangGraph's `bind_tools` did not natively accept Pydantic AI's tool objects. Before starting W2 implementation, check `langchain-ai/langgraph` and `pydantic/pydantic-ai` release notes — if either side now offers native interop, delete the adapter and use it directly. See `FUTURE_WATCH.md` for the broader watch-list.

---

**Status:** ✅ Merged 2026-05-07 in [#71](https://github.com/deshmukh-neel/mlops_city_concierge/pull/71).

Shipped:
- `app/agent/` package: `state` (ItineraryState/UserConstraints/Stop/PlaceCard), `prompts`, `tools`, `graph`, `planning`, `io` (HTTP↔agent boundary)
- `commit_itinerary` tool with grounding validation — ungrounded `place_id`s are dropped, not silently accepted
- `POST /chat` is primary; `/predict` stays on the legacy `RetrievalQA` chain (no agent fanout) so existing callers don't pay agent latency
- Async `plan` node, `add_messages` reducer on state.messages, embedding LRU cache, ToolMessage pruning before LLM calls
- Tests at all four layers: unit (`test_agent_state/planning/tools/graph/chat_endpoint`), smoke (`test_agent_smoke`), functional (`test_chat_functional`, real graph + scripted LLM), integration (`tests/integration/test_agent_graph.py`, gated on `APP_ENV=integration`)
- Manual smoke (2026-05-07) against Cloud SQL via `cloud-sql-proxy`: `/chat` ambiguous → clarifier turn (no fabricated places); `/chat` unambiguous → committed real `place_id` (e.g. `ChIJpzBpPPCAhYAR...` "The Italian Homemade Company") with planned arrival + duration in the prose; `/predict` returned 5 sources from the legacy chain with similarity scores

Deviations from the original plan:
- Dropped `pydantic-ai` runtime dep — tools are pure Python with `inspect` + `typing.get_type_hints` driving a Pydantic args schema; raises loudly on missing annotations. Adapter became unnecessary.
- `StructuredTool.from_function` instead of `Tool.from_function` (multi-arg tools).
- LangGraph 0.6's compiled graph returns a dict; the `/chat` handler reconstructs `ItineraryState(**raw)` for serialization.
- `RecommendationSource` gained `place_id` and `primary_type` so `/predict`'s legacy contract carries grounded IDs.

Deferred to W3+:
- `critique` node only does loop-bound + final-message detection — full self-correction logic lands in W3.
- `propose_booking` tool (W4); `kg_traverse` is a stub.
- W1 `neighborhood` column not yet on Cloud SQL — agent works without it; filter is a no-op until that migration deploys.
