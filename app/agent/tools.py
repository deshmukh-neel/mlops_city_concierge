"""Tool definitions exposed to the LLM via LangChain's StructuredTool.

The underlying retrieval functions remain importable from app.tools.* for
eval (W6) and tests; this module is the thin LLM-facing wrapper.
"""

from __future__ import annotations

import inspect
from typing import Any, get_type_hints

from langchain_core.tools import StructuredTool
from pydantic import create_model

from app.agent.state import Stop
from app.tools.filters import SearchFilters
from app.tools.retrieval import (
    PlaceDetails,
    PlaceHit,
)
from app.tools.retrieval import (
    get_details as _get_details,
)
from app.tools.retrieval import (
    nearby as _nearby,
)
from app.tools.retrieval import (
    semantic_search as _semantic_search,
)

COMMIT_ITINERARY_TOOL_NAME = "commit_itinerary"


def semantic_search(
    query: str,
    filters: SearchFilters | None = None,
    k: int = 8,
) -> list[PlaceHit]:
    """Search for places by meaning + structured filters.

    Use this for queries like "romantic italian in north beach under $$$ open
    Sunday at 7pm". Prefer the structured `filters` argument over packing
    constraints into `query`.
    """
    return _semantic_search(query=query, filters=filters, k=k)


def nearby(
    place_id: str,
    radius_m: int = 800,
    filters: SearchFilters | None = None,
    k: int = 8,
) -> list[PlaceHit]:
    """Find places within radius_m meters of an anchor place. Call this AFTER
    you've picked a first stop and need a second stop within walking distance."""
    return _nearby(place_id=place_id, radius_m=radius_m, filters=filters, k=k)


def get_details(place_id: str) -> PlaceDetails | None:
    """Fetch the full record for a place: hours, website, ratings count, types."""
    return _get_details(place_id=place_id)


def commit_itinerary(stops: list[Stop]) -> dict:
    """Finalize the itinerary by committing the chosen stops to state.

    Call this exactly once, after retrieval is complete and before your final
    reply. Every stop's place_id MUST come from a prior tool result — the
    graph rejects unknown place_ids to prevent hallucination. The agent's
    `act` node intercepts this call and writes `state.stops`; the return
    value here is only the bookkeeping payload the LLM sees.
    """
    return {"committed": [s.place_id for s in stops]}


def kg_traverse(place_id: str, relation: str = "co_mentioned") -> dict:
    """Traverse the editorial knowledge graph from `place_id`. NOT YET AVAILABLE.

    Stub: the KG lands in a future PR after the editorial scrape is done.
    The tool exists now so the agent's tool surface is stable.
    """
    return {
        "available": False,
        "reason": "knowledge graph not yet built; use semantic_search instead",
    }


def _args_schema_for(fn: Any):
    """Build a Pydantic args schema from the function's annotations.

    Resolved via typing.get_type_hints so PEP 563 / `from __future__ import
    annotations` strings are evaluated, and raises loudly on any missing
    annotation so a forgotten type hint can't ship a malformed tool surface.
    """
    sig = inspect.signature(fn)
    hints = get_type_hints(fn)
    fields: dict[str, tuple[Any, Any]] = {}
    for pname, param in sig.parameters.items():
        if pname not in hints:
            raise TypeError(
                f"tool {fn.__name__!r}: parameter {pname!r} is missing a type annotation"
            )
        default = param.default if param.default is not inspect.Parameter.empty else ...
        fields[pname] = (hints[pname], default)
    return create_model(f"{fn.__name__}_args", **fields)


def _to_lc_tool(name: str, description: str, fn: Any) -> StructuredTool:
    return StructuredTool.from_function(
        name=name,
        description=description or "",
        func=fn,
        args_schema=_args_schema_for(fn),
    )


_TOOLS: list[StructuredTool] = [
    _to_lc_tool("semantic_search", semantic_search.__doc__ or "", semantic_search),
    _to_lc_tool("nearby", nearby.__doc__ or "", nearby),
    _to_lc_tool("get_details", get_details.__doc__ or "", get_details),
    _to_lc_tool(COMMIT_ITINERARY_TOOL_NAME, commit_itinerary.__doc__ or "", commit_itinerary),
    _to_lc_tool("kg_traverse", kg_traverse.__doc__ or "", kg_traverse),
]


def all_tools() -> list[StructuredTool]:
    return list(_TOOLS)
