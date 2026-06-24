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
from app.tools.graph import RelatedPlace
from app.tools.retrieval import (
    PlaceDetails,
    PlaceHit,
)
from app.tools.retrieval import (
    get_details as get_details_impl,
)
from app.tools.retrieval import (
    nearby as nearby_impl,
)
from app.tools.retrieval import (
    semantic_search as semantic_search_impl,
)

COMMIT_ITINERARY_TOOL_NAME = "commit_itinerary"


def semantic_search(
    query: str,
    filters: SearchFilters | None = None,
    k: int = 8,
    slot_index: int | None = None,
) -> list[PlaceHit]:
    """Search for places by meaning + structured filters.

    Use this for queries like "romantic italian in north beach under $$$ open
    Sunday at 7pm". Prefer the structured `filters` argument over packing
    constraints into `query` — but filters REFINE the query, they DO NOT
    replace it. `query` must always stay descriptive: include at minimum the
    cuisine or vibe, the place type, and the neighborhood, even when the same
    information is also passed as a filter. A bare query like "lunch" embeds
    poorly and retrieves weak matches.
    Pass `slot_index = i` (0-based) when retrieving for stop *i* in a query
    that named per-slot categories (e.g., 'omakase, then drinks, then dessert').
    Leave None for free-text queries.
    """
    return semantic_search_impl(query=query, filters=filters, k=k)


def nearby(
    place_id: str,
    radius_m: int = 800,
    filters: SearchFilters | None = None,
    k: int = 8,
    slot_index: int | None = None,
) -> list[PlaceHit]:
    """Find places within radius_m meters of an anchor place. Call this AFTER
    you've picked a first stop and need a second stop within walking distance.
    Pass `slot_index = i` (0-based) when retrieving for stop *i* in a query
    that named per-slot categories (e.g., 'omakase, then drinks, then dessert').
    Leave None for free-text queries.
    """
    return nearby_impl(place_id=place_id, radius_m=radius_m, filters=filters, k=k)


def get_details(place_id: str) -> PlaceDetails | None:
    """Fetch the full record for a place: hours, website, ratings count, types."""
    return get_details_impl(place_id=place_id)


def commit_itinerary(stops: list[Stop]) -> dict:
    """Finalize the itinerary by committing the chosen stops to state.

    Call this exactly once, after retrieval is complete and before your final
    reply. Every stop's place_id MUST come from a prior tool result — the
    graph rejects unknown place_ids to prevent hallucination. The agent's
    `act` node intercepts this call and writes `state.stops`; the return
    value here is only the bookkeeping payload the LLM sees.
    """
    return {"committed": [s.place_id for s in stops]}


def kg_traverse(
    place_id: str,
    relation_type: str = "SIMILAR_VECTOR",
    k: int = 5,
    excluded_place_ids: list[str] | None = None,
) -> list[RelatedPlace]:
    """Traverse the knowledge graph from `place_id` along a relation_type.

    Pick the relation_type by intent:
    - SIMILAR_VECTOR: "more like this" — same vibe/category as the anchor.
    - SAME_NEIGHBORHOOD: alternates in the same SF neighborhood.
    - NEAR_LANDMARK: the anchor is near a known landmark (museum, park).
    - NEAR: geographic neighbors (~800m) without re-running `nearby`.
    - CONTAINED_IN: the parent venue (e.g. a stall inside a food hall) — rare.

    Single-hop: for multi-hop reasoning call again with the new anchor. If it
    returns empty, fall back to `semantic_search` or `nearby`.
    """
    from app.tools.graph import kg_traverse as kg_traverse_impl

    return kg_traverse_impl(
        place_id=place_id,
        relation_type=relation_type,
        k=k,
        excluded_place_ids=excluded_place_ids,
    )


def args_schema_for(fn: Any):
    """Build a Pydantic args schema from the function's annotations.

    Resolved via typing.get_type_hints so PEP 563 / `from __future__ import
    annotations` strings are evaluated, and raises loudly on any missing
    annotation so a forgotten type hint can't ship a malformed tool surface.
    """
    sig = inspect.signature(fn)
    hints = get_type_hints(fn)
    fields: dict[str, Any] = {}
    for pname, param in sig.parameters.items():
        if pname not in hints:
            raise TypeError(
                f"tool {fn.__name__!r}: parameter {pname!r} is missing a type annotation"
            )
        default = param.default if param.default is not inspect.Parameter.empty else ...
        fields[pname] = (hints[pname], default)
    return create_model(f"{fn.__name__}_args", **fields)


def to_lc_tool(name: str, description: str, fn: Any) -> StructuredTool:
    args_schema = args_schema_for(fn)
    fn.args_schema = args_schema
    return StructuredTool.from_function(
        name=name,
        description=description or "",
        func=fn,
        args_schema=args_schema,
    )


TOOLS: list[StructuredTool] = [
    to_lc_tool("semantic_search", semantic_search.__doc__ or "", semantic_search),
    to_lc_tool("nearby", nearby.__doc__ or "", nearby),
    to_lc_tool("get_details", get_details.__doc__ or "", get_details),
    to_lc_tool(COMMIT_ITINERARY_TOOL_NAME, commit_itinerary.__doc__ or "", commit_itinerary),
    to_lc_tool("kg_traverse", kg_traverse.__doc__ or "", kg_traverse),
]


def all_tools() -> list[StructuredTool]:
    return list(TOOLS)
