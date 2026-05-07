"""Tool definitions. We author tools with Pydantic AI for type safety, and
expose them as LangChain Tool instances so LangGraph's plan() node can bind
them to the LLM. Underlying Python functions remain importable from
app.tools.* for eval (W6) and tests."""

from __future__ import annotations

import inspect

from langchain_core.tools import StructuredTool
from pydantic import create_model
from pydantic_ai import RunContext

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


def semantic_search(
    ctx: RunContext[None],
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
    ctx: RunContext[None],
    place_id: str,
    radius_m: int = 800,
    filters: SearchFilters | None = None,
    k: int = 8,
) -> list[PlaceHit]:
    """Find places within radius_m meters of an anchor place. Call this AFTER
    you've picked a first stop and need a second stop within walking distance."""
    return _nearby(place_id=place_id, radius_m=radius_m, filters=filters, k=k)


def get_details(ctx: RunContext[None], place_id: str) -> PlaceDetails | None:
    """Fetch the full record for a place: hours, website, ratings count, types."""
    return _get_details(place_id=place_id)


def kg_traverse(ctx: RunContext[None], place_id: str, relation: str = "co_mentioned") -> dict:
    """Traverse the editorial knowledge graph from `place_id`. NOT YET AVAILABLE.

    Stub: the KG lands in a future PR after the editorial scrape is done.
    The tool exists now so the agent's tool surface is stable.
    """
    return {
        "available": False,
        "reason": "knowledge graph not yet built; use semantic_search instead",
    }


def _args_schema_for(fn):
    """Reuse the tool function's annotations as a Pydantic args schema. Keeps
    a single source of truth for arg validation."""
    sig = inspect.signature(fn)
    fields = {}
    for pname, param in sig.parameters.items():
        if pname == "ctx":
            continue
        ann = param.annotation if param.annotation is not inspect._empty else str
        default = param.default if param.default is not inspect._empty else ...
        fields[pname] = (ann, default)
    return create_model(f"{fn.__name__}_args", **fields)


def _to_lc_tool(name: str, description: str, fn) -> StructuredTool:
    def _runner(**kwargs):
        return fn(None, **kwargs)

    return StructuredTool.from_function(
        name=name,
        description=description or "",
        func=_runner,
        args_schema=_args_schema_for(fn),
    )


def all_tools() -> list[StructuredTool]:
    return [
        _to_lc_tool("semantic_search", semantic_search.__doc__ or "", semantic_search),
        _to_lc_tool("nearby", nearby.__doc__ or "", nearby),
        _to_lc_tool("get_details", get_details.__doc__ or "", get_details),
        _to_lc_tool("kg_traverse", kg_traverse.__doc__ or "", kg_traverse),
    ]
