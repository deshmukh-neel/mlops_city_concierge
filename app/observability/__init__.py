"""Per-request tracing for the agent product (W0).

Wired via Langfuse — self-hostable alongside MLflow if cost matters, or use
the free Cloud tier for class-project usage. Falls back to a no-op if
LANGFUSE_SECRET_KEY isn't set so local dev still works.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

try:
    from langfuse import Langfuse
    from langfuse.callback import CallbackHandler
except ImportError:  # pragma: no cover - exercised only when extras are absent
    Langfuse = None  # type: ignore[assignment,misc]
    CallbackHandler = None  # type: ignore[assignment,misc]


def get_client() -> Any | None:
    """Return a configured Langfuse client, or None if disabled."""
    if Langfuse is None:
        return None
    if not os.getenv("LANGFUSE_SECRET_KEY"):
        return None
    return Langfuse(
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )


def langgraph_callbacks() -> list[Any]:
    """Callbacks list to pass to LangGraph/LangChain `config={"callbacks": ...}`.

    Empty list when tracing is disabled — LangChain treats that as a no-op.
    """
    if CallbackHandler is None or not os.getenv("LANGFUSE_SECRET_KEY"):
        return []
    return [CallbackHandler()]


@contextmanager
def trace_request(name: str, **metadata: Any) -> Iterator[str | None]:
    """Wrap an agent invocation. Yields a trace id you can attach to logs."""
    client = get_client()
    if client is None:
        yield None
        return
    trace = client.trace(name=name, metadata=metadata)
    try:
        yield trace.id
    finally:
        client.flush()
