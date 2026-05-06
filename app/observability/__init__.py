"""Per-request tracing for the agent product (W0).

Wired via Langfuse — self-hostable alongside MLflow if cost matters, or use
the free Cloud tier for class-project usage. Falls back to a no-op if
LANGFUSE_SECRET_KEY isn't set so local dev still works.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger("city_concierge.observability")

try:
    from langfuse import Langfuse
    from langfuse.callback import CallbackHandler
except ImportError:  # pragma: no cover - exercised only when extras are absent
    Langfuse = None  # type: ignore[assignment,misc]
    CallbackHandler = None  # type: ignore[assignment,misc]

_warned_missing_package = False


def _tracing_enabled() -> bool:
    """Single source of truth for 'is per-request tracing turned on?'."""
    return Langfuse is not None and bool(os.getenv("LANGFUSE_SECRET_KEY"))


def _warn_if_package_missing_but_env_set() -> None:
    """Warn once if env says tracing is on but the langfuse package failed to import.

    Distinguishes a real misconfig (deps drift, broken image) from an intentional
    no-op (local dev with no env). Fires at most once per process.
    """
    global _warned_missing_package
    if _warned_missing_package:
        return
    if Langfuse is None and os.getenv("LANGFUSE_SECRET_KEY"):
        logger.warning(
            "LANGFUSE_SECRET_KEY is set but the 'langfuse' package is not installed; "
            "tracing is silently disabled. Check that the deployed image includes it."
        )
        _warned_missing_package = True


def get_client() -> Any | None:
    """Return a configured Langfuse client, or None if disabled.

    Returns None on any SDK construction failure (bad host, version skew, etc.) —
    tracing must never break the request path.
    """
    _warn_if_package_missing_but_env_set()
    if not _tracing_enabled():
        return None
    try:
        return Langfuse(
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
    except Exception as exc:
        logger.warning("Langfuse client construction failed: %s", exc)
        return None


def langgraph_callbacks() -> list[Any]:
    """Callbacks list to pass to LangGraph/LangChain `config={"callbacks": ...}`.

    Empty list when tracing is disabled — LangChain treats that as a no-op.
    """
    _warn_if_package_missing_but_env_set()
    if not _tracing_enabled() or CallbackHandler is None:
        return []
    return [CallbackHandler()]


@contextmanager
def trace_request(name: str, **metadata: Any) -> Iterator[str | None]:
    """Wrap an agent invocation. Yields a trace id you can attach to logs.

    Falls through to a no-op (yield None) if the SDK raises while creating the
    trace — observability must never break user requests.
    """
    client = get_client()
    if client is None:
        yield None
        return
    try:
        trace = client.trace(name=name, metadata=metadata)
    except Exception as exc:
        logger.warning("Langfuse trace() failed: %s", exc)
        yield None
        return
    try:
        yield trace.id
    finally:
        try:
            client.flush()
        except Exception as exc:
            logger.warning("Langfuse flush() failed: %s", exc)
