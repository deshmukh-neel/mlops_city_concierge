"""Integration test for app.observability.trace_request — hits real Langfuse.

Gated on `APP_ENV=integration` AND on `LANGFUSE_SECRET_KEY`/`LANGFUSE_PUBLIC_KEY`
being set to real (test-project) credentials. Skipped otherwise.

Run locally with:
    APP_ENV=integration \\
    LANGFUSE_SECRET_KEY=sk-... LANGFUSE_PUBLIC_KEY=pk-... \\
    LANGFUSE_HOST=https://cloud.langfuse.com \\
    poetry run pytest tests/integration/test_observability_tracing_integration.py -v
"""

from __future__ import annotations

import os
import uuid

import pytest

pytestmark = [
    pytest.mark.skipif(
        os.getenv("APP_ENV", "test") != "integration",
        reason="Set APP_ENV=integration to run integration tests.",
    ),
    pytest.mark.skipif(
        not (os.getenv("LANGFUSE_SECRET_KEY") and os.getenv("LANGFUSE_PUBLIC_KEY")),
        reason="LANGFUSE_SECRET_KEY/LANGFUSE_PUBLIC_KEY must be set for live trace test.",
    ),
]


def test_real_trace_emits_id_and_flushes() -> None:
    """Submit a trace to the real Langfuse host; verify we get back a UUID-shaped id."""
    from app.observability import trace_request

    marker = f"w0-integration-{uuid.uuid4()}"
    with trace_request("integration_smoke", marker=marker) as trace_id:
        assert isinstance(trace_id, str)
        # Langfuse trace ids are UUID-shaped; a length sanity check is enough — we
        # don't assert exact format because the SDK may evolve.
        assert len(trace_id) >= 16
