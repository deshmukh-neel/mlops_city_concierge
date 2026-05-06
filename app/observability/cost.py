"""Per-call token + cost telemetry for the agent product (W0 §5).

Wraps each LLM call so we get a single structured log line per request with
model, tokens_in, tokens_out, est_cost_usd, latency_ms. Lives alongside
Langfuse tracing — Langfuse gives the UI; this gives greppable Cloud Logging
entries for cost dashboards and spot checks.

Logs go to stdout, picked up by Cloud Run's log router. For dashboards, point
Cloud Logging's log-based metrics at jsonPayload.est_cost_usd.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

logger = logging.getLogger("city_concierge.cost")

# USD per 1M tokens. Update when prices change. Source of truth: each provider's
# public pricing page. Treat these numbers as estimates and refresh quarterly.
PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "gpt-4o": {"in": 2.50, "out": 10.00},
    "claude-opus-4-7": {"in": 15.00, "out": 75.00},
    "claude-sonnet-4-6": {"in": 3.00, "out": 15.00},
    "claude-haiku-4-5": {"in": 0.80, "out": 4.00},
    "gemini-2.5-flash": {"in": 0.075, "out": 0.30},
    "text-embedding-3-small": {"in": 0.02, "out": 0.0},
}


@dataclass
class CallRecord:
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: int

    @property
    def est_cost_usd(self) -> float:
        p = PRICING.get(self.model)
        if not p:
            return 0.0
        return (self.tokens_in / 1e6) * p["in"] + (self.tokens_out / 1e6) * p["out"]


@contextmanager
def record_llm_call(model: str, request_id: str | None = None) -> Iterator[CallRecord]:
    """Time an LLM call and emit a structured cost log line on exit.

    Caller is responsible for setting `tokens_in` / `tokens_out` on the yielded
    record from the provider's response usage block. Latency is captured
    automatically. The log line fires even if the wrapped block raises.
    """
    rec = CallRecord(model=model, tokens_in=0, tokens_out=0, latency_ms=0)
    start = time.monotonic()
    try:
        yield rec
    finally:
        rec.latency_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "llm_call",
            extra={
                "model": rec.model,
                "tokens_in": rec.tokens_in,
                "tokens_out": rec.tokens_out,
                "latency_ms": rec.latency_ms,
                "est_cost_usd": round(rec.est_cost_usd, 6),
                "request_id": request_id,
            },
        )
