"""Tests for app.observability.cost.

Layers covered:
  - smoke      → module imports, helpers callable.
  - unit/mock  → cost math (known/unknown models, edge cases).
  - functional → record_llm_call context-manager behavior end-to-end:
                 latency capture, log line shape, fires on exception,
                 zero-token calls still log, request_id propagation.

No integration layer: cost.py has no external dependencies — it's pure
math + stdlib logging — so an integration test would just re-run the
functional layer.
"""

from __future__ import annotations

import logging
import time
from unittest.mock import patch

import pytest

from app.observability.cost import PRICING, CallRecord, record_llm_call

# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


def test_module_exports() -> None:
    assert callable(record_llm_call)
    assert isinstance(PRICING, dict)
    assert "gpt-4o-mini" in PRICING


def test_callrecord_constructs_with_zero_tokens() -> None:
    rec = CallRecord(model="gpt-4o-mini", tokens_in=0, tokens_out=0, latency_ms=0)
    assert rec.est_cost_usd == 0.0


# ---------------------------------------------------------------------------
# Unit — cost math
# ---------------------------------------------------------------------------


def test_cost_calculation_known_model() -> None:
    rec = CallRecord(
        model="gpt-4o-mini",
        tokens_in=1_000_000,
        tokens_out=500_000,
        latency_ms=0,
    )
    # 0.15 * 1.0  +  0.60 * 0.5  =  0.45
    assert abs(rec.est_cost_usd - 0.45) < 1e-9


def test_cost_zero_for_unknown_model() -> None:
    rec = CallRecord(model="never-heard-of-it", tokens_in=999, tokens_out=999, latency_ms=0)
    assert rec.est_cost_usd == 0.0


def test_cost_handles_embedding_model_no_output() -> None:
    rec = CallRecord(
        model="text-embedding-3-small",
        tokens_in=1_000_000,
        tokens_out=0,
        latency_ms=0,
    )
    # 0.02 * 1.0 = 0.02
    assert abs(rec.est_cost_usd - 0.02) < 1e-9


def test_cost_scales_linearly_with_tokens() -> None:
    rec_small = CallRecord(model="gpt-4o", tokens_in=1_000, tokens_out=1_000, latency_ms=0)
    rec_big = CallRecord(model="gpt-4o", tokens_in=10_000, tokens_out=10_000, latency_ms=0)
    assert abs(rec_big.est_cost_usd - 10 * rec_small.est_cost_usd) < 1e-9


def test_pricing_table_has_required_columns() -> None:
    """Every model in PRICING must have both 'in' and 'out' keys."""
    for model, prices in PRICING.items():
        assert "in" in prices, f"{model} missing 'in' price"
        assert "out" in prices, f"{model} missing 'out' price"
        assert prices["in"] >= 0, f"{model} has negative input price"
        assert prices["out"] >= 0, f"{model} has negative output price"


# ---------------------------------------------------------------------------
# Functional — context-manager behavior end-to-end
# ---------------------------------------------------------------------------


def test_record_llm_call_logs_on_exit(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="city_concierge.cost")
    with record_llm_call("gpt-4o-mini", request_id="r1") as rec:
        rec.tokens_in = 100
        rec.tokens_out = 50

    matches = [r for r in caplog.records if r.message == "llm_call"]
    assert len(matches) == 1
    log = matches[0]
    assert log.model == "gpt-4o-mini"
    assert log.tokens_in == 100
    assert log.tokens_out == 50
    assert log.request_id == "r1"
    # 0.15 * 100/1M + 0.60 * 50/1M = 0.000015 + 0.000030 = 0.000045
    assert log.est_cost_usd == round(0.000045, 6)


def test_record_llm_call_captures_latency(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="city_concierge.cost")

    fake_now = [0.0]

    def fake_monotonic() -> float:
        return fake_now[0]

    with (
        patch("app.observability.cost.time.monotonic", side_effect=fake_monotonic),
        record_llm_call("gpt-4o-mini") as rec,
    ):
        fake_now[0] = 1.234  # simulate 1.234s elapsed inside the block
        rec.tokens_in = 1
        rec.tokens_out = 1

    matches = [r for r in caplog.records if r.message == "llm_call"]
    assert len(matches) == 1
    assert matches[0].latency_ms == 1234


def test_record_llm_call_fires_log_on_exception(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="city_concierge.cost")
    with (
        pytest.raises(RuntimeError, match="provider down"),
        record_llm_call("gpt-4o-mini", request_id="r-fail") as rec,
    ):
        rec.tokens_in = 7
        raise RuntimeError("provider down")

    matches = [r for r in caplog.records if r.message == "llm_call"]
    assert len(matches) == 1
    assert matches[0].tokens_in == 7
    assert matches[0].request_id == "r-fail"


def test_record_llm_call_zero_tokens_still_logs(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="city_concierge.cost")
    with record_llm_call("gpt-4o-mini"):
        pass

    matches = [r for r in caplog.records if r.message == "llm_call"]
    assert len(matches) == 1
    assert matches[0].tokens_in == 0
    assert matches[0].tokens_out == 0
    assert matches[0].est_cost_usd == 0.0
    assert matches[0].request_id is None


def test_record_llm_call_unknown_model_logs_zero_cost(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="city_concierge.cost")
    with record_llm_call("brand-new-model") as rec:
        rec.tokens_in = 1_000_000
        rec.tokens_out = 500_000

    matches = [r for r in caplog.records if r.message == "llm_call"]
    assert len(matches) == 1
    assert matches[0].est_cost_usd == 0.0
    assert matches[0].model == "brand-new-model"


def test_record_llm_call_real_clock_yields_nonneg_latency() -> None:
    """Functional sanity check using the real monotonic clock."""
    with record_llm_call("gpt-4o-mini") as rec:
        time.sleep(0.001)
    assert rec.latency_ms >= 0
