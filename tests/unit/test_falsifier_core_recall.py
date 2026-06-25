"""Unit tests for Phase 19 scoring primitives in app/loop/falsifier_core.py.

Covers:
- compute_recall_at_k + RecallAtKResult (Task 1)
- FLOOR constant + decide_loop_exit gate (Task 2)

All tests are stdlib-only / zero API cost.
"""

from __future__ import annotations

import pytest

from app.loop.falsifier_core import (
    EXIT_FAIL,
    EXIT_INFRA,
    EXIT_PASS,
    FLOOR,
    GuardResult,
    RecallAtKResult,
    compute_recall_at_k,
    decide_loop_exit,
)

# ---------------------------------------------------------------------------
# TestComputeRecallAtK
# ---------------------------------------------------------------------------


class TestComputeRecallAtK:
    def test_all_new_ids_found_across_paraphrases(self) -> None:
        """All newly ingested IDs appear in some paraphrase top-k → recall == 1.0."""
        newly_ingested_ids = {"new-1", "new-2", "new-3"}
        per_paraphrase_topk = [
            ["new-1", "old-a", "old-b", "old-c", "old-d"],
            ["new-2", "old-a", "old-b", "old-c", "old-d"],
            ["new-3", "old-a", "old-b", "old-c", "old-d"],
        ]
        result = compute_recall_at_k(per_paraphrase_topk, newly_ingested_ids)
        assert isinstance(result, RecallAtKResult)
        assert result.found_count == 3
        assert result.total_count == 3
        assert pytest.approx(result.recall) == 1.0

    def test_partial_coverage_returns_correct_recall(self) -> None:
        """2 of 4 new IDs appear across paraphrases → recall == 0.5."""
        newly_ingested_ids = {"new-1", "new-2", "new-3", "new-4"}
        per_paraphrase_topk = [
            ["new-1", "old-a", "old-b", "old-c", "old-d"],
            ["new-2", "old-a", "old-b", "old-c", "old-d"],
            ["old-a", "old-b", "old-c", "old-d", "old-e"],
            ["old-b", "old-c", "old-d", "old-e", "old-f"],
        ]
        result = compute_recall_at_k(per_paraphrase_topk, newly_ingested_ids)
        assert result.found_count == 2
        assert result.total_count == 4
        assert pytest.approx(result.recall) == 0.5

    def test_empty_newly_ingested_ids_returns_zero_no_division_error(self) -> None:
        """Empty newly_ingested_ids → RecallAtKResult(0, 0, 0.0), no ZeroDivisionError."""
        per_paraphrase_topk = [
            ["old-a", "old-b", "old-c", "old-d", "old-e"],
        ]
        result = compute_recall_at_k(per_paraphrase_topk, set())
        assert result.found_count == 0
        assert result.total_count == 0
        assert pytest.approx(result.recall) == 0.0

    def test_topk_longer_than_k_raises_assertion_error(self) -> None:
        """A top-k list longer than K raises AssertionError (IN-02)."""
        newly_ingested_ids = {"new-1"}
        per_paraphrase_topk = [
            ["old-a", "old-b", "old-c", "old-d", "old-e", "old-f"],  # 6 > K=5
        ]
        with pytest.raises(AssertionError):
            compute_recall_at_k(per_paraphrase_topk, newly_ingested_ids)

    def test_same_new_id_in_multiple_paraphrases_counts_once(self) -> None:
        """The same new ID found in multiple paraphrases counts ONCE (distinct union)."""
        newly_ingested_ids = {"new-1", "new-2"}
        per_paraphrase_topk = [
            ["new-1", "old-a", "old-b", "old-c", "old-d"],
            ["new-1", "old-a", "old-b", "old-c", "old-d"],  # same new-1, repeated
            ["new-1", "old-a", "old-b", "old-c", "old-d"],  # same new-1, again
        ]
        # Only new-1 found (new-2 never appears); found_count must be 1, not 3
        result = compute_recall_at_k(per_paraphrase_topk, newly_ingested_ids)
        assert result.found_count == 1
        assert result.total_count == 2
        assert pytest.approx(result.recall) == 0.5


# ---------------------------------------------------------------------------
# TestDecideLoopExit
# ---------------------------------------------------------------------------


class TestDecideLoopExit:
    def test_guard_violation_returns_infra(self) -> None:
        """guard_violation present with ok=False → EXIT_INFRA (highest priority)."""
        violation = GuardResult(ok=False, message="prod collision")
        result = decide_loop_exit(
            before_rate=0.0,
            after_rate=0.8,
            floor=0.0,
            guard_violation=violation,
            embed_added_count=3,
        )
        assert result == EXIT_INFRA

    def test_zero_embed_added_returns_infra(self) -> None:
        """embed_added_count == 0 → EXIT_INFRA (D-02 empty-diff loud-fail)."""
        result = decide_loop_exit(
            before_rate=0.0,
            after_rate=0.8,
            floor=0.0,
            guard_violation=None,
            embed_added_count=0,
        )
        assert result == EXIT_INFRA

    def test_positive_delta_above_floor_returns_pass(self) -> None:
        """Strictly-positive delta AND after_rate >= floor → EXIT_PASS."""
        result = decide_loop_exit(
            before_rate=0.2,
            after_rate=0.8,
            floor=0.5,
            guard_violation=None,
            embed_added_count=3,
        )
        assert result == EXIT_PASS

    def test_positive_delta_below_floor_returns_fail(self) -> None:
        """Strictly-positive delta BUT after_rate < floor → EXIT_FAIL (below calibrated bar, D-05)."""
        result = decide_loop_exit(
            before_rate=0.2,
            after_rate=0.4,
            floor=0.6,
            guard_violation=None,
            embed_added_count=3,
        )
        assert result == EXIT_FAIL

    def test_non_positive_delta_returns_fail(self) -> None:
        """Non-positive delta → EXIT_FAIL."""
        result = decide_loop_exit(
            before_rate=0.8,
            after_rate=0.8,
            floor=0.0,
            guard_violation=None,
            embed_added_count=3,
        )
        assert result == EXIT_FAIL

    def test_floor_zero_reduces_to_strict_positive_delta_only(self) -> None:
        """floor == 0.0 means any strictly-positive delta passes (first-run default, D-05)."""
        result = decide_loop_exit(
            before_rate=0.0,
            after_rate=0.2,  # very low but > 0
            floor=0.0,
            guard_violation=None,
            embed_added_count=5,
        )
        assert result == EXIT_PASS


# ---------------------------------------------------------------------------
# TestFLOORConstant
# ---------------------------------------------------------------------------


class TestFLOORConstant:
    def test_floor_default_is_zero(self) -> None:
        """FLOOR defaults to 0.0 — strict-positive-delta only for the first run (D-05)."""
        assert FLOOR == 0.0

    def test_floor_is_float(self) -> None:
        """FLOOR is a float (runtime-tunable constant, D-05)."""
        assert isinstance(FLOOR, float)
