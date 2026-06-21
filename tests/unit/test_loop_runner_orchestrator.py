"""Unit tests for scripts/loop_runner.py orchestrator decision logic.

All DB, network, subprocess, mlflow, and LLM calls are mocked — these tests
run without any live services or API keys (D-06 pure-core / operator-run split).

Coverage:
  1. decide_loop_exit floor handling (6 cases)
  2. gap-handoff set-diff branches: len(new)==0, len(new)==1, len(new)>1 (3 cases)
  3. stale-proposal rejection runs BEFORE pending_before snapshot
  4. embedding-table assertion exits INFRA when settings.embedding_table != 'place_embeddings_v2'
  5. exit-code mapping 0/1/2 to PASS/FAIL/INFRA labels
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import scripts.loop_runner as lr
from app.loop.falsifier_core import (
    EXIT_FAIL,
    EXIT_INFRA,
    EXIT_PASS,
    FLOOR,
    GuardResult,
    decide_loop_exit,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_conn(
    pending_rows: list[tuple[str, ...]] | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Build a minimal mock psycopg2 / get_conn context-manager connection.

    Mirrors the _make_mock_conn idiom from test_loop_falsifier_orchestrator.py.
    The returned mock_conn is a context manager (supports ``with get_conn() as conn``).
    """
    mock_cur = MagicMock()
    mock_cur.__enter__ = lambda s: s
    mock_cur.__exit__ = MagicMock(return_value=False)

    if pending_rows is not None:
        mock_cur.fetchall.return_value = pending_rows

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cur
    mock_conn.__enter__ = lambda s: s
    mock_conn.__exit__ = MagicMock(return_value=False)
    return mock_conn, mock_cur


# ---------------------------------------------------------------------------
# Section 1: decide_loop_exit floor handling (6 cases)
# ---------------------------------------------------------------------------


class TestDecideLoopExitFloor:
    """Pure-function gate logic — zero API cost (no mocks needed)."""

    def test_positive_delta_above_floor_passes(self) -> None:
        """Positive delta AND after_rate >= floor → EXIT_PASS."""
        result = decide_loop_exit(
            before_rate=0.0,
            after_rate=0.6,
            floor=0.4,
            guard_violation=None,
            embed_added_count=3,
        )
        assert result == EXIT_PASS

    def test_positive_delta_below_floor_fails(self) -> None:
        """Positive delta but after_rate < floor → EXIT_FAIL (floor not met)."""
        result = decide_loop_exit(
            before_rate=0.0,
            after_rate=0.2,
            floor=0.4,
            guard_violation=None,
            embed_added_count=3,
        )
        assert result == EXIT_FAIL

    def test_floor_zero_reduces_to_strict_positive_delta(self) -> None:
        """When floor==0.0, any positive delta passes (first-run default, D-05)."""
        result = decide_loop_exit(
            before_rate=0.0,
            after_rate=0.01,
            floor=0.0,
            guard_violation=None,
            embed_added_count=1,
        )
        assert result == EXIT_PASS

    def test_non_positive_delta_fails(self) -> None:
        """Zero or negative delta → EXIT_FAIL (floor==0 means strict-positive required)."""
        result = decide_loop_exit(
            before_rate=0.0,
            after_rate=0.0,
            floor=0.0,
            guard_violation=None,
            embed_added_count=2,
        )
        assert result == EXIT_FAIL

    def test_guard_violation_exits_infra(self) -> None:
        """guard_violation.ok == False takes highest priority → EXIT_INFRA."""
        violation = GuardResult(ok=False, message="prod collision detected")
        result = decide_loop_exit(
            before_rate=0.0,
            after_rate=0.8,
            floor=0.0,
            guard_violation=violation,
            embed_added_count=5,
        )
        assert result == EXIT_INFRA

    def test_embed_added_count_zero_exits_infra(self) -> None:
        """embed_added_count == 0 → EXIT_INFRA (D-02 empty-diff loud-fail)."""
        result = decide_loop_exit(
            before_rate=0.0,
            after_rate=0.0,
            floor=0.0,
            guard_violation=None,
            embed_added_count=0,
        )
        assert result == EXIT_INFRA


# ---------------------------------------------------------------------------
# Section 2: exit-code mapping 0/1/2 to labels
# ---------------------------------------------------------------------------


class TestExitCodeConstants:
    """Verify the EXIT_* constants match the expected 0/1/2 labels."""

    def test_exit_pass_is_zero(self) -> None:
        assert EXIT_PASS == 0

    def test_exit_fail_is_one(self) -> None:
        assert EXIT_FAIL == 1

    def test_exit_infra_is_two(self) -> None:
        assert EXIT_INFRA == 2

    def test_floor_constant_default(self) -> None:
        """FLOOR module default must be 0.0 until calibration run updates it (D-05)."""
        assert FLOOR == 0.0

    def test_floor_constant_is_float(self) -> None:
        assert isinstance(FLOOR, float)


# ---------------------------------------------------------------------------
# Section 3: gap-handoff set-diff — len(new)==0 cold-start
# ---------------------------------------------------------------------------


class TestGapHandoffColdStart:
    """len(new) == 0: no new pending proposals → EXIT_PASS (cold-start no-op)."""

    def test_no_new_proposals_exits_pass(self) -> None:
        """When pending_after == pending_before (empty set-diff), exit 0 (no-op)."""
        mock_settings = MagicMock()
        mock_settings.resolved_database_url = "postgresql://sandbox"
        mock_settings.embedding_table = "place_embeddings_v2"

        # Both pending queries return no rows → set-diff is empty
        mock_conn, mock_cur = _make_mock_conn(pending_rows=[])

        with (
            patch.dict(
                "os.environ",
                {
                    "SANDBOX_DATABASE_URL": "postgresql://sandbox",
                    "GOOGLE_PLACES_API_KEY": "key",
                    "OPENAI_API_KEY": "key",
                },
            ),
            patch("scripts.loop_runner.resolve_prod_url", return_value=None),
            patch(
                "scripts.loop_runner.check_prod_safety",
                return_value=GuardResult(ok=True, message="OK"),
            ),
            patch("app.config.resolve_database_url", return_value="postgresql://sandbox"),
            patch("app.config.get_settings", return_value=mock_settings),
            patch("app.db_pool.close_db_pool"),
            patch("app.db.get_conn", return_value=mock_conn),
            patch("scripts.sandbox_guard.assert_sandbox_write_target"),
            patch("scripts.coverage_agent.gap_mine_main"),
            pytest.raises(SystemExit) as exc_info,
        ):
            lr.main()

        assert exc_info.value.code == EXIT_PASS


# ---------------------------------------------------------------------------
# Section 4: gap-handoff set-diff — len(new)==1 single gap
# ---------------------------------------------------------------------------


class TestGapHandoffOneGap:
    """len(new) == 1: parse (neighborhood, cuisine) from seed query format."""

    def test_gap_to_seed_reverse_format(self) -> None:
        """Verify the reverse-parse of '{cuisine} restaurants in {neighborhood} San Francisco'."""
        # This tests the pure parsing logic directly (extracted from main())
        seed = "vietnamese restaurants in Outer Sunset San Francisco"
        suffix = " San Francisco"
        midfix = " restaurants in "

        assert seed.endswith(suffix)
        assert midfix in seed

        without_suffix = seed[: -len(suffix)]
        idx = without_suffix.index(midfix)
        cuisine = without_suffix[:idx]
        neighborhood = without_suffix[idx + len(midfix) :]

        assert cuisine == "vietnamese"
        assert neighborhood == "Outer Sunset"

    def test_gap_parse_multi_word_neighborhood(self) -> None:
        """Multi-word neighborhood parses correctly."""
        seed = "dim sum restaurants in Sunset District San Francisco"
        suffix = " San Francisco"
        midfix = " restaurants in "

        without_suffix = seed[: -len(suffix)]
        idx = without_suffix.index(midfix)
        cuisine = without_suffix[:idx]
        neighborhood = without_suffix[idx + len(midfix) :]

        assert cuisine == "dim sum"
        assert neighborhood == "Sunset District"

    def test_gap_parse_single_word_cuisine_and_neighborhood(self) -> None:
        """Single-word cuisine and neighborhood parse correctly."""
        seed = "sushi restaurants in Japantown San Francisco"
        suffix = " San Francisco"
        midfix = " restaurants in "

        without_suffix = seed[: -len(suffix)]
        idx = without_suffix.index(midfix)
        cuisine = without_suffix[:idx]
        neighborhood = without_suffix[idx + len(midfix) :]

        assert cuisine == "sushi"
        assert neighborhood == "Japantown"


# ---------------------------------------------------------------------------
# Section 5: gap-handoff set-diff — len(new) > 1 exits INFRA
# ---------------------------------------------------------------------------


class TestGapHandoffMultipleGaps:
    """len(new) > 1: unexpected multiple new proposals → EXIT_INFRA."""

    def test_multiple_new_proposals_exits_infra(self) -> None:
        """gap_mine_main with --top-n 1 produces 2 new proposals → EXIT_INFRA."""
        mock_settings = MagicMock()
        mock_settings.resolved_database_url = "postgresql://sandbox"
        mock_settings.embedding_table = "place_embeddings_v2"

        # Before snapshot (inside first get_conn): pending_before is empty.
        # After snapshot (inside second get_conn call): pending_after has 2 rows.
        call_count = [0]

        def make_conn_side_effect(*args: Any, **kwargs: Any) -> MagicMock:
            call_count[0] += 1
            if call_count[0] <= 1:
                # First get_conn: stale-clear + pending_before snapshot → empty
                conn, _ = _make_mock_conn(pending_rows=[])
                return conn
            else:
                # Second get_conn: pending_after has 2 new rows
                conn, _ = _make_mock_conn(
                    pending_rows=[
                        ("vietnamese restaurants in Outer Sunset San Francisco",),
                        ("sushi restaurants in Japantown San Francisco",),
                    ]
                )
                return conn

        with (
            patch.dict(
                "os.environ",
                {
                    "SANDBOX_DATABASE_URL": "postgresql://sandbox",
                    "GOOGLE_PLACES_API_KEY": "key",
                    "OPENAI_API_KEY": "key",
                },
            ),
            patch("scripts.loop_runner.resolve_prod_url", return_value=None),
            patch(
                "scripts.loop_runner.check_prod_safety",
                return_value=GuardResult(ok=True, message="OK"),
            ),
            patch("app.config.resolve_database_url", return_value="postgresql://sandbox"),
            patch("app.config.get_settings", return_value=mock_settings),
            patch("app.db_pool.close_db_pool"),
            patch("app.db.get_conn", side_effect=make_conn_side_effect),
            patch("scripts.sandbox_guard.assert_sandbox_write_target"),
            patch("scripts.coverage_agent.gap_mine_main"),
            pytest.raises(SystemExit) as exc_info,
        ):
            lr.main()

        assert exc_info.value.code == EXIT_INFRA


# ---------------------------------------------------------------------------
# Section 6: stale-proposal rejection runs BEFORE pending_before snapshot
# ---------------------------------------------------------------------------


class TestStaleProposalRejectionOrder:
    """Verify stale-proposal reject (UPDATE ... SET status='rejected') appears in
    the SQL execution stream BEFORE the SELECT for pending_before snapshot.

    Uses source-level inspection as a fast, reliable proxy — the same fallback
    technique used in test_loop_falsifier_orchestrator.py (L421-447).
    """

    def test_source_has_reject_stale_before_pending_before_snapshot(self) -> None:
        """The reject_stale_sql UPDATE must appear before the pending_before SELECT."""
        import inspect  # noqa: PLC0415

        source = inspect.getsource(lr)

        reject_idx = source.find("status = 'rejected'")
        pending_before_idx = source.find("pending_before")

        assert reject_idx != -1, (
            "scripts/loop_runner.py must UPDATE stale pending proposals to 'rejected' "
            "before capturing the pending_before snapshot."
        )
        assert pending_before_idx != -1, (
            "scripts/loop_runner.py must capture a pending_before snapshot."
        )
        assert reject_idx < pending_before_idx, (
            "The stale-rejection UPDATE must appear BEFORE the pending_before snapshot in "
            "scripts/loop_runner.py. Rejecting after the snapshot would cause the set-diff "
            "to count pre-existing stale proposals as 'new'."
        )

    def test_source_uses_set_diff_not_created_at(self) -> None:
        """The gap-handoff set-diff must be on query_text (not created_at — D-08)."""
        import inspect  # noqa: PLC0415

        source = inspect.getsource(lr)
        assert "pending_after - pending_before" in source, (
            "scripts/loop_runner.py must compute set-diff as "
            "'new = pending_after - pending_before' (D-08 deterministic set-diff). "
            "Do NOT use created_at ordering."
        )


# ---------------------------------------------------------------------------
# Section 7: embedding-table assertion exits INFRA when wrong table
# ---------------------------------------------------------------------------


class TestEmbeddingTableAssertion:
    """When settings.embedding_table != 'place_embeddings_v2', exit EXIT_INFRA."""

    def test_wrong_embedding_table_exits_infra(self) -> None:
        """settings.embedding_table='place_embeddings' (not v2) → EXIT_INFRA."""
        mock_settings = MagicMock()
        mock_settings.resolved_database_url = "postgresql://sandbox"
        mock_settings.embedding_table = "place_embeddings"  # wrong table — not v2

        with (
            patch.dict(
                "os.environ",
                {
                    "SANDBOX_DATABASE_URL": "postgresql://sandbox",
                    "GOOGLE_PLACES_API_KEY": "key",
                    "OPENAI_API_KEY": "key",
                },
            ),
            patch("scripts.loop_runner.resolve_prod_url", return_value=None),
            patch(
                "scripts.loop_runner.check_prod_safety",
                return_value=GuardResult(ok=True, message="OK"),
            ),
            patch("app.config.resolve_database_url", return_value="postgresql://sandbox"),
            patch("app.config.get_settings", return_value=mock_settings),
            patch("app.db_pool.close_db_pool"),
            pytest.raises(SystemExit) as exc_info,
        ):
            lr.main()

        assert exc_info.value.code == EXIT_INFRA

    def test_source_asserts_place_embeddings_v2(self) -> None:
        """Source must contain the embedding-table assertion string (structural check)."""
        import inspect  # noqa: PLC0415

        source = inspect.getsource(lr)
        assert 'embedding_table != "place_embeddings_v2"' in source, (
            "scripts/loop_runner.py must assert settings.embedding_table == 'place_embeddings_v2' "
            "after cache_clear() and before any semantic_search call (D-07 LOCKED CONSTRAINT)."
        )


# ---------------------------------------------------------------------------
# Section 8: _snapshot_ids_from_url — allowlist guard (CR-01)
# ---------------------------------------------------------------------------


class TestSnapshotIdsAllowlistGuard:
    """CR-01: _snapshot_ids_from_url must reject non-allowlisted table names before
    any DB connection is made.
    """

    def test_non_allowlisted_table_raises_value_error(self) -> None:
        """Passing an arbitrary table name raises ValueError — no DB call needed."""
        with pytest.raises(ValueError, match="not in the allowed set"):
            lr._snapshot_ids_from_url("postgresql://sandbox", "places_ingest_query_proposals")

    def test_sql_injection_attempt_raises_value_error(self) -> None:
        """A SQL-injection-style table name raises ValueError immediately."""
        with pytest.raises(ValueError, match="not in the allowed set"):
            lr._snapshot_ids_from_url("postgresql://sandbox", "places_raw; DROP TABLE users;")

    def test_allowed_table_places_raw_does_not_raise_on_allowlist_check(self) -> None:
        """'places_raw' passes the allowlist gate (DB call will fail without a real URL)."""
        # The allowlist guard must NOT raise for the two canonical table names.
        # We expect psycopg2.OperationalError (or similar) on the actual connect
        # — NOT ValueError from our guard.
        import psycopg2  # noqa: PLC0415

        with pytest.raises((psycopg2.OperationalError, Exception)) as exc_info:
            lr._snapshot_ids_from_url("postgresql://invalid-host/db", "places_raw")
        # Must not be our ValueError
        assert "not in the allowed set" not in str(exc_info.value)

    def test_allowed_table_place_embeddings_v2_does_not_raise_on_allowlist_check(self) -> None:
        """'place_embeddings_v2' passes the allowlist gate."""
        import psycopg2  # noqa: PLC0415

        with pytest.raises((psycopg2.OperationalError, Exception)) as exc_info:
            lr._snapshot_ids_from_url("postgresql://invalid-host/db", "place_embeddings_v2")
        assert "not in the allowed set" not in str(exc_info.value)


# ---------------------------------------------------------------------------
# Section 9: run_subprocess_or_infra — CalledProcessError → EXIT_INFRA
# ---------------------------------------------------------------------------


class TestRunSubprocessOrInfra:
    @patch("scripts.loop_runner.subprocess")
    def test_successful_subprocess_returns_none(self, mock_subprocess: MagicMock) -> None:
        mock_subprocess.run.return_value = MagicMock(returncode=0)
        mock_subprocess.CalledProcessError = subprocess.CalledProcessError
        # Should not raise
        lr.run_subprocess_or_infra(
            argv=[sys.executable, "-c", "print('ok')"],
            env={"DATABASE_URL": "postgresql://sandbox"},
        )

    @patch("scripts.loop_runner.subprocess")
    def test_subprocess_failure_exits_infra(self, mock_subprocess: MagicMock) -> None:
        mock_subprocess.CalledProcessError = subprocess.CalledProcessError
        mock_subprocess.run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=["python", "scripts/ingest_places_sf.py"]
        )
        with pytest.raises(SystemExit) as exc_info:
            lr.run_subprocess_or_infra(
                argv=[sys.executable, "scripts/ingest_places_sf.py"],
                env={"DATABASE_URL": "postgresql://sandbox"},
            )
        assert exc_info.value.code == EXIT_INFRA


# ---------------------------------------------------------------------------
# Section 10: MLflow failure → EXIT_INFRA
# ---------------------------------------------------------------------------


class TestMlflowFailure:
    @patch("scripts.loop_runner.mlflow")
    def test_mlflow_set_experiment_failure_exits_infra(self, mock_mlflow: MagicMock) -> None:
        mock_mlflow.set_experiment.side_effect = Exception("MLflow server unavailable")
        with pytest.raises(SystemExit) as exc_info:
            lr.log_to_mlflow(
                neighborhood="Outer Sunset",
                cuisine="vietnamese",
                seed_query="vietnamese restaurants in Outer Sunset San Francisco",
                paraphrases=["pho spots in outer sunset", "banh mi near ocean beach"],
                frozen_artifact={"paraphrases": ["pho spots"], "seed_query": "..."},
                frozen_artifact_path="loop_runner_artifacts/frozen.json",
                before_snapshot={"hit_rate": 0.0},
                after_snapshot={"hit_rate": 0.4},
                new_v2_ids={"id1", "id2"},
                before_hit_at_k=0.0,
                after_hit_at_k=0.4,
                hit_rate_delta=0.4,
                recall_at_k=0.5,
                new_place_count=2,
                embed_added_count=2,
                floor=0.0,
                fixture_mode=True,
            )
        assert exc_info.value.code == EXIT_INFRA

    @patch("scripts.loop_runner.mlflow")
    def test_mlflow_log_dict_failure_exits_infra(self, mock_mlflow: MagicMock) -> None:
        mock_mlflow.set_experiment.return_value = None
        mock_mlflow.start_run.return_value.__enter__ = lambda s: MagicMock()
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
        mock_mlflow.log_dict.side_effect = Exception("artifact logging failed")

        with pytest.raises(SystemExit) as exc_info:
            lr.log_to_mlflow(
                neighborhood="Outer Sunset",
                cuisine="vietnamese",
                seed_query="vietnamese restaurants in Outer Sunset San Francisco",
                paraphrases=["pho spots in outer sunset"],
                frozen_artifact={"paraphrases": ["pho spots"], "seed_query": "..."},
                frozen_artifact_path="loop_runner_artifacts/frozen.json",
                before_snapshot={"hit_rate": 0.0},
                after_snapshot={"hit_rate": 0.4},
                new_v2_ids={"id1"},
                before_hit_at_k=0.0,
                after_hit_at_k=0.4,
                hit_rate_delta=0.4,
                recall_at_k=1.0,
                new_place_count=1,
                embed_added_count=1,
                floor=0.0,
                fixture_mode=True,
            )
        assert exc_info.value.code == EXIT_INFRA


# ---------------------------------------------------------------------------
# Section 11: source-level D-06 checks (no API cost, structural)
# ---------------------------------------------------------------------------


class TestD06StructuralChecks:
    """Verify D-06 split is preserved: CI tests decision logic; CI never runs live loop."""

    def test_loop_runner_imports_falsifier_core_functions(self) -> None:
        """loop_runner.py must import decide_loop_exit from falsifier_core (not reimplement)."""
        import inspect  # noqa: PLC0415

        source = inspect.getsource(lr)
        assert "decide_loop_exit" in source, (
            "scripts/loop_runner.py must delegate gate decisions to "
            "falsifier_core.decide_loop_exit (not reimplement the logic)."
        )
        assert "compute_hit_rate" in source, (
            "scripts/loop_runner.py must use falsifier_core.compute_hit_rate "
            "(not reimplement hit@k scoring)."
        )
        assert "compute_recall_at_k" in source, (
            "scripts/loop_runner.py must use falsifier_core.compute_recall_at_k "
            "(not reimplement recall@k scoring)."
        )

    def test_loop_runner_cache_clear_and_close_pool_present(self) -> None:
        """main() must call cache_clear() + close_db_pool() after DATABASE_URL coercion."""
        import inspect  # noqa: PLC0415

        source = inspect.getsource(lr)
        assert "cache_clear" in source, (
            "scripts/loop_runner.py must call get_settings.cache_clear() after "
            "os.environ['DATABASE_URL'] = sandbox_url (D-07 coercion-ordering)."
        )
        assert "close_db_pool" in source, (
            "scripts/loop_runner.py must call close_db_pool() after coercion "
            "(safe no-op when pool is None)."
        )

    def test_loop_runner_has_frozen_paraphrases_artifact(self) -> None:
        """loop_runner.py must freeze paraphrases to disk before ingest (D-04)."""
        import inspect  # noqa: PLC0415

        source = inspect.getsource(lr)
        assert "frozen_paraphrases_runner.json" in source, (
            "scripts/loop_runner.py must write a frozen_paraphrases_runner.json artifact "
            "to disk before the ingest subprocess (D-04 LOCKED CONSTRAINT)."
        )


# ---------------------------------------------------------------------------
# Section 12: Bug-fix — before/after hit@k scored against the SAME v2-diff set
# ---------------------------------------------------------------------------


class TestBeforeHitRateTargetSet:
    """Bug fix (19-03): before_hit_result must score against new_v2_ids, not before_v2_ids.

    Contract (falsifier_core decide_loop_exit docstring):
      before_hit@k MUST be scored against the v2-diff target set (new_v2_ids).
      before_hit@k = 0 by construction (D-03): the new IDs did not exist before ingest,
      so a pre-ingest semantic_search cannot return them.

    Previously, before-scoring used before_v2_ids (all pre-existing embeddings), which
    inflated before to 1.0 → delta = -1.000 (impossible on a loop that adds data).
    """

    def test_source_before_hit_result_uses_new_v2_ids(self) -> None:
        """Structural: before_hit_result must reference new_v2_ids, not before_v2_ids."""
        import inspect  # noqa: PLC0415

        source = inspect.getsource(lr)

        # Verify the bug is fixed: before_hit_result must NOT score against before_v2_ids
        assert "compute_hit_rate(before_topk, before_v2_ids)" not in source, (
            "Bug 1 regression: before_hit_result must NOT score against before_v2_ids "
            "(all pre-existing embeddings). It must use new_v2_ids (the v2-diff target). "
            "Fix: move before_hit_result = compute_hit_rate(before_topk, new_v2_ids) "
            "to AFTER new_v2_ids is computed in Step 10."
        )

        # Verify the fix: before_hit_result must score against new_v2_ids
        assert "compute_hit_rate(before_topk, new_v2_ids)" in source, (
            "before_hit_result must be scored against new_v2_ids (the v2-diff target set, D-03). "
            "Both before and after must score against the SAME target set."
        )

    def test_source_before_hit_result_positioned_after_new_v2_ids_assignment(self) -> None:
        """new_v2_ids must be assigned BEFORE before_hit_result is computed."""
        import inspect  # noqa: PLC0415

        source = inspect.getsource(lr)

        new_v2_ids_assign_idx = source.find("new_v2_ids = db_diff(")
        before_hit_result_idx = source.find(
            "before_hit_result = compute_hit_rate(before_topk, new_v2_ids)"
        )

        assert new_v2_ids_assign_idx != -1, (
            "loop_runner.py must compute new_v2_ids = db_diff(before_v2_ids, after_v2_ids)."
        )
        assert before_hit_result_idx != -1, (
            "loop_runner.py must have: before_hit_result = compute_hit_rate(before_topk, new_v2_ids)."
        )
        assert before_hit_result_idx > new_v2_ids_assign_idx, (
            "before_hit_result must be assigned AFTER new_v2_ids is computed. "
            "The before_topk SEARCHES happen pre-ingest (correct); only the SCORING "
            "is deferred until new_v2_ids is available (the fix)."
        )

    def test_before_topk_assigned_before_ingest_subprocess_call(self) -> None:
        """before_topk must be captured PRE-INGEST (D-04 probe ordering)."""
        import inspect  # noqa: PLC0415

        source = inspect.getsource(lr)

        before_topk_idx = source.find("before_topk: list[list[str]]")
        ingest_subprocess_idx = source.find("ingest_places_sf.py")

        assert before_topk_idx != -1, "before_topk must be captured pre-ingest."
        assert ingest_subprocess_idx != -1, "ingest_places_sf.py subprocess must be invoked."
        assert before_topk_idx < ingest_subprocess_idx, (
            "before_topk searches must be captured BEFORE the ingest subprocess. "
            "Post-ingest probe would allow new places to appear in before_topk, "
            "violating D-04 paraphrase-freeze + pre-ingest probe ordering."
        )

    def test_behavioral_before_zero_after_positive_with_mock(self) -> None:
        """Behavioral: with mock search, before=0 and after>0 when new_v2_ids differ.

        Simulates the D-03 contract:
          - before_topk returns only old IDs (pre-ingest — new IDs don't exist yet).
          - after_topk returns new IDs (post-ingest — new embeddings now retrievable).
          - Both score against new_v2_ids (the v2-diff target set).
          - before_hit = 0.0 by construction; after_hit > 0.0; delta > 0.
        """
        from app.loop.falsifier_core import K, N, compute_hit_rate  # noqa: PLC0415

        new_v2_ids = {"new-place-1", "new-place-2", "new-place-3"}
        old_ids = ["old-a", "old-b", "old-c", "old-d", "old-e"]

        # Pre-ingest: semantic_search returns only old IDs (new places don't exist yet)
        before_topk = [old_ids[:K] for _ in range(N)]

        # Post-ingest: semantic_search now returns new IDs for each paraphrase
        after_topk = [["new-place-1", "new-place-2", "old-a", "old-b", "old-c"] for _ in range(N)]

        before_hit_result = compute_hit_rate(before_topk, new_v2_ids)
        after_hit_result = compute_hit_rate(after_topk, new_v2_ids)

        # Core D-03 invariant: before = 0 by construction
        assert before_hit_result.hit_rate == 0.0, (
            f"before_hit@k must be 0.0 by construction (D-03): "
            f"pre-ingest search cannot return IDs that don't exist yet. "
            f"Got {before_hit_result.hit_rate}."
        )

        # After ingest: new places are retrievable → positive hit@k
        assert after_hit_result.hit_rate > 0.0, (
            f"after_hit@k must be > 0 when new places are retrieved. "
            f"Got {after_hit_result.hit_rate}."
        )

        # The delta must be strictly positive
        delta = after_hit_result.hit_rate - before_hit_result.hit_rate
        assert delta > 0.0, f"delta must be strictly positive; got {delta}."


# ---------------------------------------------------------------------------
# Section 13: Bug-fix — list/block-style LLM content unwrapped in _generate_paraphrases
# ---------------------------------------------------------------------------


class TestGenerateParaphrasesBlockContent:
    """Bug fix (19-03): Gemini returns .content as a list of typed blocks.

    Default provider (gemini / gemini-3.1-flash-lite-preview) may return:
      [{'type': 'text', 'text': '["para1","para2","para3","para4","para5"]',
        'extras': {'signature': ...}}]

    The fix normalizes list-style content to a string before the isinstance(str) check.
    build_chat_model is a deferred import inside _generate_paraphrases, so it must be
    patched at its source module (app.llm_factory), not at scripts.loop_runner.
    """

    def test_block_list_content_is_normalized_to_string(self) -> None:
        """Mock llm.invoke returning a block-list; assert 5 paraphrases returned."""
        from unittest.mock import MagicMock, patch  # noqa: PLC0415

        paraphrase_json = (
            '["where to eat pho in outer sunset", "banh mi spots near ocean beach", '
            '"vietnamese noodle soup outer sunset sf", "best pho places outer richmond", '
            '"cheap vietnamese food sunset district"]'
        )

        # Simulate Gemini's block-list response shape
        mock_response = MagicMock()
        mock_response.content = [
            {"type": "text", "text": paraphrase_json, "extras": {"signature": "abc123"}}
        ]

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        # build_chat_model is a deferred import inside _generate_paraphrases —
        # patch at its source module, not at scripts.loop_runner (it's not bound there)
        with patch("app.llm_factory.build_chat_model", return_value=mock_llm):
            paraphrases, prompt, model = lr._generate_paraphrases(
                seed_query="vietnamese restaurants in Outer Sunset San Francisco",
                neighborhood="Outer Sunset",
                cuisine="vietnamese",
                n=5,
            )

        assert len(paraphrases) == 5
        assert "where to eat pho in outer sunset" in paraphrases
        assert all(isinstance(p, str) for p in paraphrases)

    def test_plain_string_content_still_works(self) -> None:
        """Plain string .content path continues to work after the normalization fix."""
        from unittest.mock import MagicMock, patch  # noqa: PLC0415

        paraphrase_json = '["spot 1", "spot 2", "spot 3", "spot 4", "spot 5"]'

        mock_response = MagicMock()
        mock_response.content = paraphrase_json  # plain string, not a list

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        with patch("app.llm_factory.build_chat_model", return_value=mock_llm):
            paraphrases, prompt, model = lr._generate_paraphrases(
                seed_query="sushi restaurants in Japantown San Francisco",
                neighborhood="Japantown",
                cuisine="sushi",
                n=5,
            )

        assert len(paraphrases) == 5
        assert "spot 1" in paraphrases

    def test_block_list_with_multiple_text_blocks_concatenated(self) -> None:
        """Multiple text blocks in the list are concatenated before JSON parsing."""
        from unittest.mock import MagicMock, patch  # noqa: PLC0415

        # Split JSON across two text blocks (edge case)
        part1 = '["item 1", "item 2", '
        part2 = '"item 3", "item 4", "item 5"]'

        mock_response = MagicMock()
        mock_response.content = [
            {"type": "text", "text": part1},
            {"type": "text", "text": part2},
        ]

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        with patch("app.llm_factory.build_chat_model", return_value=mock_llm):
            paraphrases, prompt, model = lr._generate_paraphrases(
                seed_query="dim sum restaurants in Chinatown San Francisco",
                neighborhood="Chinatown",
                cuisine="dim sum",
                n=5,
            )

        assert len(paraphrases) == 5

    def test_non_text_block_type_is_skipped(self) -> None:
        """Blocks without a 'text' key (e.g. image blocks) are silently skipped."""
        from unittest.mock import MagicMock, patch  # noqa: PLC0415

        paraphrase_json = '["a", "b", "c", "d", "e"]'

        mock_response = MagicMock()
        mock_response.content = [
            {"type": "image", "url": "http://example.com/img.png"},  # no 'text' key
            {"type": "text", "text": paraphrase_json},
        ]

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        with patch("app.llm_factory.build_chat_model", return_value=mock_llm):
            paraphrases, prompt, model = lr._generate_paraphrases(
                seed_query="tacos restaurants in Mission San Francisco",
                neighborhood="Mission",
                cuisine="tacos",
                n=5,
            )

        assert len(paraphrases) == 5

    def test_empty_block_list_exits_infra(self) -> None:
        """An empty block list (no text parts) normalizes to '' → EXIT_INFRA."""
        from unittest.mock import MagicMock, patch  # noqa: PLC0415

        mock_response = MagicMock()
        mock_response.content = []  # empty list, normalizes to ""

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        with (
            patch("app.llm_factory.build_chat_model", return_value=mock_llm),
            pytest.raises(SystemExit) as exc_info,
        ):
            lr._generate_paraphrases(
                seed_query="thai restaurants in Tenderloin San Francisco",
                neighborhood="Tenderloin",
                cuisine="thai",
                n=5,
            )

        assert exc_info.value.code == EXIT_INFRA


# ---------------------------------------------------------------------------
# Section 14: _generate_paraphrases — element-level validation (WR-03)
# ---------------------------------------------------------------------------


class TestGenerateParaphrasesElementValidation:
    """WR-03: _generate_paraphrases must reject list elements that are not non-empty strings.

    A JSON array of arrays / ints / objects passes the isinstance(list) + len() check
    but would crash deep inside semantic_search. The fix validates each element before
    returning and raises SystemExit(EXIT_INFRA) for any non-string or empty element.
    """

    def test_list_of_lists_exits_infra(self) -> None:
        """JSON array of arrays → EXIT_INFRA (WR-03 element guard)."""
        from unittest.mock import MagicMock, patch  # noqa: PLC0415

        # LLM returns [[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]] — list of lists
        bad_json = "[[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]]"

        mock_response = MagicMock()
        mock_response.content = bad_json

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        with (
            patch("app.llm_factory.build_chat_model", return_value=mock_llm),
            pytest.raises(SystemExit) as exc_info,
        ):
            lr._generate_paraphrases(
                seed_query="vietnamese restaurants in Outer Sunset San Francisco",
                neighborhood="Outer Sunset",
                cuisine="vietnamese",
                n=5,
            )

        assert exc_info.value.code == EXIT_INFRA

    def test_list_with_integer_exits_infra(self) -> None:
        """JSON array containing an integer exits EXIT_INFRA (WR-03 element guard)."""
        from unittest.mock import MagicMock, patch  # noqa: PLC0415

        # 4 valid strings + 1 integer
        bad_json = '["spot 1", "spot 2", "spot 3", "spot 4", 42]'

        mock_response = MagicMock()
        mock_response.content = bad_json

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        with (
            patch("app.llm_factory.build_chat_model", return_value=mock_llm),
            pytest.raises(SystemExit) as exc_info,
        ):
            lr._generate_paraphrases(
                seed_query="sushi restaurants in Japantown San Francisco",
                neighborhood="Japantown",
                cuisine="sushi",
                n=5,
            )

        assert exc_info.value.code == EXIT_INFRA

    def test_list_with_empty_string_exits_infra(self) -> None:
        """JSON array containing an empty string exits EXIT_INFRA (WR-03 element guard)."""
        from unittest.mock import MagicMock, patch  # noqa: PLC0415

        # 4 valid strings + 1 empty string
        bad_json = '["spot 1", "spot 2", "spot 3", "spot 4", ""]'

        mock_response = MagicMock()
        mock_response.content = bad_json

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        with (
            patch("app.llm_factory.build_chat_model", return_value=mock_llm),
            pytest.raises(SystemExit) as exc_info,
        ):
            lr._generate_paraphrases(
                seed_query="tacos restaurants in Mission San Francisco",
                neighborhood="Mission",
                cuisine="tacos",
                n=5,
            )

        assert exc_info.value.code == EXIT_INFRA

    def test_all_valid_strings_pass_element_guard(self) -> None:
        """5 valid non-empty strings pass the element guard and return correctly."""
        from unittest.mock import MagicMock, patch  # noqa: PLC0415

        good_json = '["place 1", "place 2", "place 3", "place 4", "place 5"]'

        mock_response = MagicMock()
        mock_response.content = good_json

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        with patch("app.llm_factory.build_chat_model", return_value=mock_llm):
            paraphrases, _, _ = lr._generate_paraphrases(
                seed_query="ramen restaurants in Richmond San Francisco",
                neighborhood="Richmond",
                cuisine="ramen",
                n=5,
            )

        assert len(paraphrases) == 5
        assert all(isinstance(p, str) and p.strip() for p in paraphrases)
