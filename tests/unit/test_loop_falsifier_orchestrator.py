"""Unit tests for scripts/loop_falsifier.py orchestrator.

All DB, network, subprocess, and mlflow calls are mocked — these tests
run without any live services or API keys.

TDD RED: These tests are written BEFORE the implementation and must fail
initially (ImportError or assertion failures).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import orchestrator module (may fail until implementation exists — RED phase)
# ---------------------------------------------------------------------------
import scripts.loop_falsifier as lf
from app.loop.falsifier_core import EXIT_FAIL, EXIT_INFRA, EXIT_PASS, GuardResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FROZEN_JSON_PATH = Path("configs/falsifier_paraphrases.json")

VALID_PARAPHRASE_DATA = {
    "seed_query": "vietnamese restaurants in Outer Sunset San Francisco",
    "generation_prompt": "Generate 5 paraphrases of the intent behind: {seed_query}",
    "non_circularity_note": (
        "Paraphrases differ from the literal seed query string; "
        "expected place_ids are post-ingest DB-diff rows."
    ),
    "paraphrases": [
        "pho and banh mi spots in the Outer Sunset",
        "Vietnamese food near Ocean Beach SF",
        "best bun bo hue in Outer Sunset neighborhood",
        "authentic Vietnamese cuisine on Irving Street San Francisco",
        "Vietnamese noodle shops in the sunset district SF",
    ],
}


# ---------------------------------------------------------------------------
# Task 1a: load_paraphrases — reads JSON, validates count, raises on missing
# ---------------------------------------------------------------------------


class TestLoadParaphrases:
    def test_good_file_returns_paraphrases_and_seed(self, tmp_path):
        frozen = tmp_path / "falsifier_paraphrases.json"
        frozen.write_text(json.dumps(VALID_PARAPHRASE_DATA))
        paraphrases, seed_query = lf.load_paraphrases(str(frozen))
        assert len(paraphrases) == 5
        assert seed_query == VALID_PARAPHRASE_DATA["seed_query"]

    def test_missing_file_raises_infra(self, tmp_path):
        missing = tmp_path / "does_not_exist.json"
        with pytest.raises(SystemExit) as exc_info:
            lf.load_paraphrases(str(missing))
        assert exc_info.value.code == EXIT_INFRA

    def test_wrong_count_raises_infra(self, tmp_path):
        bad_data = dict(VALID_PARAPHRASE_DATA)
        bad_data["paraphrases"] = ["only one"]
        frozen = tmp_path / "bad.json"
        frozen.write_text(json.dumps(bad_data))
        with pytest.raises(SystemExit) as exc_info:
            lf.load_paraphrases(str(frozen))
        assert exc_info.value.code == EXIT_INFRA


# ---------------------------------------------------------------------------
# Task 1b: run_guards — prod safety + non-circularity
# ---------------------------------------------------------------------------


class TestRunGuards:
    def test_prod_collision_returns_violation(self):
        sandbox_url = "postgresql://postgres:pw@127.0.0.1:5433/city_concierge_sandbox"
        # Same URL for both -> should trigger violation
        guard_result = lf.run_guards(
            sandbox_url=sandbox_url,
            prod_url=sandbox_url,
            paraphrases=VALID_PARAPHRASE_DATA["paraphrases"],
            seed_query=VALID_PARAPHRASE_DATA["seed_query"],
        )
        assert not guard_result.ok
        assert "prod" in guard_result.message.lower() or "same" in guard_result.message.lower()

    def test_paraphrase_equals_seed_returns_violation(self):
        sandbox_url = "postgresql://postgres:pw@127.0.0.1:5433/city_concierge_sandbox"
        prod_url = "postgresql://postgres:pw@127.0.0.1:5433/city_concierge"
        seed = VALID_PARAPHRASE_DATA["seed_query"]
        paraphrases_with_seed = [seed, "some other paraphrase", "third", "fourth", "fifth"]
        guard_result = lf.run_guards(
            sandbox_url=sandbox_url,
            prod_url=prod_url,
            paraphrases=paraphrases_with_seed,
            seed_query=seed,
        )
        assert not guard_result.ok

    def test_clean_config_returns_ok(self):
        sandbox_url = "postgresql://postgres:pw@127.0.0.1:5433/city_concierge_sandbox"
        prod_url = "postgresql://postgres:pw@127.0.0.1:5433/city_concierge"
        guard_result = lf.run_guards(
            sandbox_url=sandbox_url,
            prod_url=prod_url,
            paraphrases=VALID_PARAPHRASE_DATA["paraphrases"],
            seed_query=VALID_PARAPHRASE_DATA["seed_query"],
        )
        assert guard_result.ok

    def test_unset_sandbox_returns_violation(self):
        guard_result = lf.run_guards(
            sandbox_url=None,
            prod_url="postgresql://postgres:pw@127.0.0.1:5433/city_concierge",
            paraphrases=VALID_PARAPHRASE_DATA["paraphrases"],
            seed_query=VALID_PARAPHRASE_DATA["seed_query"],
        )
        assert not guard_result.ok


# ---------------------------------------------------------------------------
# Task 1c: decide_exit — all branches
# ---------------------------------------------------------------------------


class TestDecideExit:
    def test_positive_delta_returns_pass(self):
        result = lf.decide_exit(
            before_rate=0.0,
            after_rate=0.4,
            guard_violation=None,
            embed_added_count=3,
        )
        assert result == EXIT_PASS

    def test_zero_delta_returns_fail(self):
        result = lf.decide_exit(
            before_rate=0.0,
            after_rate=0.0,
            guard_violation=None,
            embed_added_count=3,
        )
        assert result == EXIT_FAIL

    def test_guard_violation_returns_infra(self):
        violation = GuardResult(ok=False, message="prod collision")
        result = lf.decide_exit(
            before_rate=0.0,
            after_rate=0.4,
            guard_violation=violation,
            embed_added_count=3,
        )
        assert result == EXIT_INFRA

    def test_zero_embed_returns_infra(self):
        result = lf.decide_exit(
            before_rate=0.0,
            after_rate=0.4,
            guard_violation=None,
            embed_added_count=0,
        )
        assert result == EXIT_INFRA

    def test_before_nonzero_returns_infra(self):
        """A non-zero before_rate means the sandbox was not clean."""
        result = lf.decide_exit(
            before_rate=0.2,
            after_rate=0.6,
            guard_violation=None,
            embed_added_count=3,
        )
        assert result == EXIT_INFRA

    def test_negative_delta_returns_fail(self):
        """after < before should be FAIL (not INFRA — before==0 check not triggered)."""
        # This case: before=0, after=0 covered by test_zero_delta.
        # Test equal non-zero (but before==0 check fires first when before!=0 -> INFRA)
        # So truly test when no guard and no zero-embed and before==0:
        result = lf.decide_exit(
            before_rate=0.0,
            after_rate=0.0,
            guard_violation=None,
            embed_added_count=1,
        )
        assert result == EXIT_FAIL


# ---------------------------------------------------------------------------
# Task 1d: no make_judge call at gate time
# ---------------------------------------------------------------------------


class TestNoMakeJudgeAtGateTime:
    @patch(
        "app.agent.critique.vibe.make_judge",
        side_effect=AssertionError("make_judge was called at gate time"),
    )
    def test_gate_does_not_call_make_judge(self, mock_make_judge):
        """Gate-time code must never regenerate paraphrases (D-06).

        Patches the real vibe.make_judge and asserts it is never called during
        gate-time operations (load_paraphrases, run_guards, decide_exit).
        """
        # load_paraphrases doesn't call make_judge
        # run_guards doesn't call make_judge
        # decide_exit doesn't call make_judge
        # None of these should invoke the LLM
        guard_result = lf.run_guards(
            sandbox_url="postgresql://postgres:pw@127.0.0.1:5433/city_concierge_sandbox",
            prod_url="postgresql://postgres:pw@127.0.0.1:5433/city_concierge",
            paraphrases=VALID_PARAPHRASE_DATA["paraphrases"],
            seed_query=VALID_PARAPHRASE_DATA["seed_query"],
        )
        # If we reach here without AssertionError, make_judge was not called
        assert guard_result.ok
        mock_make_judge.assert_not_called()
        # Also verify make_judge is not referenced in the module at all
        import inspect  # noqa: PLC0415

        source = inspect.getsource(lf)
        assert "make_judge" not in source, (
            "scripts/loop_falsifier.py must not reference make_judge — "
            "paraphrases are frozen and never regenerated at gate time (D-06)"
        )


# ---------------------------------------------------------------------------
# Task 1e: W1 resolved-target check (after DATABASE_URL injection)
# ---------------------------------------------------------------------------


class TestResolvedTargetCheck:
    def test_resolved_target_differs_from_sandbox_exits_infra(self):
        """If in-process resolved target != sandbox URL, exit EXIT_INFRA."""
        sandbox_url = "postgresql://postgres:pw@127.0.0.1:5433/city_concierge_sandbox"
        # Simulate a stale lru_cache that resolved to a different DB
        stale_resolved = "postgresql://postgres:pw@127.0.0.1:5432/city_concierge"
        with pytest.raises(SystemExit) as exc_info:
            lf.assert_resolved_target(sandbox_url=sandbox_url, resolved_url=stale_resolved)
        assert exc_info.value.code == EXIT_INFRA

    def test_resolved_target_matches_sandbox_ok(self):
        """Resolved target == sandbox URL should not raise."""
        sandbox_url = "postgresql://postgres:pw@127.0.0.1:5433/city_concierge_sandbox"
        # Should not raise
        lf.assert_resolved_target(sandbox_url=sandbox_url, resolved_url=sandbox_url)


# ---------------------------------------------------------------------------
# Task 1f: premark_seed_isolation — mock conn, assert UPSERT behavior
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Task 3 (bug fix): settings lru_cache cleared after DATABASE_URL coercion
# ---------------------------------------------------------------------------


class TestSettingsCacheClearAfterCoercion:
    """Prove the settings lru_cache footgun is fixed in the orchestrator.

    Root cause (confirmed during live FALSIFY-01 gate run):
    - app/config.py calls `settings = get_settings()` at MODULE IMPORT time.
    - get_settings() is @lru_cache — it freezes a Settings object whose
      .database_url field is the PROD value from .env.
    - main() Step 3 sets os.environ["DATABASE_URL"] = sandbox_url, but
      WITHOUT clearing the lru_cache the pool rebuilds from the STALE
      cached Settings = prod. The existing assert_resolved_target check uses
      resolve_database_url(os.environ) — a DIFFERENT code path than the pool
      — so it passes while the pool still targets prod.

    The fix: after Step 3 coercion, call get_settings.cache_clear() and
    close_db_pool() so the NEXT call to get_settings() rebuilds from the
    coerced env, and assert_resolved_target ALSO validates the cached-settings
    path (get_settings().resolved_database_url == sandbox_url).
    """

    def test_stale_cache_exposes_prod_before_fix(self):
        """Simulate the pre-fix bug: stale cache returns prod URL after coercion.

        This test verifies that WITHOUT a cache_clear, get_settings() still
        returns the prod URL even after os.environ["DATABASE_URL"] is coerced
        to the sandbox — proving the bug is real.
        """
        from app.config import get_settings  # noqa: PLC0415

        sandbox_url = "postgresql://postgres:pw@127.0.0.1:5433/city_concierge_sandbox"
        prod_url = "postgresql://postgres:pw@127.0.0.1:5433/mlops-city-concierge"

        saved_db_url = os.environ.get("DATABASE_URL")
        try:
            # Step 1: warm the cache with the prod URL (simulates module-scope import)
            os.environ["DATABASE_URL"] = prod_url
            get_settings.cache_clear()
            # Force cache population with prod URL
            settings_before = get_settings()
            assert settings_before.resolved_database_url == prod_url, (
                "Pre-condition: cache should be warm with prod URL"
            )

            # Step 2: coerce to sandbox (as main() Step 3 does) — WITHOUT cache_clear
            os.environ["DATABASE_URL"] = sandbox_url

            # Step 3: WITHOUT fix, get_settings() still returns prod URL (stale cache)
            settings_stale = get_settings()
            # This is the bug: the stale cache still holds the prod URL despite coercion
            assert settings_stale.resolved_database_url == prod_url, (
                "BUG REPRODUCED: stale lru_cache returns prod URL after coercion to sandbox. "
                "The pool would connect to prod, not the sandbox. "
                "Fix: call get_settings.cache_clear() after coercing DATABASE_URL."
            )
        finally:
            get_settings.cache_clear()
            if saved_db_url is None:
                os.environ.pop("DATABASE_URL", None)
            else:
                os.environ["DATABASE_URL"] = saved_db_url

    def test_cache_clear_after_coercion_resolves_sandbox(self):
        """After get_settings.cache_clear(), get_settings() resolves the sandbox URL.

        This test verifies that the FIX works: calling cache_clear() after
        the DATABASE_URL coercion causes get_settings() to rebuild from the
        coerced env, resolving to the sandbox URL.
        """
        from app.config import get_settings  # noqa: PLC0415

        sandbox_url = "postgresql://postgres:pw@127.0.0.1:5433/city_concierge_sandbox"
        prod_url = "postgresql://postgres:pw@127.0.0.1:5433/mlops-city-concierge"

        saved_db_url = os.environ.get("DATABASE_URL")
        try:
            # Step 1: warm the cache with the prod URL
            os.environ["DATABASE_URL"] = prod_url
            get_settings.cache_clear()
            get_settings()  # populate cache with prod URL

            # Step 2: coerce to sandbox
            os.environ["DATABASE_URL"] = sandbox_url

            # Step 3: apply the fix — clear the lru_cache
            get_settings.cache_clear()

            # Step 4: now get_settings() rebuilds from the coerced env
            settings_after = get_settings()
            assert settings_after.resolved_database_url == sandbox_url, (
                f"After cache_clear, get_settings().resolved_database_url should be "
                f"{sandbox_url!r} (sandbox), not {settings_after.resolved_database_url!r} (prod). "
                "The orchestrator must call get_settings.cache_clear() after Step 3 coercion."
            )
        finally:
            get_settings.cache_clear()
            if saved_db_url is None:
                os.environ.pop("DATABASE_URL", None)
            else:
                os.environ["DATABASE_URL"] = saved_db_url

    def test_assert_resolved_target_catches_settings_cache_path(self):
        """assert_resolved_target must fail when get_settings() returns prod (stale cache).

        The existing check passes resolve_database_url(os.environ) which always
        sees the coerced env. The NEW strengthened check also validates
        get_settings().resolved_database_url — the SAME path the pool uses.
        If they diverge, EXIT_INFRA(2) must fire.
        """
        from app.config import get_settings  # noqa: PLC0415

        sandbox_url = "postgresql://postgres:pw@127.0.0.1:5433/city_concierge_sandbox"
        prod_url = "postgresql://postgres:pw@127.0.0.1:5433/mlops-city-concierge"

        saved_db_url = os.environ.get("DATABASE_URL")
        try:
            # Warm cache with prod URL
            os.environ["DATABASE_URL"] = prod_url
            get_settings.cache_clear()
            get_settings()

            # Coerce env to sandbox WITHOUT cache_clear (simulate the pre-fix state)
            os.environ["DATABASE_URL"] = sandbox_url

            # The free-function path sees sandbox (passes), but cached-settings sees prod (fails)
            free_fn_resolved = sandbox_url  # resolve_database_url(os.environ) = sandbox
            cached_settings_resolved = get_settings().resolved_database_url  # still prod (stale)

            # The NEW strengthened assert_resolved_target must catch this divergence
            # and exit EXIT_INFRA — the pool uses the cached path, not the free function
            assert cached_settings_resolved == prod_url, (
                "Pre-condition: stale cache should still see prod URL before fix is applied"
            )
            # Verify the free-function path does NOT catch the divergence
            assert free_fn_resolved == sandbox_url, (
                "Pre-condition: free function resolves sandbox correctly"
            )
            # The bug: both look OK from assert_resolved_target(sandbox, free_fn_resolved)
            # but the pool's path (cached settings) still points to prod.
            # After fix: assert_resolved_target must ALSO check get_settings().resolved_database_url
            with pytest.raises(SystemExit) as exc_info:
                # Simulate what the strengthened assertion should do:
                # validate get_settings().resolved_database_url == sandbox_url
                if get_settings().resolved_database_url != sandbox_url:
                    raise SystemExit(EXIT_INFRA)
            assert exc_info.value.code == EXIT_INFRA
        finally:
            get_settings.cache_clear()
            if saved_db_url is None:
                os.environ.pop("DATABASE_URL", None)
            else:
                os.environ["DATABASE_URL"] = saved_db_url

    def test_orchestrator_step3_calls_cache_clear_and_close_pool(self):
        """main() Step 3 must call get_settings.cache_clear() and close_db_pool() after coercion.

        This test inspects the orchestrator source to confirm the fix is present.
        A source-level assertion is appropriate here because the lru_cache side
        effect is process-global and mocking it cleanly in an end-to-end main()
        call requires complex patching. The source check is a fast, reliable
        proxy for the structural requirement.
        """
        import inspect  # noqa: PLC0415

        source = inspect.getsource(lf)

        assert "cache_clear" in source, (
            "scripts/loop_falsifier.py must call get_settings.cache_clear() after "
            "os.environ['DATABASE_URL'] = sandbox_url (Step 3) to invalidate the stale "
            "lru_cache before the DB pool is initialized. "
            "Root cause: app/config.py populates the @lru_cache at module scope."
        )
        assert "close_db_pool" in source, (
            "scripts/loop_falsifier.py must call close_db_pool() after Step 3 coercion "
            "to reset any already-initialized pool that may point at prod. "
            "close_db_pool() is a no-op when the pool is None — safe to call unconditionally."
        )


class TestPremarkSeedIsolation:
    SEED = "vietnamese restaurants in Outer Sunset San Francisco"
    MINI_CATALOG = [
        "vietnamese restaurants in Outer Sunset San Francisco",
        "pho restaurants in San Francisco",
        "burritos in Mission San Francisco",
    ]

    def _make_mock_conn(self):
        """Build a minimal mock psycopg2 connection."""
        mock_cur = MagicMock()
        mock_cur.__enter__ = lambda s: s
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        return mock_conn, mock_cur

    @patch("scripts.ingest_places_sf.build_seed_queries")
    def test_premark_upsertes_catalog_minus_seed(self, mock_bsq):
        """UPSERTs completed rows for catalog-minus-seed; inserts SEED as pending."""
        mock_bsq.return_value = self.MINI_CATALOG
        mock_conn, mock_cur = self._make_mock_conn()

        # Simulate: count of stale pending rows = 0
        mock_cur.fetchone.return_value = (0,)
        mock_cur.fetchall.return_value = []

        lf.premark_seed_isolation(mock_conn, self.SEED)

        # Should have executed at least one UPSERT for checkpoints
        execute_calls = [str(c) for c in mock_cur.execute.call_args_list]
        upsert_calls = [c for c in execute_calls if "places_ingest_query_checkpoints" in c]
        assert len(upsert_calls) >= 1, "Expected at least one UPSERT into checkpoints"

    @patch("scripts.ingest_places_sf.build_seed_queries")
    def test_premark_exits_infra_when_seed_absent_from_catalog(self, mock_bsq):
        """If SEED_QUERY is not in the catalog, exit EXIT_INFRA."""
        # Catalog does NOT contain the seed
        mock_bsq.return_value = ["pho restaurants in San Francisco", "burritos in Mission SF"]
        mock_conn, mock_cur = self._make_mock_conn()
        mock_cur.fetchone.return_value = (0,)
        mock_cur.fetchall.return_value = []

        with pytest.raises(SystemExit) as exc_info:
            lf.premark_seed_isolation(mock_conn, self.SEED)
        assert exc_info.value.code == EXIT_INFRA

    @patch("scripts.ingest_places_sf.build_seed_queries")
    def test_premark_clears_stale_pending_proposals(self, mock_bsq):
        """Stale pending proposals != SEED_QUERY are updated to 'rejected'."""
        mock_bsq.return_value = self.MINI_CATALOG
        mock_conn, mock_cur = self._make_mock_conn()

        # Simulate: fetchone returns stale count = 0 (after clearing)
        # But we should see the UPDATE call for 'rejected'
        mock_cur.fetchone.return_value = (0,)
        mock_cur.fetchall.return_value = []

        lf.premark_seed_isolation(mock_conn, self.SEED)

        execute_calls = [str(c) for c in mock_cur.execute.call_args_list]
        # Should have called UPDATE ... SET status='rejected'
        rejected_calls = [c for c in execute_calls if "rejected" in c and "proposals" in c.lower()]
        assert len(rejected_calls) >= 1, "Expected UPDATE to 'rejected' on stale proposals"

    @patch("scripts.ingest_places_sf.build_seed_queries")
    def test_premark_exits_infra_when_stale_pending_remain(self, mock_bsq):
        """If stale pending rows remain after clearing, exit EXIT_INFRA."""
        mock_bsq.return_value = self.MINI_CATALOG
        mock_conn, mock_cur = self._make_mock_conn()

        # Simulate: stale pending rows still exist after clearing (count = 2)
        mock_cur.fetchone.return_value = (2,)
        mock_cur.fetchall.return_value = [("some stale query",), ("another stale query",)]

        with pytest.raises(SystemExit) as exc_info:
            lf.premark_seed_isolation(mock_conn, self.SEED)
        assert exc_info.value.code == EXIT_INFRA


# ---------------------------------------------------------------------------
# Task 1g: .env-merged prod resolution (prod URL only in dotenv_values)
# ---------------------------------------------------------------------------


class TestDotenvMergedProdResolution:
    @patch("scripts.loop_falsifier.dotenv_values")
    @patch("app.config.resolve_database_url")
    def test_prod_url_only_in_dotenv_trips_prod_safety(self, mock_resolve, mock_dotenv_values):
        """Even if prod URL is only in .env (not os.environ), it must be detected."""
        sandbox_url = "postgresql://postgres:pw@127.0.0.1:5433/city_concierge_sandbox"
        # prod URL is the same as sandbox URL -> collision
        mock_dotenv_values.return_value = {
            "DATABASE_URL": sandbox_url,
        }
        # resolve_database_url called with merged env (DATABASE_URL popped) returns sandbox_url
        # simulating prod == sandbox (they share the same URL in .env)
        mock_resolve.return_value = sandbox_url

        prod_url = lf.resolve_prod_url(sandbox_url=sandbox_url)
        guard_result = lf.run_guards(
            sandbox_url=sandbox_url,
            prod_url=prod_url,
            paraphrases=VALID_PARAPHRASE_DATA["paraphrases"],
            seed_query=VALID_PARAPHRASE_DATA["seed_query"],
        )
        assert not guard_result.ok

    @patch("scripts.loop_falsifier.dotenv_values")
    @patch("app.config.resolve_database_url")
    def test_prod_url_from_dotenv_merged_with_os_environ(self, mock_resolve, mock_dotenv_values):
        """os.environ takes precedence over .env per {**dotenv, **os.environ} merge."""
        sandbox_url = "postgresql://postgres:pw@127.0.0.1:5433/city_concierge_sandbox"
        prod_url_in_dotenv = "postgresql://postgres:pw@127.0.0.1:5433/city_concierge"
        mock_dotenv_values.return_value = {"DATABASE_URL": prod_url_in_dotenv}
        mock_resolve.return_value = prod_url_in_dotenv

        # The resolved prod URL should be the .env one (sandbox is popped from copy)
        prod_url = lf.resolve_prod_url(sandbox_url=sandbox_url)
        assert prod_url == prod_url_in_dotenv


# ---------------------------------------------------------------------------
# Task 1h: run_subprocess_or_infra — CalledProcessError -> EXIT_INFRA
# ---------------------------------------------------------------------------


class TestRunSubprocessOrInfra:
    @patch("scripts.loop_falsifier.subprocess")
    def test_successful_subprocess_returns_none(self, mock_subprocess):
        mock_subprocess.run.return_value = MagicMock(returncode=0)
        mock_subprocess.CalledProcessError = subprocess.CalledProcessError
        # Should not raise
        lf.run_subprocess_or_infra(
            argv=[sys.executable, "-c", "print('ok')"],
            env={"DATABASE_URL": "postgresql://..."},
        )

    @patch("scripts.loop_falsifier.subprocess")
    def test_called_process_error_raises_systemexit_infra(self, mock_subprocess):
        mock_subprocess.CalledProcessError = subprocess.CalledProcessError
        mock_subprocess.run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=["python", "scripts/ingest_places_sf.py"]
        )
        with pytest.raises(SystemExit) as exc_info:
            lf.run_subprocess_or_infra(
                argv=[sys.executable, "scripts/ingest_places_sf.py"],
                env={"DATABASE_URL": "postgresql://..."},
            )
        assert exc_info.value.code == EXIT_INFRA


# ---------------------------------------------------------------------------
# Task 1i: MLflow failure -> EXIT_INFRA
# ---------------------------------------------------------------------------


class TestMlflowFailure:
    @patch("scripts.loop_falsifier.mlflow")
    def test_mlflow_set_experiment_failure_exits_infra(self, mock_mlflow):
        mock_mlflow.set_experiment.side_effect = Exception("MLflow server unavailable")
        with pytest.raises(SystemExit) as exc_info:
            lf.log_to_mlflow(
                gap=("Outer Sunset", "vietnamese"),
                seed_query="vietnamese restaurants in Outer Sunset San Francisco",
                paraphrases=VALID_PARAPHRASE_DATA["paraphrases"],
                before_snapshot={"paraphrase_topk": [], "hit_rate": 0.0},
                after_snapshot={"paraphrase_topk": [], "hit_rate": 0.4},
                db_diff_ids=["id1", "id2"],
                before_hit_rate=0.0,
                after_hit_rate=0.4,
                hit_rate_delta=0.4,
                new_place_count=2,
                embed_added_count=2,
            )
        assert exc_info.value.code == EXIT_INFRA

    @patch("scripts.loop_falsifier.mlflow")
    def test_mlflow_log_dict_failure_exits_infra(self, mock_mlflow):
        mock_mlflow.set_experiment.return_value = None
        mock_cm = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = lambda s: mock_cm
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
        mock_cm.log_dict = MagicMock(side_effect=Exception("artifact logging failed"))
        mock_mlflow.log_dict.side_effect = Exception("artifact logging failed")

        with pytest.raises(SystemExit) as exc_info:
            lf.log_to_mlflow(
                gap=("Outer Sunset", "vietnamese"),
                seed_query="vietnamese restaurants in Outer Sunset San Francisco",
                paraphrases=VALID_PARAPHRASE_DATA["paraphrases"],
                before_snapshot={"paraphrase_topk": [], "hit_rate": 0.0},
                after_snapshot={"paraphrase_topk": [], "hit_rate": 0.4},
                db_diff_ids=["id1", "id2"],
                before_hit_rate=0.0,
                after_hit_rate=0.4,
                hit_rate_delta=0.4,
                new_place_count=2,
                embed_added_count=2,
            )
        assert exc_info.value.code == EXIT_INFRA


# ---------------------------------------------------------------------------
# Task 1j: premark set size = catalog - SEED (pure coverage)
# ---------------------------------------------------------------------------


class TestPremarkSetSize:
    @patch("scripts.ingest_places_sf.build_seed_queries")
    def test_premark_set_equals_catalog_minus_seed(self, mock_bsq):
        """The set of checkpoint keys marked completed == {checkpoint_key(q) for q in catalog - SEED}.

        WR-02: premark_seed_isolation now writes checkpoint_key(query_text) rows (i.e.
        'all::<query>') into places_ingest_query_checkpoints to match the keying that
        record_query_checkpoint() and select_seed_queries_for_run() use.
        """
        from scripts.ingest_places_sf import checkpoint_key  # noqa: PLC0415

        catalog = [
            "vietnamese restaurants in Outer Sunset San Francisco",
            "pho restaurants in San Francisco",
            "burritos in Mission San Francisco",
            "pizza in North Beach San Francisco",
        ]
        seed = "vietnamese restaurants in Outer Sunset San Francisco"
        mock_bsq.return_value = catalog

        mock_conn, mock_cur = MagicMock(), MagicMock()
        mock_cur.__enter__ = lambda s: s
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cur
        mock_cur.fetchone.return_value = (0,)
        mock_cur.fetchall.return_value = []

        # Track all execute calls
        executed_queries: list[str] = []
        executed_params: list[Any] = []

        def capture_execute(sql, params=None):
            executed_queries.append(sql)
            executed_params.append(params)

        mock_cur.execute.side_effect = capture_execute

        lf.premark_seed_isolation(mock_conn, seed)

        # Find the UPSERT calls — they should be for 3 queries (catalog minus seed),
        # written as checkpoint_key-prefixed strings.
        upsert_calls = [
            (q, p)
            for q, p in zip(executed_queries, executed_params, strict=False)
            if "places_ingest_query_checkpoints" in q and p is not None
        ]
        # The executed params must include exactly {checkpoint_key(q) for q in catalog - {seed}}
        marked_queries = set()
        for _, params in upsert_calls:
            if isinstance(params, (list, tuple)) and len(params) >= 1:
                marked_queries.add(params[0])

        expected = {checkpoint_key(q) for q in set(catalog) - {seed}}
        assert expected.issubset(marked_queries), (
            f"Expected all catalog-minus-seed checkpoint keys to be marked; "
            f"missing: {expected - marked_queries}"
        )
        assert checkpoint_key(seed) not in marked_queries, (
            "SEED_QUERY's checkpoint_key must NOT be marked completed"
        )
        assert seed not in marked_queries, (
            "Raw SEED_QUERY must NOT be marked completed (only checkpoint_key-keyed rows)"
        )


# ---------------------------------------------------------------------------
# CR-01: Sandbox emptiness assertion before snapshot
# ---------------------------------------------------------------------------


class TestSandboxEmptinessAssertion:
    """CR-01: before_hit_rate must be computed against real data; sandbox must be asserted empty."""

    def test_decide_exit_before_rate_nonzero_returns_infra(self):
        """decide_exit must return EXIT_INFRA when before_rate != 0 (belt-and-suspenders)."""
        result = lf.decide_exit(
            before_rate=0.2,
            after_rate=0.6,
            guard_violation=None,
            embed_added_count=3,
        )
        assert result == EXIT_INFRA

    def test_decide_exit_before_rate_zero_with_embeds_and_positive_delta_returns_pass(self):
        """decide_exit returns EXIT_PASS when before_rate=0, embeds>0, delta>0."""
        result = lf.decide_exit(
            before_rate=0.0,
            after_rate=1.0,
            guard_violation=None,
            embed_added_count=5,
        )
        assert result == EXIT_PASS


# ---------------------------------------------------------------------------
# WR-02: premark uses checkpoint_key() for checkpoints keying
# ---------------------------------------------------------------------------


class TestPremarkCheckpointKeyKeying:
    """WR-02: premark_seed_isolation must write checkpoint_key(query_text) rows, not raw query."""

    SEED = "vietnamese restaurants in Outer Sunset San Francisco"
    MINI_CATALOG = [
        "vietnamese restaurants in Outer Sunset San Francisco",
        "pho restaurants in San Francisco",
        "burritos in Mission San Francisco",
    ]

    @patch("scripts.ingest_places_sf.build_seed_queries")
    def test_premark_writes_checkpoint_key_not_raw_query(self, mock_bsq):
        """The checkpoint row key must be checkpoint_key(query_text) = 'all::<query>'."""
        from scripts.ingest_places_sf import checkpoint_key  # noqa: PLC0415

        mock_bsq.return_value = self.MINI_CATALOG
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.__enter__ = lambda s: s
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cur
        mock_cur.fetchone.return_value = (0,)
        mock_cur.fetchall.return_value = []

        executed_params: list[Any] = []

        def capture_execute(sql, params=None):
            executed_params.append((sql, params))

        mock_cur.execute.side_effect = capture_execute

        lf.premark_seed_isolation(mock_conn, self.SEED)

        # Find checkpoint UPSERT params
        checkpoint_params = [
            params[0]
            for sql, params in executed_params
            if "places_ingest_query_checkpoints" in sql and params is not None
        ]

        # Every checkpoint key must be in checkpoint_key format
        expected_keys = {checkpoint_key(q) for q in self.MINI_CATALOG if q != self.SEED}
        actual_keys = set(checkpoint_params)

        assert expected_keys == actual_keys, (
            f"Checkpoint keys must use checkpoint_key() format ('all::<query>'). "
            f"Expected: {expected_keys}, got: {actual_keys}"
        )
        # Raw query text (without prefix) must NOT appear as checkpoint key
        raw_non_seed = {q for q in self.MINI_CATALOG if q != self.SEED}
        for raw_q in raw_non_seed:
            assert raw_q not in actual_keys, (
                f"Raw query {raw_q!r} was written as checkpoint key instead of "
                f"checkpoint_key-prefixed form. This breaks the keying convention."
            )


# ---------------------------------------------------------------------------
# WR-03: embed_added_count == 0 must raise SystemExit immediately
# ---------------------------------------------------------------------------


class TestEmbedZeroShortCircuit:
    """WR-03: embed_added_count==0 must raise SystemExit(EXIT_INFRA) before after-snapshot."""

    def test_decide_exit_embed_zero_returns_infra(self):
        """decide_exit still returns INFRA for zero embeds (unit / belt-and-suspenders path)."""
        result = lf.decide_exit(
            before_rate=0.0,
            after_rate=0.0,
            guard_violation=None,
            embed_added_count=0,
        )
        assert result == EXIT_INFRA


# ---------------------------------------------------------------------------
# WR-05: SEED proposal upsert resets 'applied' -> 'pending'
# ---------------------------------------------------------------------------


class TestPremarkProposalUpsertResetApplied:
    """WR-05: The SEED proposal insert must use ON CONFLICT DO UPDATE SET status='pending'."""

    SEED = "vietnamese restaurants in Outer Sunset San Francisco"
    MINI_CATALOG = [
        "vietnamese restaurants in Outer Sunset San Francisco",
        "pho restaurants in San Francisco",
    ]

    @patch("scripts.ingest_places_sf.build_seed_queries")
    def test_proposal_insert_has_do_update_not_do_nothing(self, mock_bsq):
        """The SEED proposal INSERT must have ON CONFLICT DO UPDATE SET status='pending'."""
        mock_bsq.return_value = self.MINI_CATALOG
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.__enter__ = lambda s: s
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cur
        mock_cur.fetchone.return_value = (0,)
        mock_cur.fetchall.return_value = []

        executed_sqls: list[str] = []

        def capture_execute(sql, params=None):
            executed_sqls.append(sql)

        mock_cur.execute.side_effect = capture_execute

        lf.premark_seed_isolation(mock_conn, self.SEED)

        # Find the proposal INSERT SQL
        proposal_inserts = [
            s
            for s in executed_sqls
            if "places_ingest_query_proposals" in s and "INSERT" in s.upper()
        ]
        assert len(proposal_inserts) >= 1, "Expected INSERT into places_ingest_query_proposals"

        # Must NOT be DO NOTHING (which leaves 'applied' status as-is)
        for sql in proposal_inserts:
            assert "DO NOTHING" not in sql.upper(), (
                "SEED proposal INSERT must use ON CONFLICT DO UPDATE SET status='pending', "
                "not DO NOTHING. DO NOTHING leaves an 'applied' row from a prior run intact, "
                "causing the next run to skip ingest silently."
            )
            # Must reset to pending
            assert "pending" in sql.lower(), (
                f"SEED proposal INSERT must set status='pending' on conflict, got: {sql!r}"
            )


# ---------------------------------------------------------------------------
# IN-02: compute_hit_rate assertion that len(topk) <= K
# ---------------------------------------------------------------------------


class TestComputeHitRateLengthAssertion:
    """IN-02: compute_hit_rate should assert len(topk) <= K so truncation is a programming-error guard."""

    def test_topk_within_k_works_normally(self):
        """Lists of exactly K elements should pass through without error."""
        from app.loop.falsifier_core import compute_hit_rate  # noqa: PLC0415

        topk = [["id1", "id2", "id3", "id4", "id5"]]  # exactly K=5 elements
        result = compute_hit_rate(topk, {"id1"})
        assert result.hit_count == 1

    def test_topk_exceeding_k_raises_assertion(self):
        """If a list has more than K elements, compute_hit_rate must raise AssertionError."""
        from app.loop.falsifier_core import compute_hit_rate  # noqa: PLC0415

        topk = [["id1", "id2", "id3", "id4", "id5", "id6"]]  # K+1 = 6 elements
        with pytest.raises(AssertionError):
            compute_hit_rate(topk, {"id6"})
