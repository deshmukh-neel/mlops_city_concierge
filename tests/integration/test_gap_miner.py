"""Integration tests for the Phase 18 demand-driven gap miner.

Gated on APP_ENV=integration.  The shared CI/sandbox DB may not have the
Phase 17 migration (user_query_log) applied yet; those tests skip gracefully
and start passing once the schema lands — no code change required.

Strategy
--------
1. Pick a catalog (neighborhood, cuisine) pair absent from BOTH
   ``places_ingest_query_proposals`` AND the normalized completed-checkpoint
   set returned by ``ingested_query_texts(conn)`` (ROUND-2 MEDIUM-1 +
   ROUND-3 MEDIUM).  Skip with a clear reason when no such pair exists.
2. Seed ONE demand row for the chosen pair via ``insert_demand_rows``.
   The message names both the chosen neighborhood AND the chosen cuisine word
   so the demand maps lexically on BOTH axes with NO LLM dependency
   (exercising the ROUND-3 cuisine-recall path against a real DB).
3. Run ``gap_mine_main(["--days","30","--min-places","100000"])`` — the
   deliberately huge ``--min-places`` forces the seeded bucket under threshold
   so the gap is GUARANTEED regardless of real sandbox supply (REVIEW MEDIUM
   determinism).
4. Assert a real ``pending`` proposal row lands in
   ``places_ingest_query_proposals`` whose ``query_text`` equals
   ``gap_to_seed_query(chosen_n, chosen_c)``.
5. Clean up ONLY the rows this test created — the seeded demand rows (by
   unique marker session_id) and the proposal row (ONLY if the test created
   it, tracked via a pre-mining check) — in a ``finally`` using parameterised
   DELETEs.  Never blanket-delete by query_text when a pre-existing proposal
   was present (REVIEW MEDIUM scoped cleanup).
"""

from __future__ import annotations

import os
import uuid
from unittest.mock import MagicMock

import pytest

from app.db import get_conn
from scripts.coverage_agent import gap_mine_main, gap_to_seed_query, ingested_query_texts
from scripts.ingest_places_sf import CUISINES, NEIGHBORHOODS, build_seed_queries
from scripts.seed_demand_log import insert_demand_rows

pytestmark = pytest.mark.skipif(
    os.getenv("APP_ENV", "test") != "integration",
    reason="Set APP_ENV=integration and provide a real DATABASE_URL to run integration tests.",
)

# Pre-compute the full catalog seed set once at module import time
# (build_seed_queries() is pure and fast; no DB connection needed).
_CATALOG_SEEDS: frozenset[str] = frozenset(build_seed_queries())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def _proposals_table_or_skip() -> None:
    """Skip when places_ingest_query_proposals is absent or unwritable."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_name = 'places_ingest_query_proposals'"
        )
        if cur.fetchone() is None:
            pytest.skip("places_ingest_query_proposals not deployed yet")
        cur.execute(
            "SELECT has_table_privilege(current_user, "
            "'places_ingest_query_proposals', 'INSERT, DELETE')"
        )
        if not cur.fetchone()[0]:
            pytest.skip("current DB role lacks INSERT/DELETE on proposals table")


@pytest.fixture
def _user_query_log_or_skip() -> None:
    """Skip when user_query_log is absent or unwritable (Phase 17 migration not applied)."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM information_schema.tables WHERE table_name = 'user_query_log'")
        if cur.fetchone() is None:
            pytest.skip("user_query_log not deployed yet (Phase 17 migration not applied)")
        cur.execute("SELECT has_table_privilege(current_user, 'user_query_log', 'INSERT, DELETE')")
        if not cur.fetchone()[0]:
            pytest.skip("current DB role lacks INSERT/DELETE on user_query_log")


def _stub_mlflow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch all mlflow.* calls so integration test doesn't need MLflow."""
    from scripts import coverage_agent

    start_run = MagicMock()
    start_run.return_value.__enter__ = MagicMock(return_value=None)
    start_run.return_value.__exit__ = MagicMock(return_value=None)
    monkeypatch.setattr(coverage_agent.mlflow, "set_experiment", MagicMock())
    monkeypatch.setattr(coverage_agent.mlflow, "start_run", start_run)
    monkeypatch.setattr(coverage_agent.mlflow, "log_param", MagicMock())
    monkeypatch.setattr(coverage_agent.mlflow, "log_metric", MagicMock())
    monkeypatch.setattr(coverage_agent.mlflow, "log_dict", MagicMock())


def _purge_demand_rows(marker_session_id: str) -> None:
    """Delete test-seeded demand rows by their unique session_id marker."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM user_query_log WHERE session_id = %s",
            [marker_session_id],
        )
        conn.commit()


def _purge_proposal_row(query_text: str) -> None:
    """Delete a single proposal row by exact query_text (only call when test created it)."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM places_ingest_query_proposals WHERE query_text = %s",
            [query_text],
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_table_readiness(_proposals_table_or_skip, _user_query_log_or_skip) -> None:
    """Schema-deploy gate: passes once both tables are deployed, skips before."""


def test_seeded_demand_produces_pending_proposal(
    _proposals_table_or_skip,
    _user_query_log_or_skip,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: seeded sandbox demand → real pending proposal row (GAP-03).

    Deterministic (forced --min-places), checkpoint-aware target selection
    (ROUND-2 MEDIUM-1 + ROUND-3 MEDIUM), cuisine-recall via message (ROUND-3
    HIGH), scoped cleanup (REVIEW MEDIUM).
    """
    # --- Step 1: Build the normalized dedup set (proposals + completed checkpoints) ---
    # This mirrors what gap_mine_main uses internally, so the chosen target pair
    # is guaranteed not to be silently deduped before insertion.
    with get_conn() as conn:
        already_ingested = ingested_query_texts(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT query_text FROM places_ingest_query_proposals")
            existing_proposals: set[str] = {row[0] for row in cur.fetchall()}

    # --- Step 2: Pick the first catalog pair free of BOTH dedup sources ---
    chosen_n: str | None = None
    chosen_c: str | None = None
    for n in NEIGHBORHOODS:
        for c in CUISINES:
            seed = gap_to_seed_query(n, c)
            if seed not in _CATALOG_SEEDS:
                # gap_to_seed_query already asserts catalog membership; this is
                # a defensive check that the seed is a valid catalog seed.
                continue
            if seed in already_ingested:
                continue
            if seed in existing_proposals:
                continue
            chosen_n, chosen_c = n, c
            break
        if chosen_n is not None:
            break

    if chosen_n is None:
        pytest.skip(
            "no catalog pair free of proposals and normalized completed-checkpoints"
            " — the gap miner would correctly dedup every candidate; cannot run test"
        )

    seed_text = gap_to_seed_query(chosen_n, chosen_c)

    # --- Step 3: Confirm no pre-existing proposal for our chosen seed ---
    # This guards the finally block: if a proposal pre-existed we must NOT
    # delete it during cleanup (REVIEW MEDIUM scoped cleanup).
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM places_ingest_query_proposals WHERE query_text = %s",
            [seed_text],
        )
        proposal_pre_existed = cur.fetchone() is not None

    # If a proposal appeared between step 1 and step 3, skip rather than
    # risk a false assertion or an over-aggressive cleanup.
    if proposal_pre_existed:
        pytest.skip(
            f"proposal for {seed_text!r} appeared between dedup-set build and pre-check"
            " — race condition; re-run to pick a different pair"
        )

    # Unique per-run marker used to scope demand-row cleanup
    marker_session_id = f"gap-int-{uuid.uuid4().hex}"

    try:
        # --- Step 4: Seed ONE demand row for the chosen pair ---
        # Message names BOTH the chosen neighborhood AND the chosen cuisine word
        # so the demand maps lexically on BOTH axes without any LLM call
        # (exercises the ROUND-3 cuisine-recall path against a real DB).
        demand_row = {
            "message": f"{chosen_c} restaurants in {chosen_n}",
            "requested_primary_types": [],  # free-text case: empty types (ROUND-3 HIGH)
            "num_stops": 2,
            "rag_label": "itinerary",
            "session_id": marker_session_id,
        }
        with get_conn() as conn:
            insert_demand_rows([demand_row], conn=conn)

        # --- Step 5: Stub mlflow and vibe.make_judge ---
        from scripts import coverage_agent

        _stub_mlflow(monkeypatch)
        # Stub the LLM to a no-op — the lexical pre-passes on BOTH axes cover
        # the seeded row; LLM is NOT required for the integration path.
        monkeypatch.setattr(coverage_agent.vibe, "make_judge", lambda: None)

        # --- Step 6: Run the miner with forced gap ---
        # --min-places 100000 forces the seeded bucket under threshold regardless
        # of actual sandbox supply (REVIEW MEDIUM determinism guarantee).
        rc = gap_mine_main(["--days", "30", "--min-places", "100000"])
        assert rc == 0, f"gap_mine_main returned {rc}, expected 0"

        # --- Step 7: Assert the pending proposal row landed ---
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT status FROM places_ingest_query_proposals WHERE query_text = %s",
                [seed_text],
            )
            row = cur.fetchone()

        assert row is not None, f"expected a pending proposal row for {seed_text!r} but none found"
        assert row[0] == "pending", f"proposal status should be 'pending', got {row[0]!r}"

    finally:
        # --- Step 8: Scoped cleanup ---
        # Always delete the seeded demand rows (by unique marker).
        _purge_demand_rows(marker_session_id)
        # Delete the proposal ONLY if the test created it (step 3 confirmed it
        # did not pre-exist).  Never blanket-delete (REVIEW MEDIUM).
        if not proposal_pre_existed:
            _purge_proposal_row(seed_text)
