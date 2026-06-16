"""Integration smoke tests for the loop falsifier gate.

Gated behind APP_ENV=integration (skip otherwise, mirroring test_embed_v2_e2e.py).
Requires: SANDBOX_DATABASE_URL pointing to a freshly-provisioned, EMPTY sandbox.

These tests verify the BEFORE-snapshot wiring and the prod-safety guard against
a real empty sandbox — they do NOT run the live paid ingest (that is the human
checkpoint in Task 3 of 16-03).

To run:
    APP_ENV=integration SANDBOX_DATABASE_URL=... poetry run pytest \\
        tests/integration/test_loop_falsifier_smoke.py -v
"""

from __future__ import annotations

import os

import psycopg2
import pytest

from app.loop.falsifier_core import K, N
from scripts.loop_falsifier import (
    SEED_QUERY,
    load_paraphrases,
    resolve_prod_url,
    run_guards,
)

pytestmark = pytest.mark.skipif(
    os.getenv("APP_ENV", "test") != "integration",
    reason=(
        "Set APP_ENV=integration, SANDBOX_DATABASE_URL, and ensure the sandbox is empty "
        "to run loop falsifier integration smoke tests."
    ),
)


@pytest.fixture(scope="module")
def sandbox_url() -> str:
    url = os.environ.get("SANDBOX_DATABASE_URL")
    if not url:
        pytest.skip("SANDBOX_DATABASE_URL not set — cannot run integration smoke.")
    return url


@pytest.fixture(scope="module")
def sandbox_conn(sandbox_url):
    """Direct psycopg2 connection to the sandbox (not the pool)."""
    conn = psycopg2.connect(sandbox_url)
    yield conn
    conn.close()


class TestBeforeSnapshotHitRateIsZero:
    """Verify the before-snapshot is 0.0 against an empty sandbox."""

    def test_places_raw_is_empty(self, sandbox_conn):
        """places_raw must be empty for the empty-sandbox precondition to hold."""
        with sandbox_conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM places_raw")
            count = cur.fetchone()[0]
        assert count == 0, (
            f"Sandbox places_raw has {count} rows — it is not empty. "
            "Run `make sandbox-provision` with a DROP+recreate to reset the sandbox "
            "before running the loop falsifier."
        )

    def test_place_embeddings_v2_is_empty(self, sandbox_conn):
        """place_embeddings_v2 must be empty so before_hit_rate == 0 by construction."""
        with sandbox_conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM place_embeddings_v2")
            count = cur.fetchone()[0]
        assert count == 0, (
            f"Sandbox place_embeddings_v2 has {count} rows — it is not empty. "
            "Reset the sandbox before running the loop falsifier."
        )

    def test_before_snapshot_hit_rate_is_zero(self, sandbox_url):
        """With empty sandbox + empty target set, compute_hit_rate must be 0.0."""
        # Temporarily coerce DATABASE_URL so semantic_search points at sandbox
        original_db_url = os.environ.get("DATABASE_URL")
        os.environ["DATABASE_URL"] = sandbox_url
        try:
            from app.loop.falsifier_core import compute_hit_rate  # noqa: PLC0415
            from app.tools.retrieval import semantic_search  # noqa: PLC0415

            paraphrases, _ = load_paraphrases()
            before_topk = []
            for paraphrase in paraphrases:
                hits = semantic_search(paraphrase, k=K)
                before_topk.append([h.place_id for h in hits])

            result = compute_hit_rate(before_topk, set())  # empty target set
            assert result.hit_rate == 0.0, (
                f"Before-snapshot hit_rate = {result.hit_rate:.3f} (expected 0.0 for empty sandbox). "
                "The sandbox is not clean."
            )
            assert result.n == N, f"Expected {N} paraphrases, got {result.n}"
        finally:
            if original_db_url is None:
                os.environ.pop("DATABASE_URL", None)
            else:
                os.environ["DATABASE_URL"] = original_db_url


class TestProdSafetyGuardPasses:
    """Verify the prod-safety guard passes when sandbox != prod."""

    def test_prod_safety_passes_when_sandbox_differs_from_prod(self, sandbox_url):
        """Prod-safety guard must not fire when sandbox URL is truly different from prod."""
        prod_url = resolve_prod_url(sandbox_url=sandbox_url)
        guard_result = run_guards(
            sandbox_url=sandbox_url,
            prod_url=prod_url,
            paraphrases=load_paraphrases()[0],
            seed_query=SEED_QUERY,
        )
        assert guard_result.ok, (
            f"Prod-safety guard FAILED: {guard_result.message}. "
            "The SANDBOX_DATABASE_URL appears to match prod. "
            "Ensure SANDBOX_DATABASE_URL points to city_concierge_sandbox, not prod."
        )

    def test_seed_query_not_in_paraphrases(self):
        """Non-circularity: no paraphrase must be identical to the seed query."""
        paraphrases, frozen_seed = load_paraphrases()
        assert SEED_QUERY not in paraphrases, (
            f"SEED_QUERY {SEED_QUERY!r} appears in the frozen paraphrase list — "
            "this violates non-circularity (D-07). Regenerate the paraphrases."
        )
        assert frozen_seed not in paraphrases, (
            f"frozen seed_query {frozen_seed!r} appears in the frozen paraphrase list — "
            "this violates non-circularity (D-07). Regenerate the paraphrases."
        )

    def test_paraphrase_count_is_n(self):
        """Frozen paraphrase file must have exactly N=5 entries."""
        paraphrases, _ = load_paraphrases()
        assert len(paraphrases) == N, (
            f"Expected N={N} paraphrases, got {len(paraphrases)}. "
            "Regenerate configs/falsifier_paraphrases.json."
        )
