"""Unit tests for app.loop.falsifier_core.

These tests exercise every guard and metric function with zero DB, network, or LLM
calls. They serve as the machine-checkable proof that the loop-falsifier gate's decision
logic is correct before any live ingest ever runs.

TDD: these tests were written BEFORE the implementation.
"""

from __future__ import annotations

import pytest

from app.loop.falsifier_core import (
    EXIT_FAIL,
    EXIT_INFRA,
    EXIT_PASS,
    GuardResult,
    HitRateResult,
    K,
    N,
    build_premark_set,
    check_non_circularity,
    check_prod_safety,
    compute_hit_rate,
    db_diff,
    is_pass,
    is_strictly_positive_delta,
)

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    def test_k_equals_5(self) -> None:
        assert K == 5

    def test_n_equals_5(self) -> None:
        assert N == 5

    def test_exit_pass_is_0(self) -> None:
        assert EXIT_PASS == 0

    def test_exit_fail_is_1(self) -> None:
        assert EXIT_FAIL == 1

    def test_exit_infra_is_2(self) -> None:
        assert EXIT_INFRA == 2


# ---------------------------------------------------------------------------
# compute_hit_rate
# ---------------------------------------------------------------------------


class TestComputeHitRate:
    def test_one_of_five_paraphrases_hits(self) -> None:
        # place_id "new-1" is newly ingested; only first paraphrase retrieves it
        per_paraphrase_topk = [
            ["new-1", "old-a", "old-b", "old-c", "old-d"],
            ["old-a", "old-b", "old-c", "old-d", "old-e"],
            ["old-b", "old-c", "old-d", "old-e", "old-f"],
            ["old-c", "old-d", "old-e", "old-f", "old-g"],
            ["old-d", "old-e", "old-f", "old-g", "old-h"],
        ]
        newly_ingested_ids = {"new-1"}
        result = compute_hit_rate(per_paraphrase_topk, newly_ingested_ids)
        assert isinstance(result, HitRateResult)
        assert result.hit_count == 1
        assert result.n == 5
        assert pytest.approx(result.hit_rate) == 0.2

    def test_zero_matches(self) -> None:
        per_paraphrase_topk = [
            ["old-a", "old-b", "old-c", "old-d", "old-e"],
            ["old-a", "old-b", "old-c", "old-d", "old-e"],
            ["old-a", "old-b", "old-c", "old-d", "old-e"],
            ["old-a", "old-b", "old-c", "old-d", "old-e"],
            ["old-a", "old-b", "old-c", "old-d", "old-e"],
        ]
        result = compute_hit_rate(per_paraphrase_topk, {"new-1"})
        assert result.hit_count == 0
        assert result.n == 5
        assert pytest.approx(result.hit_rate) == 0.0

    def test_all_five_hit(self) -> None:
        per_paraphrase_topk = [
            ["new-1", "old-a", "old-b", "old-c", "old-d"],
            ["new-2", "old-a", "old-b", "old-c", "old-d"],
            ["new-3", "old-a", "old-b", "old-c", "old-d"],
            ["new-4", "old-a", "old-b", "old-c", "old-d"],
            ["new-5", "old-a", "old-b", "old-c", "old-d"],
        ]
        newly_ingested_ids = {"new-1", "new-2", "new-3", "new-4", "new-5"}
        result = compute_hit_rate(per_paraphrase_topk, newly_ingested_ids)
        assert result.hit_count == 5
        assert result.n == 5
        assert pytest.approx(result.hit_rate) == 1.0

    def test_empty_input_returns_zero_no_division_error(self) -> None:
        # Empty per_paraphrase_topk must return 0/0/0.0 without ZeroDivisionError
        result = compute_hit_rate([], {"new-1"})
        assert result.hit_count == 0
        assert result.n == 0
        assert pytest.approx(result.hit_rate) == 0.0

    def test_match_at_index_beyond_k_does_not_hit(self) -> None:
        # A list of exactly K=5 elements, none of which match -> no hit.
        # IN-02: compute_hit_rate now asserts len(topk) <= K instead of silently
        # truncating; the caller (semantic_search with k=K) is the single source of
        # truth for the retrieval window. We verify that a list of exactly K elements
        # with no match returns 0 — the "beyond K" is enforced at the call site.
        per_paraphrase_topk = [
            ["old-a", "old-b", "old-c", "old-d", "old-e"],  # K=5 elements, no match
        ]
        result = compute_hit_rate(per_paraphrase_topk, {"new-1"})
        assert result.hit_count == 0
        assert result.n == 1
        assert pytest.approx(result.hit_rate) == 0.0

    def test_match_exactly_at_k_boundary(self) -> None:
        # match at index K-1 = 4 (index 4 in a K=5 list) — SHOULD hit
        per_paraphrase_topk = [
            ["old-a", "old-b", "old-c", "old-d", "new-1"],  # index 4 = position 5
        ]
        result = compute_hit_rate(per_paraphrase_topk, {"new-1"})
        assert result.hit_count == 1
        assert result.n == 1
        assert pytest.approx(result.hit_rate) == 1.0


# ---------------------------------------------------------------------------
# is_pass
# ---------------------------------------------------------------------------


class TestIsPass:
    def test_zero_is_false(self) -> None:
        assert is_pass(0.0) is False

    def test_positive_is_true(self) -> None:
        assert is_pass(0.2) is True

    def test_one_is_true(self) -> None:
        assert is_pass(1.0) is True

    def test_exact_zero_float_is_false(self) -> None:
        assert is_pass(float(0)) is False

    def test_tiny_positive_is_true(self) -> None:
        assert is_pass(1e-10) is True


# ---------------------------------------------------------------------------
# is_strictly_positive_delta (FALSIFY-01(f) literal rule)
# ---------------------------------------------------------------------------


class TestIsStrictlyPositiveDelta:
    def test_zero_to_02_is_true(self) -> None:
        assert is_strictly_positive_delta(0.0, 0.2) is True

    def test_zero_to_zero_is_false(self) -> None:
        assert is_strictly_positive_delta(0.0, 0.0) is False

    def test_same_nonzero_is_false(self) -> None:
        assert is_strictly_positive_delta(0.4, 0.4) is False

    def test_increase_is_true(self) -> None:
        assert is_strictly_positive_delta(0.2, 0.6) is True

    def test_decrease_is_false(self) -> None:
        assert is_strictly_positive_delta(0.6, 0.2) is False


# ---------------------------------------------------------------------------
# db_diff
# ---------------------------------------------------------------------------


class TestDbDiff:
    def test_partial_overlap(self) -> None:
        before = {"a", "b"}
        after = {"a", "b", "c", "d"}
        assert db_diff(before, after) == {"c", "d"}

    def test_empty_before_returns_all_after(self) -> None:
        assert db_diff(set(), {"a", "b", "c"}) == {"a", "b", "c"}

    def test_identical_returns_empty(self) -> None:
        s = {"a", "b"}
        assert db_diff(s, s) == set()

    def test_empty_both(self) -> None:
        assert db_diff(set(), set()) == set()


# ---------------------------------------------------------------------------
# build_premark_set (seed-isolation fix: catalog minus chosen seed)
# ---------------------------------------------------------------------------


class TestBuildPremarkSet:
    def test_catalog_minus_seed(self) -> None:
        catalog = ["a", "b", "c", "seed"]
        result = build_premark_set(catalog, "seed")
        assert result == {"a", "b", "c"}
        assert "seed" not in result

    def test_chosen_absent_returns_full_catalog(self) -> None:
        catalog = ["a", "b", "c"]
        result = build_premark_set(catalog, "not-in-catalog")
        assert result == {"a", "b", "c"}

    def test_duplicate_entries_collapse_to_set(self) -> None:
        catalog = ["a", "a", "b", "b", "seed", "seed"]
        result = build_premark_set(catalog, "seed")
        assert result == {"a", "b"}

    def test_single_item_catalog_equals_seed(self) -> None:
        # catalog only contains the seed -> premark set is empty
        result = build_premark_set(["seed"], "seed")
        assert result == set()


# ---------------------------------------------------------------------------
# check_prod_safety
# ---------------------------------------------------------------------------


class TestCheckProdSafety:
    def test_unset_sandbox_url_is_violation(self) -> None:
        result = check_prod_safety(None, "postgresql://user:pw@prod-host:5432/proddb")
        assert isinstance(result, GuardResult)
        assert result.ok is False
        assert "SANDBOX_DATABASE_URL" in result.message

    def test_empty_sandbox_url_is_violation(self) -> None:
        result = check_prod_safety("", "postgresql://user:pw@prod-host:5432/proddb")
        assert result.ok is False
        assert "SANDBOX_DATABASE_URL" in result.message

    def test_same_host_and_db_tcp_is_violation(self) -> None:
        url = "postgresql://user:pw@shared-host:5432/mydb"
        result = check_prod_safety(url, url)
        assert result.ok is False

    def test_distinct_host_is_ok(self) -> None:
        sandbox = "postgresql://user:pw@sandbox-host:5432/mydb"
        prod = "postgresql://user:pw@prod-host:5432/mydb"
        result = check_prod_safety(sandbox, prod)
        assert result.ok is True

    def test_distinct_dbname_is_ok(self) -> None:
        sandbox = "postgresql://user:pw@same-host:5432/sandbox_db"
        prod = "postgresql://user:pw@same-host:5432/prod_db"
        result = check_prod_safety(sandbox, prod)
        assert result.ok is True

    def test_prod_url_none_is_ok(self) -> None:
        # If prod URL cannot be resolved, no collision possible -> ok
        sandbox = "postgresql://user:pw@sandbox-host:5432/sandbox_db"
        result = check_prod_safety(sandbox, None)
        assert result.ok is True

    def test_cloud_sql_socket_same_instance_is_violation(self) -> None:
        # Cloud SQL socket URL: empty netloc, host in ?host= query param
        instance = "myproject:us-central1:myinstance"
        prod_socket = f"postgresql://user:pw@/proddb?host=/cloudsql/{instance}"
        sandbox_socket = f"postgresql://user:pw@/sandbox_db?host=/cloudsql/{instance}"
        result = check_prod_safety(sandbox_socket, prod_socket)
        assert result.ok is False

    def test_cloud_sql_socket_different_instance_is_ok_when_allow_remote(self) -> None:
        # WR-01: Different Cloud SQL instances are OK when allow_remote=True.
        # Without allow_remote, ANY Cloud SQL sandbox is rejected (mirrors shell guard).
        prod_socket = "postgresql://user:pw@/proddb?host=/cloudsql/proj:reg:prod-inst"
        sandbox_socket = "postgresql://user:pw@/sandbox_db?host=/cloudsql/proj:reg:sandbox-inst"
        result = check_prod_safety(sandbox_socket, prod_socket, allow_remote=True)
        assert result.ok is True

    def test_same_cloud_sql_instance_different_db_is_still_violation(self) -> None:
        # Same instance = violation regardless of dbname
        instance = "myproject:us-central1:myinstance"
        prod_socket = f"postgresql://user:pw@/proddb?host=/cloudsql/{instance}"
        sandbox_socket = f"postgresql://user:pw@/different_sandbox?host=/cloudsql/{instance}"
        result = check_prod_safety(sandbox_socket, prod_socket)
        assert result.ok is False

    # WR-01: Cloud SQL sandbox URL must be rejected unless allow_remote=True
    def test_cloud_sql_sandbox_url_rejected_when_allow_remote_false(self) -> None:
        """WR-01: A Cloud SQL sandbox URL must be rejected when allow_remote=False."""
        prod_tcp = "postgresql://user:pw@127.0.0.1:5432/proddb"
        # Different Cloud SQL instance from prod — but still Cloud SQL socket
        sandbox_socket = "postgresql://user:pw@/sandbox_db?host=/cloudsql/proj:reg:sandbox-inst"
        result = check_prod_safety(sandbox_socket, prod_tcp, allow_remote=False)
        assert result.ok is False
        assert "Cloud SQL" in result.message or "cloudsql" in result.message.lower()

    def test_cloud_sql_sandbox_url_allowed_when_allow_remote_true(self) -> None:
        """WR-01: allow_remote=True bypasses the Cloud SQL rejection."""
        prod_tcp = "postgresql://user:pw@127.0.0.1:5432/proddb"
        sandbox_socket = "postgresql://user:pw@/sandbox_db?host=/cloudsql/proj:reg:sandbox-inst"
        result = check_prod_safety(sandbox_socket, prod_tcp, allow_remote=True)
        # No Cloud SQL rejection; sandbox and prod are different instances -> ok
        assert result.ok is True

    def test_local_sandbox_not_rejected_by_cloud_sql_check(self) -> None:
        """WR-01: A plain TCP sandbox URL must NOT be rejected as Cloud SQL."""
        sandbox = "postgresql://postgres:pw@127.0.0.1:5433/city_concierge_sandbox"
        prod = "postgresql://postgres:pw@127.0.0.1:5432/city_concierge"
        result = check_prod_safety(sandbox, prod, allow_remote=False)
        assert result.ok is True

    def test_allow_remote_defaults_to_false(self) -> None:
        """WR-01: allow_remote defaults to False; old callers passing 2 args get the strict guard."""
        prod_tcp = "postgresql://user:pw@127.0.0.1:5432/proddb"
        sandbox_socket = "postgresql://user:pw@/sandbox_db?host=/cloudsql/proj:reg:sandbox-inst"
        # Call with 2 positional args (old call site style)
        result = check_prod_safety(sandbox_socket, prod_tcp)
        assert result.ok is False


# ---------------------------------------------------------------------------
# check_non_circularity
# ---------------------------------------------------------------------------


class TestCheckNonCircularity:
    def test_disjoint_paraphrases_is_ok(self) -> None:
        paraphrases = ["best pho in outer sunset", "top vietnamese spots near 19th ave"]
        forbidden = ["vietnamese restaurants outer sunset"]
        result = check_non_circularity(paraphrases, forbidden)
        assert isinstance(result, GuardResult)
        assert result.ok is True

    def test_exact_match_with_seed_is_violation(self) -> None:
        seed = "vietnamese restaurants outer sunset"
        paraphrases = [seed, "top pho near golden gate park"]
        result = check_non_circularity(paraphrases, [seed])
        assert result.ok is False
        # message must name both the offending paraphrase AND the forbidden source
        assert seed in result.message

    def test_violation_message_names_both_strings(self) -> None:
        paraphrase = "lunch spots mission district"
        forbidden_query = "lunch spots mission district"
        result = check_non_circularity([paraphrase], [forbidden_query])
        assert result.ok is False
        assert paraphrase in result.message
        assert forbidden_query in result.message

    def test_topk_exceeding_k_raises_assertion(self) -> None:
        """IN-02: lists longer than K must raise AssertionError — single source of truth."""
        per_paraphrase_topk = [
            ["a", "b", "c", "d", "e", "f"],  # 6 elements > K=5
        ]
        with pytest.raises(AssertionError):
            compute_hit_rate(per_paraphrase_topk, {"f"})

    def test_whitespace_sensitive_no_match(self) -> None:
        # Trailing space makes it a different string -> ok (exact-string match)
        paraphrase = "lunch spots mission district "
        forbidden = ["lunch spots mission district"]
        result = check_non_circularity([paraphrase], forbidden)
        assert result.ok is True

    def test_case_sensitive_no_match(self) -> None:
        paraphrase = "Lunch Spots Mission District"
        forbidden = ["lunch spots mission district"]
        result = check_non_circularity([paraphrase], forbidden)
        assert result.ok is True

    def test_empty_paraphrases_is_ok(self) -> None:
        result = check_non_circularity([], ["seed query"])
        assert result.ok is True

    def test_empty_forbidden_is_ok(self) -> None:
        result = check_non_circularity(["some paraphrase"], [])
        assert result.ok is True
