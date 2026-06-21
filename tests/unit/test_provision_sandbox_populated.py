"""Pure source-assertion tests for the --populated / --reset modes added to
provision_sandbox.sh in Phase 19 Plan 02.

Zero cost: no subprocess execution, no live DB, no API calls.
All assertions are string-index comparisons on the shell and Makefile source text.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
PROVISION_SCRIPT = REPO_ROOT / "scripts" / "provision_sandbox.sh"
MAKEFILE = REPO_ROOT / "Makefile"


def _load_script() -> str:
    return PROVISION_SCRIPT.read_text()


def _load_makefile() -> str:
    return MAKEFILE.read_text()


class TestFlagRoutingProvisionScript:
    """Assertion 1: --populated sets both RESET_MODE and POPULATE_BASELINE.
    Assertion 2: --reset sets RESET_MODE (but NOT POPULATE_BASELINE independently).
    """

    def test_populated_sets_reset_mode(self) -> None:
        """--populated must set RESET_MODE=1 (it is the full idempotent reset)."""
        script = _load_script()
        # The --populated arm must assign RESET_MODE="1"
        assert 'RESET_MODE="1"' in script, (
            '--populated block must set RESET_MODE="1"; not found in provision_sandbox.sh'
        )

    def test_populated_sets_populate_baseline(self) -> None:
        """--populated must set POPULATE_BASELINE=1 (activates ingest+embed)."""
        script = _load_script()
        assert 'POPULATE_BASELINE="1"' in script, (
            '--populated block must set POPULATE_BASELINE="1"; not found in provision_sandbox.sh'
        )

    def test_reset_flag_present(self) -> None:
        """--reset case must exist in the argument parser."""
        script = _load_script()
        assert "--reset" in script, "--reset flag must be handled in argument parsing"

    def test_populated_flag_present(self) -> None:
        """--populated case must exist in the argument parser."""
        script = _load_script()
        assert "--populated" in script, "--populated flag must be handled in argument parsing"

    def test_reset_mode_initialised_to_empty(self) -> None:
        """RESET_MODE must be initialised empty so bare invocation is unchanged."""
        script = _load_script()
        assert 'RESET_MODE=""' in script, (
            "RESET_MODE must be initialised to empty string (preserves bare-invocation path)"
        )

    def test_populate_baseline_initialised_to_empty(self) -> None:
        """POPULATE_BASELINE must be initialised empty so bare invocation is unchanged."""
        script = _load_script()
        assert 'POPULATE_BASELINE=""' in script, (
            "POPULATE_BASELINE must be initialised to empty string (preserves bare-invocation path)"
        )


class TestGuardBeforeDropOrdering:
    """Assertion 3: 'Prod-safety guard PASSED' appears BEFORE 'DROP DATABASE IF EXISTS'.

    Guards must run before any destructive DDL (T-19-02-01).
    """

    def test_guard_marker_before_drop(self) -> None:
        script = _load_script()
        guard_idx = script.find("Prod-safety guard PASSED")
        drop_idx = script.find("DROP DATABASE IF EXISTS")

        assert guard_idx != -1, (
            "'Prod-safety guard PASSED' message not found in provision_sandbox.sh"
        )
        assert drop_idx != -1, "'DROP DATABASE IF EXISTS' not found in provision_sandbox.sh"
        assert guard_idx < drop_idx, (
            f"Guard message (index {guard_idx}) must appear BEFORE "
            f"DROP DATABASE (index {drop_idx}); got guard AFTER drop — "
            "prod-safety invariant violated"
        )


class TestIngestEmbedUnderPopulateBaselineGate:
    """Assertion 4: ingest + embed invocations are gated under POPULATE_BASELINE,
    NOT under RESET_MODE alone.

    This proves --reset is schema-only and only --populated re-ingests/re-embeds.
    """

    def test_ingest_under_populate_baseline_gate(self) -> None:
        """The ingest call (ingest_places_sf.py) must appear AFTER the POPULATE_BASELINE gate."""
        script = _load_script()
        populate_gate_idx = script.find('if [[ "${POPULATE_BASELINE:-}" == "1" ]]')
        ingest_idx = script.find("ingest_places_sf.py")

        assert populate_gate_idx != -1, (
            "POPULATE_BASELINE gate block not found in provision_sandbox.sh"
        )
        assert ingest_idx != -1, "ingest_places_sf.py invocation not found in provision_sandbox.sh"
        assert populate_gate_idx < ingest_idx, (
            f"POPULATE_BASELINE gate (index {populate_gate_idx}) must appear BEFORE "
            f"ingest_places_sf.py call (index {ingest_idx}); "
            "ingest must be under the POPULATE_BASELINE gate only"
        )

    def test_embed_under_populate_baseline_gate(self) -> None:
        """The embed call (embed_places_pgvector_v2) must appear AFTER the POPULATE_BASELINE gate."""
        script = _load_script()
        populate_gate_idx = script.find('if [[ "${POPULATE_BASELINE:-}" == "1" ]]')
        embed_idx = script.find("embed_places_pgvector_v2")

        assert populate_gate_idx != -1, (
            "POPULATE_BASELINE gate block not found in provision_sandbox.sh"
        )
        assert embed_idx != -1, (
            "embed_places_pgvector_v2 invocation not found in provision_sandbox.sh"
        )
        assert populate_gate_idx < embed_idx, (
            f"POPULATE_BASELINE gate (index {populate_gate_idx}) must appear BEFORE "
            f"embed_places_pgvector_v2 call (index {embed_idx}); "
            "embed must be under the POPULATE_BASELINE gate only"
        )

    def test_drop_database_under_reset_mode_gate(self) -> None:
        """DROP DATABASE must be under the RESET_MODE gate, not POPULATE_BASELINE."""
        script = _load_script()
        reset_gate_idx = script.find('if [[ "${RESET_MODE:-}" == "1" ]]')
        drop_idx = script.find("DROP DATABASE IF EXISTS")

        assert reset_gate_idx != -1, "RESET_MODE gate block not found in provision_sandbox.sh"
        assert drop_idx != -1, "DROP DATABASE IF EXISTS not found in provision_sandbox.sh"
        assert reset_gate_idx < drop_idx, (
            f"RESET_MODE gate (index {reset_gate_idx}) must appear BEFORE "
            f"DROP DATABASE (index {drop_idx})"
        )

    def test_sandbox_database_url_used_for_ingest(self) -> None:
        """The ingest call must use DATABASE_URL=\"${{SANDBOX_DATABASE_URL}}\" (T-19-02-02)."""
        script = _load_script()
        # Look for the explicit SANDBOX_DATABASE_URL env-var prefix on the ingest invocation
        assert (
            'DATABASE_URL="${SANDBOX_DATABASE_URL}" poetry run python scripts/ingest_places_sf.py'
            in script
        ), (
            'Ingest must be called with DATABASE_URL="${SANDBOX_DATABASE_URL}" '
            "to prevent prod leakage (T-19-02-02)"
        )

    def test_sandbox_database_url_used_for_embed(self) -> None:
        """The embed call must use DATABASE_URL=\"${{SANDBOX_DATABASE_URL}}\" (T-19-02-02)."""
        script = _load_script()
        assert (
            'DATABASE_URL="${SANDBOX_DATABASE_URL}" poetry run python -m scripts.embed_places_pgvector_v2'
            in script
        ), (
            'Embed must be called with DATABASE_URL="${SANDBOX_DATABASE_URL}" '
            "to prevent prod leakage (T-19-02-02)"
        )


class TestInversionGapBucketExclusionKeys:
    """Assertion 5 (part): Gap-bucket exclusion keys on LOOP_GAP_NEIGHBORHOOD /
    LOOP_GAP_CUISINE — confirming the INVERSION marks ONLY the gap bucket completed
    (not the non-gap catalog).
    """

    def test_loop_gap_neighborhood_env_var_used(self) -> None:
        """LOOP_GAP_NEIGHBORHOOD env var must appear in the populate block."""
        script = _load_script()
        assert "LOOP_GAP_NEIGHBORHOOD" in script, (
            "LOOP_GAP_NEIGHBORHOOD env var must be used in the populate block "
            "to parameterise the gap-bucket exclusion (D-02)"
        )

    def test_loop_gap_cuisine_env_var_used(self) -> None:
        """LOOP_GAP_CUISINE env var must appear in the populate block."""
        script = _load_script()
        assert "LOOP_GAP_CUISINE" in script, (
            "LOOP_GAP_CUISINE env var must be used in the populate block "
            "to parameterise the gap-bucket exclusion (D-02)"
        )

    def test_inversion_comment_present(self) -> None:
        """A comment must make clear that the non-gap catalog is INGESTED (not marked completed)."""
        script = _load_script()
        # The inversion must be documented in the script
        assert "INVERSION" in script, (
            "The script must contain an 'INVERSION' comment explaining "
            "that Phase 19 marks ONLY the gap bucket (not the non-gap catalog)"
        )

    def test_exclusion_covers_citywide_and_neighborhood(self) -> None:
        """The gap-bucket exclusion must cover both per-neighborhood and citywide queries (D-02)."""
        script = _load_script()
        # The Python exclusion block should reference 'san francisco' (citywide coverage)
        # and neighborhood-scoped filtering
        assert "san francisco" in script.lower(), (
            "The exclusion logic must reference citywide 'San Francisco' queries (D-02)"
        )
        assert "neighborhood_lower" in script, (
            "The exclusion logic must reference the neighborhood dimension (D-02)"
        )

    def test_exclusion_upserts_completed_status(self) -> None:
        """The exclusion step must upsert 'completed' status (not any other status)."""
        script = _load_script()
        assert "'completed'" in script, (
            "The gap-bucket exclusion step must upsert status='completed' "
            "so ingest SKIPS_COMPLETED_QUERIES path skips the gap"
        )


class TestUrlParserStderrSuppression:
    """Assertion (CR-02): The Python URL-parser subshell must redirect 2>/dev/null,
    not 2>&1, so Poetry's own stderr output cannot corrupt _PARSED_FIELDS.

    The guard-result block lower in the file already uses 2>/dev/null correctly;
    the URL-parser block must match it.
    """

    def test_url_parser_uses_devnull_not_merge(self) -> None:
        """The URL-parser subshell must suppress Poetry stderr with 2>/dev/null."""
        script = _load_script()

        # Locate the URL-parser block (WR-06 comment anchors it)
        parser_block_start = script.find("WR-06: URL parsing is delegated")
        assert parser_block_start != -1, (
            "WR-06 comment block not found — cannot locate URL-parser subshell"
        )

        # Find the first 2>/dev/null or 2>&1 AFTER the parser block start
        parser_region = script[parser_block_start : parser_block_start + 600]

        devnull_idx = parser_region.find("2>/dev/null")
        merge_idx = parser_region.find("2>&1")

        assert devnull_idx != -1, (
            "URL-parser subshell (WR-06 block) must use '2>/dev/null' to suppress "
            "Poetry's own stderr from bleeding into _PARSED_FIELDS (CR-02). "
            "Currently missing; the _GUARD_RESULT block further down already uses "
            "2>/dev/null — match that pattern here."
        )
        # If both appear, 2>/dev/null must come first (i.e. the merge redirect is gone)
        if merge_idx != -1:
            assert devnull_idx < merge_idx, (
                "URL-parser block has both '2>/dev/null' and '2>&1'; "
                "the '2>&1' must have been removed (CR-02 fix)."
            )

    def test_guard_result_block_uses_devnull(self) -> None:
        """Regression: _GUARD_RESULT block must still use 2>/dev/null (unchanged)."""
        script = _load_script()
        guard_block_start = script.find("_GUARD_RESULT=")
        assert guard_block_start != -1, "_GUARD_RESULT block not found"
        guard_region = script[guard_block_start : guard_block_start + 400]
        assert "2>/dev/null" in guard_region, (
            "_GUARD_RESULT block must still use 2>/dev/null (should be unchanged by CR-02)"
        )


class TestMakefileTarget:
    """Assertion 5 (Makefile): sandbox-provision-populated target exists with the
    standard SANDBOX_DATABASE_URL env-guard and calls provision_sandbox.sh --populated.
    """

    def test_target_exists(self) -> None:
        makefile = _load_makefile()
        assert "sandbox-provision-populated:" in makefile, (
            "Makefile must contain 'sandbox-provision-populated:' target"
        )

    def test_calls_provision_script_with_populated(self) -> None:
        makefile = _load_makefile()
        assert "bash scripts/provision_sandbox.sh --populated" in makefile, (
            "sandbox-provision-populated target must call "
            "'bash scripts/provision_sandbox.sh --populated'"
        )

    def test_sandbox_database_url_guard_present(self) -> None:
        makefile = _load_makefile()
        assert "SANDBOX_DATABASE_URL" in makefile, (
            "Makefile sandbox-provision-populated target must have SANDBOX_DATABASE_URL env-guard"
        )

    def test_phony_declaration(self) -> None:
        makefile = _load_makefile()
        assert ".PHONY: sandbox-provision-populated" in makefile, (
            "sandbox-provision-populated must be declared .PHONY in Makefile"
        )
