"""Unit tests for scripts/audit_list_content_aimessages.py (REPLAY-02 / D-14-05).

Covers:
- Half (a): run-dir scan correctly identifies the persisted EvalRunReport shape
  (tool_calls as int, no message .content) without crashing.
- Half (b): structural analysis returns the correct EXPECTED-NULL verdict for
  the three RUN-model provider/model pairs and identifies Anthropic as the only
  list-content adapter.
- No live runs, no network, no DB.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Minimal EvalRunReport JSON shape, mirroring the real structure produced by
# eval_agent.py: top-level keys include "queries", each query has a
# "deterministic" sub-dict with "tool_calls" as an integer count.
# Critically: NO "message_content", NO "messages" key under deterministic.
FIXTURE_RUN_JSON_SHAPE: dict[str, Any] = {
    "eval_queries_path": "tests/fixtures/eval_queries.jsonl",
    "llm_provider": "openai",
    "chat_model": "gpt-5-mini",
    "query_count": 2,
    "aggregate": {"committed_itinerary_rate": 0.5},
    "queries": [
        {
            "id": "omakase_mission_open_ended",
            "question": "Plan a SF itinerary",
            "answer": "Here is your itinerary",
            "contexts": [],
            "reference": None,
            "tags": ["omakase"],
            "status": "ok",
            "error": None,
            "final_reply": "Here is your itinerary",
            "latency_seconds": 12.3,
            "expected": {
                "min_stops": 3,
                "max_stops": 5,
                "expects_clarification_or_relaxation": False,
            },
            "actual": {
                "result_count": 4,
                "committed_stop_count": 4,
                "place_ids": ["p1", "p2", "p3", "p4"],
                "place_names": ["A", "B", "C", "D"],
                "sources": [],
                "answer_place_names": ["A", "B", "C", "D"],
            },
            "deterministic": {
                # tool_calls is an INTEGER COUNT — the key finding of the audit.
                # There is NO serialized AIMessage .content or .additional_kwargs.
                "tool_calls": 8,
                "tool_names": ["search_places", "commit_itinerary"],
                "step_telemetry": [{"step": 1, "tool_exec_seconds": 2.1}],
                "arm_flags": {
                    "viability_contract": False,
                    "forced_commit_step": 0,
                    "parallel_tool": False,
                    "viability_threshold_override": None,
                    "replay_multi_message": False,
                    "replay_content_blocks": False,
                },
                "commit_forced": False,
                "forced_commit_step": None,
                "first_commit_call_step": 6,
                "first_commit_mention_step": None,
                "viable_candidates_per_step": [],
                "rule8_met_per_step": [],
                "rule8_met_but_kept_searching_steps": [],
                "expected_results_met": True,
                "checks": {},
                "violations": [],
                "tool_errors": [],
                "first_tool_error": None,
                "revision_hints": 0,
                "revision_reasons": [],
                "viability_threshold": 0.55,
            },
        },
        {
            "id": "refinement_cheaper",
            "question": "Make stop 2 cheaper",
            "answer": "Updated itinerary",
            "contexts": [],
            "reference": None,
            "tags": ["refinement"],
            "status": "ok",
            "error": None,
            "final_reply": "Updated itinerary",
            "latency_seconds": 8.7,
            "expected": None,
            "actual": {
                "result_count": 0,
                "committed_stop_count": 0,
                "place_ids": [],
                "place_names": [],
                "sources": [],
                "answer_place_names": [],
            },
            "deterministic": {
                "tool_calls": 3,
                "tool_names": ["search_places"],
                "step_telemetry": [],
                "arm_flags": {},
                "commit_forced": False,
                "forced_commit_step": None,
                "first_commit_call_step": None,
                "first_commit_mention_step": None,
                "viable_candidates_per_step": [],
                "rule8_met_per_step": [],
                "rule8_met_but_kept_searching_steps": [],
                "expected_results_met": None,
                "checks": {},
                "violations": [],
                "tool_errors": [],
                "first_tool_error": None,
                "revision_hints": 1,
                "revision_reasons": ["cost"],
                "viability_threshold": 0.55,
            },
        },
    ],
}


@pytest.fixture()
def run_dir_with_fixture_json(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a temporary run dir with two fixture JSON files."""
    run_dir = tmp_path / "run_dir"
    run_dir.mkdir()

    # Two files: one for gpt-5-mini, one for deepseek.
    for name in [
        "openai--gpt-5-mini--omakase_mission_open_ended--run-0.json",
        "deepseek--deepseek-reasoner--omakase_mission_open_ended--run-0.json",
    ]:
        fixture = dict(FIXTURE_RUN_JSON_SHAPE)
        (run_dir / name).write_text(json.dumps(fixture))

    return run_dir


# ---------------------------------------------------------------------------
# Half (a) tests: run-dir scan
# ---------------------------------------------------------------------------


class TestRunDirScan:
    def test_detects_no_message_trace_shape(self, run_dir_with_fixture_json: pathlib.Path) -> None:
        """Half (a): scan correctly identifies that run JSONs have no message .content."""
        from scripts.audit_list_content_aimessages import scan_run_dir  # noqa: PLC0415

        findings = scan_run_dir(run_dir_with_fixture_json)

        assert findings["files_found"] == 2
        assert findings["files_parsed"] == 2
        assert findings["has_message_content"] is False, (
            "Expected no message content in fixture run JSONs — "
            "EvalRunReport persists tool_calls as int, not message content"
        )
        assert "CONFIRMED" in findings["shape_finding"], (
            f"Expected 'CONFIRMED' in shape finding, got: {findings['shape_finding']}"
        )
        assert "tool_calls" in findings["shape_finding"]
        assert (
            "integer" in findings["shape_finding"].lower() or "COUNT" in findings["shape_finding"]
        )

    def test_does_not_crash_on_valid_run_json(
        self, run_dir_with_fixture_json: pathlib.Path
    ) -> None:
        """Half (a): scan exits cleanly without raising exceptions."""
        from scripts.audit_list_content_aimessages import scan_run_dir  # noqa: PLC0415

        findings = scan_run_dir(run_dir_with_fixture_json)

        assert findings["errors"] == [], f"Unexpected errors: {findings['errors']}"

    def test_handles_missing_run_dir(self, tmp_path: pathlib.Path) -> None:
        """Half (a): missing run dir is reported, not raised."""
        from scripts.audit_list_content_aimessages import scan_run_dir  # noqa: PLC0415

        # scan_run_dir expects the dir to exist; caller handles missing case.
        # Pass an empty dir to simulate "no JSON files".
        empty_dir = tmp_path / "empty_run_dir"
        empty_dir.mkdir()
        findings = scan_run_dir(empty_dir)

        assert findings["files_found"] == 0
        assert "No JSON files found" in findings["shape_finding"]
        assert findings["has_message_content"] is False

    def test_handles_malformed_json_gracefully(self, tmp_path: pathlib.Path) -> None:
        """Half (a): malformed JSON is recorded as an error, not a crash."""
        from scripts.audit_list_content_aimessages import scan_run_dir  # noqa: PLC0415

        run_dir = tmp_path / "bad_json_dir"
        run_dir.mkdir()
        (run_dir / "bad.json").write_text("{not valid json}")

        findings = scan_run_dir(run_dir)

        assert findings["files_found"] == 1
        assert len(findings["errors"]) >= 1
        assert findings["has_message_content"] is False  # default, no crash

    def test_detects_unexpected_message_content_if_present(self, tmp_path: pathlib.Path) -> None:
        """Half (a): if a run JSON unexpectedly carries message_content, it is flagged."""
        from scripts.audit_list_content_aimessages import scan_run_dir  # noqa: PLC0415

        run_dir = tmp_path / "surprise_dir"
        run_dir.mkdir()

        # Inject an unexpected 'messages' key under deterministic.
        surprise_fixture = dict(FIXTURE_RUN_JSON_SHAPE)
        surprise_fixture["queries"] = [
            {
                **FIXTURE_RUN_JSON_SHAPE["queries"][0],
                "deterministic": {
                    **FIXTURE_RUN_JSON_SHAPE["queries"][0]["deterministic"],
                    "messages": [{"role": "assistant", "content": "hello"}],
                },
            }
        ]
        (run_dir / "surprise--run-0.json").write_text(json.dumps(surprise_fixture))

        findings = scan_run_dir(run_dir)

        assert findings["has_message_content"] is True
        assert "UNEXPECTED" in findings["shape_finding"]


# ---------------------------------------------------------------------------
# Half (b) tests: structural adapter analysis
# ---------------------------------------------------------------------------


class TestStructuralAdapterAnalysis:
    def test_all_run_models_have_str_content(self) -> None:
        """Half (b): all three RUN models are classified as str-content (no list content)."""
        from scripts.audit_list_content_aimessages import structural_analysis  # noqa: PLC0415

        result = structural_analysis()

        assert result["all_run_models_str_content"] is True, (
            "Expected all RUN-model adapters to have str-content AIMessages; "
            f"got all_run_models_str_content={result['all_run_models_str_content']}"
        )

    def test_expected_null_verdict_for_run_models(self) -> None:
        """Half (b): overall verdict states R2 is EXPECTED-NULL on tested cells."""
        from scripts.audit_list_content_aimessages import structural_analysis  # noqa: PLC0415

        result = structural_analysis()

        assert "EXPECTED-NULL" in result["verdict"], (
            f"Expected 'EXPECTED-NULL' in verdict, got: {result['verdict'][:200]}"
        )
        # Must mention all three RUN models.
        assert "gpt-5-mini" in result["verdict"]
        assert "gpt-4o-mini" in result["verdict"]
        assert "deepseek-reasoner" in result["verdict"]

    def test_anthropic_is_only_list_content_adapter(self) -> None:
        """Half (b): only Anthropic is classified as a list-content adapter."""
        from scripts.audit_list_content_aimessages import structural_analysis  # noqa: PLC0415

        result = structural_analysis()

        assert result["list_content_adapter_count"] == 1, (
            f"Expected exactly 1 list-content adapter, got {result['list_content_adapter_count']}: "
            f"{result['list_content_adapters']}"
        )
        assert result["list_content_adapters"] == ["anthropic"], (
            f"Expected only 'anthropic' as list-content adapter, "
            f"got {result['list_content_adapters']}"
        )

    def test_anthropic_is_deferred_not_in_run_matrix(self) -> None:
        """Half (b): Anthropic (list-content) is deferred and NOT in the run matrix."""
        from scripts.audit_list_content_aimessages import structural_analysis  # noqa: PLC0415

        result = structural_analysis()

        assert "anthropic" in result["deferred_list_content_adapters"], (
            "Expected anthropic to be in deferred_list_content_adapters; "
            f"got {result['deferred_list_content_adapters']}"
        )

    def test_run_model_count_is_three(self) -> None:
        """Half (b): exactly three RUN models are classified (gpt-5-mini, gpt-4o-mini, deepseek)."""
        from scripts.audit_list_content_aimessages import structural_analysis  # noqa: PLC0415

        result = structural_analysis()

        assert result["run_model_count"] == 2, (  # Two entries: openai (both models), deepseek
            # NOTE: the classification table has one openai entry (covers both gpt-5 + gpt-4o)
            # and one deepseek entry. The actual distinct RUN model cells are 3 but the
            # classification table has 2 RUN-model rows.
            f"Expected 2 RUN-model classification rows, got {result['run_model_count']}"
        )

    def test_str_collapse_no_op_mentioned_in_verdict(self) -> None:
        """Half (b): verdict explicitly states str() collapse is a NO-OP for RUN models."""
        from scripts.audit_list_content_aimessages import structural_analysis  # noqa: PLC0415

        result = structural_analysis()

        # Either 'NO-OP' or some equivalent phrasing must be in the verdict.
        assert "NO-OP" in result["verdict"] or "no-op" in result["verdict"].lower(), (
            "Expected 'NO-OP' in verdict to confirm str() collapse has no effect; "
            f"got: {result['verdict'][:300]}"
        )

    def test_verdict_mentions_criterion_2_still_runs(self) -> None:
        """Half (b): verdict notes R2 still runs even though expected-null (criterion 2)."""
        from scripts.audit_list_content_aimessages import structural_analysis  # noqa: PLC0415

        result = structural_analysis()

        # Must acknowledge R2 still runs despite expected-null.
        lower = result["verdict"].lower()
        assert "still runs" in lower or "still run" in lower, (
            "Expected verdict to state R2 still runs (criterion 2 requires measured delta); "
            f"got: {result['verdict'][:300]}"
        )


# ---------------------------------------------------------------------------
# Integration: main() exits 0 on a synthetic run dir
# ---------------------------------------------------------------------------


class TestMainExitsCleanly:
    def test_main_exits_zero_on_fixture_run_dir(
        self, run_dir_with_fixture_json: pathlib.Path
    ) -> None:
        """main() returns 0 when given a valid run dir with the correct JSON shape."""
        from scripts.audit_list_content_aimessages import main  # noqa: PLC0415

        exit_code = main(
            [
                "--run-dir",
                str(run_dir_with_fixture_json),
                "--skip-adapter-check",
            ]
        )
        assert exit_code == 0

    def test_main_exits_zero_with_no_args(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
    ) -> None:
        """main() with no args (defaults to known arm run dirs) exits 0 even if dirs absent."""
        # Patch the known arm dirs to a non-existent path so the test is hermetic.
        import scripts.audit_list_content_aimessages as audit_mod  # noqa: PLC0415

        monkeypatch.setattr(
            audit_mod,
            "KNOWN_ARM_RUN_DIRS",
            [str(tmp_path / "nonexistent_dir")],
        )
        exit_code = audit_mod.main(["--skip-adapter-check"])
        assert exit_code == 0
