"""Unit tests for the W5 coverage agent (pure logic, no real DB or LLM)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from scripts.coverage_agent import (
    CoverageStat,
    ProposedQuery,
    _parse_proposals,
    find_gaps,
    insert_pending,
    propose_queries,
)


class TestFindGaps:
    def test_filters_by_min_places(self) -> None:
        stats = [
            CoverageStat("neighborhood:Mission", 200, 0, None),
            CoverageStat("neighborhood:Outer Sunset", 2, 0, None),
            CoverageStat("cuisine:italian", 100, 0, None),
            CoverageStat("cuisine:burmese", 1, 0, None),
            CoverageStat("recent_query", 50, 30, None),
        ]
        gaps = find_gaps(stats, min_place_count=5)
        assert {g.bucket for g in gaps} == {
            "neighborhood:Outer Sunset",
            "cuisine:burmese",
        }

    def test_recent_query_bucket_never_a_gap(self) -> None:
        stats = [CoverageStat("recent_query", 0, 0, None)]
        assert find_gaps(stats, min_place_count=5) == []

    def test_empty_input_returns_empty(self) -> None:
        assert find_gaps([], min_place_count=5) == []


class TestParseProposals:
    def _payload(self) -> str:
        return json.dumps(
            [
                {
                    "query_text": "burmese restaurants in San Francisco",
                    "field_mode": "enriched",
                    "rationale": "burmese only has 1 place",
                }
            ]
        )

    def test_plain_json(self) -> None:
        proposals = _parse_proposals(self._payload())
        assert len(proposals) == 1
        assert proposals[0].query_text == "burmese restaurants in San Francisco"

    def test_strips_code_fences(self) -> None:
        wrapped = f"```json\n{self._payload()}\n```"
        assert len(_parse_proposals(wrapped)) == 1

    def test_strips_unlabeled_code_fences(self) -> None:
        wrapped = f"```\n{self._payload()}\n```"
        assert len(_parse_proposals(wrapped)) == 1

    def test_invalid_json_returns_empty(self) -> None:
        assert _parse_proposals("not json at all") == []

    def test_empty_string_returns_empty(self) -> None:
        assert _parse_proposals("") == []

    def test_non_list_payload_returns_empty(self) -> None:
        assert _parse_proposals('{"query_text": "x"}') == []

    def test_skips_items_missing_required_keys(self) -> None:
        raw = json.dumps(
            [
                {"query_text": "ok", "field_mode": "enriched", "rationale": "fine"},
                {"query_text": "missing-rest"},
                {"query_text": "x", "field_mode": "y"},
            ]
        )
        proposals = _parse_proposals(raw)
        assert len(proposals) == 1
        assert proposals[0].query_text == "ok"

    def test_skips_items_with_non_string_values(self) -> None:
        raw = json.dumps(
            [
                {"query_text": 42, "field_mode": "enriched", "rationale": "n"},
                {"query_text": "x", "field_mode": "enriched", "rationale": None},
            ]
        )
        assert _parse_proposals(raw) == []

    def test_defaults_blank_field_mode_to_enriched(self) -> None:
        raw = json.dumps([{"query_text": "q", "field_mode": "", "rationale": "r"}])
        proposals = _parse_proposals(raw)
        assert proposals[0].field_mode == "enriched"


class TestProposeQueries:
    def test_returns_empty_when_no_gaps(self) -> None:
        llm = MagicMock()
        assert propose_queries([], llm) == []
        llm.invoke.assert_not_called()

    def test_returns_empty_when_llm_is_none(self) -> None:
        gaps = [CoverageStat("cuisine:burmese", 1, 0, None)]
        assert propose_queries(gaps, None) == []

    def test_parses_llm_json_response(self) -> None:
        llm = MagicMock()
        llm.invoke.return_value.content = json.dumps(
            [
                {
                    "query_text": "burmese restaurants in San Francisco",
                    "field_mode": "enriched",
                    "rationale": "burmese only has 1 place",
                }
            ]
        )
        gaps = [CoverageStat("cuisine:burmese", 1, 0, None)]
        proposals = propose_queries(gaps, llm)
        assert len(proposals) == 1
        assert proposals[0].query_text == "burmese restaurants in San Francisco"
        assert llm.invoke.call_count == 1

    def test_non_string_content_returns_empty(self) -> None:
        llm = MagicMock()
        llm.invoke.return_value.content = ["not", "a", "string"]
        gaps = [CoverageStat("cuisine:burmese", 1, 0, None)]
        assert propose_queries(gaps, llm) == []


class TestInsertPending:
    def test_dry_run_inserts_nothing_and_prints(self, capsys) -> None:
        proposals = [ProposedQuery("x", "enriched", "y")]
        n = insert_pending(proposals, dry_run=True)
        assert n == 0
        out = capsys.readouterr().out
        assert "[dry-run]" in out
        assert "'x'" in out

    def test_no_proposals_returns_zero(self) -> None:
        assert insert_pending([], dry_run=False) == 0
