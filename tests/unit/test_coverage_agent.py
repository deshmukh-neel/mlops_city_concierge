"""Unit tests for the W5 coverage agent (pure logic, no real DB or LLM)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from scripts import coverage_agent
from scripts.coverage_agent import (
    CoverageStat,
    ProposedQuery,
    _build_proposal_prompt,
    _fill_missing_cuisines,
    _parse_proposals,
    filter_already_covered,
    find_gaps,
    gather_stats,
    insert_pending,
    propose_queries,
)
from scripts.ingest_places_sf import CUISINES


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


class _CapturingCursor:
    def __init__(self, captured: list[tuple]) -> None:
        self._captured = captured

    def __enter__(self) -> _CapturingCursor:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, sql: str, params: object) -> None:
        self._captured.append((sql, params))

    def fetchall(self) -> list[tuple]:
        return []


class _CapturingConn:
    def __init__(self, captured: list[tuple]) -> None:
        self._captured = captured

    def __enter__(self) -> _CapturingConn:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def cursor(self) -> _CapturingCursor:
        return _CapturingCursor(self._captured)


class TestGatherStatsSql:
    """Lock the SQL contract: parameter shape + bucket allowlist."""

    def _captured_executes(self, monkeypatch) -> list[tuple]:
        captured: list[tuple] = []
        monkeypatch.setattr(coverage_agent, "get_conn", lambda: _CapturingConn(captured))
        gather_stats(days=14)
        return captured

    def test_passes_cuisine_allowlist_and_cutoff(self, monkeypatch) -> None:
        captured = self._captured_executes(monkeypatch)
        assert len(captured) == 1
        sql, params = captured[0]
        assert len(params) == 2
        cuisines, cutoff = params
        assert isinstance(cuisines, list) and "italian" in cuisines and "vietnamese" in cuisines
        assert "google" not in cuisines  # raw Google types must not leak in
        assert cutoff is not None  # the time bound for recent_query_diversity

    def test_neighborhood_regex_filters_non_matching_addresses(self, monkeypatch) -> None:
        sql, _ = self._captured_executes(monkeypatch)[0]
        # The non-matching guard lives in the WHERE clause of the neighborhoods CTE,
        # not just the regexp_replace. Without this clause, garbage addresses pass
        # through unchanged and pollute the bucket list.
        assert "formatted_address ~" in sql


class TestFillMissingCuisines:
    def test_zero_coverage_cuisines_are_synthesized(self) -> None:
        stats = [CoverageStat("cuisine:italian", 100, 0, None)]
        out = _fill_missing_cuisines(stats)
        bucket_set = {s.bucket for s in out}
        # italian is preserved, every other CUISINES entry shows up at 0
        assert "cuisine:italian" in bucket_set
        for c in CUISINES:
            assert f"cuisine:{c}" in bucket_set
        # synthesized rows have place_count=0 and last_ingest=None
        zeros = [
            s for s in out if s.bucket != "cuisine:italian" and s.bucket.startswith("cuisine:")
        ]
        assert all(s.place_count == 0 and s.last_ingest is None for s in zeros)

    def test_zero_coverage_cuisine_is_visible_to_find_gaps(self) -> None:
        # Without the synthesis, a cuisine with 0 ingested places is invisible
        # — exactly the gap the agent most needs to see.
        stats = _fill_missing_cuisines([CoverageStat("cuisine:italian", 100, 0, None)])
        gaps = find_gaps(stats, min_place_count=5)
        zero_cuisine_gaps = [
            g for g in gaps if g.bucket.startswith("cuisine:") and g.place_count == 0
        ]
        assert zero_cuisine_gaps, "expected synthesized 0-coverage cuisine gaps to surface"


class TestPromptFormat:
    def test_lines_tag_axis_so_llm_does_not_parse_prefix(self) -> None:
        gaps = [
            CoverageStat("neighborhood:Mission", 4, 0, None),
            CoverageStat("cuisine:burmese", 0, 0, None),
        ]
        prompt = _build_proposal_prompt(gaps)
        assert "type=neighborhood name='Mission' place_count=4" in prompt
        assert "type=cuisine name='burmese' place_count=0" in prompt
        assert "type=neighborhood gaps" in prompt
        assert "type=cuisine gaps" in prompt


class TestProposalsSchemaOwnership:
    def test_ingest_does_not_inline_create_proposals_table(self) -> None:
        # Alembic owns the schema for the W5 proposals table. The ingest
        # script must not silently fall back to a CREATE TABLE IF NOT EXISTS
        # that would drift from the migration (no CHECK constraint, no index).
        from scripts import ingest_places_sf

        assert not hasattr(ingest_places_sf, "ensure_query_proposals_table")


class TestFilterAlreadyCovered:
    def test_drops_proposals_already_in_seed_list(self) -> None:
        # `vietnamese restaurants in San Francisco` is emitted by the static
        # seed list — the LLM should not be able to claim it as a new gap.
        existing = {"vietnamese restaurants in San Francisco"}
        proposals = [
            ProposedQuery("vietnamese restaurants in San Francisco", "enriched", "x"),
            ProposedQuery("burmese restaurants in San Francisco", "enriched", "y"),
        ]
        kept, dropped = filter_already_covered(proposals, existing)
        assert [p.query_text for p in kept] == ["burmese restaurants in San Francisco"]
        assert [p.query_text for p in dropped] == ["vietnamese restaurants in San Francisco"]

    def test_empty_proposals(self) -> None:
        kept, dropped = filter_already_covered([], {"x"})
        assert kept == [] and dropped == []

    def test_static_seed_queries_are_in_existing_set(self) -> None:
        # Sanity check: the helper that builds `existing` actually pulls from
        # build_seed_queries, so the canonical "<cuisine> restaurants in SF"
        # form is covered.
        from scripts.ingest_places_sf import build_seed_queries

        seeds = set(build_seed_queries())
        assert "vietnamese restaurants in San Francisco" in seeds


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
