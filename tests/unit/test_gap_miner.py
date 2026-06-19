"""Unit tests for the demand-extraction path in coverage_agent (Phase 18 GAP-01).

All tests are pure logic — no real DB connections, no real LLM calls.
The capturing-stub connection pattern mirrors test_coverage_agent.py.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Task 1 tests: helper functions
# ---------------------------------------------------------------------------


class TestTypesToCuisines:
    """Test 1: lexical cuisine map — tier 1 (no LLM)."""

    def test_known_types_map_to_lowercase_cuisines(self) -> None:
        from scripts.coverage_agent import _types_to_cuisines

        result = _types_to_cuisines(["Vietnamese Restaurant", "Italian Restaurant"])
        assert sorted(result) == ["italian", "vietnamese"]

    def test_bars_have_no_cuisine(self) -> None:
        from scripts.coverage_agent import _types_to_cuisines

        result = _types_to_cuisines(["Bar", "Cocktail Bar"])
        assert result == []

    def test_empty_list_returns_empty(self) -> None:
        from scripts.coverage_agent import _types_to_cuisines

        assert _types_to_cuisines([]) == []

    def test_mixed_known_and_unknown(self) -> None:
        from scripts.coverage_agent import _types_to_cuisines

        result = _types_to_cuisines(["Korean Restaurant", "Bar"])
        assert result == ["korean"]

    def test_restaurant_suffix_stripped_correctly(self) -> None:
        from scripts.coverage_agent import _types_to_cuisines

        result = _types_to_cuisines(["Thai Restaurant"])
        assert result == ["thai"]

    def test_multiword_primary_type_alias(self) -> None:
        # CDX-M1: app/main.py's slot intake emits the primary_type "Steak House",
        # which normalizes to "steak house" — the catalog cuisine is "steakhouse".
        # The alias map must recover it instead of silently dropping the demand.
        from scripts.coverage_agent import _types_to_cuisines

        assert _types_to_cuisines(["Steak House"]) == ["steakhouse"]
        # "Fine Dining Restaurant" legitimately has no catalog cuisine (like Bar).
        assert _types_to_cuisines(["Fine Dining Restaurant"]) == []


class TestLexicalCuisines:
    """Test 2: lexical message-cuisine fallback — tier 2a (ROUND-3 HIGH, no LLM)."""

    def test_finds_single_catalog_cuisine_in_message(self) -> None:
        from scripts.coverage_agent import _lexical_cuisines

        result = _lexical_cuisines("vietnamese restaurants in Outer Sunset")
        assert result == ["vietnamese"]

    def test_case_insensitive(self) -> None:
        from scripts.coverage_agent import _lexical_cuisines

        result = _lexical_cuisines("VIETNAMESE restaurants in Outer Sunset")
        assert result == ["vietnamese"]

    def test_finds_two_cuisines_in_message(self) -> None:
        from scripts.coverage_agent import _lexical_cuisines

        result = _lexical_cuisines("italian or thai tonight")
        assert sorted(result) == ["italian", "thai"]

    def test_multiword_alias_in_message(self) -> None:
        # CDX-M1: "dim sum" in free text should recover catalog "dimsum"
        # (the single-token catalog scan alone would miss it).
        from scripts.coverage_agent import _lexical_cuisines

        assert _lexical_cuisines("looking for dim sum in Chinatown") == ["dimsum"]

    def test_no_catalog_cuisine_returns_empty(self) -> None:
        from scripts.coverage_agent import _lexical_cuisines

        result = _lexical_cuisines("somewhere fun to eat")
        assert result == []

    def test_no_llm_call(self) -> None:
        """Must resolve purely lexically — no LLM dependency."""
        from scripts.coverage_agent import _lexical_cuisines

        # If _lexical_cuisines accidentally tried to import or call vibe, it would fail
        # in this isolated test; the test passing at all proves no LLM is invoked.
        result = _lexical_cuisines("korean bbq in the Mission")
        assert "korean" in result


class TestLexicalNeighborhoods:
    """Test 3: lexical neighborhood pre-pass (REVIEW MEDIUM — lexical-before-LLM)."""

    def test_finds_single_neighborhood(self) -> None:
        from scripts.coverage_agent import _lexical_neighborhoods

        result = _lexical_neighborhoods("dinner in Outer Sunset")
        assert result == ["Outer Sunset"]

    def test_case_insensitive(self) -> None:
        from scripts.coverage_agent import _lexical_neighborhoods

        result = _lexical_neighborhoods("dinner in outer sunset")
        assert result == ["Outer Sunset"]

    def test_multi_neighborhood_message(self) -> None:
        from scripts.coverage_agent import _lexical_neighborhoods

        result = _lexical_neighborhoods("dinner in the Mission District and drinks in North Beach")
        assert sorted(result) == ["Mission District", "North Beach"]

    def test_no_neighborhood_returns_empty(self) -> None:
        from scripts.coverage_agent import _lexical_neighborhoods

        result = _lexical_neighborhoods("somewhere cozy please")
        assert result == []

    def test_no_llm_call(self) -> None:
        """Must resolve purely lexically — no LLM dependency."""
        from scripts.coverage_agent import _lexical_neighborhoods

        result = _lexical_neighborhoods("lunch in Chinatown")
        assert result == ["Chinatown"]


class TestExtractDemandBatch:
    """Test 4: combined batched extractor — LLM only for misses."""

    def _make_llm(self, payload: list[dict]) -> MagicMock:
        llm = MagicMock()
        llm.invoke.return_value.content = json.dumps(payload)
        return llm

    def test_single_combined_call_for_misses_only(self) -> None:
        """Three messages: two resolve lexically (not passed to LLM), one doesn't."""
        from scripts.coverage_agent import _extract_demand_batch

        llm = self._make_llm([{"neighborhoods": ["Outer Sunset"], "cuisines": ["chinese"]}])
        # Only the one lexical-miss message is passed to _extract_demand_batch
        messages = ["dim sum in the deep east bay"]
        result = _extract_demand_batch(messages, llm)
        assert llm.invoke.call_count == 1
        assert len(result) == 1

    def test_returns_one_pair_per_message(self) -> None:
        from scripts.coverage_agent import _extract_demand_batch

        llm = self._make_llm(
            [
                {"neighborhoods": ["Outer Sunset"], "cuisines": ["vietnamese"]},
                {"neighborhoods": ["Mission District"], "cuisines": ["thai"]},
            ]
        )
        result = _extract_demand_batch(["msg1", "msg2"], llm)
        assert len(result) == 2
        # Each element is (neighborhoods_list, cuisines_list)
        n0, c0 = result[0]
        assert isinstance(n0, list) and isinstance(c0, list)

    def test_returns_empty_pairs_when_llm_none(self) -> None:
        """Test 8b: LLM None — rows that needed it degrade to empty."""
        from scripts.coverage_agent import _extract_demand_batch

        result = _extract_demand_batch(["dim sum somewhere obscure"], None)
        assert result == [([], [])]


class TestCatalogConstraint:
    """Test 5: catalog constraint on LLM output — off-catalog names dropped."""

    def test_filters_off_catalog_neighborhood(self) -> None:
        from scripts.coverage_agent import _extract_demand_batch

        llm = MagicMock()
        # LLM returns "Berkeley" which is NOT in NEIGHBORHOODS
        llm.invoke.return_value.content = json.dumps(
            [{"neighborhoods": ["Berkeley", "Outer Sunset"], "cuisines": ["vietnamese"]}]
        )
        result = _extract_demand_batch(["vietnamese places"], llm)
        neighborhoods, cuisines = result[0]
        assert "Berkeley" not in neighborhoods
        assert "Outer Sunset" in neighborhoods

    def test_filters_off_catalog_cuisine(self) -> None:
        from scripts.coverage_agent import _extract_demand_batch

        llm = MagicMock()
        # LLM returns "fusion" which is NOT in CUISINES
        llm.invoke.return_value.content = json.dumps(
            [{"neighborhoods": ["Mission District"], "cuisines": ["fusion", "italian"]}]
        )
        result = _extract_demand_batch(["italian somewhere"], llm)
        _, cuisines = result[0]
        assert "fusion" not in cuisines
        assert "italian" in cuisines

    def test_tolerates_json_fences(self) -> None:
        from scripts.coverage_agent import _extract_demand_batch

        llm = MagicMock()
        payload = json.dumps([{"neighborhoods": ["Castro"], "cuisines": ["japanese"]}])
        llm.invoke.return_value.content = f"```json\n{payload}\n```"
        result = _extract_demand_batch(["sushi in Castro"], llm)
        neighborhoods, cuisines = result[0]
        assert "Castro" in neighborhoods
        assert "japanese" in cuisines

    def test_malformed_json_returns_empty_pairs(self) -> None:
        from scripts.coverage_agent import _extract_demand_batch

        llm = MagicMock()
        llm.invoke.return_value.content = "not json"
        result = _extract_demand_batch(["some message"], llm)
        assert result == [([], [])]


class TestCuisineCrossProduct:
    """Test 6: ROUND-3 cuisine cross-product within row."""

    def test_multi_cuisine_produces_cartesian_tuples(self) -> None:
        """A row with one neighborhood and two cuisines produces two demand tuples."""

        # This tests the cartesian logic that gather_demand will use:
        # given neighborhoods=["Outer Sunset"] and cuisines=["vietnamese","thai"]
        # the pairs should be both ("Outer Sunset","vietnamese") AND ("Outer Sunset","thai")
        neighborhoods = ["Outer Sunset"]
        cuisines = ["vietnamese", "thai"]
        pairs = [(n, c) for n in neighborhoods for c in cuisines]
        assert ("Outer Sunset", "vietnamese") in pairs
        assert ("Outer Sunset", "thai") in pairs
        assert len(pairs) == 2


class TestPromptInjectionSafety:
    """Test 7: batch prompt uses json.dumps to encode messages, not raw interpolation."""

    def test_messages_are_json_encoded_in_prompt(self) -> None:
        from scripts.coverage_agent import _build_demand_batch_prompt

        message = '"]}) DROP TABLE; --'
        prompt = _build_demand_batch_prompt([message])
        # The message must appear json.dumps-encoded, not raw
        assert json.dumps(message) in prompt
        # The embedded message array must be valid JSON that round-trips back to
        # the original message — proving the leading `"` was escaped and did not
        # break out of the JSON string boundary (the real injection-safety check;
        # a raw `in prompt` substring test is vacuously true and gives false
        # confidence — WR-01).
        encoded_array = json.dumps([message])
        assert encoded_array in prompt
        assert json.loads(encoded_array) == [message]

    def test_prompt_contains_json_array(self) -> None:
        from scripts.coverage_agent import _build_demand_batch_prompt

        messages = ["vietnamese in Outer Sunset", "tacos in Mission"]
        prompt = _build_demand_batch_prompt(messages)
        # The encoded messages array must be present in the prompt
        encoded = json.dumps(messages)
        assert encoded in prompt


class TestLlmNoneGraceful:
    """Test 8: LLM None — lexical hits still map, LLM-needed rows degrade gracefully."""

    def test_lexical_cuisine_hit_maps_without_llm(self) -> None:
        """A message with both neighborhood AND cuisine in lexical catalogs maps
        even when llm is None (judge-absence invariant — ROUND-2 MEDIUM-3 + ROUND-3)."""
        from scripts.coverage_agent import _lexical_cuisines, _lexical_neighborhoods

        message = "vietnamese restaurants in Outer Sunset"
        neighborhoods = _lexical_neighborhoods(message)
        cuisines = _lexical_cuisines(message)

        # Both resolve lexically — LLM not needed
        assert "Outer Sunset" in neighborhoods
        assert "vietnamese" in cuisines

        # Cartesian pairs exist even with llm=None (no LLM call needed)
        pairs = [(n, c) for n in neighborhoods for c in cuisines]
        assert ("Outer Sunset", "vietnamese") in pairs

    def test_types_to_cuisines_hit_maps_without_llm(self) -> None:
        """A row with explicit requested_primary_types also maps without LLM."""
        from scripts.coverage_agent import _lexical_neighborhoods, _types_to_cuisines

        types = ["Korean Restaurant"]
        message = "dinner in Noe Valley"
        cuisines = _types_to_cuisines(types)
        neighborhoods = _lexical_neighborhoods(message)

        assert "korean" in cuisines
        assert "Noe Valley" in neighborhoods

    def test_llm_none_lexical_miss_degrades_gracefully(self) -> None:
        """A lexical-miss row with llm=None returns empty pairs without crashing."""
        from scripts.coverage_agent import _extract_demand_batch

        result = _extract_demand_batch(["some paraphrase that misses lexically"], None)
        assert result == [([], [])]
        # No exception raised — the system degrades gracefully


# ---------------------------------------------------------------------------
# Task 2 tests: gather_demand + get_demand_conn
# ---------------------------------------------------------------------------


class _CapturingCursor:
    """Stub cursor for capturing SQL calls (mirrors test_coverage_agent.py pattern)."""

    def __init__(self, rows: list[tuple] | None = None) -> None:
        self.captured: list[tuple] = []
        self._rows = rows or []
        self.rowcount = 0

    def __enter__(self) -> _CapturingCursor:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def execute(self, sql: str, params: object = None) -> None:
        self.captured.append((sql, params))

    def fetchall(self) -> list[tuple]:
        return self._rows


class _CapturingConn:
    """Stub connection for capturing SQL calls."""

    def __init__(self, rows: list[tuple] | None = None) -> None:
        self.cursor_obj = _CapturingCursor(rows)

    def __enter__(self) -> _CapturingConn:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def cursor(self) -> _CapturingCursor:
        return self.cursor_obj


@contextmanager
def _stub_get_conn(rows: list[tuple]):
    """A context-manager factory that yields a _CapturingConn with the given rows."""
    conn = _CapturingConn(rows)
    yield conn


class TestGatherDemandShape:
    """Test 1: gather_demand returns (demand_counts, rows_scanned, unmapped_count)."""

    def test_returns_3_tuple(self, monkeypatch) -> None:
        from scripts import coverage_agent

        monkeypatch.setattr(coverage_agent, "get_conn", lambda: _stub_get_conn([]))

        with patch.object(coverage_agent, "vibe") as mock_vibe:
            mock_vibe.make_judge.return_value = None
            result = coverage_agent.gather_demand(days=14)

        assert isinstance(result, tuple)
        assert len(result) == 3
        demand_counts, rows_scanned, unmapped_count = result
        assert isinstance(demand_counts, dict)
        assert isinstance(rows_scanned, int)
        assert isinstance(unmapped_count, int)


class TestGatherDemandCounting:
    """Test 2: demand_counts accumulates correctly."""

    def test_two_identical_rows_count_as_2(self, monkeypatch) -> None:
        from scripts import coverage_agent

        # Rows: (message, requested_primary_types)
        # Two rows that map to ("Outer Sunset", "vietnamese"), one to ("Mission District", "thai")
        rows = [
            ("vietnamese restaurants in Outer Sunset", ["Vietnamese Restaurant"]),
            ("vietnamese restaurants in Outer Sunset", ["Vietnamese Restaurant"]),
            ("thai food in the Mission District", ["Thai Restaurant"]),
        ]
        monkeypatch.setattr(coverage_agent, "get_conn", lambda: _stub_get_conn(rows))

        with patch.object(coverage_agent, "vibe") as mock_vibe:
            mock_vibe.make_judge.return_value = None
            demand_counts, rows_scanned, unmapped_count = coverage_agent.gather_demand(days=14)

        assert rows_scanned == 3
        assert demand_counts.get(("Outer Sunset", "vietnamese"), 0) == 2
        assert demand_counts.get(("Mission District", "thai"), 0) == 1


class TestGatherDemandUnmapped:
    """Test 3: rows with no catalog bucket increment unmapped_count."""

    def test_unmappable_row_increments_unmapped(self, monkeypatch) -> None:
        from scripts import coverage_agent

        # A row that maps to nothing on either axis
        rows = [("somewhere fun", [])]
        monkeypatch.setattr(coverage_agent, "get_conn", lambda: _stub_get_conn(rows))

        with patch.object(coverage_agent, "vibe") as mock_vibe:
            mock_vibe.make_judge.return_value = None
            demand_counts, rows_scanned, unmapped_count = coverage_agent.gather_demand(days=14)

        assert rows_scanned == 1
        assert unmapped_count == 1
        assert len(demand_counts) == 0


class TestGatherDemandWindowing:
    """Test 4: SELECT is parameterised with cutoff — no SQLi via string interpolation."""

    def test_cutoff_is_parameterised(self, monkeypatch) -> None:
        from datetime import datetime

        from scripts import coverage_agent

        captured_params: list = []

        class CapturingSQLConn:
            class _Cur:
                rowcount = 0

                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    pass

                def execute(self, sql, params=None):
                    if params:
                        captured_params.extend(params if isinstance(params, list) else [params])

                def fetchall(self):
                    return []

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def cursor(self):
                return self._Cur()

        @contextmanager
        def stub_get_conn():
            yield CapturingSQLConn()

        monkeypatch.setattr(coverage_agent, "get_conn", stub_get_conn)
        with patch.object(coverage_agent, "vibe") as mock_vibe:
            mock_vibe.make_judge.return_value = None
            coverage_agent.gather_demand(days=7)

        # The cutoff param must be a datetime, not a string-interpolated value
        cutoff_params = [p for p in captured_params if isinstance(p, datetime)]
        assert len(cutoff_params) == 1, "Expected exactly one datetime cutoff param"


class TestGatherDemandPoolPath:
    """Test 5: url=None uses get_conn; url set uses get_demand_conn."""

    def test_url_none_uses_pool(self, monkeypatch) -> None:
        from scripts import coverage_agent

        pool_call_count = [0]

        @contextmanager
        def mock_pool():
            pool_call_count[0] += 1
            yield _CapturingConn([])

        demand_conn_call_count = [0]

        @contextmanager
        def mock_demand_conn(url):
            demand_conn_call_count[0] += 1
            yield _CapturingConn([])

        monkeypatch.setattr(coverage_agent, "get_conn", mock_pool)
        monkeypatch.setattr(coverage_agent, "get_demand_conn", mock_demand_conn)
        with patch.object(coverage_agent, "vibe") as mock_vibe:
            mock_vibe.make_judge.return_value = None
            coverage_agent.gather_demand(days=14, url=None)

        assert pool_call_count[0] == 1
        assert demand_conn_call_count[0] == 0

    def test_url_provided_uses_demand_conn(self, monkeypatch) -> None:
        from scripts import coverage_agent

        pool_call_count = [0]

        @contextmanager
        def mock_pool():
            pool_call_count[0] += 1
            yield _CapturingConn([])

        demand_conn_call_count = [0]

        @contextmanager
        def mock_demand_conn(url):
            demand_conn_call_count[0] += 1
            yield _CapturingConn([])

        monkeypatch.setattr(coverage_agent, "get_conn", mock_pool)
        monkeypatch.setattr(coverage_agent, "get_demand_conn", mock_demand_conn)
        with patch.object(coverage_agent, "vibe") as mock_vibe:
            mock_vibe.make_judge.return_value = None
            coverage_agent.gather_demand(days=14, url="postgresql://fake/db")

        assert pool_call_count[0] == 0
        assert demand_conn_call_count[0] == 1


class TestGatherDemandMultiIntent:
    """Test 6: multi-intent cartesian — REVIEW MEDIUM + ROUND-3."""

    def test_multi_neighborhood_single_cuisine_cross_product(self, monkeypatch) -> None:
        from scripts import coverage_agent

        # "italian in Mission District and North Beach" should produce 2 pairs
        rows = [("italian in Mission District and North Beach", ["Italian Restaurant"])]
        monkeypatch.setattr(coverage_agent, "get_conn", lambda: _stub_get_conn(rows))

        with patch.object(coverage_agent, "vibe") as mock_vibe:
            mock_vibe.make_judge.return_value = None
            demand_counts, rows_scanned, unmapped_count = coverage_agent.gather_demand(days=14)

        assert demand_counts.get(("Mission District", "italian"), 0) == 1
        assert demand_counts.get(("North Beach", "italian"), 0) == 1


class TestGatherDemandRound3HighLexical:
    """Test 7: ROUND-3 HIGH — empty requested_primary_types, cuisine from message."""

    def test_free_text_row_maps_via_message_lexical(self, monkeypatch) -> None:
        from scripts import coverage_agent

        # The free-text case: app/main.py returns requested_primary_types=[]
        # The cuisine must be recovered from the message via _lexical_cuisines
        rows = [("vietnamese restaurants in Outer Sunset", [])]
        monkeypatch.setattr(coverage_agent, "get_conn", lambda: _stub_get_conn(rows))

        with patch.object(coverage_agent, "vibe") as mock_vibe:
            mock_vibe.make_judge.return_value = None
            demand_counts, rows_scanned, unmapped_count = coverage_agent.gather_demand(days=14)

        assert demand_counts.get(("Outer Sunset", "vietnamese"), 0) == 1
        assert unmapped_count == 0, "Lexically-resolved row must NOT land in unmapped_count"


class TestGatherDemandRound3HighLlm:
    """Test 8: ROUND-3 HIGH — cuisine resolved via LLM when message lexical misses."""

    def test_llm_resolves_paraphrase_cuisine(self, monkeypatch) -> None:
        from scripts import coverage_agent

        # A row with empty types and a message whose cuisine is NOT in lexical CUISINES
        # (paraphrase — e.g. "pho place" doesn't contain "vietnamese" literally)
        rows = [("a pho place in Outer Sunset", [])]
        monkeypatch.setattr(coverage_agent, "get_conn", lambda: _stub_get_conn(rows))

        # The LLM returns the catalog cuisine for the paraphrase
        llm_response = json.dumps([{"neighborhoods": ["Outer Sunset"], "cuisines": ["vietnamese"]}])
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = llm_response

        with patch.object(coverage_agent, "vibe") as mock_vibe:
            mock_vibe.make_judge.return_value = mock_llm
            demand_counts, rows_scanned, unmapped_count = coverage_agent.gather_demand(days=14)

        assert demand_counts.get(("Outer Sunset", "vietnamese"), 0) == 1
        # The batched extractor must have been invoked
        assert mock_llm.invoke.call_count >= 1


class TestGatherDemandJudgeNone:
    """Test 9: judge None — lexically resolved rows still count (ROUND-2 MEDIUM-3 + ROUND-3)."""

    def test_lexical_rows_count_when_judge_none(self, monkeypatch) -> None:
        from scripts import coverage_agent

        rows = [
            # Lexical hit on both axes (via _types_to_cuisines + _lexical_neighborhoods)
            ("dinner in Noe Valley", ["Korean Restaurant"]),
            # Lexical hit via _lexical_cuisines + _lexical_neighborhoods
            ("vietnamese food in Outer Sunset", []),
            # Lexical miss — needs LLM but judge is None → unmapped
            ("a great pho place near the park", []),
        ]
        monkeypatch.setattr(coverage_agent, "get_conn", lambda: _stub_get_conn(rows))

        with patch.object(coverage_agent, "vibe") as mock_vibe:
            mock_vibe.make_judge.return_value = None
            demand_counts, rows_scanned, unmapped_count = coverage_agent.gather_demand(days=14)

        assert rows_scanned == 3
        # The two lexical rows must map even with judge=None
        assert demand_counts.get(("Noe Valley", "korean"), 0) == 1
        assert demand_counts.get(("Outer Sunset", "vietnamese"), 0) == 1
        # Only the LLM-needed row is unmapped
        assert unmapped_count == 1


class TestGatherDemandSingleBatchCall:
    """Test 10: _extract_demand_batch is called AT MOST ONCE (ROUND-3 — no per-row round-trip)."""

    def test_batch_called_at_most_once(self, monkeypatch) -> None:
        from scripts import coverage_agent

        rows = [
            # Resolves both axes lexically — should NOT go to batch
            ("vietnamese restaurants in Outer Sunset", []),
            # Also resolves lexically
            ("thai food in Mission District", ["Thai Restaurant"]),
            # Needs LLM for cuisine (paraphrase)
            ("pho place in Chinatown", []),
            # Needs LLM for neighborhood
            ("vietnamese noodle shop downtown", []),
        ]
        monkeypatch.setattr(coverage_agent, "get_conn", lambda: _stub_get_conn(rows))

        batch_calls: list[list[str]] = []

        def mock_extract_demand_batch(messages, llm):
            batch_calls.append(messages)
            # Return empty pairs for each message
            return [([], [])] * len(messages)

        monkeypatch.setattr(coverage_agent, "_extract_demand_batch", mock_extract_demand_batch)

        with patch.object(coverage_agent, "vibe") as mock_vibe:
            mock_vibe.make_judge.return_value = MagicMock()
            coverage_agent.gather_demand(days=14)

        assert len(batch_calls) <= 1, (
            f"_extract_demand_batch must be called at most once, got {len(batch_calls)}"
        )


# ---------------------------------------------------------------------------
# Task 3 tests (18-03): DemandGap + gather_pair_supply + find_demand_gaps
#                       + gap_to_seed_query
# ---------------------------------------------------------------------------


class TestPairLevelGate:
    """Test 1 (HIGH-1): TRUE pair-level supply — cuisine city-wide but absent in
    demanded neighborhood IS flagged as a gap (the Outer Sunset / Vietnamese case)."""

    def test_outer_sunset_vietnamese_zero_pair_supply_is_gap(self, monkeypatch) -> None:
        from scripts.coverage_agent import DemandGap, find_demand_gaps

        # pair supply: Outer Sunset/vietnamese = 0 (never ingested for this pair)
        pair_supply = {("Outer Sunset", "vietnamese"): 0}
        demand_counts = {("Outer Sunset", "vietnamese"): 5}

        gaps = find_demand_gaps(demand_counts, pair_supply, min_place_count=5)

        assert len(gaps) == 1
        g = gaps[0]
        assert isinstance(g, DemandGap)
        assert g.neighborhood == "Outer Sunset"
        assert g.cuisine == "vietnamese"
        assert g.place_count == 0
        assert g.demand_count == 5


class TestSaturatedPairExcluded:
    """Test 2: pair with pair_place_count >= min_places is NOT a gap."""

    def test_mission_italian_saturated_not_a_gap(self) -> None:
        from scripts.coverage_agent import find_demand_gaps

        pair_supply = {("Mission District", "italian"): 40}
        demand_counts = {("Mission District", "italian"): 3}

        gaps = find_demand_gaps(demand_counts, pair_supply, min_place_count=5)

        assert gaps == []


class TestDemandGatesGap:
    """Test 3: pair with demand_count == 0 is NOT a gap even if supply is 0."""

    def test_zero_demand_pair_not_a_gap(self) -> None:
        from scripts.coverage_agent import find_demand_gaps

        pair_supply = {("Outer Sunset", "thai"): 0}
        demand_counts = {("Outer Sunset", "thai"): 0}

        gaps = find_demand_gaps(demand_counts, pair_supply, min_place_count=5)

        assert gaps == []


class TestDemandDescendingRanking:
    """Test 4: find_demand_gaps returns gaps ordered by demand_count descending."""

    def test_higher_demand_comes_first(self) -> None:
        from scripts.coverage_agent import find_demand_gaps

        pair_supply = {
            ("Outer Sunset", "vietnamese"): 0,
            ("Mission District", "thai"): 1,
        }
        demand_counts = {
            ("Outer Sunset", "vietnamese"): 5,
            ("Mission District", "thai"): 9,
        }

        gaps = find_demand_gaps(demand_counts, pair_supply, min_place_count=5)

        assert len(gaps) == 2
        assert gaps[0].demand_count == 9
        assert gaps[0].neighborhood == "Mission District"
        assert gaps[1].demand_count == 5


class TestGatherPairSupplySQL:
    """Test 5: gather_pair_supply issues a parameterised SELECT from place_query_hits."""

    def test_sql_shape_and_parameterised_seeds(self) -> None:
        from scripts.coverage_agent import gather_pair_supply

        captured_sql: list[str] = []
        captured_params: list = []

        class StubCursor:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def execute(self, sql, params=None):
                captured_sql.append(sql)
                captured_params.append(params)

            def fetchall(self):
                return []

        class StubConn:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def cursor(self):
                return StubCursor()

        from contextlib import contextmanager

        @contextmanager
        def stub_get_conn():
            yield StubConn()

        import scripts.coverage_agent as ca

        original_get_conn = ca.get_conn
        ca.get_conn = stub_get_conn
        try:
            result = gather_pair_supply([("Outer Sunset", "vietnamese")])
        finally:
            ca.get_conn = original_get_conn

        # Must SELECT from place_query_hits
        assert any("place_query_hits" in sql for sql in captured_sql)
        # Must count DISTINCT place_id
        assert any("DISTINCT place_id" in sql for sql in captured_sql)
        # Seed strings are passed as param, not interpolated
        assert any(params is not None for params in captured_params)
        # pair with no rows returns 0
        assert result.get(("Outer Sunset", "vietnamese"), -1) == 0


class TestDemandGapDataclass:
    """Test 6 (REVIEW MEDIUM): find_demand_gaps returns DemandGap instances with
    explicit fields, not CoverageStat rows with bucket='demand:...' strings."""

    def test_returns_demand_gap_instances(self) -> None:
        from scripts.coverage_agent import DemandGap, find_demand_gaps

        pair_supply = {("Outer Sunset", "vietnamese"): 2}
        demand_counts = {("Outer Sunset", "vietnamese"): 7}

        gaps = find_demand_gaps(demand_counts, pair_supply, min_place_count=5)

        assert len(gaps) == 1
        g = gaps[0]
        assert isinstance(g, DemandGap)
        # Explicit fields — no string parsing required
        assert g.neighborhood == "Outer Sunset"
        assert g.cuisine == "vietnamese"
        assert g.place_count == 2
        assert g.demand_count == 7
        # Must NOT have a 'bucket' attribute encoding "demand:..."
        assert not hasattr(g, "bucket")


class TestSeedFormatExactness:
    """Test 7: gap_to_seed_query returns the exact seed format AND it is a catalog member."""

    def test_outer_sunset_vietnamese_seed(self) -> None:
        from scripts.coverage_agent import gap_to_seed_query
        from scripts.ingest_places_sf import build_seed_queries

        seed = gap_to_seed_query("Outer Sunset", "vietnamese")
        assert seed == "vietnamese restaurants in Outer Sunset San Francisco"
        assert seed in set(build_seed_queries())


class TestSeedFormatOffCatalogRaises:
    """Test 8 (catalog assertion): gap_to_seed_query raises on off-catalog inputs."""

    def test_off_catalog_neighborhood_raises(self) -> None:
        import pytest

        from scripts.coverage_agent import gap_to_seed_query

        with pytest.raises((AssertionError, ValueError, KeyError, RuntimeError)):
            gap_to_seed_query("Berkeley", "vietnamese")

    def test_off_catalog_cuisine_raises(self) -> None:
        import pytest

        from scripts.coverage_agent import gap_to_seed_query

        with pytest.raises((AssertionError, ValueError, KeyError, RuntimeError)):
            gap_to_seed_query("Outer Sunset", "fusion")


# ---------------------------------------------------------------------------
# Task 4 tests (18-03): ingested_query_texts + insert_pending conn= +
#                       gap_mine_main + sandbox guard + cold-start
# ---------------------------------------------------------------------------


class _MultiQueryConn:
    """A stub connection that can serve different rows to different SELECT calls.

    The ``rows_by_query`` dict maps a substring of the SQL to the rows to return.
    If no key matches, returns [].
    """

    def __init__(self, rows_by_query: dict[str, list[tuple]]) -> None:
        self._rows_by_query = rows_by_query
        self.execute_calls: list[tuple[str, object]] = []
        self.insert_calls: list[tuple[str, object]] = []
        self.committed = False
        # current_database() stub
        self._dbname = "city_concierge_sandbox"

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def cursor(self):
        return _MultiQueryCursor(self)

    def commit(self):
        self.committed = True


class _MultiQueryCursor:
    def __init__(self, conn: _MultiQueryConn) -> None:
        self._conn = conn
        self._rows: list[tuple] = []
        self.rowcount = 1

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def execute(self, sql: str, params=None) -> None:
        self._conn.execute_calls.append((sql, params))
        # Dispatch rows based on SQL fragment
        self._rows = []
        for key, rows in self._conn._rows_by_query.items():
            if key in sql:
                self._rows = rows
                break
        # Support current_database() call from sandbox guard
        if "current_database" in sql:
            self._rows = [(self._conn._dbname,)]
        # Track INSERT calls
        if "INSERT" in sql.upper():
            self._conn.insert_calls.append((sql, params))

    def fetchall(self) -> list[tuple]:
        return self._rows

    def fetchone(self) -> tuple | None:
        return self._rows[0] if self._rows else None


def _make_multi_conn(
    checkpoint_rows: list[tuple] = (),
    proposal_rows: list[tuple] = (),
    pqh_rows: list[tuple] = (),
    dbname: str = "city_concierge_sandbox",
) -> _MultiQueryConn:
    """Build a multi-query stub connection with preset rows for each table."""
    conn = _MultiQueryConn(
        rows_by_query={
            "places_ingest_query_checkpoints": list(checkpoint_rows),
            "places_ingest_query_proposals": list(proposal_rows),
            "place_query_hits": list(pqh_rows),
            "user_query_log": [],
        }
    )
    conn._dbname = dbname
    return conn


class _StatusAwareCheckpointCursor:
    """Cursor stub that returns an `incomplete` checkpoint row ONLY when the
    SELECT omits the `status = 'completed'` predicate (CDX-L1)."""

    def __init__(self, prefixed_row: str) -> None:
        self._prefixed_row = prefixed_row
        self._rows: list[tuple] = []

    def __enter__(self) -> _StatusAwareCheckpointCursor:
        return self

    def __exit__(self, *a: object) -> None:
        pass

    def execute(self, sql: str, params: object = None) -> None:
        if "places_ingest_query_checkpoints" in sql:
            # The incomplete row leaks only if the completed-status filter is absent.
            self._rows = [] if "status = 'completed'" in sql else [(self._prefixed_row,)]
        else:
            self._rows = []

    def fetchall(self) -> list[tuple]:
        return self._rows

    def fetchone(self) -> tuple | None:
        return self._rows[0] if self._rows else None


class _StatusAwareCheckpointConn:
    """Connection stub backing the CDX-L1 status-filter test."""

    _dbname = "city_concierge_sandbox"

    def __init__(self, prefixed_row: str) -> None:
        self._prefixed_row = prefixed_row

    def cursor(self) -> _StatusAwareCheckpointCursor:
        return _StatusAwareCheckpointCursor(self._prefixed_row)


class TestIngestedQueryTextsHigh2:
    """Test 1 (HIGH-2): a catalog-valid, not-yet-ingested proposal SURVIVES
    filter_already_covered when deduping against ingested_query_texts (not
    existing_query_texts which includes the static catalog)."""

    def test_catalog_valid_gap_survives_ingested_dedup(self, monkeypatch) -> None:
        from scripts.coverage_agent import (
            ProposedQuery,
            filter_already_covered,
            ingested_query_texts,
        )

        # No checkpoints, no proposals → ingested set is empty
        conn = _make_multi_conn()

        ingested = ingested_query_texts(conn)

        # A known catalog seed that has NOT been ingested
        seed = "vietnamese restaurants in Outer Sunset San Francisco"
        # Must NOT be in the ingested set (it is in build_seed_queries() but not ingested)
        assert seed not in ingested

        proposal = ProposedQuery(seed, "enriched", "test")
        kept, dropped = filter_already_covered([proposal], ingested)
        assert proposal in kept
        assert proposal not in dropped


class TestIngestedQueryTextsExcludesCatalog:
    """Test 2: ingested_query_texts does NOT include build_seed_queries() members."""

    def test_catalog_only_seed_absent_from_ingested(self) -> None:
        from scripts.coverage_agent import ingested_query_texts
        from scripts.ingest_places_sf import build_seed_queries

        # No rows in either table
        conn = _make_multi_conn()
        ingested = ingested_query_texts(conn)

        # Any catalog seed must be absent from ingested (it's not in DB tables)
        catalog_seeds = set(build_seed_queries())
        assert len(ingested & catalog_seeds) == 0, (
            "ingested_query_texts must NOT include static catalog seeds"
        )


class TestCheckpointPrefixDedup:
    """Test 3 (ROUND-2 NEW HIGH): completed checkpoint with FIELD_MODE:: prefix is
    normalized — the raw seed is in ingested set, so the mined proposal is deduped."""

    def test_prefixed_completed_checkpoint_dedupes_raw_seed(self, monkeypatch) -> None:
        from scripts.coverage_agent import (
            ProposedQuery,
            filter_already_covered,
            ingested_query_texts,
        )

        raw_seed = "vietnamese restaurants in Outer Sunset San Francisco"
        prefixed = f"all::{raw_seed}"

        # Completed checkpoint with FIELD_MODE:: prefix
        conn = _make_multi_conn(checkpoint_rows=[(prefixed,), ("status_col_dummy",)])
        # Override _rows_by_query to also have status='completed' filter support
        conn._rows_by_query["places_ingest_query_checkpoints"] = [(prefixed,)]
        conn._rows_by_query["places_ingest_query_proposals"] = []

        ingested = ingested_query_texts(conn)

        # The RAW seed (prefix stripped) must be in ingested
        assert raw_seed in ingested, (
            f"Expected {raw_seed!r} to be in ingested after prefix normalization"
        )

        # No-:: row is returned as-is (defensive)
        conn2 = _make_multi_conn(checkpoint_rows=[("no_prefix_seed",)])
        conn2._rows_by_query["places_ingest_query_checkpoints"] = [("no_prefix_seed",)]
        conn2._rows_by_query["places_ingest_query_proposals"] = []
        ingested2 = ingested_query_texts(conn2)
        assert "no_prefix_seed" in ingested2

        # Proposal for the already-ingested seed is DEDUPED
        proposal = ProposedQuery(raw_seed, "enriched", "test")
        kept, dropped = filter_already_covered([proposal], ingested)
        assert proposal in dropped
        assert proposal not in kept


class TestCheckpointStatusFilter:
    """Test 4 (ROUND-3 MEDIUM): incomplete checkpoint does NOT contribute to
    ingested set — OPPOSITE of Test 3 (completed → deduped; incomplete → kept)."""

    def test_incomplete_checkpoint_does_not_dedupe(self, monkeypatch) -> None:
        from scripts.coverage_agent import (
            ProposedQuery,
            filter_already_covered,
            ingested_query_texts,
        )

        raw_seed = "vietnamese restaurants in Outer Sunset San Francisco"
        prefixed = f"all::{raw_seed}"

        # Status-aware stub: this checkpoint row is `incomplete`, so it is
        # returned ONLY when the SELECT does NOT constrain `status = 'completed'`.
        # If ingested_query_texts correctly carries the completed-status filter,
        # the row is suppressed and the seed is absent. If the filter were ever
        # dropped, the row would leak and this test would FAIL — proving the
        # filter exists, not merely that an empty stub returned nothing (CDX-L1).
        ingested = ingested_query_texts(_StatusAwareCheckpointConn(prefixed))

        # The seed must be ABSENT (incomplete checkpoint filtered out by status)
        assert raw_seed not in ingested

        # Proposal for this seed is NOT deduped — it lands in kept
        proposal = ProposedQuery(raw_seed, "enriched", "test")
        kept, dropped = filter_already_covered([proposal], ingested)
        assert proposal in kept
        assert proposal not in dropped


class TestSameConnectionGuardAndInsert:
    """Test 5 (ROUND-3 LOW): gap_mine_main uses the SAME conn object for both
    assert_sandbox_write_target(conn) and insert_pending(..., conn=conn)."""

    def test_same_conn_instance_for_guard_and_insert(self, monkeypatch) -> None:
        import scripts.coverage_agent as ca

        seen_guard_conns: list = []
        seen_insert_conns: list = []

        def fake_guard(conn=None):
            seen_guard_conns.append(conn)

        def fake_insert(proposals, dry_run, conn=None):
            seen_insert_conns.append(conn)
            return 0

        monkeypatch.setattr(ca, "assert_sandbox_write_target", fake_guard)
        monkeypatch.setattr(ca, "insert_pending", fake_insert)

        # Stub gather_demand to return one mappable pair
        monkeypatch.setattr(
            ca,
            "gather_demand",
            lambda days, url=None: ({("Outer Sunset", "vietnamese"): 5}, 1, 0),
        )
        # Stub gather_pair_supply to return 0 (gap)
        monkeypatch.setattr(
            ca, "gather_pair_supply", lambda pairs, conn=None: {p: 0 for p in pairs}
        )
        # Stub ingested_query_texts to return empty set (no dedup)
        monkeypatch.setattr(ca, "ingested_query_texts", lambda conn: set())
        # Stub log_to_mlflow
        monkeypatch.setattr(ca, "log_to_mlflow", lambda *a, **kw: None)

        # Stub get_conn to yield a single traceable object
        sentinel_conn = _make_multi_conn()

        from contextlib import contextmanager

        @contextmanager
        def fake_get_conn():
            yield sentinel_conn

        monkeypatch.setattr(ca, "get_conn", fake_get_conn)

        ca.gap_mine_main([])

        # Both guard and insert must have been called with the SAME conn object
        assert len(seen_guard_conns) >= 1, "assert_sandbox_write_target not called"
        assert len(seen_insert_conns) >= 1, "insert_pending not called"
        assert seen_guard_conns[0] is sentinel_conn, "guard used wrong conn"
        assert seen_insert_conns[0] is sentinel_conn, "insert used wrong conn"


class TestInsertPendingBackwardCompat:
    """Test 6: insert_pending without conn= still self-opens get_conn (backward compat)."""

    def test_no_conn_self_opens_get_conn(self, monkeypatch) -> None:
        import scripts.coverage_agent as ca

        get_conn_calls = [0]

        class FakeCursor:
            rowcount = 1

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def execute(self, sql, params=None):
                pass

        class FakeConn:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def cursor(self):
                return FakeCursor()

            def commit(self):
                pass

        from contextlib import contextmanager

        @contextmanager
        def fake_get_conn():
            get_conn_calls[0] += 1
            yield FakeConn()

        monkeypatch.setattr(ca, "get_conn", fake_get_conn)

        from scripts.coverage_agent import ProposedQuery, insert_pending

        proposals = [ProposedQuery("some query", "enriched", "reason")]
        insert_pending(proposals, dry_run=False)  # no conn= kwarg

        assert get_conn_calls[0] == 1, "Expected insert_pending to self-open get_conn"


class TestGuardImportedNotRedefined:
    """Test 7 (HIGH-3): assert_sandbox_write_target is imported from scripts.sandbox_guard,
    not redefined; raising it prevents any inserts."""

    def test_guard_raise_prevents_inserts(self, monkeypatch) -> None:
        import scripts.coverage_agent as ca

        insert_called = [False]

        def raising_guard(conn=None):
            raise RuntimeError("not sandbox — refused")

        monkeypatch.setattr(ca, "assert_sandbox_write_target", raising_guard)

        def should_not_be_called(proposals, dry_run, conn=None):
            insert_called[0] = True
            return 0

        monkeypatch.setattr(ca, "insert_pending", should_not_be_called)
        monkeypatch.setattr(
            ca,
            "gather_demand",
            lambda days, url=None: ({("Outer Sunset", "vietnamese"): 5}, 1, 0),
        )
        monkeypatch.setattr(
            ca, "gather_pair_supply", lambda pairs, conn=None: {p: 0 for p in pairs}
        )
        monkeypatch.setattr(ca, "ingested_query_texts", lambda conn: set())
        monkeypatch.setattr(ca, "log_to_mlflow", lambda *a, **kw: None)

        from contextlib import contextmanager

        @contextmanager
        def fake_get_conn():
            yield _make_multi_conn()

        monkeypatch.setattr(ca, "get_conn", fake_get_conn)

        import pytest

        with pytest.raises(RuntimeError, match="not sandbox"):
            ca.gap_mine_main([])

        assert not insert_called[0], "insert_pending must not be called when guard raises"

    def test_guard_runs_before_insert(self, monkeypatch) -> None:
        """Assert call ORDER: guard before insert."""
        import scripts.coverage_agent as ca

        call_order: list[str] = []

        def tracking_guard(conn=None):
            call_order.append("guard")

        def tracking_insert(proposals, dry_run, conn=None):
            call_order.append("insert")
            return 0

        monkeypatch.setattr(ca, "assert_sandbox_write_target", tracking_guard)
        monkeypatch.setattr(ca, "insert_pending", tracking_insert)
        monkeypatch.setattr(
            ca,
            "gather_demand",
            lambda days, url=None: ({("Outer Sunset", "vietnamese"): 3}, 1, 0),
        )
        monkeypatch.setattr(
            ca, "gather_pair_supply", lambda pairs, conn=None: {p: 0 for p in pairs}
        )
        monkeypatch.setattr(ca, "ingested_query_texts", lambda conn: set())
        monkeypatch.setattr(ca, "log_to_mlflow", lambda *a, **kw: None)

        from contextlib import contextmanager

        @contextmanager
        def fake_get_conn():
            yield _make_multi_conn()

        monkeypatch.setattr(ca, "get_conn", fake_get_conn)

        ca.gap_mine_main([])

        assert call_order.index("guard") < call_order.index("insert"), (
            "assert_sandbox_write_target must be called before insert_pending"
        )


class TestColdStart:
    """Test 9 (D-04): empty demand → gaps_found=0 logged, return 0, zero inserts."""

    def test_empty_demand_returns_0_no_inserts(self, monkeypatch) -> None:
        import scripts.coverage_agent as ca

        insert_called = [False]
        logged_metrics: dict = {}

        def fake_guard(conn=None):
            pass  # allow (sandbox)

        def fake_insert(proposals, dry_run, conn=None):
            insert_called[0] = True
            return 0

        monkeypatch.setattr(ca, "assert_sandbox_write_target", fake_guard)
        monkeypatch.setattr(ca, "insert_pending", fake_insert)
        monkeypatch.setattr(ca, "gather_demand", lambda days, url=None: ({}, 0, 0))
        monkeypatch.setattr(ca, "gather_pair_supply", lambda pairs, conn=None: {})
        monkeypatch.setattr(ca, "ingested_query_texts", lambda conn: set())

        import mlflow as _mlflow

        def fake_log_metric(key, val):
            logged_metrics[key] = val

        monkeypatch.setattr(_mlflow, "log_metric", fake_log_metric)
        monkeypatch.setattr(ca, "log_to_mlflow", lambda *a, **kw: None)

        from contextlib import contextmanager

        @contextmanager
        def fake_get_conn():
            yield _make_multi_conn()

        monkeypatch.setattr(ca, "get_conn", fake_get_conn)

        result = ca.gap_mine_main([])
        assert result == 0
        assert not insert_called[0], "insert_pending must not be called on cold start"


class TestHappyPath:
    """Test 10: happy path produces a proposal with the exact gap_to_seed_query output."""

    def test_proposal_query_text_equals_seed_format(self, monkeypatch) -> None:
        import scripts.coverage_agent as ca

        inserted_proposals: list = []

        def fake_guard(conn=None):
            pass

        def fake_insert(proposals, dry_run, conn=None):
            inserted_proposals.extend(proposals)
            return len(proposals)

        monkeypatch.setattr(ca, "assert_sandbox_write_target", fake_guard)
        monkeypatch.setattr(ca, "insert_pending", fake_insert)
        monkeypatch.setattr(
            ca,
            "gather_demand",
            lambda days, url=None: ({("Outer Sunset", "vietnamese"): 5}, 1, 0),
        )
        monkeypatch.setattr(
            ca, "gather_pair_supply", lambda pairs, conn=None: {p: 0 for p in pairs}
        )
        monkeypatch.setattr(ca, "ingested_query_texts", lambda conn: set())
        monkeypatch.setattr(ca, "log_to_mlflow", lambda *a, **kw: None)

        from contextlib import contextmanager

        @contextmanager
        def fake_get_conn():
            yield _make_multi_conn()

        monkeypatch.setattr(ca, "get_conn", fake_get_conn)

        result = ca.gap_mine_main([])
        assert result == 0

        assert len(inserted_proposals) >= 1
        expected_seed = "vietnamese restaurants in Outer Sunset San Francisco"
        assert any(p.query_text == expected_seed for p in inserted_proposals), (
            f"Expected proposal with query_text={expected_seed!r}; got: "
            f"{[p.query_text for p in inserted_proposals]}"
        )


class TestDryRun:
    """Test 11: --dry-run runs the full path but insert_pending inserts nothing."""

    def test_dry_run_zero_inserts(self, monkeypatch) -> None:
        import scripts.coverage_agent as ca

        insert_dry_run_values: list[bool] = []

        def fake_guard(conn=None):
            pass

        def fake_insert(proposals, dry_run, conn=None):
            insert_dry_run_values.append(dry_run)
            return 0  # nothing inserted on dry run

        monkeypatch.setattr(ca, "assert_sandbox_write_target", fake_guard)
        monkeypatch.setattr(ca, "insert_pending", fake_insert)
        monkeypatch.setattr(
            ca,
            "gather_demand",
            lambda days, url=None: ({("Outer Sunset", "vietnamese"): 5}, 1, 0),
        )
        monkeypatch.setattr(
            ca, "gather_pair_supply", lambda pairs, conn=None: {p: 0 for p in pairs}
        )
        monkeypatch.setattr(ca, "ingested_query_texts", lambda conn: set())
        monkeypatch.setattr(ca, "log_to_mlflow", lambda *a, **kw: None)

        from contextlib import contextmanager

        @contextmanager
        def fake_get_conn():
            yield _make_multi_conn()

        monkeypatch.setattr(ca, "get_conn", fake_get_conn)

        result = ca.gap_mine_main(["--dry-run"])
        assert result == 0
        # insert_pending called with dry_run=True
        assert len(insert_dry_run_values) >= 1
        assert all(v is True for v in insert_dry_run_values)


class TestTopNAfterDedup:
    """Test 12: --top-n is applied AFTER filter_already_covered (post-dedup cap)."""

    def test_top_n_caps_post_dedup_list(self, monkeypatch) -> None:
        import scripts.coverage_agent as ca

        inserted_proposals: list = []

        def fake_guard(conn=None):
            pass

        def fake_insert(proposals, dry_run, conn=None):
            inserted_proposals.extend(proposals)
            return len(proposals)

        monkeypatch.setattr(ca, "assert_sandbox_write_target", fake_guard)
        monkeypatch.setattr(ca, "insert_pending", fake_insert)

        # 3 demand gaps surviving dedup — only 2 should be inserted with --top-n 2
        demand = {
            ("Outer Sunset", "vietnamese"): 9,
            ("Mission District", "thai"): 7,
            ("Castro", "italian"): 5,
        }
        monkeypatch.setattr(ca, "gather_demand", lambda days, url=None: (demand, 3, 0))
        monkeypatch.setattr(
            ca, "gather_pair_supply", lambda pairs, conn=None: {p: 0 for p in pairs}
        )
        monkeypatch.setattr(ca, "ingested_query_texts", lambda conn: set())
        monkeypatch.setattr(ca, "log_to_mlflow", lambda *a, **kw: None)

        from contextlib import contextmanager

        @contextmanager
        def fake_get_conn():
            yield _make_multi_conn()

        monkeypatch.setattr(ca, "get_conn", fake_get_conn)

        ca.gap_mine_main(["--top-n", "2"])

        # Only 2 proposals must have been inserted
        assert len(inserted_proposals) == 2, (
            f"Expected 2 proposals after --top-n 2; got {len(inserted_proposals)}"
        )


class TestMLflowDemandMetrics:
    """Test 13: log_to_mlflow logs demand_rows_scanned, unmapped_count, gaps_found,
    proposals_inserted, and a demand_gaps.json artifact."""

    def test_demand_metrics_logged(self, monkeypatch) -> None:

        logged_metrics: dict[str, float] = {}
        logged_dicts: dict[str, object] = {}

        import mlflow as _mlflow

        monkeypatch.setattr(_mlflow, "set_experiment", lambda *a, **kw: None)
        monkeypatch.setattr(_mlflow, "log_param", lambda *a, **kw: None)
        monkeypatch.setattr(_mlflow, "log_dict", lambda d, name: logged_dicts.update({name: d}))
        monkeypatch.setattr(
            _mlflow, "log_metric", lambda key, val: logged_metrics.update({key: val})
        )

        class FakeRun:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

        monkeypatch.setattr(_mlflow, "start_run", lambda **kw: FakeRun())

        from scripts.coverage_agent import (
            DemandGap,
            ProposedQuery,
            log_to_mlflow,
        )

        gaps = [DemandGap("Outer Sunset", "vietnamese", 0, 5)]
        proposals = [
            ProposedQuery(
                "vietnamese restaurants in Outer Sunset San Francisco", "enriched", "test"
            )
        ]

        log_to_mlflow(
            stats=[],
            gaps=[],
            proposals=proposals,
            dropped=[],
            inserted=1,
            dry_run=False,
            demand_rows_scanned=10,
            unmapped_count=2,
            demand_gaps=gaps,
        )

        assert "demand_rows_scanned" in logged_metrics, "Missing demand_rows_scanned metric"
        assert "unmapped_count" in logged_metrics, "Missing unmapped_count metric"
        assert "proposals_inserted" in logged_metrics, "Missing proposals_inserted metric"
        assert "demand_gaps.json" in logged_dicts, "Missing demand_gaps.json artifact"


class TestJudgeNoneStillMinesLexical:
    """Test 14 (ROUND-2 MEDIUM-3): judge=None with lexically-mappable demand still
    produces a gap and reaches insert_pending with the exact seed."""

    def test_judge_none_lexical_demand_mines(self, monkeypatch) -> None:
        import scripts.coverage_agent as ca

        inserted_proposals: list = []

        def fake_guard(conn=None):
            pass

        def fake_insert(proposals, dry_run, conn=None):
            inserted_proposals.extend(proposals)
            return len(proposals)

        monkeypatch.setattr(ca, "assert_sandbox_write_target", fake_guard)
        monkeypatch.setattr(ca, "insert_pending", fake_insert)

        # gather_demand (already-tested) returns lexically-mappable demand
        # even when judge=None; we stub it here to isolate gap_mine_main behavior
        monkeypatch.setattr(
            ca,
            "gather_demand",
            lambda days, url=None: ({("Outer Sunset", "vietnamese"): 3}, 1, 0),
        )
        monkeypatch.setattr(
            ca, "gather_pair_supply", lambda pairs, conn=None: {p: 0 for p in pairs}
        )
        monkeypatch.setattr(ca, "ingested_query_texts", lambda conn: set())
        monkeypatch.setattr(ca, "log_to_mlflow", lambda *a, **kw: None)

        from contextlib import contextmanager

        @contextmanager
        def fake_get_conn():
            yield _make_multi_conn()

        monkeypatch.setattr(ca, "get_conn", fake_get_conn)

        # Simulate judge being None
        with patch.object(ca, "vibe") as mock_vibe:
            mock_vibe.make_judge.return_value = None
            result = ca.gap_mine_main([])

        assert result == 0
        expected_seed = "vietnamese restaurants in Outer Sunset San Francisco"
        assert any(p.query_text == expected_seed for p in inserted_proposals), (
            "judge=None must not suppress lexically-mappable demand"
        )
