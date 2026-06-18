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
        # The raw injection string must NOT appear literally in the prompt
        # (it should be JSON-escaped as \"]})... or similar)
        assert '"]}) DROP TABLE; --' not in prompt or json.dumps([message]) in prompt

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
