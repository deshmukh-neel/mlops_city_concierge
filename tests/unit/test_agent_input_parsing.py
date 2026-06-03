from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.agent.input_parsing import (
    SlotExtractionResult,
    explicit_num_stops_from_conversation,
    explicit_num_stops_from_text,
    has_slot_structure,
    is_refinement_request,
    parse_closure_decision,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("plan a 3-stop omakase date night", 3),
        ("make it 4 stops", 4),
        ("three spots near Japantown", 3),
        ("just find sushi", None),
    ],
)
def test_explicit_num_stops_from_text(text: str, expected: int | None) -> None:
    assert explicit_num_stops_from_text(text) == expected


@dataclass
class _Msg:
    role: str
    content: str


def test_conversation_uses_history_when_current_message_lacks_count() -> None:
    history = [_Msg("user", "plan a 3-stop omakase date night"), _Msg("assistant", "Done.")]
    assert explicit_num_stops_from_conversation(history, "make it cheaper") == 3


def test_conversation_current_message_overrides_history() -> None:
    history = [_Msg("user", "plan a 3-stop omakase date night"), _Msg("assistant", "Done.")]
    # Conservative parser requires a stop-noun; "make it 4" alone wouldn't match.
    assert explicit_num_stops_from_conversation(history, "actually make it 4 stops") == 4


def test_conversation_ignores_assistant_messages() -> None:
    """Assistant might echo numbers; don't latch onto them."""
    history = [_Msg("user", "plan a date"), _Msg("assistant", "I'll find 5 spots")]
    assert explicit_num_stops_from_conversation(history, "go ahead") is None


def test_conversation_latest_user_count_wins_when_user_revises_twice() -> None:
    history = [
        _Msg("user", "plan a 3-stop omakase date night"),
        _Msg("assistant", "Done."),
        _Msg("user", "actually 5 places"),
        _Msg("assistant", "Updated."),
    ]
    assert explicit_num_stops_from_conversation(history, "go ahead") == 5


def test_conversation_returns_none_when_neither_history_nor_message_has_count() -> None:
    history = [_Msg("user", "plan a date"), _Msg("assistant", "How many?")]
    assert explicit_num_stops_from_conversation(history, "you decide") is None


def test_conversation_bare_number_after_stops_question_is_count() -> None:
    """The user replies with just '3' after the assistant asks 'how many stops?'.
    Without this, the deterministic count guardrail never fires and gpt-4o-mini
    can over-loop, surfacing as 'I hit the planning step limit'."""
    history = [
        _Msg("user", "plan me a date in mission, gf likes omakase"),
        _Msg(
            "assistant",
            "How many stops would you like for your date? Would you prefer just "
            "dinner, or would you like to include drinks or dessert afterward?",
        ),
    ]
    assert explicit_num_stops_from_conversation(history, "3") == 3


def test_conversation_bare_word_number_after_stops_question_is_count() -> None:
    """Same idea but the user wrote 'three' instead of '3'."""
    history = [
        _Msg("user", "plan a date"),
        _Msg("assistant", "Sure — how many stops would you like?"),
    ]
    assert explicit_num_stops_from_conversation(history, "three") == 3


def test_conversation_bare_number_ignored_when_assistant_didnt_ask_about_stops() -> None:
    """A bare number unrelated to a stops-question should NOT latch — the user
    might be saying 'arrive at 3' or 'I'm 30'. Only activate when the prior
    assistant turn was clearly a stops-count question."""
    history = [
        _Msg("user", "plan a date"),
        _Msg("assistant", "Got it. What neighborhood?"),
    ]
    assert explicit_num_stops_from_conversation(history, "3") is None


def test_conversation_bare_number_out_of_range_ignored() -> None:
    """Even after a stops question, '47' isn't a plausible stop count.
    Bound to the same 1..10 range the noun-based parser uses."""
    history = [
        _Msg("user", "plan a date"),
        _Msg("assistant", "How many stops would you like?"),
    ]
    assert explicit_num_stops_from_conversation(history, "47") is None


def test_conversation_bare_number_after_question_loses_to_history_explicit_count() -> None:
    """If the user said '3 stops' earlier AND the assistant later asks again
    and the user replies with bare '4', the LATEST hit (current message=4)
    still wins — same precedence rule as `actually make it 4 stops`."""
    history = [
        _Msg("user", "plan a 3-stop date"),
        _Msg("assistant", "Done."),
        _Msg("user", "make it different"),
        _Msg("assistant", "Sure, how many stops do you want now?"),
    ]
    assert explicit_num_stops_from_conversation(history, "4") == 4


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # accept — first-token rule, anything starting with a yes-ish token
        ("yes", "accept"),
        ("Yes", "accept"),
        ("yes!", "accept"),
        ("yes, make it 4 stops", "accept"),
        ("yeah do it", "accept"),
        ("yep", "accept"),
        ("sure thing", "accept"),
        ("ok let's go", "accept"),
        ("okay", "accept"),
        ("y", "accept"),
        ("👍", "accept"),
        # decline
        ("no", "decline"),
        ("No thanks", "decline"),
        ("nope", "decline"),
        ("nah", "decline"),
        ("n", "decline"),
        # alternative — anything else
        ("find something cheaper instead", "alternative"),
        ("what about ramen?", "alternative"),
        ("pick a different one", "alternative"),
        ("", "alternative"),
        ("   ", "alternative"),
    ],
)
def test_parse_closure_decision(text: str, expected: str) -> None:
    assert parse_closure_decision(text) == expected


# ─── has_slot_structure (Phase 4 / D-04-01) ──────────────────────────────


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Positives — comma-separated vocab list (2+ distinct vocab words)
        ("dinner, drinks, dessert", True),
        ("dinner, drinks, dessert in Hayes Valley around 7pm Saturday", True),
        # Positives — "X then Y [then Z]" pattern
        ("omakase then ramen then dessert", True),
        ("omakase followed by ramen", True),
        # Positives — numbered structure
        ("1. dinner spot 2. drinks", True),
        ("1) dinner spot 2) drinks", True),
        # Positives — vocab + planning verb (omakase + plan an / 3 stops shape)
        ("Plan an omakase night in the Mission, 3 stops", True),
        # Negatives — single-slot free text
        ("plan me a date", False),
        ("find good tacos in the mission", False),
        ("affordable italian", False),
        # Edge — empty string
        ("", False),
        # Edge — repeated SAME vocab word should not count as 2 distinct
        ("dinner dinner dinner", False),
        # Negative — non-vocab comma list (e.g. address components)
        ("San Francisco, CA, USA", False),
    ],
)
def test_has_slot_structure(text: str, expected: bool) -> None:
    assert has_slot_structure(text) is expected


def test_has_slot_structure_regexes_compile_at_module_load() -> None:
    """The regex constants must be module-level (compiled once) — verifiable
    by importing the module and checking the attribute is a compiled pattern,
    not a function call result."""
    import re as _re

    from app.agent import input_parsing as _ip

    assert isinstance(_ip._THEN_PATTERN_RE, _re.Pattern)
    assert isinstance(_ip._NUMBERED_SLOT_RE, _re.Pattern)
    assert isinstance(_ip._SLOT_VOCAB_RE, _re.Pattern)


def test_slot_extraction_result_default_constructs_with_empty_list() -> None:
    s = SlotExtractionResult()
    assert s.requested_primary_types == []


def test_slot_extraction_result_accepts_title_case_primary_types() -> None:
    s = SlotExtractionResult(
        requested_primary_types=["Sushi Restaurant", "Cocktail Bar", "Dessert Shop"]
    )
    assert s.requested_primary_types == ["Sushi Restaurant", "Cocktail Bar", "Dessert Shop"]


# ─── is_refinement_request (Phase 6 / D-06-03) ───────────────────────────


class TestIsRefinementRequest:
    """Deterministic regex pre-check that gates structured-plan injection.

    Per D-06-03 the helper is conservative on purpose — false negatives are
    cheaper than false positives because a false positive on turn 1 would
    clobber first-turn behavior (REF-04). These tests pin the conservative
    contract: the four pattern families fire, first-turn intent does not,
    and the 1-indexed target slot is extracted correctly.
    """

    def test_returns_true_with_slot_for_canonical_refinement_cheaper(self) -> None:
        # Exact scenario string from configs/eval_queries.yaml refinement_cheaper.
        assert is_refinement_request("make stop 2 cheaper") == (True, 2)

    @pytest.mark.parametrize(
        "modifier",
        ["cheaper", "different", "fancier", "earlier", "later"],
    )
    def test_make_stop_n_patterns_match_all_modifiers(self, modifier: str) -> None:
        """All five modifiers documented in D-06-03 fire for 'make stop N <mod>'."""
        assert is_refinement_request(f"make stop 2 {modifier}") == (True, 2)

    @pytest.mark.parametrize(
        ("text", "expected_slot"),
        [
            ("stop 3 cheaper", 3),
            ("stop 1 different", 1),
            ("stop 2 instead a dessert spot", 2),
            ("stop 4 fancier", 4),
            ("stop 2 earlier", 2),
            ("stop 1 later", 1),
        ],
    )
    def test_stop_n_bare_patterns_match(self, text: str, expected_slot: int) -> None:
        """Bare 'stop N <mod|instead>' (no leading 'make') still triggers."""
        assert is_refinement_request(text) == (True, expected_slot)

    @pytest.mark.parametrize(
        ("text", "expected_slot"),
        [
            ("swap stop 1", 1),
            ("swap stop 3", 3),
        ],
    )
    def test_swap_stop_n_matches(self, text: str, expected_slot: int) -> None:
        """'swap stop N' is a refinement (replace the candidate at slot N)."""
        assert is_refinement_request(text) == (True, expected_slot)

    @pytest.mark.parametrize(
        ("text", "expected_slot"),
        [
            ("instead for stop 2", 2),
            ("different for stop 1", 1),
            ("instead in stop 3", 3),
            ("different in stop 4", 4),
        ],
    )
    def test_instead_for_stop_n_matches(self, text: str, expected_slot: int) -> None:
        """'(instead|different) (for|in) stop N' captures the trailing-prep shape."""
        assert is_refinement_request(text) == (True, expected_slot)

    @pytest.mark.parametrize(
        "text",
        [
            "Plan a date night in Hayes Valley",
            "dinner then drinks then dessert",
            "I want sushi in the Mission",
            "find me a cocktail bar",
        ],
    )
    def test_first_turn_intent_does_not_trigger(self, text: str) -> None:
        """First-turn plan/find queries MUST NOT trigger (REF-04 protection)."""
        assert is_refinement_request(text) == (False, None)

    @pytest.mark.parametrize("text", ["", "   ", "\n\t"])
    def test_empty_and_whitespace_return_false_none(self, text: str) -> None:
        """Empty / whitespace-only inputs return (False, None)."""
        assert is_refinement_request(text) == (False, None)

    def test_make_it_cheaper_without_slot_number_returns_false_none(self) -> None:
        """Per D-06-03 conservatism: no slot number -> no match.

        We deliberately do NOT assume slot 1; the user must name the slot
        explicitly. A pre-check false positive here would clobber free-text
        refinement behavior on ambiguous turns.
        """
        assert is_refinement_request("make it cheaper") == (False, None)

    @pytest.mark.parametrize(
        ("text", "expected_slot"),
        [
            ("MAKE STOP 2 CHEAPER", 2),
            ("Stop 1 Different", 1),
            ("Swap Stop 3", 3),
            ("Instead For Stop 2", 2),
        ],
    )
    def test_case_insensitive(self, text: str, expected_slot: int) -> None:
        """All patterns compile with re.IGNORECASE — mixed/upper case still matches."""
        assert is_refinement_request(text) == (True, expected_slot)

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8, 9])
    def test_slot_range_extraction_one_through_nine(self, n: int) -> None:
        """Regex extracts every single-digit slot 1..9 without off-by-one bugs."""
        assert is_refinement_request(f"make stop {n} cheaper") == (True, n)

    @pytest.mark.parametrize(
        "text",
        [
            "my flight is nonstop",
            "the stops were great",
            "nonstop service to LAX",
        ],
    )
    def test_does_not_match_nonstop_or_stops(self, text: str) -> None:
        """Word boundaries (\\b) prevent false positives on 'nonstop' / 'stops'."""
        assert is_refinement_request(text) == (False, None)

    def test_pattern_family_coverage_explicit_assertions(self) -> None:
        """Explicit assertions pinning the four pattern families documented in
        the helper's docstring. Mirrors the D-06-03 behavior table 1:1 so a
        regression to any family fails this single test with a clear name.
        """
        # Family 1: 'make stop N <mod>'
        assert is_refinement_request("make stop 2 cheaper") == (True, 2)
        assert is_refinement_request("make stop 1 different") == (True, 1)
        # Family 2: 'stop N <mod|instead>'
        assert is_refinement_request("stop 3 cheaper please") == (True, 3)
        assert is_refinement_request("stop 4 fancier") == (True, 4)
        # Family 3: 'swap stop N'
        assert is_refinement_request("swap stop 3") == (True, 3)
        # Family 4: '(instead|different) (for|in) stop N'
        assert is_refinement_request("different for stop 1") == (True, 1)
        assert is_refinement_request("instead for stop 2") == (True, 2)
