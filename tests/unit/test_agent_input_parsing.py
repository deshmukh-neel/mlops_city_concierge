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
