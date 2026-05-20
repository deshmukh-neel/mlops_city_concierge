from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.agent.input_parsing import (
    explicit_num_stops_from_conversation,
    explicit_num_stops_from_text,
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
