from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.agent.input_parsing import (
    explicit_num_stops_from_conversation,
    explicit_num_stops_from_text,
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
