from __future__ import annotations

import pytest

from app.agent.input_parsing import explicit_num_stops_from_text


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
