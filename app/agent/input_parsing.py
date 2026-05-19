"""Lightweight deterministic parsing for user constraints.

This is deliberately conservative. The LLM still owns nuanced interpretation;
these helpers only extract constraints that are explicit enough to use as
guardrails when the model tries to ask a question the user already answered.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Protocol

_MAX_EXPLICIT_STOPS = 10
_STOP_NOUN = r"(?:stop|stops|spot|spots|place|places)"
_DIGIT_STOP_RE = re.compile(rf"\b([1-9]|10)\s*[- ]?\s*{_STOP_NOUN}\b", re.IGNORECASE)
_WORD_TO_INT: dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
_WORD_STOP_RE = re.compile(
    rf"\b({'|'.join(_WORD_TO_INT)})\s*[- ]?\s*{_STOP_NOUN}\b",
    re.IGNORECASE,
)


class _HasRoleContent(Protocol):
    role: str
    content: str


def explicit_num_stops_from_conversation(
    history: Iterable[_HasRoleContent], current_message: str
) -> int | None:
    """The latest explicit user-stated stop count across the whole turn.

    /chat is stateless — each POST rebuilds ItineraryState — so parsing only
    `current_message` loses the count on every follow-up turn. Walks history's
    user messages in order, then the current message; the LAST hit wins so a
    mid-conversation revision ("actually make it 4") overrides an earlier "3".
    """
    latest: int | None = None
    for m in history:
        if getattr(m, "role", None) == "user":
            found = explicit_num_stops_from_text(getattr(m, "content", "") or "")
            if found is not None:
                latest = found
    found = explicit_num_stops_from_text(current_message)
    if found is not None:
        latest = found
    return latest


def explicit_num_stops_from_text(text: str) -> int | None:
    """Return an explicit stop count from user text, if one is stated.

    Examples matched: "3-stop", "3 stops", "three spots". Counts above 10
    are intentionally ignored because the current itinerary agent is optimized
    for small walking plans, not all-day route generation.
    """
    digit_match = _DIGIT_STOP_RE.search(text)
    if digit_match:
        value = int(digit_match.group(1))
        return value if 1 <= value <= _MAX_EXPLICIT_STOPS else None

    word_match = _WORD_STOP_RE.search(text)
    if word_match:
        return _WORD_TO_INT[word_match.group(1).lower()]

    return None
