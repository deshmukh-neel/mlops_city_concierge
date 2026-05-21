"""Lightweight deterministic parsing for user constraints.

This is deliberately conservative. The LLM still owns nuanced interpretation;
these helpers only extract constraints that are explicit enough to use as
guardrails when the model tries to ask a question the user already answered.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Literal, Protocol

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

# Detects "the assistant just asked a how-many-stops question."
# Conservative — requires both an interrogative phrase ("how many") and a
# stop-noun nearby, so a stray "how many drinks?" or a comment like
# "I'd want 3 stops" doesn't activate the bare-number capture below.
_ASSISTANT_STOPS_QUESTION_RE = re.compile(
    rf"how\s+many\s+\w*\s*{_STOP_NOUN}\b",
    re.IGNORECASE,
)

# Bare-integer or bare-word-integer in [1, _MAX_EXPLICIT_STOPS]. Used only
# when the prior assistant turn was a stops-count question.
_BARE_NUMBER_RE = re.compile(r"^\s*([1-9]|10)\s*[.!]?\s*$")
_BARE_WORD_NUMBER_RE = re.compile(
    rf"^\s*({'|'.join(_WORD_TO_INT)})\s*[.!]?\s*$",
    re.IGNORECASE,
)


class _HasRoleContent(Protocol):
    # Read-only attribute protocol so concrete classes with narrower types
    # (e.g. ChatMessage with role: Literal["user", "assistant"]) still match.
    @property
    def role(self) -> str: ...
    @property
    def content(self) -> str: ...


def explicit_num_stops_from_conversation(
    history: Iterable[_HasRoleContent], current_message: str
) -> int | None:
    """The latest explicit user-stated stop count across the whole turn.

    /chat is stateless — each POST rebuilds ItineraryState — so parsing only
    `current_message` loses the count on every follow-up turn. Walks history's
    user messages in order, then the current message; the LAST hit wins so a
    mid-conversation revision ("actually make it 4") overrides an earlier "3".

    Bare-number fallback (current_message only): if the prior assistant turn
    was clearly a how-many-stops question (e.g. "How many stops would you
    like?") and the current message is a bare integer or bare word-integer
    (e.g. "3", "three", "3.", "Three!"), treat it as the count. This
    closes the gap where gpt-4o-mini asked the count question and the user
    replied with just a number — without it, no count guardrail kicks in
    and the model can over-loop into the step-limit short-circuit. The
    bare-number rule applies ONLY to the current message, not to history,
    because old assistant turns are no longer the prompt at the front of
    the model's attention.
    """
    history_list = list(history)
    latest: int | None = None
    for m in history_list:
        if getattr(m, "role", None) == "user":
            found = explicit_num_stops_from_text(getattr(m, "content", "") or "")
            if found is not None:
                latest = found
    found = explicit_num_stops_from_text(current_message)
    if found is not None:
        return found
    # Bare-number fallback: only activate if the LAST assistant turn (the
    # one the user is replying to) was a how-many-stops question.
    last_assistant = next(
        (
            getattr(m, "content", "") or ""
            for m in reversed(history_list)
            if getattr(m, "role", None) == "assistant"
        ),
        None,
    )
    if last_assistant and _ASSISTANT_STOPS_QUESTION_RE.search(last_assistant):
        bare = _parse_bare_count(current_message)
        if bare is not None:
            return bare
    return latest


def _parse_bare_count(text: str) -> int | None:
    """Bare digit / word number in [1, 10], or None.

    Anchored match (`^...$`): "3 places" doesn't qualify (the noun-based
    parser handles that); only standalone numbers or word-numbers.
    """
    m = _BARE_NUMBER_RE.match(text)
    if m:
        value = int(m.group(1))
        return value if 1 <= value <= _MAX_EXPLICIT_STOPS else None
    m = _BARE_WORD_NUMBER_RE.match(text)
    if m:
        return _WORD_TO_INT[m.group(1).lower()]
    return None


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


# First-token sets for `parse_closure_decision`. The rule is intentionally
# conservative: only act when the user's first word reads as a yes/no token,
# so "yes, make it 4 stops" routes to accept (the count update is handled
# separately by `explicit_num_stops_from_conversation`) while
# "find something cheaper" — a question disguised as a hint — routes to
# alternative for the LLM to interpret in context.
_ACCEPT_TOKENS: frozenset[str] = frozenset({"yes", "yeah", "yep", "sure", "ok", "okay", "y", "👍"})
_DECLINE_TOKENS: frozenset[str] = frozenset({"no", "nope", "n", "nah"})


def parse_closure_decision(text: str) -> Literal["accept", "decline", "alternative"]:
    """Conservative parser for a user's reply to a closure question.

    First-token rule resolves "yes + revision" (e.g. "yes! make it 4 stops")
    unambiguously as accept; the existing `explicit_num_stops_from_conversation`
    separately handles the count update. Empty / whitespace / questions / free
    text all bucket to "alternative" (no auto-accept on silence).
    """
    if not text or not text.strip():
        return "alternative"
    first = text.strip().split()[0].lower()
    # Strip surrounding punctuation so "Yes!" / "yes," / "ok." also match.
    stripped = first.strip(".,!?;:\"'()[]{}")
    if stripped in _ACCEPT_TOKENS or first in _ACCEPT_TOKENS:
        return "accept"
    if stripped in _DECLINE_TOKENS or first in _DECLINE_TOKENS:
        return "decline"
    return "alternative"
