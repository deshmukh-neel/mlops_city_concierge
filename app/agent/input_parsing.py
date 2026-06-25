"""Lightweight deterministic parsing for user constraints.

This is deliberately conservative. The LLM still owns nuanced interpretation;
these helpers only extract constraints that are explicit enough to use as
guardrails when the model tries to ask a question the user already answered.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Literal, Protocol

from pydantic import BaseModel, Field

MAX_EXPLICIT_STOPS = 10
STOP_NOUN = r"(?:stop|stops|spot|spots|place|places)"
DIGIT_STOP_RE = re.compile(rf"\b([1-9]|10)\s*[- ]?\s*{STOP_NOUN}\b", re.IGNORECASE)
WORD_TO_INT: dict[str, int] = {
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
WORD_STOP_RE = re.compile(
    rf"\b({'|'.join(WORD_TO_INT)})\s*[- ]?\s*{STOP_NOUN}\b",
    re.IGNORECASE,
)

# Detects "the assistant just asked a how-many-stops question."
# Conservative — requires both an interrogative phrase ("how many") and a
# stop-noun nearby, so a stray "how many drinks?" or a comment like
# "I'd want 3 stops" doesn't activate the bare-number capture below.
ASSISTANT_STOPS_QUESTION_RE = re.compile(
    rf"how\s+many\s+\w*\s*{STOP_NOUN}\b",
    re.IGNORECASE,
)

# Bare-integer or bare-word-integer in [1, MAX_EXPLICIT_STOPS]. Used only
# when the prior assistant turn was a stops-count question.
BARE_NUMBER_RE = re.compile(r"^\s*([1-9]|10)\s*[.!]?\s*$")
BARE_WORD_NUMBER_RE = re.compile(
    rf"^\s*({'|'.join(WORD_TO_INT)})\s*[.!]?\s*$",
    re.IGNORECASE,
)


# Gate the extra intake LLM call to messages that look slot-structured.
SLOT_VOCABULARY: frozenset[str] = frozenset(
    {
        "dinner",
        "drinks",
        "dessert",
        "brunch",
        "lunch",
        "breakfast",
        "cocktails",
        "coffee",
        "nightcap",
        "omakase",
        "sushi",
        "ramen",
        "tacos",
        "pizza",
        "bar",
        "cafe",
    }
)
SLOT_VOCAB_RE = re.compile(
    rf"\b(?:{'|'.join(sorted(SLOT_VOCABULARY))})\b",
    re.IGNORECASE,
)
THEN_PATTERN_RE = re.compile(
    r"\b\w+\s+(?:then|followed\s+by|->|>)\s+\w+",
    re.IGNORECASE,
)
NUMBERED_SLOT_RE = re.compile(r"\b1[.)]\s.*?\b2[.)]", re.IGNORECASE | re.DOTALL)
PLANNING_VERB_RE = re.compile(
    r"\b(?:plan|schedule|book|do)\b",
    re.IGNORECASE,
)


# Refinement detection stays literal: the user must name a numbered stop.
REFINE_MAKE_STOP_RE = re.compile(
    r"\bmake\s+stop\s+([1-9])\s+(?:cheaper|different|fancier|earlier|later)\b",
    re.IGNORECASE,
)
REFINE_STOP_BARE_RE = re.compile(
    r"\bstop\s+([1-9])\s+(?:cheaper|different|fancier|earlier|later|instead)\b",
    re.IGNORECASE,
)
REFINE_SWAP_STOP_RE = re.compile(
    r"\bswap\s+stop\s+([1-9])\b",
    re.IGNORECASE,
)
REFINE_INSTEAD_FOR_STOP_RE = re.compile(
    r"\b(?:instead|different)\s+(?:for|in)\s+stop\s+([1-9])\b",
    re.IGNORECASE,
)

REFINE_PATTERNS: tuple[re.Pattern[str], ...] = (
    REFINE_MAKE_STOP_RE,
    REFINE_STOP_BARE_RE,
    REFINE_SWAP_STOP_RE,
    REFINE_INSTEAD_FOR_STOP_RE,
)


def is_refinement_request(text: str) -> tuple[bool, int | None]:
    """Detect a refinement turn and return its 1-indexed target slot."""
    if not text or not text.strip():
        return False, None
    for pattern in REFINE_PATTERNS:
        match = pattern.search(text)
        if match:
            return True, int(match.group(1))
    return False, None


def has_slot_structure(text: str) -> bool:
    """Return True when `text` looks like ordered category slots."""
    if not text or not text.strip():
        return False
    if THEN_PATTERN_RE.search(text):
        return True
    if NUMBERED_SLOT_RE.search(text):
        return True
    vocab_hits = {m.group(0).lower() for m in SLOT_VOCAB_RE.finditer(text)}
    if len(vocab_hits) >= 2:
        return True
    return bool(vocab_hits and PLANNING_VERB_RE.search(text))


class SlotExtractionResult(BaseModel):
    """Structured-output shape for the intake LLM."""

    requested_primary_types: list[str] = Field(
        default_factory=list,
        description=(
            "Per-slot Google primary_type values, Title Case "
            "(e.g. 'Sushi Restaurant', 'Cocktail Bar', 'Dessert Shop'); "
            "validated downstream against family_of() in "
            "app.tools.filters — unmappable entries are dropped."
        ),
    )


class HasRoleContent(Protocol):
    # Read-only attribute protocol so concrete classes with narrower types
    # (e.g. ChatMessage with role: Literal["user", "assistant"]) still match.
    @property
    def role(self) -> str: ...
    @property
    def content(self) -> str: ...


def explicit_num_stops_from_conversation(
    history: Iterable[HasRoleContent], current_message: str
) -> int | None:
    """Return the latest explicit stop count from history plus this turn."""
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
    last_assistant = next(
        (
            getattr(m, "content", "") or ""
            for m in reversed(history_list)
            if getattr(m, "role", None) == "assistant"
        ),
        None,
    )
    if last_assistant and ASSISTANT_STOPS_QUESTION_RE.search(last_assistant):
        bare = parse_bare_count(current_message)
        if bare is not None:
            return bare
    return latest


def parse_bare_count(text: str) -> int | None:
    """Bare digit / word number in [1, 10], or None.

    Anchored match (`^...$`): "3 places" doesn't qualify (the noun-based
    parser handles that); only standalone numbers or word-numbers.
    """
    m = BARE_NUMBER_RE.match(text)
    if m:
        value = int(m.group(1))
        return value if 1 <= value <= MAX_EXPLICIT_STOPS else None
    m = BARE_WORD_NUMBER_RE.match(text)
    if m:
        return WORD_TO_INT[m.group(1).lower()]
    return None


def explicit_num_stops_from_text(text: str) -> int | None:
    """Return an explicit stop count from user text, if one is stated.

    Examples matched: "3-stop", "3 stops", "three spots". Counts above 10
    are intentionally ignored because the current itinerary agent is optimized
    for small walking plans, not all-day route generation.
    """
    digit_match = DIGIT_STOP_RE.search(text)
    if digit_match:
        value = int(digit_match.group(1))
        return value if 1 <= value <= MAX_EXPLICIT_STOPS else None

    word_match = WORD_STOP_RE.search(text)
    if word_match:
        return WORD_TO_INT[word_match.group(1).lower()]

    return None


# First-token sets for `parse_closure_decision`. The rule is intentionally
# conservative: only act when the user's first word reads as a yes/no token,
# so "yes, make it 4 stops" routes to accept (the count update is handled
# separately by `explicit_num_stops_from_conversation`) while
# "find something cheaper" — a question disguised as a hint — routes to
# alternative for the LLM to interpret in context.
ACCEPT_TOKENS: frozenset[str] = frozenset({"yes", "yeah", "yep", "sure", "ok", "okay", "y", "👍"})
DECLINE_TOKENS: frozenset[str] = frozenset({"no", "nope", "n", "nah"})


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
    if stripped in ACCEPT_TOKENS or first in ACCEPT_TOKENS:
        return "accept"
    if stripped in DECLINE_TOKENS or first in DECLINE_TOKENS:
        return "decline"
    return "alternative"
