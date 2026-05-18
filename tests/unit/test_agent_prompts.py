from __future__ import annotations

import pytest

from app.agent.prompts import CLARIFYING_STOPS_COUNT_TEMPLATE, SYSTEM_PROMPT


def test_system_prompt_renders_with_max_steps() -> None:
    rendered = SYSTEM_PROMPT.format(max_steps=8)
    assert "8" in rendered
    assert "{max_steps}" not in rendered


def test_system_prompt_format_round_trips() -> None:
    """str.format() must succeed without KeyError. The prompt deliberately
    contains JSON-shaped examples wrapped in `{{ }}` that render as `{ }`
    in the output — that is intended, but a single unescaped brace in the
    raw template would raise here.
    """
    SYSTEM_PROMPT.format(max_steps=8)


def test_system_prompt_only_substitutes_max_steps() -> None:
    """If anyone adds a new `{foo}` placeholder without updating the .format()
    call site, this test fails fast. We do this by formatting with max_steps
    only and asserting no other unfilled `{name}` patterns remain.
    """
    import re

    rendered = SYSTEM_PROMPT.format(max_steps=8)
    # A bare `{word}` after rendering would be an un-substituted placeholder
    # (legit JSON braces are doubled and render as `{` and `}` separately).
    leftover = re.findall(r"\{[a-zA-Z_]\w*\}", rendered)
    assert leftover == [], f"unfilled placeholders in prompt: {leftover}"


def test_system_prompt_missing_max_steps_raises() -> None:
    """Defensive: removing the {max_steps} substitution must not silently
    drop the loop bound — `.format()` would raise KeyError, which the test
    locks in."""
    with pytest.raises(KeyError):
        SYSTEM_PROMPT.format()


def test_system_prompt_contains_relation_type_guidance() -> None:
    """SYSTEM_PROMPT must name all five relation_type values + the tool name
    so the agent knows when to pick which edge (TOOL-06)."""
    s = SYSTEM_PROMPT.lower()
    assert "kg_traverse" in s
    assert "similar_vector" in s
    assert "same_neighborhood" in s
    assert "near_landmark" in s
    assert "contained_in" in s


def test_system_prompt_requires_rich_semantic_query() -> None:
    """SYSTEM_PROMPT must teach a minimum semantic-query richness and that
    structured filters REFINE the query rather than REPLACE it.

    Root cause (W10): Gemini 3 stripped `query` to bare keywords ('lunch')
    while max-stuffing filters, tanking cosine similarity to ~0.28-0.35 and
    never converging — while gpt-4o-mini wrote richer queries and committed
    6/6. The fix is an explicit query-construction contract: the semantic
    query must always carry cuisine/vibe + place-type + neighborhood, and
    filters must not be treated as a substitute for query content.
    """
    s = SYSTEM_PROMPT.lower()
    # The query must never be a bare keyword — it must carry semantic content.
    assert "filters refine" in s or "filters do not replace" in s
    # Concrete minimum-content guidance the model can follow.
    assert "cuisine" in s
    assert "neighborhood" in s and "place type" in s


def test_clarifying_stops_template_is_a_static_string() -> None:
    assert isinstance(CLARIFYING_STOPS_COUNT_TEMPLATE, str)
    assert "stops" in CLARIFYING_STOPS_COUNT_TEMPLATE
