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


def test_clarifying_stops_template_is_a_static_string() -> None:
    assert isinstance(CLARIFYING_STOPS_COUNT_TEMPLATE, str)
    assert "stops" in CLARIFYING_STOPS_COUNT_TEMPLATE
