from __future__ import annotations

import pytest

from app.agent.prompts import CLARIFYING_STOPS_COUNT_TEMPLATE, SYSTEM_PROMPT

_FMT = {"max_steps": 8, "current_datetime": "2026-05-18 19:00 PDT (Monday)"}


def test_system_prompt_renders_with_max_steps() -> None:
    rendered = SYSTEM_PROMPT.format(**_FMT)
    assert "8" in rendered
    assert "{max_steps}" not in rendered


def test_system_prompt_format_round_trips() -> None:
    """str.format() must succeed without KeyError. The prompt deliberately
    contains JSON-shaped examples wrapped in `{{ }}` that render as `{ }`
    in the output — that is intended, but a single unescaped brace in the
    raw template would raise here.
    """
    SYSTEM_PROMPT.format(**_FMT)


def test_system_prompt_only_substitutes_known_placeholders() -> None:
    """If anyone adds a new `{foo}` placeholder without updating the .format()
    call site, this test fails fast: format with the known keys and assert no
    other unfilled `{name}` patterns remain.
    """
    import re

    rendered = SYSTEM_PROMPT.format(**_FMT)
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


def test_system_prompt_has_decisive_commit_contract() -> None:
    """SYSTEM_PROMPT must give an explicit EARLY-commit criterion: once you
    have one viable option per requested stop, call commit_itinerary — do
    NOT keep optimizing geometry/walkability.

    Root cause (convergence investigation): with the bogus low_similarity
    critique fixed, Kimi still looped 4/6 — its trajectories contained a
    complete viable itinerary by step ~4 but it kept re-searching to perfect
    the route ('Casements is north of my lunch... let me try a different
    structure') and never committed. DeepSeek converged only because it was
    more decisive. Rule 8 framed stopping as a last resort at the step
    ceiling, with no positive 'you have enough, commit' signal. Add one.
    """
    s = SYSTEM_PROMPT.lower()
    assert "commit_itinerary" in s
    # An explicit good-enough / don't-keep-optimizing stopping criterion.
    assert "one viable option" in s or "good enough" in s
    assert "do not keep" in s or "don't keep" in s or "stop optimizing" in s


def test_system_prompt_injects_current_datetime() -> None:
    """Root cause (temporal_coherence caveat): the model was never told the
    current date, so gpt-4o-mini hallucinated a training-era date
    (2023-10-06) for arrival_time. temporal_coherence then checked
    place_is_open against that stale date and the caveat fired. The prompt
    must carry an explicit 'today is ...' anchor the model uses for
    arrival_time when the user gives no date."""
    rendered = SYSTEM_PROMPT.format(max_steps=8, current_datetime="2026-05-18 19:00 PDT (Monday)")
    assert "2026-05-18 19:00 PDT (Monday)" in rendered
    # The prompt must instruct the model to anchor scheduling to this, not
    # invent a date.
    low = rendered.lower()
    assert "current date" in low or "today" in low
    assert "{current_datetime}" not in rendered


def test_system_prompt_requires_both_placeholders() -> None:
    """Both substitutions are mandatory — dropping either must raise KeyError
    rather than silently shipping an unanchored or unbounded prompt."""
    with pytest.raises(KeyError):
        SYSTEM_PROMPT.format(max_steps=8)  # missing current_datetime
    with pytest.raises(KeyError):
        SYSTEM_PROMPT.format(current_datetime="x")  # missing max_steps


def test_clarifying_stops_template_is_a_static_string() -> None:
    assert isinstance(CLARIFYING_STOPS_COUNT_TEMPLATE, str)
    assert "stops" in CLARIFYING_STOPS_COUNT_TEMPLATE
