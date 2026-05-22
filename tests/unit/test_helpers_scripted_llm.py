"""Unit tests for tests/_helpers/scripted_llm.py (Plan 03-11 / WR-03 hoist).

Pins the contract of the shared `ScriptedLLM` + `RecordingScriptedLLM` test
helpers so future edits cannot silently reintroduce the singleton-fallback bug
(CR-02) or drop the recording behavior.

The helper is intentionally stricter than the production
`app.llm_factory.ScriptedChatModel`: it RAISES on exhaustion (loud-fail) rather
than synthesizing a fresh fallback AIMessage. Tests should be exhaustive about
their scripts; surprises should localize immediately.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from tests._helpers.scripted_llm import RecordingScriptedLLM, ScriptedLLM


def test_scripted_llm_pops_in_order() -> None:
    """ScriptedLLM consumes messages FIFO across consecutive _generate calls."""
    m = ScriptedLLM(scripted=[AIMessage(content="a"), AIMessage(content="b")])

    first = m._generate(messages=[])
    second = m._generate(messages=[])

    assert first.generations[0].message.content == "a"
    assert second.generations[0].message.content == "b"


def test_scripted_llm_raises_when_exhausted() -> None:
    """Empty scripted list raises IndexError so the test author sees a clean,
    early failure instead of a silent singleton or a confusing downstream
    AttributeError in the agent graph."""
    m = ScriptedLLM(scripted=[])

    with pytest.raises(IndexError, match="exhausted"):
        m._generate(messages=[])


def test_recording_scripted_llm_captures_seen_messages() -> None:
    """RecordingScriptedLLM snapshots the messages it saw on each call before
    delegating to the pop logic. The snapshot uses `list(messages)` so later
    mutation of the caller's list doesn't retroactively change `seen`."""
    rec = RecordingScriptedLLM(scripted=[AIMessage(content="a")])

    rec._generate(messages=[HumanMessage(content="hello")])

    assert len(rec.seen) == 1
    assert rec.seen[0][0].content == "hello"


def test_recording_scripted_llm_default_seen_is_empty_list() -> None:
    """The `seen` field defaults to a fresh list via Field(default_factory=list).

    This is the WR-04 fix — the 5 outer-scope `seen: list[list[BaseMessage]] = []`
    vars in test_eval_agent.py are dead because Pydantic deep-copies during
    validation. With a default_factory the helper instance has a clean,
    instance-owned list with no kwarg required.
    """
    rec = RecordingScriptedLLM(scripted=[])

    assert rec.seen == []
    # Second instance must NOT share the first's list (default_factory, not
    # default=[] — the latter would alias the list across instances).
    rec_two = RecordingScriptedLLM(scripted=[])
    assert rec.seen is not rec_two.seen


def test_bind_tools_returns_self() -> None:
    """Both classes treat bind_tools as a no-op (return self). The agent graph
    calls .bind_tools(...) on the LLM; tests don't care about real tool binding."""
    plain = ScriptedLLM(scripted=[])
    rec = RecordingScriptedLLM(scripted=[])

    assert plain.bind_tools(["any"]) is plain
    assert rec.bind_tools(["any"]) is rec
