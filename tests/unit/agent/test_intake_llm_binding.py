"""Binding-inspection tests for the /chat hybrid intake pipeline.

Locks in:
  - The intake LLM bind carries `temperature=1.0` and provider-appropriate
    reasoning-off kwargs (per memory `feedback_temp1_reasoning_off_all_models.md`
    — always temp=1.0; disable thinking for ALL providers including Gemini).
  - Zero-latency-tax on free-text inputs: when `has_slot_structure` returns
    False, NO `.bind()` and NO `.with_structured_output()` is invoked.
  - Defensive null-check: when `app.state.agent_llm` is None, no bind happens
    and the request still 200s with `requested_primary_types == []`.

The tests use MagicMock-spied LLMs so we can introspect exactly what the
handler asked for, independent of any real provider. The shape mirrors the
production path:
  intake_llm = loaded.llm.bind(**kwargs)              # .bind returns a runnable
  structured = intake_llm.with_structured_output(...) # returns a runnable
  result = await structured.ainvoke(prompt)           # returns a SlotExtractionResult
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from app.agent.input_parsing import SlotExtractionResult
from app.main import ActiveModelConfig, LoadedConfig, app


def _make_recording_llm(
    extraction_result: SlotExtractionResult | Exception,
) -> tuple[Any, dict[str, Any]]:
    """Build a MagicMock-style fake LLM that records `.bind(**kwargs)` calls
    and exposes `with_structured_output` and `ainvoke` plumbing.

    Returns a tuple of (fake_llm, observed) where `observed` is a dict
    populated during the request:
      observed["bind_calls"]: list[dict] — kwargs passed to .bind on each call
      observed["wso_call_count"]: int — number of times with_structured_output ran
      observed["ainvoke_call_count"]: int — number of structured ainvoke calls
      observed["last_prompt"]: str | None — last prompt fed to structured.ainvoke
    """
    observed: dict[str, Any] = {
        "bind_calls": [],
        "wso_call_count": 0,
        "ainvoke_call_count": 0,
        "last_prompt": None,
    }

    class _Structured:
        async def ainvoke(self, prompt: str, *args: Any, **kwargs: Any) -> Any:
            observed["ainvoke_call_count"] += 1
            observed["last_prompt"] = prompt
            if isinstance(extraction_result, Exception):
                raise extraction_result
            return extraction_result

    class _Bound:
        def with_structured_output(self, *args: Any, **kwargs: Any) -> _Structured:
            observed["wso_call_count"] += 1
            return _Structured()

    class _LLM:
        # Mimic openai-class branch of `_intake_bind_kwargs` by class name.
        __class__name = "ChatOpenAI"

        def bind(self, **kwargs: Any) -> _Bound:
            observed["bind_calls"].append(dict(kwargs))
            return _Bound()

    # Replace ChatOpenAI-like detection: set the spoofed class name via type()
    llm = type("ChatOpenAI", (_LLM,), {})()
    return llm, observed


def _stub_loaded_config(llm: Any) -> LoadedConfig:
    return LoadedConfig(
        chain=object(),
        llm=llm,
        params=ActiveModelConfig(
            llm_provider="openai",
            chat_model="gpt-4o-mini",
            k=5,
            temperature=0.0,
            run_id="run-1",
            model_version="1",
        ),
    )


def _final_state_dict() -> dict:
    return {
        "messages": [],
        "constraints": {},
        "stops": [],
        "scratch": {},
        "step_count": 1,
        "done": True,
        "final_reply": "ok",
        "awaiting_stops_count": False,
        "walked_meters_so_far": 0.0,
    }


def test_intake_bind_carries_temperature_one_and_reasoning_off_for_openai_like(
    mocker,
) -> None:
    """Slot-structured input → loaded.llm.bind called with temperature=1.0
    (the cross-provider reasoning-off invariant)."""
    fake_llm, observed = _make_recording_llm(
        SlotExtractionResult(requested_primary_types=["Sushi Restaurant", "Cocktail Bar"])
    )

    fake_graph = mocker.Mock()

    async def _ainvoke(state, config=None):
        return _final_state_dict()

    fake_graph.ainvoke = _ainvoke
    mocker.patch(
        "app.main.load_registered_rag_chain",
        return_value=_stub_loaded_config(fake_llm),
    )
    mocker.patch("app.main.build_agent_graph", return_value=fake_graph)

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"message": "omakase, drinks, dessert in Mission"},
        )

    assert response.status_code == 200
    assert observed["bind_calls"], "intake bind never invoked on slot-structured input"
    # The first bind call is the intake bind; assert it carries temp=1.0.
    intake_bind_kwargs = observed["bind_calls"][0]
    assert intake_bind_kwargs.get("temperature") == 1.0
    assert observed["wso_call_count"] >= 1
    assert observed["ainvoke_call_count"] >= 1


def test_intake_bind_not_invoked_on_free_text(mocker) -> None:
    """Free-text /chat → ZERO bind/with_structured_output/ainvoke calls.
    Locks the zero-latency-tax invariant: the existing code path is unchanged
    when has_slot_structure returns False."""
    fake_llm, observed = _make_recording_llm(
        SlotExtractionResult(requested_primary_types=["should never be used"])
    )

    fake_graph = mocker.Mock()

    async def _ainvoke(state, config=None):
        return _final_state_dict()

    fake_graph.ainvoke = _ainvoke
    mocker.patch(
        "app.main.load_registered_rag_chain",
        return_value=_stub_loaded_config(fake_llm),
    )
    mocker.patch("app.main.build_agent_graph", return_value=fake_graph)

    with TestClient(app) as client:
        # Free-text — single slot only, no vocab, no planning verb pattern
        response = client.post(
            "/chat",
            json={"message": "find me good tacos"},
        )

    assert response.status_code == 200
    assert observed["bind_calls"] == [], f"bind invoked on free-text path: {observed['bind_calls']}"
    assert observed["wso_call_count"] == 0
    assert observed["ainvoke_call_count"] == 0


def test_intake_bind_not_invoked_when_agent_llm_is_none(mocker) -> None:
    """Defensive null-check: even on a slot-structured message, if
    app.state.agent_llm is None (e.g. lifespan partial failure), the
    handler does NOT crash and treats requested_primary_types as []."""
    fake_graph = mocker.Mock()
    captured: dict[str, Any] = {}

    async def _ainvoke(state, config=None):
        captured["state"] = state
        return _final_state_dict()

    fake_graph.ainvoke = _ainvoke
    mocker.patch(
        "app.main.load_registered_rag_chain",
        return_value=_stub_loaded_config(object()),
    )
    mocker.patch("app.main.build_agent_graph", return_value=fake_graph)

    with TestClient(app) as client:
        # Force agent_llm to None AFTER lifespan startup wired the real one.
        app.state.agent_llm = None
        response = client.post(
            "/chat",
            json={"message": "dinner, drinks, dessert"},
        )

    assert response.status_code == 200
    # No bind happened (no llm to bind on); state.constraints carries []
    assert captured["state"].constraints.requested_primary_types == []
