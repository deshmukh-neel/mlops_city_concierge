from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.config import get_settings
from app.main import ActiveModelConfig, LoadedConfig, lifespan

TEST_DATABASE_URL = "postgresql://postgres:test@localhost:5432/city_concierge_test"


@pytest.fixture(autouse=True)
def _force_test_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", TEST_DATABASE_URL)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_lifespan_initializes_and_closes_db_pool(mocker) -> None:
    fake_app = SimpleNamespace(state=SimpleNamespace())
    fake_chain = object()
    fake_llm = object()
    fake_graph = object()
    model_config = ActiveModelConfig(
        llm_provider="openai",
        chat_model="gpt-4o-mini",
        k=5,
        temperature=0.0,
        run_id="run-123",
        model_version="7",
    )
    init_db_pool = mocker.patch("app.main.init_db_pool")
    close_db_pool = mocker.patch("app.main.close_db_pool")
    load_registered_rag_chain = mocker.patch(
        "app.main.load_registered_rag_chain",
        return_value=LoadedConfig(chain=fake_chain, llm=fake_llm, params=model_config),
    )
    build_agent_graph = mocker.patch("app.main.build_agent_graph", return_value=fake_graph)

    async with lifespan(fake_app):
        assert fake_app.state.rag_chain is fake_chain
        assert fake_app.state.active_model_config is model_config
        assert fake_app.state.agent_graph is fake_graph
        assert fake_app.state.rag_label == "openai:gpt-4o-mini"

    init_db_pool.assert_called_once_with(
        TEST_DATABASE_URL,
        0,
        10,
    )
    load_registered_rag_chain.assert_called_once_with()
    build_agent_graph.assert_called_once_with(fake_llm)
    close_db_pool.assert_called_once_with()


@pytest.mark.asyncio
async def test_lifespan_preserves_degraded_mode_when_rag_load_fails(mocker) -> None:
    fake_app = SimpleNamespace(state=SimpleNamespace())
    init_db_pool = mocker.patch("app.main.init_db_pool")
    close_db_pool = mocker.patch("app.main.close_db_pool")
    load_registered_rag_chain = mocker.patch(
        "app.main.load_registered_rag_chain",
        side_effect=RuntimeError("mlflow unavailable"),
    )
    logger = mocker.patch("app.main.logger")

    async with lifespan(fake_app):
        assert fake_app.state.rag_chain is None
        assert fake_app.state.active_model_config is None
        assert fake_app.state.agent_graph is None

    init_db_pool.assert_called_once_with(
        TEST_DATABASE_URL,
        0,
        10,
    )
    load_registered_rag_chain.assert_called_once_with()
    logger.warning.assert_called_once()
    close_db_pool.assert_called_once_with()
