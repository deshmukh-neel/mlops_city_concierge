from __future__ import annotations

import pytest

from app.config import csv_list, env_flag, get_settings, resolve_database_url


def test_resolve_database_url_prefers_explicit_database_url() -> None:
    env = {
        "DATABASE_URL": "postgresql://user:pass@db.example.com:5432/city_concierge",
        "POSTGRES_USER": "ignored",
        "POSTGRES_PASSWORD": "ignored",
        "POSTGRES_DB": "ignored",
    }

    assert resolve_database_url(env) == env["DATABASE_URL"]


def test_resolve_database_url_builds_standard_postgres_url() -> None:
    env = {
        "POSTGRES_USER": "postgres",
        "POSTGRES_PASSWORD": "p@ss word",
        "POSTGRES_DB": "city_concierge",
        "POSTGRES_HOST": "db.example.com",
        "POSTGRES_PORT": "5432",
    }

    assert (
        resolve_database_url(env)
        == "postgresql://postgres:p%40ss+word@db.example.com:5432/city_concierge"
    )


def test_resolve_database_url_includes_optional_ssl_settings() -> None:
    env = {
        "POSTGRES_USER": "postgres",
        "POSTGRES_PASSWORD": "secret",
        "POSTGRES_DB": "city_concierge",
        "POSTGRES_HOST": "db.example.com",
        "POSTGRES_PORT": "5432",
        "POSTGRES_SSLMODE": "require",
        "POSTGRES_SSLROOTCERT": "/certs/server-ca.pem",
    }

    assert (
        resolve_database_url(env)
        == "postgresql://postgres:secret@db.example.com:5432/city_concierge"
        "?sslmode=require&sslrootcert=%2Fcerts%2Fserver-ca.pem"
    )


def test_resolve_database_url_builds_cloud_sql_socket_url() -> None:
    env = {
        "POSTGRES_USER": "postgres",
        "POSTGRES_PASSWORD": "secret",
        "POSTGRES_DB": "city_concierge",
        "CLOUD_SQL_INSTANCE_CONNECTION_NAME": "project:region:instance",
        "CLOUD_SQL_SOCKET_DIR": "/cloudsql",
    }

    assert (
        resolve_database_url(env)
        == "postgresql://postgres:secret@/city_concierge?host=%2Fcloudsql%2Fproject%3Aregion%3Ainstance"
    )


def test_google_directions_api_key_defaults_empty() -> None:
    """Empty key is the first-class 'use haversine fallback' signal — unit
    tests must get '' by default so the no-key branch is exercised for free."""
    get_settings.cache_clear()
    assert get_settings().google_directions_api_key == ""


def test_cors_defaults_allow_localhost_and_vercel_preview() -> None:
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.cors_origins == ["http://localhost:5173", "http://localhost:3000"]
    assert settings.cors_allow_origin_regex == r"https://.*\.vercel\.app$"


def test_cors_origins_read_comma_separated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "CORS_ALLOW_ORIGINS",
        "https://city-concierge.vercel.app, https://preview.example.com ",
    )
    get_settings.cache_clear()
    assert get_settings().cors_origins == [
        "https://city-concierge.vercel.app",
        "https://preview.example.com",
    ]


def test_csv_list_discards_blank_items() -> None:
    assert csv_list(" https://a.example, ,https://b.example,, ") == [
        "https://a.example",
        "https://b.example",
    ]


def test_embedding_table_defaults_to_v2() -> None:
    """v2 (richer compose_embedding_text_v2) is the standard for dev/test/prod.
    v1 (place_embeddings) is noisy and was never meant to be the default; the
    KG SIMILAR_VECTOR edges already read v2 only. Lock the default so test,
    development, and prod all retrieve against the same v2 corpus."""
    get_settings.cache_clear()
    assert get_settings().embedding_table == "place_embeddings_v2"


def test_resolve_llm_api_key_supports_deepseek_and_kimi(monkeypatch) -> None:
    from app.config import get_settings, resolve_llm_api_key

    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
    monkeypatch.setenv("MOONSHOT_API_KEY", "ms-key")
    get_settings.cache_clear()
    assert resolve_llm_api_key("deepseek") == "ds-key"
    assert resolve_llm_api_key("kimi") == "ms-key"


def test_rag_model_override_defaults_none(monkeypatch) -> None:
    """Unset RAG_MODEL_OVERRIDE -> Settings.rag_model_override is None.

    Backward-compat sentinel (OVR-03): every existing test path must see None
    so load_registered_rag_chain takes the 'production' alias branch.
    """
    monkeypatch.delenv("RAG_MODEL_OVERRIDE", raising=False)
    get_settings.cache_clear()
    assert get_settings().rag_model_override is None


def test_rag_model_override_reads_env_var(monkeypatch) -> None:
    """RAG_MODEL_OVERRIDE env var flows through pydantic-settings (OVR-01, OVR-05).

    Uses monkeypatch.setenv, never os.environ[...] (OVR-05 contract).
    """
    monkeypatch.setenv("RAG_MODEL_OVERRIDE", "version:7")
    get_settings.cache_clear()
    assert get_settings().rag_model_override == "version:7"


# ---------------------------------------------------------------------------
# env_flag (WR-09 DRY) — single source of truth for boolean-flag parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    ["1", "true", "yes", "on", "TRUE", "YES", "ON", "True", " true ", " 1 ", " ON "],
)
def test_env_flag_truthy_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """env_flag returns True for every canonical truthy token and case/whitespace variants."""
    monkeypatch.setenv("_TEST_ENV_FLAG", value)
    assert env_flag("_TEST_ENV_FLAG") is True


@pytest.mark.parametrize(
    "value",
    ["0", "false", "off", "no", "FALSE", "OFF", "NO", "2", "yes1", ""],
)
def test_env_flag_falsy_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """env_flag returns False for any value not in the truthy set."""
    monkeypatch.setenv("_TEST_ENV_FLAG", value)
    assert env_flag("_TEST_ENV_FLAG") is False


def test_env_flag_unset_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """env_flag returns False when the variable is not set at all."""
    monkeypatch.delenv("_TEST_ENV_FLAG", raising=False)
    assert env_flag("_TEST_ENV_FLAG") is False
