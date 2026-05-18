from __future__ import annotations

from app.config import get_settings, resolve_database_url


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


def test_embedding_table_defaults_to_v2() -> None:
    """v2 (richer compose_embedding_text_v2) is the standard for dev/test/prod.
    v1 (place_embeddings) is noisy and was never meant to be the default; the
    KG SIMILAR_VECTOR edges already read v2 only. Lock the default so test,
    development, and prod all retrieve against the same v2 corpus."""
    get_settings.cache_clear()
    assert get_settings().embedding_table == "place_embeddings_v2"
