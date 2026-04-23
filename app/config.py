from __future__ import annotations

from collections.abc import Mapping
from urllib.parse import quote_plus, urlencode

from pydantic_settings import BaseSettings, SettingsConfigDict


def build_database_url(
    *,
    user: str,
    password: str,
    dbname: str,
    host: str = "localhost",
    port: str | int = "5432",
    cloud_sql_instance: str | None = None,
    cloud_sql_socket_dir: str = "/cloudsql",
    sslmode: str | None = None,
    sslrootcert: str | None = None,
) -> str:
    encoded_password = quote_plus(password)
    query_params: dict[str, str] = {}

    if sslmode:
        query_params["sslmode"] = sslmode
    if sslrootcert:
        query_params["sslrootcert"] = sslrootcert

    if cloud_sql_instance:
        socket_path = f"{cloud_sql_socket_dir.rstrip('/')}/{cloud_sql_instance}"
        query_params["host"] = socket_path
        query_string = urlencode(query_params, quote_via=quote_plus)
        return f"postgresql://{user}:{encoded_password}@/{dbname}?{query_string}"

    database_url = f"postgresql://{user}:{encoded_password}@{host}:{port}/{dbname}"
    if query_params:
        query_string = urlencode(query_params, quote_via=quote_plus)
        return f"{database_url}?{query_string}"

    return database_url


def resolve_database_url(env: Mapping[str, str]) -> str | None:
    explicit = env.get("DATABASE_URL")
    if explicit:
        return explicit

    user = env.get("POSTGRES_USER")
    password = env.get("POSTGRES_PASSWORD")
    dbname = env.get("POSTGRES_DB")
    host = env.get("POSTGRES_HOST", "localhost")
    port = env.get("POSTGRES_PORT", "5432")
    cloud_sql_instance = env.get("CLOUD_SQL_INSTANCE_CONNECTION_NAME")
    cloud_sql_socket_dir = env.get("CLOUD_SQL_SOCKET_DIR", "/cloudsql")
    sslmode = env.get("POSTGRES_SSLMODE")
    sslrootcert = env.get("POSTGRES_SSLROOTCERT")

    if not (user and password and dbname):
        return None

    return build_database_url(
        user=user,
        password=password,
        dbname=dbname,
        host=host,
        port=port,
        cloud_sql_instance=cloud_sql_instance,
        cloud_sql_socket_dir=cloud_sql_socket_dir,
        sslmode=sslmode,
        sslrootcert=sslrootcert,
    )


class Settings(BaseSettings):
    database_url: str | None = None
    postgres_db: str | None = None
    postgres_user: str | None = None
    postgres_password: str | None = None
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    cloud_sql_instance_connection_name: str | None = None
    cloud_sql_socket_dir: str = "/cloudsql"
    postgres_sslmode: str | None = None
    postgres_sslrootcert: str | None = None
    api_title: str = "City Concierge API"
    api_version: str = "0.1.0"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def resolved_database_url(self) -> str | None:
        if self.database_url:
            return self.database_url

        if not (self.postgres_user and self.postgres_password and self.postgres_db):
            return None

        return build_database_url(
            user=self.postgres_user,
            password=self.postgres_password,
            dbname=self.postgres_db,
            host=self.postgres_host,
            port=self.postgres_port,
            cloud_sql_instance=self.cloud_sql_instance_connection_name,
            cloud_sql_socket_dir=self.cloud_sql_socket_dir,
            sslmode=self.postgres_sslmode,
            sslrootcert=self.postgres_sslrootcert,
        )


settings = Settings()
