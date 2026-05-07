from __future__ import annotations

from functools import lru_cache

import psycopg2
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from .config import get_settings


def vector_to_pg(embedding: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in embedding) + "]"


@lru_cache(maxsize=4096)
def _embed_cached(query: str, embedding_model: str, api_key: str) -> tuple[float, ...]:
    """Cache OpenAI embeddings keyed on (query, model). The api_key is part of
    the key only for safety; it doesn't affect output for a given (query, model).
    Returns a tuple so the cached value is immutable."""
    embeddings = OpenAIEmbeddings(model=embedding_model, api_key=SecretStr(api_key))
    return tuple(embeddings.embed_query(query))


def build_embedding(
    query: str, embedding_model: str, openai_api_key: str | None = None
) -> list[float]:
    """Generate an OpenAI embedding for a query string. Reused by agent tools.

    Embeddings are deterministic per (query, model), so we cache them via
    _embed_cached to skip duplicate OpenAI round-trips within and across
    sessions.
    """
    settings = get_settings()
    api_key = openai_api_key or settings.openai_api_key
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY for query embedding generation.")
    return list(_embed_cached(query, embedding_model, api_key))


class PgVectorRetriever(BaseRetriever):
    connection_string: str
    embedding_model: str = "text-embedding-3-small"
    k: int = 5
    openai_api_key: str | None = None

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        del run_manager

        settings = get_settings()
        api_key = self.openai_api_key or settings.openai_api_key
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY for query embedding generation.")

        embeddings = OpenAIEmbeddings(model=self.embedding_model, api_key=SecretStr(api_key))
        vector_literal = vector_to_pg(embeddings.embed_query(query))

        # settings.embedding_table is validated against an allowlist at config load
        # (see ALLOWED_EMBEDDING_TABLES in app/config.py), so f-stringing is safe.
        sql = f"""
        SELECT
            e.embedding_text,
            p.place_id,
            p.name,
            p.rating,
            p.formatted_address,
            p.primary_type,
            1 - (e.embedding <=> %s::vector) AS similarity
        FROM {settings.embedding_table} e
        JOIN places_raw p ON p.place_id = e.place_id
        WHERE e.embedding_model = %s
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s
        """  # noqa: S608

        with psycopg2.connect(self.connection_string) as conn, conn.cursor() as cur:
            cur.execute(sql, (vector_literal, self.embedding_model, vector_literal, self.k))
            rows = cur.fetchall()

        return [
            Document(
                page_content=row[0],
                metadata={
                    "place_id": row[1],
                    "name": row[2],
                    "rating": row[3],
                    "address": row[4],
                    "primary_type": row[5],
                    "similarity": row[6],
                },
            )
            for row in rows
        ]
