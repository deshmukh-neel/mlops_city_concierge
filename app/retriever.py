from __future__ import annotations

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from .config import get_settings
from .db import borrow_connection


def vector_to_pg(embedding: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in embedding) + "]"


class PgVectorRetriever(BaseRetriever):
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

        api_key = self.openai_api_key or get_settings().openai_api_key
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY for query embedding generation.")

        embeddings = OpenAIEmbeddings(model=self.embedding_model, api_key=SecretStr(api_key))
        vector_literal = vector_to_pg(embeddings.embed_query(query))

        sql = """
        SELECT
            e.embedding_text,
            p.place_id,
            p.name,
            p.rating,
            p.formatted_address,
            p.primary_type,
            1 - (e.embedding <=> %s::vector) AS similarity
        FROM place_embeddings e
        JOIN places_raw p ON p.place_id = e.place_id
        WHERE e.embedding_model = %s
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s
        """

        with borrow_connection() as conn, conn.cursor() as cur:
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
