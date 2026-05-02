from __future__ import annotations

from unittest.mock import MagicMock

from pydantic import SecretStr

from app.retriever import PgVectorRetriever


class FakeCursor:
    def __init__(self, rows: list[tuple]) -> None:
        self.rows = rows
        self.executed_sql = ""
        self.executed_params: tuple | None = None

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, sql: str, params: tuple) -> None:
        self.executed_sql = sql
        self.executed_params = params

    def fetchall(self) -> list[tuple]:
        return self.rows


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self._cursor = cursor

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def cursor(self) -> FakeCursor:
        return self._cursor


def test_get_relevant_documents_formats_vector_and_maps_metadata(mocker) -> None:
    rows = [
        (
            "Name: Taqueria Example\nPrimary Type: mexican_restaurant",
            "abc123",
            "Taqueria Example",
            4.7,
            "123 Mission St, San Francisco, CA",
            "mexican_restaurant",
            0.982,
        )
    ]
    fake_cursor = FakeCursor(rows)
    fake_connection = FakeConnection(fake_cursor)
    mocker.patch("app.retriever.borrow_connection", return_value=fake_connection)

    embeddings = mocker.Mock()
    embeddings.embed_query.return_value = [0.125, 0.5, 1.0]
    embeddings_cls = mocker.patch("app.retriever.OpenAIEmbeddings", return_value=embeddings)

    retriever = PgVectorRetriever(
        embedding_model="text-embedding-3-small",
        k=3,
        openai_api_key="test-key",
    )

    documents = retriever._get_relevant_documents("best tacos", run_manager=MagicMock())

    embeddings_cls.assert_called_once_with(
        model="text-embedding-3-small",
        api_key=SecretStr("test-key"),
    )
    embeddings.embed_query.assert_called_once_with("best tacos")
    assert "JOIN places_raw" in fake_cursor.executed_sql
    assert "ORDER BY e.embedding <=> %s::vector" in fake_cursor.executed_sql
    assert fake_cursor.executed_params == (
        "[0.125,0.5,1.0]",
        "text-embedding-3-small",
        "[0.125,0.5,1.0]",
        3,
    )

    assert len(documents) == 1
    document = documents[0]
    assert document.page_content == rows[0][0]
    assert document.metadata == {
        "place_id": "abc123",
        "name": "Taqueria Example",
        "rating": 4.7,
        "address": "123 Mission St, San Francisco, CA",
        "primary_type": "mexican_restaurant",
        "similarity": 0.982,
    }
