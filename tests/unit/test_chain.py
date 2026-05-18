from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.chain import BuiltChain, build_rag_chain, build_retrieval_qa


def test_build_rag_chain_supports_openai(mocker) -> None:
    retriever = object()
    chain = mocker.Mock()
    chain.invoke.return_value = {"result": "Try Taqueria Example.", "source_documents": []}

    retriever_cls = mocker.patch("app.chain.PgVectorRetriever", return_value=retriever)
    factory = mocker.patch("app.chain.build_chat_model", return_value="openai-llm")
    build_qa = mocker.patch("app.chain.build_retrieval_qa", return_value=chain)

    built = build_rag_chain(
        connection_string="postgresql://example",
        api_key="openai-key",
        llm_provider="openai",
        chat_model="gpt-4o-mini",
        k=4,
        temperature=0.2,
    )

    assert isinstance(built, BuiltChain)
    assert built.llm == "openai-llm"
    retriever_cls.assert_called_once()
    factory.assert_called_once_with("openai", "gpt-4o-mini", temperature=0.2)
    build_qa.assert_called_once_with(retriever=retriever, llm="openai-llm")
    assert built.chain.invoke({"query": "Best tacos"}) == {
        "result": "Try Taqueria Example.",
        "source_documents": [],
    }


def test_build_rag_chain_supports_gemini(mocker) -> None:
    retriever = object()
    chain = mocker.Mock()
    chain.invoke.return_value = {"result": "Try Del Popolo.", "source_documents": []}

    mocker.patch("app.chain.PgVectorRetriever", return_value=retriever)
    factory = mocker.patch("app.chain.build_chat_model", return_value="gemini-llm")
    mocker.patch("app.chain.build_retrieval_qa", return_value=chain)

    built = build_rag_chain(
        connection_string="postgresql://example",
        api_key="gemini-key",
        llm_provider="gemini",
        chat_model="gemini-2.5-flash",
        k=2,
        temperature=0.1,
    )

    assert built.llm == "gemini-llm"
    factory.assert_called_once_with("gemini", "gemini-2.5-flash", temperature=0.1)
    assert built.chain.invoke({"query": "Pizza"}) == {
        "result": "Try Del Popolo.",
        "source_documents": [],
    }


def test_build_rag_chain_rejects_invalid_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported llm_provider"):
        build_rag_chain(
            connection_string="postgresql://example",
            api_key="unused",
            llm_provider="anthropic",
            chat_model="claude",
            k=3,
        )


class _FakeRetriever:
    """Minimal retriever stub exposing the BaseRetriever.invoke() contract."""

    def __init__(self, docs: list[Document]) -> None:
        self._docs = docs

    def invoke(self, query: str) -> list[Document]:  # noqa: ARG002
        return self._docs


def test_build_retrieval_qa_preserves_legacy_output_contract() -> None:
    docs = [
        Document(page_content="Taqueria Example serves great tacos."),
        Document(page_content="Open until 10pm."),
    ]
    llm = FakeListChatModel(responses=["You should try Taqueria Example."])

    chain = build_retrieval_qa(retriever=_FakeRetriever(docs), llm=llm)
    out = chain.invoke({"query": "Best tacos?"})

    assert out == {
        "result": "You should try Taqueria Example.",
        "source_documents": docs,
    }
