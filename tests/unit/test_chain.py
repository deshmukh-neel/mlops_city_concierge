from __future__ import annotations

import pytest
from pydantic import SecretStr

from app.chain import BuiltChain, build_rag_chain


def test_build_rag_chain_supports_openai(mocker) -> None:
    retriever = object()
    chain = mocker.Mock()
    chain.invoke.return_value = {"result": "Try Taqueria Example.", "source_documents": []}

    retriever_cls = mocker.patch("app.chain.PgVectorRetriever", return_value=retriever)
    openai_cls = mocker.patch("app.chain.ChatOpenAI", return_value="openai-llm")
    mocker.patch("app.chain.ChatGoogleGenerativeAI")
    from_chain = mocker.patch("app.chain.RetrievalQA.from_chain_type", return_value=chain)

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
    openai_cls.assert_called_once_with(
        model="gpt-4o-mini",
        api_key=SecretStr("openai-key"),
        temperature=0.2,
    )
    from_chain.assert_called_once_with(
        llm="openai-llm",
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    assert built.chain.invoke({"query": "Best tacos"}) == {
        "result": "Try Taqueria Example.",
        "source_documents": [],
    }


def test_build_rag_chain_supports_gemini(mocker) -> None:
    retriever = object()
    chain = mocker.Mock()
    chain.invoke.return_value = {"result": "Try Del Popolo.", "source_documents": []}

    mocker.patch("app.chain.PgVectorRetriever", return_value=retriever)
    mocker.patch("app.chain.ChatOpenAI")
    gemini_cls = mocker.patch(
        "app.chain.ChatGoogleGenerativeAI",
        return_value="gemini-llm",
    )
    mocker.patch("app.chain.RetrievalQA.from_chain_type", return_value=chain)

    built = build_rag_chain(
        connection_string="postgresql://example",
        api_key="gemini-key",
        llm_provider="gemini",
        chat_model="gemini-2.5-flash",
        k=2,
        temperature=0.1,
    )

    assert built.llm == "gemini-llm"
    gemini_cls.assert_called_once_with(
        model="gemini-2.5-flash",
        google_api_key=SecretStr("gemini-key"),
        temperature=0.1,
    )
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
