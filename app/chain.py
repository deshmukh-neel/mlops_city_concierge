from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from .config import get_settings
from .gemini_compat import patch_langchain_google_genai_for_gemini3
from .retriever import PgVectorRetriever

# Default prompt of the legacy langchain RetrievalQA "stuff" chain, preserved
# verbatim so migrating off langchain.chains (removed in langchain 1.x) does not
# change model behavior.
_STUFF_PROMPT = ChatPromptTemplate.from_template(
    "Use the following pieces of context to answer the question at the end. "
    "If you don't know the answer, just say that you don't know, don't try to "
    "make up an answer.\n\n{context}\n\nQuestion: {question}\nHelpful Answer:"
)


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_retrieval_qa(retriever: BaseRetriever, llm: BaseChatModel) -> Runnable:
    """LCEL equivalent of ``RetrievalQA.from_chain_type(chain_type="stuff",
    return_source_documents=True)``.

    langchain 1.x removed the legacy ``langchain.chains`` module. This composes
    the same behaviour from stable ``langchain_core`` primitives and preserves
    the original output contract: ``invoke({"query": q})`` returns
    ``{"result": <answer>, "source_documents": [<Document>, ...]}``.
    """
    answer = _STUFF_PROMPT | llm | StrOutputParser()

    def _invoke(payload: dict[str, Any]) -> dict[str, Any]:
        query = payload["query"]
        docs = retriever.invoke(query)
        result = answer.invoke({"context": _format_docs(docs), "question": query})
        return {"result": result, "source_documents": docs}

    return RunnableLambda(_invoke)


@dataclass
class BuiltChain:
    chain: Runnable
    llm: BaseChatModel


def build_rag_chain(
    connection_string: str,
    api_key: str,
    llm_provider: str,
    chat_model: str,
    k: int,
    temperature: float = 0.0,
) -> BuiltChain:
    settings = get_settings()
    retriever = PgVectorRetriever(
        connection_string=connection_string,
        embedding_model=settings.openai_embedding_model,
        k=k,
        openai_api_key=settings.openai_api_key,
    )

    provider = llm_provider.lower()
    llm: BaseChatModel
    if provider == "openai":
        llm = ChatOpenAI(model=chat_model, api_key=SecretStr(api_key), temperature=temperature)
    elif provider == "gemini":
        patch_langchain_google_genai_for_gemini3()
        llm = ChatGoogleGenerativeAI(
            model=chat_model,
            google_api_key=SecretStr(api_key),
            temperature=temperature,
        )
    else:
        raise ValueError(f"Unsupported llm_provider: {llm_provider}")

    chain = build_retrieval_qa(retriever=retriever, llm=llm)
    return BuiltChain(chain=chain, llm=llm)
