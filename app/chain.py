from __future__ import annotations

from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from .config import get_settings
from .retriever import PgVectorRetriever


def build_rag_chain(
    connection_string: str,
    api_key: str,
    llm_provider: str,
    chat_model: str,
    k: int,
    temperature: float = 0.0,
) -> RetrievalQA:
    settings = get_settings()
    retriever = PgVectorRetriever(
        connection_string=connection_string,
        embedding_model=settings.openai_embedding_model,
        k=k,
        openai_api_key=settings.openai_api_key,
    )

    provider = llm_provider.lower()
    if provider == "openai":
        llm = ChatOpenAI(model=chat_model, api_key=api_key, temperature=temperature)
    elif provider == "gemini":
        llm = ChatGoogleGenerativeAI(
            model=chat_model,
            google_api_key=api_key,
            temperature=temperature,
        )
    else:
        raise ValueError(f"Unsupported llm_provider: {llm_provider}")

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
