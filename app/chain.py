from __future__ import annotations

from langchain.chains import RetrievalQA

from .config import get_settings
from .providers import get_provider
from .retriever import PgVectorRetriever


def build_rag_chain(
    api_key: str,
    llm_provider: str,
    chat_model: str,
    k: int,
    temperature: float = 0.0,
) -> RetrievalQA:
    settings = get_settings()
    retriever = PgVectorRetriever(
        embedding_model=settings.openai_embedding_model,
        k=k,
        openai_api_key=settings.openai_api_key,
    )

    llm = get_provider(llm_provider).build_llm(chat_model, api_key, temperature)

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
