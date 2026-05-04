from __future__ import annotations

try:
    import pysqlite3  # type: ignore
    import sys

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    OPENROUTER_BASE_URL,
    load_settings,
)


def get_embeddings() -> Embeddings:
    settings = load_settings()
    api_key = settings["openrouter_api_key"]

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is required for embeddings.")
    if not EMBEDDING_MODEL.endswith(":free"):
        raise ValueError("Only OpenRouter free embedding models are allowed.")

    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=api_key,
        openai_api_base=OPENROUTER_BASE_URL,
    )


def ingest_documents(
    documents: list[Document],
    reset: bool = False,
) -> Chroma:
    if reset and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )

    persist = getattr(vectorstore, "persist", None)
    if callable(persist):
        persist()

    return vectorstore


def load_vector_store() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def get_retriever():
    vectorstore = load_vector_store()
    return vectorstore.as_retriever(search_kwargs={"k": 4})
