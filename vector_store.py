from __future__ import annotations

try:
    import pysqlite3  # type: ignore
    import sys

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import shutil

import httpx
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    OPENROUTER_BASE_URL,
    load_settings,
)


class OpenRouterFreeEmbeddings(Embeddings):
    def __init__(self, api_key: str, batch_size: int = 16):
        self.api_key = api_key
        self.batch_size = batch_size
        self.endpoint = f"{OPENROUTER_BASE_URL}/embeddings"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            embeddings.extend(self._embed_batch(batch))
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = httpx.post(
            self.endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": EMBEDDING_MODEL, "input": texts},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") or []

        if len(data) != len(texts):
            raise ValueError(
                f"Expected {len(texts)} embeddings, received {len(data)}."
            )

        return [item["embedding"] for item in data]


def get_embeddings() -> Embeddings:
    settings = load_settings()
    api_key = settings["openrouter_api_key"]

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is required for embeddings.")
    if not EMBEDDING_MODEL.endswith(":free"):
        raise ValueError("Only OpenRouter free embedding models are allowed.")

    return OpenRouterFreeEmbeddings(api_key=api_key)


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
