from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    DirectoryLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR


DOCUMENT_TYPE_BY_EXTENSION = {
    ".docx": "document",
    ".pdf": "pdf",
    ".txt": "text",
}


def load_raw_documents(docs_dir: Path = DOCS_DIR) -> list[Document]:
    if not docs_dir.exists():
        raise FileNotFoundError(f"Document folder not found: {docs_dir}")

    loaders = [
        DirectoryLoader(
            str(docs_dir),
            glob="**/*.docx",
            loader_cls=Docx2txtLoader,
            show_progress=False,
            use_multithreading=True,
        ),
        DirectoryLoader(
            str(docs_dir),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=False,
            use_multithreading=True,
        ),
        DirectoryLoader(
            str(docs_dir),
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "cp1252"},
            show_progress=False,
            use_multithreading=True,
        ),
    ]

    documents: list[Document] = []
    for loader in loaders:
        documents.extend(loader.load())

    if not documents:
        raise ValueError(f"No DOCX, PDF, or TXT documents found in {docs_dir}")

    return documents


def load_and_split_documents(docs_dir: Path = DOCS_DIR) -> list[Document]:
    raw_documents = load_raw_documents(docs_dir)
    documents_by_source = _group_by_source(raw_documents)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunked_documents: list[Document] = []
    ingestion_timestamp = _now_iso()

    for source, source_documents in documents_by_source.items():
        source_path = Path(source)
        file_metadata = _build_file_metadata(source_path, source_documents)
        chunk_index = 0

        for source_document in source_documents:
            page_content = source_document.page_content.strip()
            if not page_content:
                continue

            page_number = _extract_page_number(source_document)
            chunks = splitter.split_text(page_content)

            for chunk_text in chunks:
                metadata = {
                    **file_metadata,
                    "chunk_index": chunk_index,
                    "chunk_size": len(chunk_text),
                    "chunk_overlap": CHUNK_OVERLAP,
                    "ingestion_timestamp": ingestion_timestamp,
                    "page_number": page_number,
                    "section_title": _detect_section_title(chunk_text),
                }
                chunked_documents.append(
                    Document(page_content=chunk_text, metadata=metadata)
                )
                chunk_index += 1

    return chunked_documents


def _group_by_source(documents: list[Document]) -> dict[str, list[Document]]:
    grouped: dict[str, list[Document]] = {}
    for document in documents:
        source = document.metadata.get("source")
        if not source:
            continue
        grouped.setdefault(source, []).append(document)
    return grouped


def _build_file_metadata(source_path: Path, documents: list[Document]) -> dict[str, object]:
    stats = source_path.stat()
    extension = source_path.suffix.lower()
    character_count = sum(len(document.page_content) for document in documents)
    creation_timestamp = getattr(stats, "st_birthtime", stats.st_ctime)

    return {
        "file_name": source_path.name,
        "file_extension": extension,
        "file_size_bytes": stats.st_size,
        "character_count": character_count,
        "document_type": DOCUMENT_TYPE_BY_EXTENSION.get(extension, "document"),
        "creation_date": _timestamp_to_iso(creation_timestamp),
        "last_modified": _timestamp_to_iso(stats.st_mtime),
    }


def _extract_page_number(document: Document) -> int:
    raw_page = document.metadata.get("page")
    if raw_page is None:
        return 1

    try:
        return int(raw_page) + 1
    except (TypeError, ValueError):
        return 1


def _detect_section_title(text: str) -> str:
    for line in text.splitlines():
        candidate = line.strip()
        if 3 <= len(candidate) <= 90:
            return candidate
    return ""


def _timestamp_to_iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
