from pathlib import Path

from dotenv import load_dotenv
import os


PROJECT_ROOT = Path(__file__).resolve().parent
DOCS_DIR = PROJECT_ROOT / "hr_documents_pack" / "initial_docs"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "vbo-aillm-bc-rag"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
CHAT_MODEL = "openai:openrouter/free"
EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
DEFAULT_THREAD_ID = "local-cli-session"


def load_settings() -> dict[str, str]:
    load_dotenv(PROJECT_ROOT / ".env")

    return {
        "openrouter_api_key": os.getenv("OPENROUTER_API_KEY", "").strip(),
        "db_uri": os.getenv(
            "DB_URI",
            "postgresql://postgres:postgres@localhost:5432/hr_rag",
        ).strip(),
    }
