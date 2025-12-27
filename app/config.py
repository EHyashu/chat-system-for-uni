from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    # Directories
    base_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    documents_dir: Path = Field(default_factory=lambda: Path("data/documents"))
    index_dir: Path = Field(default_factory=lambda: Path("data/index"))

    # Embeddings
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # LLM configuration
    llm_provider: Literal["openai", "dummy"] = "openai"
    llm_model_name: str = "gpt-4o-mini"
    openai_api_key: str | None = None

    # Retrieval
    top_k: int = 5
    similarity_threshold: float = 0.2  # Lowered from 0.3 for better recall

    # RAG behavior
    max_context_chars: int = 6000

    class Config:
        env_prefix = "UNI_RAG_"
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    settings.documents_dir = (settings.base_dir / settings.documents_dir).resolve()
    settings.index_dir = (settings.base_dir / settings.index_dir).resolve()
    settings.documents_dir.mkdir(parents=True, exist_ok=True)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    return settings
