"""
Central configuration module.

Loads all settings from environment variables (via .env) and exposes:
  - `settings`   — a singleton Settings instance
  - `get_logger` — factory that returns a named, pre-configured logger
"""

import logging
import sys
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolved once at import time; all path defaults derive from this.
BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ─────────────────────────────────────────────────────────────────
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model_name: str = Field(
        default="qwen/qwen3.8-27b", alias="GROQ_MODEL_NAME"
    )

    # ── Embeddings ───────────────────────────────────────────────────────────
    embedding_model_name: str = Field(
        default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL_NAME"
    )

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    chroma_collection_name: str = Field(
        default="parking_knowledge", alias="CHROMA_COLLECTION_NAME"
    )

    # ── RAG ──────────────────────────────────────────────────────────────────
    retrieval_k: int = Field(default=4, alias="RETRIEVAL_K")

    # ── Admin ─────────────────────────────────────────────────────────────────
    admin_password: str = Field(default="", alias="ADMIN_PASSWORD")

    # ── MCP (Stage 3) ─────────────────────────────────────────────────────────
    # Separate random secret used to sign time-based MCP auth tokens.
    # Generate with: python -c "import secrets; print(secrets.token_hex(32))"
    # Must NOT be derived from or equal to ADMIN_PASSWORD.
    mcp_shared_secret: str = Field(default="", alias="MCP_SHARED_SECRET")

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # ── Derived paths (not from env — computed below) ────────────────────────
    # These are declared as properties rather than fields so they always
    # reflect the true BASE_DIR regardless of working directory.

    @property
    def base_dir(self) -> Path:
        return BASE_DIR

    @property
    def data_dir(self) -> Path:
        return BASE_DIR / "data"

    @property
    def documents_dir(self) -> Path:
        return BASE_DIR / "data" / "documents"

    @property
    def evaluation_dir(self) -> Path:
        return BASE_DIR / "data" / "evaluation"

    @property
    def storage_dir(self) -> Path:
        return BASE_DIR / "storage"

    @property
    def chroma_dir(self) -> Path:
        return BASE_DIR / "storage" / "chroma"

    @property
    def db_path(self) -> Path:
        return BASE_DIR / "storage" / "parking.db"

    @property
    def reports_dir(self) -> Path:
        return BASE_DIR / "reports"

    @property
    def approval_log_file(self) -> Path:
        import os
        custom = os.environ.get("APPROVAL_LOG_FILE", "").strip()
        if custom:
            return Path(custom)
        return BASE_DIR / "storage" / "approved_reservations.txt"

    def ensure_directories(self) -> None:
        """Create all required runtime directories if they do not exist."""
        dirs = [
            self.data_dir,
            self.documents_dir,
            self.evaluation_dir,
            self.storage_dir,
            self.chroma_dir,
            self.reports_dir,
        ]
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def _load_settings() -> Settings:
    """Load and cache the settings singleton."""
    return Settings()


# Public singleton — import this everywhere.
settings: Settings = _load_settings()


# ── Logging ──────────────────────────────────────────────────────────────────

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_logging_configured = False


def _configure_root_logger() -> None:
    """Configure the root logger once for the whole application."""
    global _logging_configured
    if _logging_configured:
        return

    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid adding duplicate handlers if called multiple times (e.g. in tests).
    if not root.handlers:
        root.addHandler(handler)

    # Silence noisy third-party loggers at WARNING unless debug mode is active.
    if level > logging.DEBUG:
        for noisy in (
            "httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers",
            "presidio-analyzer", "presidio_analyzer",
        ):
            logging.getLogger(noisy).setLevel(logging.ERROR)

    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.  Configures the root logger on first call."""
    _configure_root_logger()
    return logging.getLogger(name)
