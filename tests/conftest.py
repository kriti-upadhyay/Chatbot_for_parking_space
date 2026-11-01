"""
Shared pytest fixtures for all test modules.

Fixtures here are available to every test file without importing.
"""

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def set_test_env_vars():
    """
    Inject safe test values for required env vars before any module imports
    settings.  This prevents failures when GROQ_API_KEY is not set locally.
    """
    os.environ.setdefault("GROQ_API_KEY", "test-key-not-real")
    os.environ.setdefault("LOG_LEVEL", "WARNING")  # keep test output quiet


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    """Temporary directory that is cleaned up after each test."""
    return tmp_path


@pytest.fixture()
def tmp_storage(tmp_path: Path) -> dict[str, Path]:
    """Isolated storage paths pointing to a temp directory."""
    chroma = tmp_path / "chroma"
    chroma.mkdir()
    db = tmp_path / "parking.db"
    docs = tmp_path / "documents"
    docs.mkdir()
    return {"chroma_dir": chroma, "db_path": db, "documents_dir": docs}
