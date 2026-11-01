"""Tests for src/config/settings.py."""

import logging
import os
from pathlib import Path
from unittest.mock import patch

import pytest


class TestSettings:
    def test_default_values_are_applied(self):
        """Settings should have sensible defaults without a .env file."""
        from src.config.settings import Settings

        # Provide required key; strip LOG_LEVEL so the default kicks in.
        env = {"GROQ_API_KEY": "test-key"}
        with patch.dict(os.environ, env, clear=False):
            env_backup = os.environ.pop("LOG_LEVEL", None)
            try:
                s = Settings()
            finally:
                if env_backup is not None:
                    os.environ["LOG_LEVEL"] = env_backup

        assert s.groq_model_name == "qwen/qwen3.8-27b"
        assert s.embedding_model_name == "all-MiniLM-L6-v2"
        assert s.chroma_collection_name == "parking_knowledge"
        assert s.retrieval_k == 4
        assert s.log_level == "INFO"

    def test_env_vars_override_defaults(self):
        """Env vars should override every default."""
        from src.config.settings import Settings

        overrides = {
            "GROQ_API_KEY": "my-real-key",
            "GROQ_MODEL_NAME": "llama-3.1-8b-instant",
            "RETRIEVAL_K": "6",
            "LOG_LEVEL": "DEBUG",
        }
        with patch.dict(os.environ, overrides, clear=False):
            s = Settings()

        assert s.groq_api_key == "my-real-key"
        assert s.groq_model_name == "llama-3.1-8b-instant"
        assert s.retrieval_k == 6
        assert s.log_level == "DEBUG"

    def test_path_properties_are_under_base_dir(self):
        """All derived paths must be children of base_dir."""
        from src.config.settings import BASE_DIR, Settings

        s = Settings()
        paths = [
            s.data_dir,
            s.documents_dir,
            s.evaluation_dir,
            s.storage_dir,
            s.chroma_dir,
            s.db_path,
            s.reports_dir,
        ]
        for p in paths:
            assert str(p).startswith(str(BASE_DIR)), (
                f"{p} is not under BASE_DIR ({BASE_DIR})"
            )

    def test_ensure_directories_creates_missing_dirs(self, tmp_path, monkeypatch):
        """ensure_directories() should create all runtime directories."""
        import sys

        # `from .settings import settings` in __init__.py shadows the submodule
        # name on the package, so we must go through sys.modules for the real
        # module object.
        import src.config.settings  # noqa: F401 — ensures entry in sys.modules
        settings_mod = sys.modules["src.config.settings"]

        monkeypatch.setattr(settings_mod, "BASE_DIR", tmp_path)

        s = settings_mod.Settings()
        s.ensure_directories()

        assert (tmp_path / "data" / "documents").exists()
        assert (tmp_path / "data" / "evaluation").exists()
        assert (tmp_path / "storage" / "chroma").exists()
        assert (tmp_path / "reports").exists()


class TestGetLogger:
    def test_returns_named_logger(self):
        """get_logger should return a Logger with the given name."""
        from src.config import get_logger

        log = get_logger("test.module")
        assert isinstance(log, logging.Logger)
        assert log.name == "test.module"

    def test_multiple_calls_return_same_logger(self):
        """Calling get_logger twice with the same name returns identical objects."""
        from src.config import get_logger

        log1 = get_logger("same.name")
        log2 = get_logger("same.name")
        assert log1 is log2

    def test_root_logger_has_handler_after_get_logger(self):
        """Root logger should have at least one handler after get_logger is called."""
        from src.config import get_logger

        get_logger("any.name")
        assert len(logging.getLogger().handlers) >= 1
