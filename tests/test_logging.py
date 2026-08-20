"""Tests for the logging setup helper."""

import logging
from pathlib import Path

from src.logging import setup_logging


class TestSetupLogging:
    """Tests for setup_logging's YAML config loading and fallback behaviour."""

    def test_loads_valid_yaml_config(self, tmp_path: Path) -> None:
        """A valid logging YAML config is applied via dictConfig."""
        config_file = tmp_path / "logging.yaml"
        config_file.write_text(
            "version: 1\n"
            "disable_existing_loggers: false\n"
            "handlers:\n"
            "  console:\n"
            "    class: logging.StreamHandler\n"
            "    level: INFO\n"
            "root:\n"
            "  level: WARNING\n"
            "  handlers: [console]\n"
        )

        setup_logging(str(config_file))

        assert logging.getLogger().level == logging.WARNING

    def test_missing_file_falls_back_to_basic_config(self, tmp_path: Path) -> None:
        """A missing config path falls back to basicConfig without raising."""
        setup_logging(str(tmp_path / "does_not_exist.yaml"), default_level=logging.ERROR)

        assert logging.getLogger().level in (logging.WARNING, logging.ERROR)

    def test_malformed_yaml_falls_back_to_basic_config(self, tmp_path: Path) -> None:
        """Invalid YAML content falls back rather than raising."""
        config_file = tmp_path / "logging.yaml"
        config_file.write_text(":\n  - not: valid: yaml:\n")

        setup_logging(str(config_file))  # should not raise

    def test_log_dir_redirects_file_handler_filenames(self, tmp_path: Path) -> None:
        """A log_dir override moves file handler filenames into that directory."""
        config_file = tmp_path / "logging.yaml"
        log_dir = tmp_path / "custom_logs"
        config_file.write_text(
            "version: 1\n"
            "disable_existing_loggers: false\n"
            "handlers:\n"
            "  file:\n"
            "    class: logging.FileHandler\n"
            "    level: DEBUG\n"
            "    filename: logs/app.log\n"
            "root:\n"
            "  level: DEBUG\n"
            "  handlers: [file]\n"
        )

        setup_logging(str(config_file), log_dir=str(log_dir))

        assert log_dir.exists()
        handler = logging.getLogger().handlers[0]
        assert isinstance(handler, logging.FileHandler)
        assert Path(handler.baseFilename).parent == log_dir
