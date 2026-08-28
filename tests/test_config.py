"""Tests for application configuration loading."""

from pathlib import Path

import pytest

from config import AppConfig, load_config


class TestLoadConfig:
    """Tests for load_config's YAML parsing and env-var overrides."""

    def test_missing_file_returns_defaults(self, tmp_path: Path) -> None:
        """A config path that doesn't exist yields plain dataclass defaults."""
        config = load_config(str(tmp_path / "does_not_exist.yaml"))

        assert config == AppConfig()

    def test_loads_sub_config_values(self, tmp_path: Path) -> None:
        """Values under a sub-config key override that dataclass's defaults."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "discovery:\n  top_k: 7\n  confidence_threshold: 0.5\nrag:\n  chunk_size: 1000\n"
        )

        config = load_config(str(config_file))

        assert config.discovery.top_k == 7
        assert config.discovery.confidence_threshold == 0.5
        assert config.rag.chunk_size == 1000
        assert config.rag.chunk_overlap == 100  # untouched default

    def test_loads_top_level_values(self, tmp_path: Path) -> None:
        """Top-level scalar keys override AppConfig's own defaults."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("data_dir: custom_data\nlog_level: DEBUG\n")

        config = load_config(str(config_file))

        assert config.data_dir == "custom_data"
        assert config.log_level == "DEBUG"

    def test_malformed_yaml_falls_back_to_defaults(self, tmp_path: Path) -> None:
        """A YAML file that doesn't parse to a dict is treated as absent."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("- just\n- a\n- list\n")

        config = load_config(str(config_file))

        assert config == AppConfig()

    def test_database_url_env_var_overrides_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DATABASE_URL in the environment takes precedence over the YAML value."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("database_url: postgresql://from-yaml/db\n")
        monkeypatch.setenv("DATABASE_URL", "postgresql://from-env/db")

        config = load_config(str(config_file))

        assert config.database_url == "postgresql://from-env/db"

    def test_database_url_env_var_unset_keeps_yaml_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without DATABASE_URL set, the YAML-configured value is used."""
        monkeypatch.delenv("DATABASE_URL", raising=False)
        config_file = tmp_path / "config.yaml"
        config_file.write_text("database_url: postgresql://from-yaml/db\n")

        config = load_config(str(config_file))

        assert config.database_url == "postgresql://from-yaml/db"
