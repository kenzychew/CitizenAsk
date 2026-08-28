"""Application configuration, loaded from configs/config.yaml plus .env overrides."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


@dataclass
class DiscoveryConfig:
    """Configuration for the catalog discovery tool.

    Attributes:
        top_k: Number of candidate datasets to return.
        confidence_threshold: Minimum lexical match score to treat a
            dataset as a genuine match rather than abstaining.
    """

    top_k: int = 3
    confidence_threshold: float = 4.5


@dataclass
class DataGovSgConfig:
    """Configuration for the data.gov.sg API client.

    Attributes:
        base_url: Base URL for the CKAN-style action API.
        timeout_seconds: Per-request timeout.
        max_retries: Number of retries on transient failures.
        default_limit: Default row limit for datastore_search calls.
    """

    base_url: str = "https://data.gov.sg/api/action"
    timeout_seconds: float = 15.0
    max_retries: int = 3
    default_limit: int = 100


@dataclass
class RagConfig:
    """Configuration for the pgvector-backed RAG fallback.

    Attributes:
        embedding_model: Name of the sentence-transformers model.
        embedding_dim: Dimensionality of the embedding vectors.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlapping characters between adjacent chunks.
        top_k: Number of chunks retrieved per query.
        table_name: Name of the chunks table in Postgres.
    """

    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 4
    table_name: str = "doc_chunks"


@dataclass
class GenerationConfig:
    """Configuration for the OpenAI generation step.

    Attributes:
        model: OpenAI chat model name.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in the generated response.
    """

    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 1024


@dataclass
class AppConfig:
    """Top-level application configuration.

    Attributes:
        discovery: Catalog discovery config.
        datagovsg: data.gov.sg client config.
        rag: RAG fallback config.
        generation: Generation config.
        database_url: PostgreSQL connection string for pgvector.
        data_dir: Directory containing RAG source documents.
        log_level: Logging level.
    """

    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    datagovsg: DataGovSgConfig = field(default_factory=DataGovSgConfig)
    rag: RagConfig = field(default_factory=RagConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    database_url: str = "postgresql://citizenask:changeme@localhost:5432/citizenask"
    data_dir: str = "data"
    log_level: str = "INFO"


def _sub_config(cfg: dict[str, object], key: str) -> dict[str, Any]:
    """Extract a sub-config dict, returning an empty dict if missing or malformed.

    Args:
        cfg: Parent config dictionary.
        key: Key to extract.

    Returns:
        The sub-config dict, or an empty dict. Typed as dict[str, Any] rather
        than dict[str, object] so callers can pass it as **kwargs into a
        dataclass constructor; the YAML source means its true value types
        are unknown until the dataclass fields validate them.
    """
    value = cfg.get(key)
    return value if isinstance(value, dict) else {}


def load_config(config_path: str = "configs/config.yaml") -> AppConfig:
    """Load application config from YAML, with .env-driven overrides.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        A populated AppConfig. Falls back to dataclass defaults for any
        value missing from the YAML file, and the file itself is optional.
    """
    load_dotenv(".env", override=False)

    cfg: dict[str, object] = {}
    path = Path(config_path)
    if path.exists():
        with open(path, encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
            if isinstance(loaded, dict):
                cfg = loaded

    top_level_keys = ["database_url", "data_dir", "log_level"]
    top_level_overrides = {k: str(cfg[k]) for k in top_level_keys if k in cfg}

    app_config = AppConfig(
        discovery=DiscoveryConfig(**_sub_config(cfg, "discovery")),
        datagovsg=DataGovSgConfig(**_sub_config(cfg, "datagovsg")),
        rag=RagConfig(**_sub_config(cfg, "rag")),
        generation=GenerationConfig(**_sub_config(cfg, "generation")),
        **top_level_overrides,
    )

    database_url = os.getenv("DATABASE_URL")
    if database_url:
        app_config.database_url = database_url

    return app_config
