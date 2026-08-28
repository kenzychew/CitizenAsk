"""Custom exception hierarchy for CitizenAsk."""


class CitizenAskError(Exception):
    """Base exception for all CitizenAsk errors."""


class ConfigError(CitizenAskError):
    """Raised when configuration is invalid."""


class DiscoveryError(CitizenAskError):
    """Raised when dataset discovery fails."""


class DataGovSgError(CitizenAskError):
    """Raised when a data.gov.sg API call fails."""


class DatasetNotFoundError(DataGovSgError):
    """Raised when a dataset_id has no live datastore_search resource."""


class RateLimitError(DataGovSgError):
    """Raised when data.gov.sg returns HTTP 429. Retried internally by the client."""


class RagError(CitizenAskError):
    """Raised when the RAG fallback pipeline fails."""


class IngestionError(RagError):
    """Raised when document ingestion into pgvector fails."""


class EmbeddingError(RagError):
    """Raised when local embedding generation fails."""


class RetrievalError(RagError):
    """Raised when vector retrieval fails."""


class GenerationError(CitizenAskError):
    """Raised when LLM generation fails."""


class AbstentionError(CitizenAskError):
    """Raised when the agent should abstain but the caller ignored it."""
