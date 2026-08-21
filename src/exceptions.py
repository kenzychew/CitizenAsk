"""Custom exception hierarchy for supportivebot."""


class SupportiveBotError(Exception):
    """Base exception for all supportivebot errors."""


class ConfigError(SupportiveBotError):
    """Raised when configuration is invalid."""


class DiscoveryError(SupportiveBotError):
    """Raised when dataset discovery fails."""


class DataGovSgError(SupportiveBotError):
    """Raised when a data.gov.sg API call fails."""


class DatasetNotFoundError(DataGovSgError):
    """Raised when a dataset_id has no live datastore_search resource."""


class RateLimitError(DataGovSgError):
    """Raised when data.gov.sg returns HTTP 429. Retried internally by the client."""


class RagError(SupportiveBotError):
    """Raised when the RAG fallback pipeline fails."""


class IngestionError(RagError):
    """Raised when document ingestion into pgvector fails."""


class EmbeddingError(RagError):
    """Raised when local embedding generation fails."""


class RetrievalError(RagError):
    """Raised when vector retrieval fails."""


class GenerationError(SupportiveBotError):
    """Raised when LLM generation fails."""


class AbstentionError(SupportiveBotError):
    """Raised when the agent should abstain but the caller ignored it."""
