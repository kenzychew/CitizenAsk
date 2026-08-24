"""Shared data models for CitizenAsk."""

from dataclasses import dataclass, field
from enum import StrEnum


class DatasetKind(StrEnum):
    """Whether a curated dataset is a queryable table or a document corpus."""

    STRUCTURED = "structured"
    DOCUMENT = "document"


@dataclass
class DatasetEntry:
    """One curated data.gov.sg dataset in the registry.

    Attributes:
        dataset_id: The data.gov.sg resource_id (e.g. "d_ebc5..."), used as
            the datastore_search resource_id for structured entries. Empty
            for document entries with no live datastore resource.
        title: Human-readable dataset title.
        agency: Owning government agency.
        description: Short description of the dataset's contents.
        tags: Topic tags used for lexical discovery matching.
        kind: Whether this is a structured (queryable) or document dataset.
        fields: Column names, for structured datasets only.
        source_url: The data.gov.sg dataset page URL.
    """

    dataset_id: str
    title: str
    agency: str
    description: str
    tags: list[str]
    kind: DatasetKind
    fields: list[str] = field(default_factory=list)
    source_url: str = ""


@dataclass
class DiscoveryMatch:
    """A scored dataset match returned by the discovery tool.

    Attributes:
        dataset: The matched registry entry.
        score: Lexical match score, higher is better. Not bounded to [0, 1].
    """

    dataset: DatasetEntry
    score: float


@dataclass
class DatastoreQueryResult:
    """Result of a structured query against a data.gov.sg dataset.

    Attributes:
        dataset_id: The resource_id queried.
        records: Raw rows returned, each a field-name to value mapping.
        total: Total number of matching rows on the server, which can exceed
            len(records) when the query was paginated.
    """

    dataset_id: str
    records: list[dict[str, str]]
    total: int


@dataclass
class DocChunk:
    """A chunk of a RAG document with its embedding and provenance.

    Attributes:
        chunk_id: Unique identifier for this chunk.
        text: The chunk text content.
        source: Source document identifier (e.g. filename).
        index: Position index within the source document.
        embedding: Dense vector embedding, empty before the embedding step.
    """

    chunk_id: str
    text: str
    source: str
    index: int
    embedding: list[float] = field(default_factory=list)


@dataclass
class RetrievedChunk:
    """A RAG chunk returned from retrieval, with its similarity score."""

    chunk: DocChunk
    score: float


@dataclass
class AgentAnswer:
    """The final answer returned by the agent for one query.

    Attributes:
        answer: The generated answer text, or an abstention message.
        abstained: Whether the agent declined to answer.
        route: Which path produced the answer ("structured", "rag", or "abstain").
        dataset_id: The dataset_id used, if any.
        citations: Source identifiers backing the answer (dataset title, or
            document sources for the RAG path).
    """

    answer: str
    abstained: bool
    route: str
    dataset_id: str | None = None
    citations: list[str] = field(default_factory=list)
