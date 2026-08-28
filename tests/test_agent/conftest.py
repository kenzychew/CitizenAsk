"""Fakes for agent graph tests: discovery, data.gov.sg client, and retriever."""

from exceptions import DataGovSgError
from schemas import DatastoreQueryResult, DiscoveryMatch, RetrievedChunk


class StubDiscovery:
    """Returns a fixed list of matches regardless of the question asked."""

    def __init__(self, matches: list[DiscoveryMatch]) -> None:
        """Store the canned matches."""
        self._matches = matches

    def discover(self, question: str, top_k: int = 3) -> list[DiscoveryMatch]:
        """Return up to top_k of the canned matches."""
        return self._matches[:top_k]

    def __len__(self) -> int:
        """Return the number of canned matches, matching CatalogDiscovery's interface."""
        return len(self._matches)


class StubDataGovSgClient:
    """Returns a canned result or raises a canned error from fetch_all_matching."""

    def __init__(
        self,
        result: DatastoreQueryResult | None = None,
        error: DataGovSgError | None = None,
        sample: list[dict[str, str]] | None = None,
        sample_error: DataGovSgError | None = None,
    ) -> None:
        """Store the canned result or error."""
        self._result = result
        self._error = error
        self._sample = sample if sample is not None else []
        self._sample_error = sample_error
        self.calls: list[tuple[str, dict[str, str] | None]] = []

    async def fetch_all_matching(
        self, dataset_id: str, filters: dict[str, str] | None = None, **kwargs: object
    ) -> DatastoreQueryResult:
        """Record the call and return the canned result, or raise the canned error."""
        self.calls.append((dataset_id, filters))
        if self._error:
            raise self._error
        assert self._result is not None
        return self._result

    async def sample_rows(self, dataset_id: str, limit: int = 3) -> list[dict[str, str]]:
        """Return the canned sample rows, or raise the canned sample-fetch error."""
        if self._sample_error:
            raise self._sample_error
        return self._sample


class StubRetriever:
    """Returns a fixed list of chunks regardless of the query asked."""

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        """Store the canned chunks."""
        self._chunks = chunks

    async def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Return the canned chunks."""
        return self._chunks
