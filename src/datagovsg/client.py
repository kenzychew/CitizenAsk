"""Thin async client for the real data.gov.sg datastore_search API.

No API key is required for read access. See
https://guide.data.gov.sg/developer-guide/dataset-apis for the underlying
CKAN-style `datastore_search` action this wraps.
"""

import json
import logging

import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import DataGovSgConfig
from src.exceptions import DataGovSgError, DatasetNotFoundError, RateLimitError
from src.schemas import DatastoreQueryResult

logger = logging.getLogger(__name__)


class DataGovSgClient:
    """Async wrapper around the data.gov.sg datastore_search API."""

    def __init__(self, config: DataGovSgConfig) -> None:
        """Initialize the client.

        Args:
            config: data.gov.sg client configuration.
        """
        self._config = config

    async def _get(self, params: dict[str, str]) -> dict[str, object]:
        """Issue one GET request against the datastore_search action.

        Retried internally with backoff, up to the configured max_retries,
        on transport errors and HTTP 429 responses.

        Args:
            params: Query parameters to send.

        Returns:
            The parsed JSON response body.

        Raises:
            DatasetNotFoundError: If the API returns HTTP 404, which is what
                an unknown or non-datastore-active resource_id produces.
            RateLimitError: If the API returns HTTP 429 on every retried
                attempt.
            DataGovSgError: If the request otherwise fails or the response
                is malformed.
        """
        retryer = AsyncRetrying(
            retry=retry_if_exception_type((httpx.TransportError, RateLimitError)),
            stop=stop_after_attempt(self._config.max_retries + 1),
            wait=wait_exponential(multiplier=2, min=2, max=15),
            reraise=True,
        )
        return await retryer(self._get_once, params)

    async def _get_once(self, params: dict[str, str]) -> dict[str, object]:
        """Issue a single GET attempt against the datastore_search action.

        Args:
            params: Query parameters to send.

        Returns:
            The parsed JSON response body.

        Raises:
            DatasetNotFoundError: If the API returns HTTP 404, which is what
                an unknown or non-datastore-active resource_id produces.
            RateLimitError: If the API returns HTTP 429.
            DataGovSgError: If the request otherwise fails or the response
                is malformed.
        """
        url = f"{self._config.base_url}/datastore_search"
        async with httpx.AsyncClient(timeout=self._config.timeout_seconds) as client:
            response = await client.get(url, params=params)

        if response.status_code == 404:
            raise DatasetNotFoundError(
                f"No live datastore resource for resource_id={params.get('resource_id')!r}"
            )
        if response.status_code == 429:
            raise RateLimitError("data.gov.sg rate limit exceeded, retrying with backoff")
        if response.status_code != 200:
            raise DataGovSgError(
                f"data.gov.sg returned HTTP {response.status_code} for {url}: {response.text[:200]}"
            )

        body = response.json()
        if not isinstance(body, dict):
            raise DataGovSgError(f"Unexpected response shape from {url}: {body!r}")
        return body

    async def datastore_search(
        self,
        dataset_id: str,
        filters: dict[str, str] | None = None,
        q: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> DatastoreQueryResult:
        """Query one page of a dataset's rows via datastore_search.

        Args:
            dataset_id: The data.gov.sg resource_id.
            filters: Exact-match column filters, e.g. {"town": "ANG MO KIO"}.
            q: Optional full-text search query.
            limit: Maximum rows to return in this page. Defaults to the
                configured default_limit.
            offset: Row offset for pagination.

        Returns:
            The query result for this page, including the server-reported
            total matching row count.

        Raises:
            DatasetNotFoundError: If dataset_id has no live datastore resource.
            DataGovSgError: If the request otherwise fails.
        """
        params: dict[str, str] = {
            "resource_id": dataset_id,
            "limit": str(limit if limit is not None else self._config.default_limit),
            "offset": str(offset),
        }
        if filters:
            params["filters"] = json.dumps(filters)
        if q:
            params["q"] = q

        body = await self._get(params)

        if not body.get("success"):
            error = body.get("error", {})
            message = error.get("message") if isinstance(error, dict) else str(error)
            raise DatasetNotFoundError(
                f"datastore_search failed for dataset_id={dataset_id!r}: {message}"
            )

        result = body.get("result")
        if not isinstance(result, dict):
            raise DataGovSgError(f"Missing 'result' in datastore_search response: {body!r}")

        records = result.get("records", [])
        if not isinstance(records, list):
            raise DataGovSgError(f"Malformed 'records' in datastore_search response: {result!r}")

        return DatastoreQueryResult(
            dataset_id=dataset_id,
            records=records,
            total=int(result.get("total", len(records))),
        )

    async def fetch_all_matching(
        self,
        dataset_id: str,
        filters: dict[str, str] | None = None,
        q: str | None = None,
        page_size: int = 1000,
        max_records: int = 10000,
    ) -> DatastoreQueryResult:
        """Paginate through all rows matching filters, up to a safety cap.

        Structured-query questions (counts, averages) need every matching
        row rather than one page, but unbounded pagination against a
        287k-row dataset is not something a single query should ever do
        by accident, hence max_records.

        Args:
            dataset_id: The data.gov.sg resource_id.
            filters: Exact-match column filters.
            q: Optional full-text search query.
            page_size: Rows requested per underlying API call.
            max_records: Hard cap on total rows fetched.

        Returns:
            A DatastoreQueryResult whose records list holds every matching
            row up to max_records, and whose total is the server-reported
            true total (which may exceed len(records)).
        """
        all_records: list[dict[str, str]] = []
        offset = 0
        total = 0

        while len(all_records) < max_records:
            page = await self.datastore_search(
                dataset_id, filters=filters, q=q, limit=page_size, offset=offset
            )
            total = page.total
            if not page.records:
                break
            all_records.extend(page.records)
            offset += len(page.records)
            if len(page.records) < page_size:
                break

        return DatastoreQueryResult(
            dataset_id=dataset_id,
            records=all_records[:max_records],
            total=total,
        )
