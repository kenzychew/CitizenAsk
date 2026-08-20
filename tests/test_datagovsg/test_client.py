"""Tests for the data.gov.sg datastore_search client."""

import httpx
import pytest
import respx

from src.config import DataGovSgConfig
from src.datagovsg.client import DataGovSgClient
from src.exceptions import DataGovSgError, DatasetNotFoundError

HDB_RESALE_DATASET_ID = "d_ebc5ab87086db484f88045b47411ebc5"


class TestDatastoreSearchMocked:
    """Tests against a mocked HTTP transport, no network required."""

    @pytest.fixture
    def config(self) -> DataGovSgConfig:
        """Client config pointed at the real base URL for realistic mocking."""
        return DataGovSgConfig()

    @pytest.fixture
    def client(self, config: DataGovSgConfig) -> DataGovSgClient:
        """A client instance under test."""
        return DataGovSgClient(config)

    @pytest.mark.asyncio
    @respx.mock
    async def test_datastore_search_returns_records(self, client: DataGovSgClient) -> None:
        """A successful response is parsed into a DatastoreQueryResult."""
        respx.get("https://data.gov.sg/api/action/datastore_search").mock(
            return_value=httpx.Response(
                200,
                json={
                    "success": True,
                    "result": {
                        "resource_id": HDB_RESALE_DATASET_ID,
                        "records": [{"town": "ANG MO KIO", "resale_price": "9000"}],
                        "total": 1,
                    },
                },
            )
        )

        result = await client.datastore_search(HDB_RESALE_DATASET_ID)

        assert result.dataset_id == HDB_RESALE_DATASET_ID
        assert result.records == [{"town": "ANG MO KIO", "resale_price": "9000"}]
        assert result.total == 1

    @pytest.mark.asyncio
    @respx.mock
    async def test_datastore_search_sends_filters_as_json(self, client: DataGovSgClient) -> None:
        """Filters are JSON-encoded into the filters query parameter."""
        route = respx.get("https://data.gov.sg/api/action/datastore_search").mock(
            return_value=httpx.Response(
                200, json={"success": True, "result": {"records": [], "total": 0}}
            )
        )

        await client.datastore_search(HDB_RESALE_DATASET_ID, filters={"town": "ANG MO KIO"})

        sent_url = route.calls.last.request.url
        assert sent_url.params["filters"] == '{"town": "ANG MO KIO"}'

    @pytest.mark.asyncio
    @respx.mock
    async def test_datastore_search_404_raises_dataset_not_found(
        self, client: DataGovSgClient
    ) -> None:
        """An HTTP 404 (unknown resource_id) raises DatasetNotFoundError."""
        respx.get("https://data.gov.sg/api/action/datastore_search").mock(
            return_value=httpx.Response(404, text="404 Not Found")
        )

        with pytest.raises(DatasetNotFoundError):
            await client.datastore_search("d_does_not_exist")

    @pytest.mark.asyncio
    @respx.mock
    async def test_datastore_search_success_false_raises_dataset_not_found(
        self, client: DataGovSgClient
    ) -> None:
        """A 200 response with success: false also raises DatasetNotFoundError."""
        respx.get("https://data.gov.sg/api/action/datastore_search").mock(
            return_value=httpx.Response(
                200, json={"success": False, "error": {"message": "Resource not found"}}
            )
        )

        with pytest.raises(DatasetNotFoundError):
            await client.datastore_search(HDB_RESALE_DATASET_ID)

    @pytest.mark.asyncio
    @respx.mock
    async def test_datastore_search_retries_on_rate_limit_then_succeeds(
        self, client: DataGovSgClient
    ) -> None:
        """A 429 is retried with backoff and succeeds once the limit clears."""
        route = respx.get("https://data.gov.sg/api/action/datastore_search")
        route.side_effect = [
            httpx.Response(429, json={"errorMsg": "Rate limit exceeded"}),
            httpx.Response(
                200, json={"success": True, "result": {"records": [{"town": "BISHAN"}], "total": 1}}
            ),
        ]

        result = await client.datastore_search(HDB_RESALE_DATASET_ID)

        assert result.records == [{"town": "BISHAN"}]
        assert route.call_count == 2

    @pytest.mark.asyncio
    @respx.mock
    async def test_datastore_search_server_error_raises_datagovsg_error(
        self, client: DataGovSgClient
    ) -> None:
        """A non-404 error status raises the generic DataGovSgError."""
        respx.get("https://data.gov.sg/api/action/datastore_search").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )

        with pytest.raises(DataGovSgError):
            await client.datastore_search(HDB_RESALE_DATASET_ID)

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_all_matching_paginates(self, client: DataGovSgClient) -> None:
        """fetch_all_matching walks pages until a short page ends pagination."""
        route = respx.get("https://data.gov.sg/api/action/datastore_search")
        route.side_effect = [
            httpx.Response(
                200,
                json={
                    "success": True,
                    "result": {"records": [{"_id": str(i)} for i in range(2)], "total": 3},
                },
            ),
            httpx.Response(
                200,
                json={
                    "success": True,
                    "result": {"records": [{"_id": "2"}], "total": 3},
                },
            ),
        ]

        result = await client.fetch_all_matching(HDB_RESALE_DATASET_ID, page_size=2)

        assert len(result.records) == 3
        assert result.total == 3

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_all_matching_respects_max_records(self, client: DataGovSgClient) -> None:
        """fetch_all_matching stops at max_records even if more rows exist."""
        respx.get("https://data.gov.sg/api/action/datastore_search").mock(
            return_value=httpx.Response(
                200,
                json={
                    "success": True,
                    "result": {"records": [{"_id": str(i)} for i in range(5)], "total": 1000},
                },
            )
        )

        result = await client.fetch_all_matching(HDB_RESALE_DATASET_ID, page_size=5, max_records=5)

        assert len(result.records) == 5
        assert result.total == 1000


@pytest.mark.integration
class TestDatastoreSearchLive:
    """Tests that hit the real, live data.gov.sg API. No API key required."""

    @pytest.mark.asyncio
    async def test_hdb_resale_dataset_returns_real_rows(self) -> None:
        """The curated HDB resale dataset_id resolves to real live data."""
        client = DataGovSgClient(DataGovSgConfig())

        result = await client.datastore_search(
            HDB_RESALE_DATASET_ID, filters={"town": "ANG MO KIO"}, limit=5
        )

        assert result.total > 0
        assert len(result.records) == 5
        assert all(record["town"] == "ANG MO KIO" for record in result.records)

    @pytest.mark.asyncio
    async def test_unknown_dataset_id_raises_not_found(self) -> None:
        """An unknown resource_id against the live API raises DatasetNotFoundError."""
        client = DataGovSgClient(DataGovSgConfig())

        with pytest.raises(DatasetNotFoundError):
            await client.datastore_search("d_this_is_not_a_real_dataset_id")
