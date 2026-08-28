"""Tests for the FastAPI /health, /datasets, and /query endpoints."""

import httpx
import pytest

from agent.graph import AgentDependencies
from api import dependencies as deps_module
from catalog.registry import REGISTRY
from config import AppConfig
from schemas import DatasetEntry, DatasetKind, DiscoveryMatch, DocChunk, RetrievedChunk
from tests.test_agent.conftest import StubDataGovSgClient, StubDiscovery, StubRetriever


class TestHealthEndpoint:
    """Tests for GET /health."""

    @pytest.mark.asyncio
    async def test_health_reports_registry_size_and_rag_availability(
        self, client: httpx.AsyncClient
    ) -> None:
        """Health response reflects the registry size and RAG availability."""
        response = await client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["rag_available"] is False


class TestDatasetsEndpoint:
    """Tests for GET /datasets."""

    @pytest.mark.asyncio
    async def test_lists_every_registry_entry(self, client: httpx.AsyncClient) -> None:
        """The datasets endpoint returns one entry per registry item."""
        response = await client.get("/datasets")

        assert response.status_code == 200
        body = response.json()
        assert len(body["datasets"]) == len(REGISTRY)
        assert {d["kind"] for d in body["datasets"]} == {"structured", "document"}


class TestQueryEndpoint:
    """Tests for GET /query, an SSE stream."""

    @pytest.mark.asyncio
    async def test_abstains_when_no_dataset_matches(self, client: httpx.AsyncClient) -> None:
        """With the default empty-match stub, the endpoint streams an abstention."""
        response = await client.get("/query", params={"q": "anything at all"})

        assert response.status_code == 200
        assert "abstain" in response.text
        assert "don't have a dataset" in response.text

    @pytest.mark.asyncio
    async def test_confident_match_streams_token_then_metadata(
        self, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A confident document match streams a token event then a metadata event."""
        dataset = DatasetEntry(
            dataset_id="",
            title="CPF LIFE Payout Guide",
            agency="CPF Board",
            description="Explainer.",
            tags=["cpf"],
            kind=DatasetKind.DOCUMENT,
        )
        chunks = [
            RetrievedChunk(
                chunk=DocChunk(
                    chunk_id="c1", text="Payouts start at 65.", source="cpf.txt", index=0
                ),
                score=0.9,
            )
        ]

        async def generate_answer(messages: object) -> str:
            return "Payouts start at age 65."

        confident_deps = AgentDependencies(
            discovery=StubDiscovery([DiscoveryMatch(dataset=dataset, score=5.0)]),
            datagovsg_client=StubDataGovSgClient(),
            retriever=StubRetriever(chunks),
            plan_query=deps_module._container.agent_deps.plan_query,  # type: ignore[union-attr]
            generate_answer=generate_answer,
            config=AppConfig(),
        )
        monkeypatch.setattr(deps_module._container, "agent_deps", confident_deps)

        response = await client.get("/query", params={"q": "When does CPF LIFE pay out?"})

        assert response.status_code == 200
        assert "event: token" in response.text
        assert "Payouts start at age 65." in response.text
        assert "event: metadata" in response.text
        assert '"route": "rag"' in response.text
