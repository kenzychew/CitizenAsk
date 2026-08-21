"""Tests for pgvector similarity retrieval, against a fake pool."""

import pytest

from src.config import RagConfig
from src.rag.ingest import Embedder
from src.rag.retriever import DocRetriever
from tests.test_rag.conftest import FakePool


class TestDocRetriever:
    """Tests for DocRetriever.retrieve."""

    @pytest.fixture(scope="module")
    def embedder(self) -> Embedder:
        """A real local embedder, loaded once per class."""
        return Embedder(RagConfig(embedding_model="all-MiniLM-L6-v2", embedding_dim=384))

    @pytest.fixture
    def config(self) -> RagConfig:
        """RAG config with a small top_k for test assertions."""
        return RagConfig(table_name="doc_chunks", top_k=2)

    @pytest.mark.asyncio
    async def test_retrieve_maps_rows_to_retrieved_chunks(
        self, embedder: Embedder, config: RagConfig
    ) -> None:
        """Rows returned by the fake connection are mapped into RetrievedChunk objects."""
        pool = FakePool(
            fetch_rows=[
                {
                    "chunk_id": "c1",
                    "text": "CPF LIFE pays out monthly.",
                    "source": "cpf-life-payouts.txt",
                    "chunk_index": 0,
                    "score": 0.83,
                },
            ]
        )
        retriever = DocRetriever(pool, embedder, config)  # type: ignore[arg-type]

        results = await retriever.retrieve("When does CPF LIFE pay out?")

        assert len(results) == 1
        assert results[0].chunk.chunk_id == "c1"
        assert results[0].chunk.source == "cpf-life-payouts.txt"
        assert results[0].score == pytest.approx(0.83)

    @pytest.mark.asyncio
    async def test_retrieve_passes_configured_top_k_to_query(
        self, embedder: Embedder, config: RagConfig
    ) -> None:
        """The configured top_k (or an explicit override) is passed as the SQL LIMIT."""
        pool = FakePool(fetch_rows=[])
        retriever = DocRetriever(pool, embedder, config)  # type: ignore[arg-type]

        await retriever.retrieve("some query", top_k=5)

        _, args = pool.connection.fetch_calls[0]
        assert args[1] == 5

    @pytest.mark.asyncio
    async def test_retrieve_defaults_to_configured_top_k(
        self, embedder: Embedder, config: RagConfig
    ) -> None:
        """Without an explicit top_k, the configured default is used."""
        pool = FakePool(fetch_rows=[])
        retriever = DocRetriever(pool, embedder, config)  # type: ignore[arg-type]

        await retriever.retrieve("some query")

        _, args = pool.connection.fetch_calls[0]
        assert args[1] == config.top_k

    @pytest.mark.asyncio
    async def test_retrieve_empty_result_returns_empty_list(
        self, embedder: Embedder, config: RagConfig
    ) -> None:
        """No matching rows returns an empty list rather than raising."""
        pool = FakePool(fetch_rows=[])
        retriever = DocRetriever(pool, embedder, config)  # type: ignore[arg-type]

        results = await retriever.retrieve("nothing matches this")

        assert results == []
