"""Tests for document loading, chunking, embedding, and pgvector indexing."""

from pathlib import Path

import pytest

from src.config import RagConfig
from src.exceptions import EmbeddingError, IngestionError
from src.rag.ingest import Embedder, VectorIndexer, chunk_text, load_documents
from src.schemas import DocChunk
from tests.test_rag.conftest import FakePool


class TestLoadDocuments:
    """Tests for loading .txt/.md documents from a directory."""

    def test_loads_txt_and_md_files(self, tmp_path: Path) -> None:
        """Both .txt and .md files in the directory are loaded."""
        (tmp_path / "a.txt").write_text("Hello from a.")
        (tmp_path / "b.md").write_text("Hello from b.")
        (tmp_path / "c.json").write_text("{}")

        documents = load_documents(tmp_path)

        sources = {source for source, _ in documents}
        assert sources == {"a.txt", "b.md"}

    def test_skips_empty_files(self, tmp_path: Path) -> None:
        """A whitespace-only file is skipped rather than yielding an empty document."""
        (tmp_path / "empty.txt").write_text("   \n  ")
        (tmp_path / "real.txt").write_text("Real content.")

        documents = load_documents(tmp_path)

        assert len(documents) == 1
        assert documents[0][0] == "real.txt"

    def test_missing_directory_returns_empty_list(self, tmp_path: Path) -> None:
        """A data_dir that doesn't exist yet returns no documents, not an error."""
        documents = load_documents(tmp_path / "does_not_exist")

        assert documents == []


class TestChunkText:
    """Tests for the recursive paragraph/sentence-aware chunker."""

    def test_short_text_yields_single_chunk(self) -> None:
        """Text shorter than chunk_size stays as one chunk."""
        chunks = chunk_text("A short sentence.", "doc.txt", chunk_size=200, chunk_overlap=20)

        assert len(chunks) == 1
        assert chunks[0].text == "A short sentence."
        assert chunks[0].source == "doc.txt"
        assert chunks[0].index == 0

    def test_long_text_splits_into_multiple_chunks(self) -> None:
        """Text longer than chunk_size is split into more than one chunk."""
        paragraph = "This is one sentence. " * 40
        chunks = chunk_text(paragraph, "doc.txt", chunk_size=200, chunk_overlap=20)

        assert len(chunks) > 1
        assert all(len(c.text) <= 220 for c in chunks)

    def test_chunk_ids_are_deterministic(self) -> None:
        """Chunking the same text twice produces identical chunk_ids."""
        text = "Paragraph one.\n\nParagraph two, which is a bit longer than one."

        first = chunk_text(text, "doc.txt", chunk_size=50, chunk_overlap=5)
        second = chunk_text(text, "doc.txt", chunk_size=50, chunk_overlap=5)

        assert [c.chunk_id for c in first] == [c.chunk_id for c in second]

    def test_chunk_indices_are_sequential(self) -> None:
        """Chunk index increases sequentially from 0."""
        paragraph = "Sentence number here. " * 30
        chunks = chunk_text(paragraph, "doc.txt", chunk_size=100, chunk_overlap=10)

        assert [c.index for c in chunks] == list(range(len(chunks)))

    def test_empty_text_raises_ingestion_error(self) -> None:
        """Chunking whitespace-only text raises IngestionError."""
        with pytest.raises(IngestionError):
            chunk_text("   ", "doc.txt", chunk_size=100, chunk_overlap=10)


class TestEmbedder:
    """Tests for the local sentence-transformers embedder."""

    @pytest.fixture(scope="module")
    def embedder(self) -> Embedder:
        """A real embedder using the small default model, loaded once per class."""
        return Embedder(RagConfig(embedding_model="all-MiniLM-L6-v2", embedding_dim=384))

    def test_embed_texts_returns_correct_dimension(self, embedder: Embedder) -> None:
        """Each embedding vector matches the model's configured dimension."""
        vectors = embedder.embed_texts(["CPF LIFE pays out monthly.", "COE is valid for 10 years."])

        assert len(vectors) == 2
        assert all(len(v) == 384 for v in vectors)

    def test_embed_texts_empty_list_returns_empty(self, embedder: Embedder) -> None:
        """Embedding an empty list returns an empty list without calling the model."""
        assert embedder.embed_texts([]) == []

    def test_similar_texts_have_higher_cosine_similarity(self, embedder: Embedder) -> None:
        """Semantically similar texts embed closer together than unrelated ones."""
        import numpy as np

        anchor, similar, different = embedder.embed_texts(
            [
                "CPF LIFE provides a monthly retirement payout for life.",
                "CPF LIFE gives retirees income every month until they pass away.",
                "The Aedes mosquito breeds in stagnant water.",
            ]
        )

        def cosine(a: list[float], b: list[float]) -> float:
            a_arr, b_arr = np.array(a), np.array(b)
            return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))

        assert cosine(anchor, similar) > cosine(anchor, different)

    def test_embed_chunks_populates_embedding_field(self, embedder: Embedder) -> None:
        """embed_chunks fills in the embedding field on each chunk in place."""
        chunks = [DocChunk(chunk_id="c1", text="Some text.", source="doc.txt", index=0)]

        result = embedder.embed_chunks(chunks)

        assert result[0].embedding
        assert len(result[0].embedding) == 384

    def test_embed_texts_wraps_model_errors(self, embedder: Embedder) -> None:
        """A failure inside the underlying model surfaces as EmbeddingError."""

        def _raise(*args: object, **kwargs: object) -> None:
            raise RuntimeError("model exploded")

        embedder._model.encode = _raise  # type: ignore[method-assign]

        with pytest.raises(EmbeddingError):
            embedder.embed_texts(["anything"])


class TestVectorIndexer:
    """Tests for pgvector table creation and chunk upsert, against a fake pool."""

    @pytest.fixture
    def config(self) -> RagConfig:
        """RAG config with a test table name."""
        return RagConfig(table_name="doc_chunks", embedding_dim=3)

    @pytest.mark.asyncio
    async def test_ensure_table_runs_extension_table_and_index_ddl(self, config: RagConfig) -> None:
        """ensure_table issues the extension, table, and index statements in order."""
        pool = FakePool()
        indexer = VectorIndexer(pool, config)  # type: ignore[arg-type]

        await indexer.ensure_table()

        statements = [sql for sql, _ in pool.connection.executed]
        assert "CREATE EXTENSION IF NOT EXISTS vector" in statements
        assert any("CREATE TABLE IF NOT EXISTS doc_chunks" in s for s in statements)
        assert any("USING hnsw" in s for s in statements)

    @pytest.mark.asyncio
    async def test_upsert_chunks_sends_one_row_per_chunk(self, config: RagConfig) -> None:
        """upsert_chunks batches all chunks into a single executemany call."""
        pool = FakePool()
        indexer = VectorIndexer(pool, config)  # type: ignore[arg-type]
        chunks = [
            DocChunk(chunk_id="c1", text="a", source="doc.txt", index=0, embedding=[0.1, 0.2, 0.3]),
            DocChunk(chunk_id="c2", text="b", source="doc.txt", index=1, embedding=[0.4, 0.5, 0.6]),
        ]

        count = await indexer.upsert_chunks(chunks)

        assert count == 2
        assert len(pool.connection.executed_many) == 1
        _, rows = pool.connection.executed_many[0]
        assert len(rows) == 2

    @pytest.mark.asyncio
    async def test_upsert_empty_list_is_a_noop(self, config: RagConfig) -> None:
        """Upserting an empty chunk list returns 0 without touching the connection."""
        pool = FakePool()
        indexer = VectorIndexer(pool, config)  # type: ignore[arg-type]

        count = await indexer.upsert_chunks([])

        assert count == 0
        assert pool.connection.executed_many == []

    @pytest.mark.asyncio
    async def test_upsert_chunk_missing_embedding_raises(self, config: RagConfig) -> None:
        """A chunk without an embedding is rejected before hitting the database."""
        pool = FakePool()
        indexer = VectorIndexer(pool, config)  # type: ignore[arg-type]
        chunks = [DocChunk(chunk_id="c1", text="a", source="doc.txt", index=0)]

        with pytest.raises(IngestionError):
            await indexer.upsert_chunks(chunks)
