"""Document loading, chunking, embedding, and pgvector storage for RAG fallback."""

import hashlib
import json
import logging
from pathlib import Path

import asyncpg
import numpy as np
from sentence_transformers import SentenceTransformer

from config import RagConfig
from exceptions import EmbeddingError, IngestionError
from schemas import DocChunk

logger = logging.getLogger(__name__)

_SEPARATORS = ["\n\n", "\n", ". ", " "]

CREATE_EXTENSION_SQL = "CREATE EXTENSION IF NOT EXISTS vector"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS {table} (
    chunk_id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    source TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    embedding vector({dim}) NOT NULL
)
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS {table}_embedding_idx
ON {table} USING hnsw (embedding vector_cosine_ops)
"""

UPSERT_SQL = """
INSERT INTO {table} (chunk_id, text, source, chunk_index, embedding)
VALUES ($1, $2, $3, $4, $5)
ON CONFLICT (chunk_id) DO UPDATE SET
    text = EXCLUDED.text,
    source = EXCLUDED.source,
    chunk_index = EXCLUDED.chunk_index,
    embedding = EXCLUDED.embedding
"""


def load_documents(data_dir: Path) -> list[tuple[str, str]]:
    """Load plain-text documents from a directory.

    Args:
        data_dir: Directory to scan for .txt and .md files (non-recursive).

    Returns:
        List of (source_filename, content) pairs, skipping empty files.
    """
    documents: list[tuple[str, str]] = []
    if not data_dir.exists():
        return documents

    for path in sorted(list(data_dir.glob("*.txt")) + list(data_dir.glob("*.md"))):
        content = path.read_text(encoding="utf-8").strip()
        if content:
            documents.append((path.name, content))

    return documents


def _chunk_id(source: str, index: int) -> str:
    """Deterministically derive a chunk id from its source and position.

    Args:
        source: Source document identifier.
        index: Chunk index within the document.

    Returns:
        A short, stable hex id, deterministic across re-ingestion runs.
    """
    return hashlib.sha256(f"{source}::{index}".encode()).hexdigest()[:24]


def chunk_text(text: str, source: str, chunk_size: int, chunk_overlap: int) -> list[DocChunk]:
    """Split text into overlapping chunks along paragraph/sentence boundaries.

    Recursively tries progressively finer separators so a chunk boundary
    lands on a paragraph or sentence break where possible, falling back
    to a hard character split only when no separator fits.

    Args:
        text: The full document text.
        source: Source document identifier, stored on every chunk.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlapping characters carried into the next chunk.

    Returns:
        Chunks in document order, each with a deterministic chunk_id.

    Raises:
        IngestionError: If text is empty.
    """
    if not text.strip():
        raise IngestionError(f"Cannot chunk empty document: {source}")

    pieces = _split_recursive(text, chunk_size, list(_SEPARATORS))
    merged = _merge_with_overlap(pieces, chunk_size, chunk_overlap)

    return [
        DocChunk(chunk_id=_chunk_id(source, i), text=piece, source=source, index=i)
        for i, piece in enumerate(merged)
    ]


def _split_recursive(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Split text on the first separator that yields pieces within chunk_size.

    Args:
        text: Text to split.
        chunk_size: Target maximum piece length.
        separators: Separators to try, in order from coarsest to finest.

    Returns:
        Text pieces, recursively split further if still oversized.
    """
    if len(text) <= chunk_size or not separators:
        return [text] if text.strip() else []

    separator, remaining = separators[0], separators[1:]
    parts = [p for p in text.split(separator) if p.strip()]

    pieces: list[str] = []
    for part in parts:
        if len(part) > chunk_size:
            pieces.extend(_split_recursive(part, chunk_size, remaining))
        else:
            pieces.append(part)
    return pieces


def _merge_with_overlap(pieces: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """Greedily pack small pieces into chunks up to chunk_size, with overlap.

    Args:
        pieces: Small text pieces from _split_recursive, in order.
        chunk_size: Maximum characters per merged chunk.
        chunk_overlap: Characters of trailing text carried into the next chunk.

    Returns:
        Merged chunks, each at most chunk_size characters (barring a single
        oversized piece, which passes through unmerged).
    """
    if not pieces:
        return []

    merged: list[str] = []
    current = pieces[0]

    for piece in pieces[1:]:
        candidate = f"{current} {piece}"
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            merged.append(current)
            overlap_tail = current[-chunk_overlap:] if chunk_overlap else ""
            current = f"{overlap_tail} {piece}".strip()

    merged.append(current)
    return merged


class Embedder:
    """Generates dense vector embeddings locally via sentence-transformers.

    Runs entirely on-device, no API key required.

    Attributes:
        model_name: Name of the loaded sentence-transformers model.
    """

    def __init__(self, config: RagConfig) -> None:
        """Load the embedding model.

        Args:
            config: RAG configuration naming the embedding model.
        """
        self.model_name = config.embedding_model
        self._model = SentenceTransformer(self.model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: Texts to embed.

        Returns:
            One embedding vector per input text, same order.

        Raises:
            EmbeddingError: If the underlying model call fails.
        """
        if not texts:
            return []
        try:
            embeddings = self._model.encode(texts, show_progress_bar=False)
        except (RuntimeError, ValueError) as exc:
            raise EmbeddingError(f"Failed to embed {len(texts)} texts: {exc}") from exc
        return [vector.tolist() for vector in embeddings]

    def embed_chunks(self, chunks: list[DocChunk]) -> list[DocChunk]:
        """Embed chunks in place and return them.

        Args:
            chunks: Chunks to embed.

        Returns:
            The same chunk objects, now with embedding populated.
        """
        vectors = self.embed_texts([chunk.text for chunk in chunks])
        for chunk, vector in zip(chunks, vectors, strict=True):
            chunk.embedding = vector
        return chunks

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string.

        Args:
            query: Query text.

        Returns:
            The query's embedding vector.
        """
        return self.embed_texts([query])[0]


class VectorIndexer:
    """Creates the pgvector table and upserts chunk embeddings.

    Attributes:
        table_name: Name of the chunks table.
        embedding_dim: Dimensionality of stored embedding vectors.
    """

    def __init__(self, pool: asyncpg.Pool, config: RagConfig) -> None:
        """Initialize the indexer.

        Args:
            pool: asyncpg connection pool.
            config: RAG configuration with table name and embedding dim.
        """
        self._pool = pool
        self.table_name = config.table_name
        self.embedding_dim = config.embedding_dim

    async def ensure_table(self) -> None:
        """Create the pgvector extension, chunks table, and HNSW index if missing.

        Raises:
            IngestionError: If DDL execution fails.
        """
        try:
            async with self._pool.acquire() as conn, conn.transaction():
                await conn.execute(CREATE_EXTENSION_SQL)
                await conn.execute(
                    CREATE_TABLE_SQL.format(table=self.table_name, dim=self.embedding_dim)
                )
                await conn.execute(CREATE_INDEX_SQL.format(table=self.table_name))
        except (OSError, asyncpg.PostgresError) as exc:
            raise IngestionError(f"Failed to ensure pgvector table: {exc}") from exc

    async def upsert_chunks(self, chunks: list[DocChunk]) -> int:
        """Upsert embedded chunks into the pgvector table.

        Args:
            chunks: Chunks with populated embeddings.

        Returns:
            Number of chunks upserted.

        Raises:
            IngestionError: If any chunk lacks an embedding, or the upsert fails.
        """
        if not chunks:
            return 0
        if any(not chunk.embedding for chunk in chunks):
            raise IngestionError("Cannot upsert chunks with missing embeddings")

        rows = [
            (
                chunk.chunk_id,
                chunk.text,
                chunk.source,
                chunk.index,
                np.array(chunk.embedding, dtype=np.float32),
            )
            for chunk in chunks
        ]

        try:
            async with self._pool.acquire() as conn, conn.transaction():
                await conn.executemany(UPSERT_SQL.format(table=self.table_name), rows)
        except (OSError, asyncpg.PostgresError) as exc:
            raise IngestionError(f"Failed to upsert {len(chunks)} chunks: {exc}") from exc

        logger.info(json.dumps({"event": "chunks_upserted", "count": len(chunks)}))
        return len(chunks)
