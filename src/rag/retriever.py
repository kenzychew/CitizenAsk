"""pgvector similarity search over ingested document chunks."""

import asyncpg
import numpy as np

from src.config import RagConfig
from src.exceptions import RetrievalError
from src.rag.ingest import Embedder
from src.schemas import DocChunk, RetrievedChunk

SIMILARITY_SEARCH_SQL = """
SELECT chunk_id, text, source, chunk_index, 1 - (embedding <=> $1) AS score
FROM {table}
ORDER BY embedding <=> $1
LIMIT $2
"""


class DocRetriever:
    """Retrieves the most similar chunks to a query via pgvector cosine distance."""

    def __init__(self, pool: asyncpg.Pool, embedder: Embedder, config: RagConfig) -> None:
        """Initialize the retriever.

        Args:
            pool: asyncpg connection pool.
            embedder: Embedder used to vectorize the query text.
            config: RAG configuration with table name and default top_k.
        """
        self._pool = pool
        self._embedder = embedder
        self.table_name = config.table_name
        self.top_k = config.top_k

    async def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Find the chunks most similar to a query.

        Args:
            query: The natural-language query.
            top_k: Number of chunks to return. Defaults to the configured top_k.

        Returns:
            Retrieved chunks ordered by descending similarity score.

        Raises:
            RetrievalError: If the similarity search query fails.
        """
        k = top_k if top_k is not None else self.top_k
        query_vector = np.array(self._embedder.embed_query(query), dtype=np.float32)

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    SIMILARITY_SEARCH_SQL.format(table=self.table_name), query_vector, k
                )
        except (OSError, asyncpg.PostgresError) as exc:
            raise RetrievalError(f"Similarity search failed for query {query!r}: {exc}") from exc

        return [
            RetrievedChunk(
                chunk=DocChunk(
                    chunk_id=row["chunk_id"],
                    text=row["text"],
                    source=row["source"],
                    index=row["chunk_index"],
                ),
                score=float(row["score"]),
            )
            for row in rows
        ]
