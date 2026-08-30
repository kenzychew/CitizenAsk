"""CLI entry point: chunk, embed, and upsert every document under data/ into pgvector.

Usage:
    uv run python -m scripts.ingest
"""

import asyncio
import logging
from pathlib import Path

import asyncpg
from pgvector.asyncpg import register_vector

from app_logging import setup_logging
from config import load_config
from rag.ingest import Embedder, VectorIndexer, chunk_text, load_documents

logger = logging.getLogger(__name__)


async def _init_pg_connection(conn: asyncpg.Connection) -> None:
    await register_vector(conn)


async def main() -> None:
    """Ingest every document under the configured data_dir into pgvector."""
    setup_logging(log_dir=None)
    config = load_config()

    documents = load_documents(Path(config.data_dir))
    if not documents:
        logger.warning("No documents found under %s", config.data_dir)
        return

    embedder = Embedder(config.rag)
    chunks = [
        chunk
        for source, text in documents
        for chunk in chunk_text(text, source, config.rag.chunk_size, config.rag.chunk_overlap)
    ]
    embedder.embed_chunks(chunks)

    # The vector extension must exist before any pooled connection's init
    # hook tries to register the vector type codec, so create it via a
    # standalone bootstrap connection ahead of pool creation (mirrors
    # api/dependencies.py's _try_create_pool).
    bootstrap_conn = await asyncpg.connect(dsn=config.database_url)
    try:
        await bootstrap_conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    finally:
        await bootstrap_conn.close()

    pool = await asyncpg.create_pool(
        dsn=config.database_url, min_size=1, max_size=5, init=_init_pg_connection
    )
    try:
        indexer = VectorIndexer(pool, config.rag)
        await indexer.ensure_table()
        count = await indexer.upsert_chunks(chunks)
        logger.info("Ingested %d chunks from %d documents", count, len(documents))
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
