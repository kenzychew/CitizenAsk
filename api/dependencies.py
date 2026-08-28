"""FastAPI dependency container: builds and serves every pipeline component."""

import logging

import asyncpg
from pgvector.asyncpg import register_vector

from agent.graph import AgentDependencies
from catalog.discovery import CatalogDiscovery
from catalog.registry import REGISTRY
from config import AppConfig
from datagovsg.client import DataGovSgClient
from generation.llm import get_chat_model, make_generate_answer, make_plan_query
from rag.ingest import Embedder
from rag.retriever import DocRetriever

logger = logging.getLogger(__name__)


class DependencyContainer:
    """Singleton holder for every initialized pipeline component.

    Attributes:
        config: Application configuration.
        pool: asyncpg connection pool, or None if Postgres was unreachable
            at startup (the RAG route degrades gracefully rather than
            crashing the whole app when this happens).
        agent_deps: The bundle passed into every agent graph run.
    """

    def __init__(self) -> None:
        """Initialize the container with every field unset."""
        self.config: AppConfig | None = None
        self.pool: asyncpg.Pool | None = None
        self.agent_deps: AgentDependencies | None = None


_container = DependencyContainer()


async def _init_pg_connection(conn: asyncpg.Connection) -> None:
    """Register the pgvector type codec on each new pooled connection."""
    await register_vector(conn)


async def _try_create_pool(database_url: str) -> asyncpg.Pool | None:
    """Attempt to connect to Postgres, returning None rather than raising on failure.

    Args:
        database_url: PostgreSQL connection string.

    Returns:
        A connection pool with pgvector registered, or None if Postgres is
        unreachable. The RAG route reports itself unavailable rather than
        the whole application failing to start.
    """
    try:
        bootstrap_conn = await asyncpg.connect(dsn=database_url)
        try:
            await bootstrap_conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        finally:
            await bootstrap_conn.close()

        return await asyncpg.create_pool(
            dsn=database_url, min_size=1, max_size=10, init=_init_pg_connection
        )
    except (OSError, asyncpg.PostgresError) as exc:
        logger.warning("Postgres unavailable, RAG fallback will be degraded: %s", exc)
        return None


async def init_dependencies(config: AppConfig) -> None:
    """Initialize every pipeline component as a process-wide singleton.

    Args:
        config: Application configuration.
    """
    _container.config = config
    _container.pool = await _try_create_pool(config.database_url)

    discovery = CatalogDiscovery(REGISTRY)
    datagovsg_client = DataGovSgClient(config.datagovsg)
    embedder = Embedder(config.rag)
    retriever = DocRetriever(_container.pool, embedder, config.rag) if _container.pool else None

    llm = get_chat_model(config.generation)

    _container.agent_deps = AgentDependencies(
        discovery=discovery,
        datagovsg_client=datagovsg_client,
        retriever=retriever,
        plan_query=make_plan_query(llm),
        generate_answer=make_generate_answer(llm),
        config=config,
    )
    logger.info("All dependencies initialized")


async def close_pool() -> None:
    """Close the asyncpg connection pool, if one was created."""
    if _container.pool is not None:
        try:
            await _container.pool.close()
        finally:
            _container.pool = None


def get_config() -> AppConfig:
    """Get the application configuration.

    Returns:
        The application configuration instance.

    Raises:
        RuntimeError: If dependencies have not been initialized yet.
    """
    if _container.config is None:
        raise RuntimeError("Config not initialized")
    return _container.config


def get_agent_deps() -> AgentDependencies:
    """Get the agent graph's dependency bundle.

    Returns:
        The initialized AgentDependencies.

    Raises:
        RuntimeError: If dependencies have not been initialized yet.
    """
    if _container.agent_deps is None:
        raise RuntimeError("Agent dependencies not initialized")
    return _container.agent_deps


def rag_available() -> bool:
    """Report whether the pgvector-backed RAG fallback is usable right now.

    Returns:
        True if a Postgres connection pool was successfully created.
    """
    return _container.pool is not None
