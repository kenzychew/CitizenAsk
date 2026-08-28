"""FastAPI application exposing the agent over a streaming /query endpoint."""

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from src.agent.graph import run_agent
from src.api.dependencies import close_pool, get_agent_deps, init_dependencies, rag_available
from src.catalog.registry import REGISTRY
from src.config import load_config
from src.exceptions import CitizenAskError
from src.logging import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize dependencies at startup and close them at shutdown."""
    config = load_config()
    setup_logging(log_dir=None)
    logger.info("Starting CitizenAsk")
    await init_dependencies(config)

    try:
        yield
    finally:
        await close_pool()
        logger.info("CitizenAsk shutdown complete")


app = FastAPI(
    title="citizenask",
    description="General-purpose agentic assistant over Singapore government open data",
    version="0.1.0",
    lifespan=lifespan,
)


class HealthResponse(BaseModel):
    """Response model for the health check endpoint.

    Attributes:
        status: Overall system status.
        rag_available: Whether the pgvector-backed RAG fallback is usable.
        registry_size: Number of curated datasets in the registry.
    """

    status: str
    rag_available: bool
    registry_size: int


class DatasetSummary(BaseModel):
    """One curated dataset, as exposed by the /datasets listing endpoint.

    Attributes:
        title: Dataset title.
        agency: Owning agency.
        kind: "structured" or "document".
        tags: Topic tags used for discovery.
    """

    title: str
    agency: str
    kind: str
    tags: list[str]


class DatasetsResponse(BaseModel):
    """Response model for the /datasets listing endpoint."""

    datasets: list[DatasetSummary]


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Report whether the agent's dependencies are ready to serve queries.

    Returns:
        HealthResponse describing RAG availability and registry size.
    """
    deps = get_agent_deps()
    return HealthResponse(
        status="healthy",
        rag_available=rag_available(),
        registry_size=len(deps.discovery),
    )


@app.get("/datasets", response_model=DatasetsResponse)
async def list_datasets() -> DatasetsResponse:
    """List every dataset in the curated registry.

    Returns:
        DatasetsResponse with each dataset's title, agency, kind, and tags.
    """
    return DatasetsResponse(
        datasets=[
            DatasetSummary(title=e.title, agency=e.agency, kind=e.kind.value, tags=e.tags)
            for e in REGISTRY
        ]
    )


@app.get("/query")
async def query_agent(
    q: str = Query(..., description="The user's natural-language question"),
) -> EventSourceResponse:
    """Answer a question via the agent graph, streamed as Server-Sent Events.

    Emits a "token" event carrying the full answer text (generation is not
    yet streamed token-by-token, see README), followed by a "metadata" event
    with the route taken, citations, and whether the agent abstained.

    Args:
        q: The user's natural-language question.

    Returns:
        An SSE stream with "token", "metadata", and (on failure) "error" events.
    """
    deps = get_agent_deps()

    async def event_generator() -> AsyncIterator[dict[str, str]]:
        try:
            answer = await run_agent(deps, q)
            yield {"event": "token", "data": answer.answer}
            yield {
                "event": "metadata",
                "data": json.dumps(
                    {
                        "route": answer.route,
                        "abstained": answer.abstained,
                        "dataset_id": answer.dataset_id,
                        "citations": answer.citations,
                    }
                ),
            }
        except CitizenAskError as exc:
            logger.error("Query failed", exc_info=exc)
            yield {"event": "error", "data": json.dumps({"error": str(exc)})}

    return EventSourceResponse(event_generator())
