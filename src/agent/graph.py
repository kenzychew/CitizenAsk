"""LangGraph StateGraph orchestrating discovery, structured query, RAG, and abstention."""

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agent.tools import compute_structured_result
from src.catalog.discovery import CatalogDiscovery
from src.config import AppConfig
from src.datagovsg.client import DataGovSgClient
from src.exceptions import DataGovSgError
from src.generation.prompt import (
    ABSTENTION_MESSAGE,
    QueryPlan,
    build_rag_answer_messages,
    build_structured_answer_messages,
)
from src.rag.retriever import DocRetriever
from src.schemas import (
    AgentAnswer,
    DatasetEntry,
    DatasetKind,
    DatastoreQueryResult,
    DiscoveryMatch,
)

logger = logging.getLogger(__name__)

NO_RAG_CONTEXT_MESSAGE = (
    "I found a related topic in my registry, but couldn't find relevant document "
    "content to answer this specific question."
)

RAG_UNAVAILABLE_MESSAGE = (
    "I found a related topic in my registry, but document search is temporarily "
    "unavailable, so I can't answer this specific question right now."
)

PlanFn = Callable[[str, DatasetEntry], Awaitable[QueryPlan]]
AnswerFn = Callable[[list[BaseMessage]], Awaitable[str]]


@dataclass
class AgentDependencies:
    """Everything the graph nodes need, injected so tests can swap in stubs.

    Attributes:
        discovery: Catalog discovery over the curated registry.
        datagovsg_client: Client for the real data.gov.sg datastore_search API.
        retriever: pgvector document retriever for the RAG fallback, or None
            if Postgres was unreachable at startup, in which case the rag
            node degrades gracefully instead of crashing.
        plan_query: Async callable turning (question, dataset) into a QueryPlan,
            wrapping an LLM structured-output call in production.
        generate_answer: Async callable turning a message list into answer text,
            wrapping a plain LLM completion call in production.
        config: Application configuration.
    """

    discovery: CatalogDiscovery
    datagovsg_client: DataGovSgClient
    retriever: DocRetriever | None
    plan_query: PlanFn
    generate_answer: AnswerFn
    config: AppConfig


@dataclass
class AgentState:
    """State threaded through the agent graph for one query.

    Attributes:
        question: The user's natural-language question.
        discovery_matches: Ranked candidate datasets from the discovery node.
        best_match: The top match, or None if it fell below the confidence
            threshold and the graph should abstain.
        answer: The final answer, set by whichever terminal node ran.
    """

    question: str
    discovery_matches: list[DiscoveryMatch] = field(default_factory=list)
    best_match: DatasetEntry | None = None
    answer: AgentAnswer | None = None


def route_after_discover(state: AgentState) -> Literal["structured", "rag", "abstain"]:
    """Decide which path to take based on the top discovery match.

    Args:
        state: Current graph state, populated by the discover node.

    Returns:
        "abstain" if there is no confident match, otherwise "structured" or
        "rag" depending on the matched dataset's kind.
    """
    if state.best_match is None:
        return "abstain"
    if state.best_match.kind == DatasetKind.STRUCTURED:
        return "structured"
    return "rag"


def build_graph(
    deps: AgentDependencies,
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    """Build and compile the agent's StateGraph.

    Args:
        deps: Injected dependencies used by every node.

    Returns:
        A compiled LangGraph graph. Invoke with an AgentState carrying only
        `question` set; the other fields are populated as the graph runs.
    """

    def discover_node(state: AgentState) -> dict[str, object]:
        """Score the question against the curated registry and pick a match."""
        matches = deps.discovery.discover(state.question, top_k=deps.config.discovery.top_k)
        best_match = None
        if matches and matches[0].score >= deps.config.discovery.confidence_threshold:
            best_match = matches[0].dataset
        return {"discovery_matches": matches, "best_match": best_match}

    async def structured_node(state: AgentState) -> dict[str, object]:
        """Plan, execute, and answer a structured datastore_search query."""
        dataset = state.best_match
        assert dataset is not None  # route_after_discover guarantees this

        try:
            plan = await deps.plan_query(state.question, dataset)
            fetched = await deps.datagovsg_client.fetch_all_matching(
                dataset.dataset_id, filters=plan.filters
            )
            records, total = compute_structured_result(fetched.records, fetched.total, plan)
        except DataGovSgError as exc:
            logger.warning("Structured query failed: %s", exc)
            return {
                "answer": AgentAnswer(
                    answer=(
                        f"I found a matching dataset ({dataset.title}) but couldn't "
                        f"compute an answer from it: {exc}"
                    ),
                    abstained=False,
                    route="structured",
                    dataset_id=dataset.dataset_id,
                    citations=[dataset.title],
                )
            }

        result = DatastoreQueryResult(dataset_id=dataset.dataset_id, records=records, total=total)
        messages = build_structured_answer_messages(state.question, dataset, plan, result)
        answer_text = await deps.generate_answer(messages)

        return {
            "answer": AgentAnswer(
                answer=answer_text,
                abstained=False,
                route="structured",
                dataset_id=dataset.dataset_id,
                citations=[dataset.title],
            )
        }

    async def rag_node(state: AgentState) -> dict[str, object]:
        """Retrieve document chunks and synthesize a cited answer."""
        if deps.retriever is None:
            return {
                "answer": AgentAnswer(
                    answer=RAG_UNAVAILABLE_MESSAGE, abstained=True, route="rag", citations=[]
                )
            }

        chunks = await deps.retriever.retrieve(state.question, top_k=deps.config.rag.top_k)

        if not chunks:
            return {
                "answer": AgentAnswer(
                    answer=NO_RAG_CONTEXT_MESSAGE, abstained=True, route="rag", citations=[]
                )
            }

        messages = build_rag_answer_messages(state.question, chunks)
        answer_text = await deps.generate_answer(messages)
        citations = sorted({rc.chunk.source for rc in chunks})

        return {
            "answer": AgentAnswer(
                answer=answer_text, abstained=False, route="rag", citations=citations
            )
        }

    def abstain_node(state: AgentState) -> dict[str, object]:
        """Decline to answer without ever calling the LLM."""
        return {
            "answer": AgentAnswer(
                answer=ABSTENTION_MESSAGE, abstained=True, route="abstain", citations=[]
            )
        }

    graph = StateGraph(AgentState)
    graph.add_node("discover", discover_node)
    graph.add_node("structured", structured_node)
    graph.add_node("rag", rag_node)
    graph.add_node("abstain", abstain_node)

    graph.set_entry_point("discover")
    graph.add_conditional_edges(
        "discover",
        route_after_discover,
        {"structured": "structured", "rag": "rag", "abstain": "abstain"},
    )
    graph.add_edge("structured", END)
    graph.add_edge("rag", END)
    graph.add_edge("abstain", END)

    return graph.compile()


async def run_agent(deps: AgentDependencies, question: str) -> AgentAnswer:
    """Run the compiled graph for one question and return its answer.

    Args:
        deps: Injected dependencies for the graph.
        question: The user's natural-language question.

    Returns:
        The final AgentAnswer produced by whichever terminal node ran.
    """
    compiled = build_graph(deps)
    result = await compiled.ainvoke(AgentState(question=question))
    answer = result["answer"]
    assert isinstance(answer, AgentAnswer)
    return answer
