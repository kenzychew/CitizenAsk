"""Tests for the agent's LangGraph routing, tool selection, and abstention.

The LLM and data.gov.sg/pgvector boundaries are stubbed throughout, matching
how the OpenAI dependency is handled elsewhere: the graph logic is real and
fully exercised, only the network calls at its edges are replaced.
"""

import pytest

from agent.graph import (
    NO_RAG_CONTEXT_MESSAGE,
    RAG_UNAVAILABLE_MESSAGE,
    AgentDependencies,
    AgentState,
    route_after_discover,
    run_agent,
)
from config import AppConfig
from exceptions import DatasetNotFoundError
from generation.prompt import ABSTENTION_MESSAGE, QueryPlan
from schemas import (
    DatasetEntry,
    DatasetKind,
    DatastoreQueryResult,
    DiscoveryMatch,
    DocChunk,
    RetrievedChunk,
)
from tests.test_agent.conftest import StubDataGovSgClient, StubDiscovery, StubRetriever

STRUCTURED_DATASET = DatasetEntry(
    dataset_id="d_ebc5ab87086db484f88045b47411ebc5",
    title="HDB Resale Flat Prices",
    agency="HDB",
    description="Resale transactions.",
    tags=["housing"],
    kind=DatasetKind.STRUCTURED,
    fields=["town", "resale_price"],
)

DOCUMENT_DATASET = DatasetEntry(
    dataset_id="",
    title="CPF LIFE Payout Guide",
    agency="CPF Board",
    description="Explainer on CPF LIFE payouts.",
    tags=["cpf"],
    kind=DatasetKind.DOCUMENT,
)


def _never_called_plan_query():
    """A plan_query stub that fails the test if the LLM boundary is touched."""

    async def plan_query(
        question: str, dataset: DatasetEntry, sample_rows: list[dict[str, str]]
    ) -> QueryPlan:
        raise AssertionError("plan_query should not be called on this path")

    return plan_query


def _never_called_generate_answer():
    """A generate_answer stub that fails the test if the LLM boundary is touched."""

    async def generate_answer(messages: object) -> str:
        raise AssertionError("generate_answer should not be called on this path")

    return generate_answer


class TestRouteAfterDiscover:
    """Tests for the pure routing decision function."""

    def test_no_match_routes_to_abstain(self) -> None:
        """A state with no best_match routes to abstain."""
        state = AgentState(question="anything", best_match=None)

        assert route_after_discover(state) == "abstain"

    def test_structured_match_routes_to_structured(self) -> None:
        """A structured-kind best_match routes to the structured node."""
        state = AgentState(question="q", best_match=STRUCTURED_DATASET)

        assert route_after_discover(state) == "structured"

    def test_document_match_routes_to_rag(self) -> None:
        """A document-kind best_match routes to the rag node."""
        state = AgentState(question="q", best_match=DOCUMENT_DATASET)

        assert route_after_discover(state) == "rag"


class TestAbstentionPath:
    """Tests for the abstention route, which must never touch the LLM."""

    @pytest.mark.asyncio
    async def test_low_confidence_discovery_abstains_without_calling_llm(self) -> None:
        """A discovery score below the threshold abstains and never calls the LLM."""
        deps = AgentDependencies(
            discovery=StubDiscovery([DiscoveryMatch(dataset=STRUCTURED_DATASET, score=0.01)]),
            datagovsg_client=StubDataGovSgClient(),
            retriever=StubRetriever([]),
            plan_query=_never_called_plan_query(),
            generate_answer=_never_called_generate_answer(),
            config=AppConfig(),
        )

        answer = await run_agent(deps, "What is the meaning of life?")

        assert answer.abstained is True
        assert answer.route == "abstain"
        assert answer.answer == ABSTENTION_MESSAGE

    @pytest.mark.asyncio
    async def test_no_discovery_matches_abstains(self) -> None:
        """An empty discovery result also abstains."""
        deps = AgentDependencies(
            discovery=StubDiscovery([]),
            datagovsg_client=StubDataGovSgClient(),
            retriever=StubRetriever([]),
            plan_query=_never_called_plan_query(),
            generate_answer=_never_called_generate_answer(),
            config=AppConfig(),
        )

        answer = await run_agent(deps, "???")

        assert answer.abstained is True
        assert answer.route == "abstain"


class TestStructuredPath:
    """Tests for the structured datastore_search route."""

    @pytest.mark.asyncio
    async def test_confident_structured_match_produces_cited_answer(self) -> None:
        """A confident structured match plans, queries, computes, and cites the dataset."""

        async def plan_query(
            question: str, dataset: DatasetEntry, sample_rows: list[dict[str, str]]
        ) -> QueryPlan:
            return QueryPlan(operation="count", filters={"town": "BISHAN"})

        async def generate_answer(messages: object) -> str:
            return "There were 42 matching transactions in Bishan."

        client = StubDataGovSgClient(
            result=DatastoreQueryResult(
                dataset_id=STRUCTURED_DATASET.dataset_id,
                records=[{"town": "BISHAN"} for _ in range(42)],
                total=42,
            )
        )
        deps = AgentDependencies(
            discovery=StubDiscovery([DiscoveryMatch(dataset=STRUCTURED_DATASET, score=5.0)]),
            datagovsg_client=client,
            retriever=StubRetriever([]),
            plan_query=plan_query,
            generate_answer=generate_answer,
            config=AppConfig(),
        )

        answer = await run_agent(deps, "How many resale transactions happened in Bishan?")

        assert answer.abstained is False
        assert answer.route == "structured"
        assert answer.dataset_id == STRUCTURED_DATASET.dataset_id
        assert answer.citations == [STRUCTURED_DATASET.title]
        assert "42" in answer.answer
        assert client.calls == [(STRUCTURED_DATASET.dataset_id, {"town": "BISHAN"})]

    @pytest.mark.asyncio
    async def test_plan_query_receives_sample_rows(self) -> None:
        """The sample rows fetched from the client reach plan_query before filters are chosen."""
        received_samples: list[list[dict[str, str]]] = []

        async def plan_query(
            question: str, dataset: DatasetEntry, sample_rows: list[dict[str, str]]
        ) -> QueryPlan:
            received_samples.append(sample_rows)
            return QueryPlan(operation="count", filters={"month": "1990-01"})

        async def generate_answer(messages: object) -> str:
            return "There were 151 matching transactions."

        client = StubDataGovSgClient(
            result=DatastoreQueryResult(
                dataset_id=STRUCTURED_DATASET.dataset_id,
                records=[{"month": "1990-01"} for _ in range(151)],
                total=151,
            ),
            sample=[{"month": "1990-01", "town": "BISHAN"}],
        )
        deps = AgentDependencies(
            discovery=StubDiscovery([DiscoveryMatch(dataset=STRUCTURED_DATASET, score=5.0)]),
            datagovsg_client=client,
            retriever=StubRetriever([]),
            plan_query=plan_query,
            generate_answer=generate_answer,
            config=AppConfig(),
        )

        await run_agent(deps, "How many resale transactions happened in Bishan in January 1990?")

        assert received_samples == [[{"month": "1990-01", "town": "BISHAN"}]]

    @pytest.mark.asyncio
    async def test_sample_fetch_failure_falls_back_to_planning_without_sample(self) -> None:
        """A sample-row fetch failure degrades to an empty sample instead of failing the query."""
        received_samples: list[list[dict[str, str]]] = []

        async def plan_query(
            question: str, dataset: DatasetEntry, sample_rows: list[dict[str, str]]
        ) -> QueryPlan:
            received_samples.append(sample_rows)
            return QueryPlan(operation="count", filters={"town": "BISHAN"})

        async def generate_answer(messages: object) -> str:
            return "There were 42 matching transactions in Bishan."

        client = StubDataGovSgClient(
            result=DatastoreQueryResult(
                dataset_id=STRUCTURED_DATASET.dataset_id,
                records=[{"town": "BISHAN"} for _ in range(42)],
                total=42,
            ),
            sample_error=DatasetNotFoundError("no live resource"),
        )
        deps = AgentDependencies(
            discovery=StubDiscovery([DiscoveryMatch(dataset=STRUCTURED_DATASET, score=5.0)]),
            datagovsg_client=client,
            retriever=StubRetriever([]),
            plan_query=plan_query,
            generate_answer=generate_answer,
            config=AppConfig(),
        )

        answer = await run_agent(deps, "How many resale transactions happened in Bishan?")

        assert received_samples == [[]]
        assert answer.abstained is False
        assert answer.route == "structured"
        assert "42" in answer.answer

    @pytest.mark.asyncio
    async def test_datagovsg_failure_produces_error_answer_not_abstention(self) -> None:
        """A data.gov.sg failure surfaces as an explanatory answer, not a silent abstention."""

        async def plan_query(
            question: str, dataset: DatasetEntry, sample_rows: list[dict[str, str]]
        ) -> QueryPlan:
            return QueryPlan(operation="count", filters={})

        client = StubDataGovSgClient(error=DatasetNotFoundError("no live resource"))
        deps = AgentDependencies(
            discovery=StubDiscovery([DiscoveryMatch(dataset=STRUCTURED_DATASET, score=5.0)]),
            datagovsg_client=client,
            retriever=StubRetriever([]),
            plan_query=plan_query,
            generate_answer=_never_called_generate_answer(),
            config=AppConfig(),
        )

        answer = await run_agent(deps, "How many resale transactions happened?")

        assert answer.abstained is False
        assert answer.route == "structured"
        assert "couldn't compute an answer" in answer.answer


class TestRagPath:
    """Tests for the document RAG fallback route."""

    @pytest.mark.asyncio
    async def test_confident_document_match_produces_cited_answer(self) -> None:
        """A confident document match retrieves chunks and cites their sources."""
        chunks = [
            RetrievedChunk(
                chunk=DocChunk(
                    chunk_id="c1",
                    text="CPF LIFE pays out monthly from age 65.",
                    source="cpf-life-payouts.txt",
                    index=0,
                ),
                score=0.9,
            )
        ]

        async def generate_answer(messages: object) -> str:
            return "CPF LIFE payouts start at age 65 [1]."

        deps = AgentDependencies(
            discovery=StubDiscovery([DiscoveryMatch(dataset=DOCUMENT_DATASET, score=5.0)]),
            datagovsg_client=StubDataGovSgClient(),
            retriever=StubRetriever(chunks),
            plan_query=_never_called_plan_query(),
            generate_answer=generate_answer,
            config=AppConfig(),
        )

        answer = await run_agent(deps, "When does CPF LIFE start paying out?")

        assert answer.abstained is False
        assert answer.route == "rag"
        assert answer.citations == ["cpf-life-payouts.txt"]
        assert "65" in answer.answer

    @pytest.mark.asyncio
    async def test_no_retrieved_chunks_abstains_without_calling_llm(self) -> None:
        """A document match with no relevant chunks abstains rather than guessing."""
        deps = AgentDependencies(
            discovery=StubDiscovery([DiscoveryMatch(dataset=DOCUMENT_DATASET, score=5.0)]),
            datagovsg_client=StubDataGovSgClient(),
            retriever=StubRetriever([]),
            plan_query=_never_called_plan_query(),
            generate_answer=_never_called_generate_answer(),
            config=AppConfig(),
        )

        answer = await run_agent(deps, "Tell me about CPF LIFE.")

        assert answer.abstained is True
        assert answer.route == "rag"
        assert answer.answer == NO_RAG_CONTEXT_MESSAGE

    @pytest.mark.asyncio
    async def test_no_retriever_degrades_gracefully_without_calling_llm(self) -> None:
        """A document match with no retriever configured degrades rather than crashing."""
        deps = AgentDependencies(
            discovery=StubDiscovery([DiscoveryMatch(dataset=DOCUMENT_DATASET, score=5.0)]),
            datagovsg_client=StubDataGovSgClient(),
            retriever=None,
            plan_query=_never_called_plan_query(),
            generate_answer=_never_called_generate_answer(),
            config=AppConfig(),
        )

        answer = await run_agent(deps, "Tell me about CPF LIFE.")

        assert answer.abstained is True
        assert answer.route == "rag"
        assert answer.answer == RAG_UNAVAILABLE_MESSAGE
