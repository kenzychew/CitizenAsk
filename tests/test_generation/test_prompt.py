"""Tests for prompt construction and structured-output schemas."""

from langchain_core.messages import HumanMessage, SystemMessage

from src.generation.prompt import (
    QueryPlan,
    build_plan_messages,
    build_rag_answer_messages,
    build_structured_answer_messages,
)
from src.schemas import DatasetEntry, DatasetKind, DatastoreQueryResult, DocChunk, RetrievedChunk


class TestBuildPlanMessages:
    """Tests for the structured-query planning prompt."""

    def test_includes_dataset_fields_and_question(self) -> None:
        """The dataset's columns and the question both appear in the human message."""
        dataset = DatasetEntry(
            dataset_id="d_ebc5ab87086db484f88045b47411ebc5",
            title="HDB Resale Flat Prices",
            agency="HDB",
            description="Resale transactions.",
            tags=["housing"],
            kind=DatasetKind.STRUCTURED,
            fields=["town", "resale_price", "flat_type"],
        )

        messages = build_plan_messages("What is the average resale price in Bishan?", dataset)

        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)
        human_text = str(messages[1].content)
        assert "town, resale_price, flat_type" in human_text
        assert "average resale price in Bishan" in human_text


class TestQueryPlanSchema:
    """Tests for the QueryPlan pydantic model's validation behaviour."""

    def test_accepts_minimal_count_plan(self) -> None:
        """A count operation needs no numeric_field."""
        plan = QueryPlan(operation="count", filters={"town": "BISHAN"})

        assert plan.operation == "count"
        assert plan.numeric_field is None

    def test_defaults_filters_to_empty_dict(self) -> None:
        """Omitting filters defaults to an empty dict rather than None."""
        plan = QueryPlan(operation="list")

        assert plan.filters == {}


class TestBuildStructuredAnswerMessages:
    """Tests for the final structured-answer synthesis prompt."""

    def test_includes_computed_result_and_dataset_name(self) -> None:
        """The dataset title and the computed result rows both appear."""
        dataset = DatasetEntry(
            dataset_id="d_ebc5ab87086db484f88045b47411ebc5",
            title="HDB Resale Flat Prices",
            agency="HDB",
            description="Resale transactions.",
            tags=["housing"],
            kind=DatasetKind.STRUCTURED,
            fields=["town", "resale_price"],
        )
        plan = QueryPlan(
            operation="average", filters={"town": "BISHAN"}, numeric_field="resale_price"
        )
        result = DatastoreQueryResult(
            dataset_id=dataset.dataset_id, records=[{"average": "512345.67"}], total=1
        )

        messages = build_structured_answer_messages("avg price in Bishan?", dataset, plan, result)

        human_text = str(messages[1].content)
        assert "HDB Resale Flat Prices" in human_text
        assert "512345.67" in human_text


class TestBuildRagAnswerMessages:
    """Tests for the RAG answer synthesis prompt."""

    def test_numbers_excerpts_for_citation(self) -> None:
        """Each chunk is prefixed with its 1-indexed citation marker."""
        chunks = [
            RetrievedChunk(
                chunk=DocChunk(
                    chunk_id="c1", text="CPF LIFE pays out monthly.", source="cpf.txt", index=0
                ),
                score=0.9,
            ),
            RetrievedChunk(
                chunk=DocChunk(
                    chunk_id="c2", text="Payouts start at age 65.", source="cpf.txt", index=1
                ),
                score=0.8,
            ),
        ]

        messages = build_rag_answer_messages("When do CPF LIFE payouts start?", chunks)

        human_text = str(messages[1].content)
        assert "[1]" in human_text
        assert "[2]" in human_text
        assert "CPF LIFE pays out monthly." in human_text
