"""Tests for prompt construction and structured-output schemas."""

from langchain_core.messages import HumanMessage, SystemMessage

from src.generation.prompt import (
    QueryFilter,
    QueryPlan,
    QueryPlanLLM,
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

    def test_omits_example_rows_when_none_given(self) -> None:
        """No sample_rows argument means no "Example rows" section in the prompt."""
        dataset = DatasetEntry(
            dataset_id="d_ebc5ab87086db484f88045b47411ebc5",
            title="HDB Resale Flat Prices",
            agency="HDB",
            description="Resale transactions.",
            tags=["housing"],
            kind=DatasetKind.STRUCTURED,
            fields=["month", "town"],
        )

        messages = build_plan_messages("Average price in Bishan?", dataset)

        assert "Example rows" not in str(messages[1].content)

    def test_includes_example_rows_when_given(self) -> None:
        """Sample rows appear in the human message so the LLM can mirror their format."""
        dataset = DatasetEntry(
            dataset_id="d_ebc5ab87086db484f88045b47411ebc5",
            title="HDB Resale Flat Prices",
            agency="HDB",
            description="Resale transactions.",
            tags=["housing"],
            kind=DatasetKind.STRUCTURED,
            fields=["month", "town"],
        )
        sample_rows = [{"month": "1990-01", "town": "BISHAN"}]

        messages = build_plan_messages(
            "Transactions in Bishan in January 1990?", dataset, sample_rows
        )

        human_text = str(messages[1].content)
        assert "Example rows" in human_text
        assert "1990-01" in human_text


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


class TestQueryPlanLLMSchema:
    """Tests for QueryPlanLLM, the strict-mode-safe schema used for the LLM call."""

    def test_converts_filter_list_to_query_plan_dict(self) -> None:
        """QueryPlanLLM's column/value pairs convert to QueryPlan's filters dict."""
        raw_plan = QueryPlanLLM(
            operation="average",
            filters=[QueryFilter(column="town", value="BISHAN")],
            numeric_field="resale_price",
        )

        plan = QueryPlan(
            operation=raw_plan.operation,
            filters={f.column: f.value for f in raw_plan.filters},
            numeric_field=raw_plan.numeric_field,
        )

        assert plan.filters == {"town": "BISHAN"}
        assert plan.operation == "average"
        assert plan.numeric_field == "resale_price"

    def test_defaults_filters_to_empty_list(self) -> None:
        """Omitting filters defaults to an empty list rather than None."""
        raw_plan = QueryPlanLLM(operation="list")

        assert raw_plan.filters == []

    def test_json_schema_has_no_bare_dict_types(self) -> None:
        """The generated JSON schema never uses an open-ended dict type.

        OpenAI's strict structured-output mode rejects any object schema
        with no fixed "properties" list (the shape Pydantic emits for a
        bare `dict[str, str]` field), so the schema must not contain one.
        """
        schema = QueryPlanLLM.model_json_schema()

        def assert_no_wildcard_object(node: object) -> None:
            if isinstance(node, dict):
                if node.get("type") == "object" and "properties" not in node:
                    raise AssertionError(f"found wildcard object schema: {node}")
                for value in node.values():
                    assert_no_wildcard_object(value)
            elif isinstance(node, list):
                for item in node:
                    assert_no_wildcard_object(item)

        assert_no_wildcard_object(schema)


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
