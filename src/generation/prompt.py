"""Prompt templates and structured-output schemas for the generation nodes."""

from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.schemas import DatasetEntry, DatastoreQueryResult, RetrievedChunk

ABSTENTION_MESSAGE = (
    "I don't have a dataset in my curated registry that genuinely matches this "
    "question, so I can't give you a reliable answer. Try rephrasing, or ask "
    "about transport, housing, health, environment, or economic data published "
    "on data.gov.sg."
)


class QueryPlan(BaseModel):
    """A plan for querying a structured data.gov.sg dataset via datastore_search.

    Attributes:
        operation: The aggregation or lookup to perform over matching rows.
        filters: Exact-match column filters to send to datastore_search.
        numeric_field: The column to aggregate over, required for count/sum/
            average/min/max operations other than plain row counting.
    """

    operation: Literal["count", "average", "sum", "min", "max", "list"]
    filters: dict[str, str] = Field(default_factory=dict)
    numeric_field: str | None = None


_PLAN_SYSTEM_PROMPT = """You turn a user's question into a query plan against a \
single structured government dataset.

Given the dataset's title, description, and column names, choose:
- "operation": what to compute over matching rows ("count", "average", "sum", \
"min", "max", or "list" to return raw matching rows).
- "filters": a dict of column name to exact value, using only column names \
that exist in the dataset. Use the exact value casing implied by the question \
where possible (e.g. Singapore town names are upper case).
- "numeric_field": the numeric column to aggregate, required unless operation \
is "count" or "list".

Only use column names from the dataset's field list. If the question does not \
need a filter on a given column, omit it."""


def build_plan_messages(question: str, dataset: DatasetEntry) -> list[BaseMessage]:
    """Build the messages that ask the LLM to produce a QueryPlan.

    Args:
        question: The user's natural-language question.
        dataset: The matched structured dataset to plan a query against.

    Returns:
        A system + human message pair for a structured-output LLM call.
    """
    context = (
        f"Dataset: {dataset.title}\n"
        f"Agency: {dataset.agency}\n"
        f"Description: {dataset.description}\n"
        f"Columns: {', '.join(dataset.fields)}\n\n"
        f"Question: {question}"
    )
    return [SystemMessage(content=_PLAN_SYSTEM_PROMPT), HumanMessage(content=context)]


_STRUCTURED_ANSWER_SYSTEM_PROMPT = """You answer questions using only the \
computed result from a government dataset query. Be precise and concise. \
State the number or values plainly, and name the dataset as your source. \
Do not speculate beyond the given data."""


def build_structured_answer_messages(
    question: str, dataset: DatasetEntry, plan: QueryPlan, result: DatastoreQueryResult
) -> list[BaseMessage]:
    """Build the messages that ask the LLM to synthesize a final answer.

    Args:
        question: The user's original question.
        dataset: The dataset the query ran against.
        plan: The query plan that was executed.
        result: The raw datastore_search result (already computed/aggregated
            upstream; records may be a single summary row or a short list).

    Returns:
        A system + human message pair for the final answer generation call.
    """
    context = (
        f"Dataset: {dataset.title} ({dataset.agency})\n"
        f"Operation: {plan.operation}"
        + (f" over {plan.numeric_field}" if plan.numeric_field else "")
        + (f"\nFilters: {plan.filters}" if plan.filters else "")
        + f"\nResult rows: {result.records}\n"
        f"Total matching rows on server: {result.total}\n\n"
        f"Question: {question}"
    )
    return [
        SystemMessage(content=_STRUCTURED_ANSWER_SYSTEM_PROMPT),
        HumanMessage(content=context),
    ]


_RAG_ANSWER_SYSTEM_PROMPT = """You answer questions using only the provided \
document excerpts. Cite the source document for every claim using [n] \
notation matching the excerpt numbers. If the excerpts don't contain the \
answer, say so plainly instead of guessing."""


def build_rag_answer_messages(question: str, chunks: list[RetrievedChunk]) -> list[BaseMessage]:
    """Build the messages that ask the LLM to answer from retrieved chunks.

    Args:
        question: The user's original question.
        chunks: Retrieved document chunks, ordered by relevance.

    Returns:
        A system + human message pair for the RAG answer generation call.
    """
    excerpts = "\n\n".join(
        f"[{i + 1}] (Source: {rc.chunk.source})\n{rc.chunk.text}" for i, rc in enumerate(chunks)
    )
    context = f"Excerpts:\n{excerpts}\n\nQuestion: {question}"
    return [SystemMessage(content=_RAG_ANSWER_SYSTEM_PROMPT), HumanMessage(content=context)]
