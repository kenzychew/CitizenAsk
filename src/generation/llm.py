"""OpenAI chat model client factory.

Reads the API key from the environment (never hardcoded) and returns a
LangChain ChatOpenAI instance used by every LLM-calling node in the agent
graph. No live key is required to import or unit-test this module; a key
is only needed when a returned model is actually invoked.
"""

import os
from collections.abc import Awaitable, Callable

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from src.config import GenerationConfig
from src.exceptions import GenerationError
from src.generation.prompt import QueryPlan, build_plan_messages
from src.schemas import DatasetEntry


def get_chat_model(config: GenerationConfig) -> ChatOpenAI:
    """Build a ChatOpenAI client from the OpenAI API direct integration.

    Args:
        config: Generation configuration (model name, temperature, max tokens).

    Returns:
        A configured ChatOpenAI client.

    Raises:
        GenerationError: If OPENAI_API_KEY is not set in the environment.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise GenerationError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
        )

    return ChatOpenAI(
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        api_key=api_key,
    )


def make_plan_query(
    llm: ChatOpenAI,
) -> Callable[[str, DatasetEntry], Awaitable[QueryPlan]]:
    """Build the agent graph's plan_query callable from a chat model.

    Args:
        llm: The chat model to use for structured-output planning calls.

    Returns:
        An async callable turning (question, dataset) into a QueryPlan.
    """
    structured_llm = llm.with_structured_output(QueryPlan)

    async def plan_query(question: str, dataset: DatasetEntry) -> QueryPlan:
        messages = build_plan_messages(question, dataset)
        plan = await structured_llm.ainvoke(messages)
        assert isinstance(plan, QueryPlan)
        return plan

    return plan_query


def make_generate_answer(llm: ChatOpenAI) -> Callable[[list[BaseMessage]], Awaitable[str]]:
    """Build the agent graph's generate_answer callable from a chat model.

    Args:
        llm: The chat model to use for final answer synthesis.

    Returns:
        An async callable turning a message list into answer text.
    """

    async def generate_answer(messages: list[BaseMessage]) -> str:
        response = await llm.ainvoke(messages)
        return str(response.content)

    return generate_answer
