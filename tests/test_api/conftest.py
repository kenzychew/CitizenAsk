"""Fixtures for API tests: an ASGI test client with dependencies pre-populated.

The app's lifespan (real Postgres connection, real embedder load, real
OPENAI_API_KEY check) is never triggered here -- httpx's ASGITransport
doesn't run it unless explicitly asked to, so tests populate the
dependency container directly with stubs instead.
"""

from collections.abc import AsyncIterator

import httpx
import pytest

from src.agent.graph import AgentDependencies
from src.api import dependencies as deps_module
from src.api.main import app
from src.config import AppConfig
from tests.test_agent.conftest import StubDataGovSgClient, StubDiscovery, StubRetriever


@pytest.fixture
def stub_agent_deps() -> AgentDependencies:
    """AgentDependencies built entirely from stubs, no network or LLM calls."""

    async def plan_query(question: str, dataset: object) -> object:
        raise AssertionError("not expected to be called in these tests")

    async def generate_answer(messages: object) -> str:
        return "stub answer"

    return AgentDependencies(
        discovery=StubDiscovery([]),
        datagovsg_client=StubDataGovSgClient(),
        retriever=StubRetriever([]),
        plan_query=plan_query,  # type: ignore[arg-type]
        generate_answer=generate_answer,
        config=AppConfig(),
    )


@pytest.fixture
async def client(
    stub_agent_deps: AgentDependencies, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[httpx.AsyncClient]:
    """An httpx client against the app, with stub dependencies pre-populated."""
    monkeypatch.setattr(deps_module._container, "config", AppConfig())
    monkeypatch.setattr(deps_module._container, "agent_deps", stub_agent_deps)
    monkeypatch.setattr(deps_module._container, "pool", None)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as test_client:
        yield test_client
