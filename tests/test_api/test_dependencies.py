"""Tests for init_dependencies' graceful degradation when secrets are unset.

Regression coverage for the bug found while adding railway.toml's
healthcheck: init_dependencies used to call get_chat_model unconditionally,
which raises GenerationError when OPENAI_API_KEY is unset, crashing the
FastAPI lifespan at startup so /health was never reachable on a fresh
deploy before secrets are configured.
"""

import pytest

from api import dependencies as deps_module
from config import AppConfig
from exceptions import GenerationError


@pytest.fixture
def isolated_container(monkeypatch: pytest.MonkeyPatch) -> deps_module.DependencyContainer:
    """A fresh DependencyContainer swapped in for the duration of the test."""
    container = deps_module.DependencyContainer()
    monkeypatch.setattr(deps_module, "_container", container)
    return container


class TestInitDependenciesWithoutOpenAiKey:
    """init_dependencies must degrade rather than crash when OPENAI_API_KEY is unset."""

    @pytest.mark.asyncio
    async def test_does_not_raise_at_startup(
        self,
        isolated_container: deps_module.DependencyContainer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With no OPENAI_API_KEY and no reachable Postgres, startup still succeeds."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        await deps_module.init_dependencies(AppConfig())

        assert isolated_container.agent_deps is not None
        assert isolated_container.pool is None

    @pytest.mark.asyncio
    async def test_plan_query_raises_generation_error_only_when_invoked(
        self,
        isolated_container: deps_module.DependencyContainer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The degraded plan_query stub only fails at first real use, not at startup."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        await deps_module.init_dependencies(AppConfig())

        assert isolated_container.agent_deps is not None
        with pytest.raises(GenerationError):
            await isolated_container.agent_deps.plan_query("question", object(), [])  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_generate_answer_raises_generation_error_only_when_invoked(
        self,
        isolated_container: deps_module.DependencyContainer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The degraded generate_answer stub only fails at first real use, not at startup."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        await deps_module.init_dependencies(AppConfig())

        assert isolated_container.agent_deps is not None
        with pytest.raises(GenerationError):
            await isolated_container.agent_deps.generate_answer([])
