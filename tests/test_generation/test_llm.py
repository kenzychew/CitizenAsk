"""Tests for the OpenAI chat model factory."""

import pytest
from langchain_openai import ChatOpenAI

from src.config import GenerationConfig
from src.exceptions import GenerationError
from src.generation.llm import get_chat_model


class TestGetChatModel:
    """Tests for get_chat_model, which never makes a network call itself."""

    def test_raises_when_api_key_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Building a client without OPENAI_API_KEY set raises GenerationError."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(GenerationError):
            get_chat_model(GenerationConfig())

    def test_builds_client_with_configured_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A configured client carries the model name and temperature through."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")

        client = get_chat_model(GenerationConfig(model="gpt-4o-mini", temperature=0.2))

        assert isinstance(client, ChatOpenAI)
        assert client.model_name == "gpt-4o-mini"
        assert client.temperature == 0.2
