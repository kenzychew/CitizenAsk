"""Tests for the scripts/ingest.py CLI entry point.

Exercises the real `main()` coroutine against fakes standing in for asyncpg,
since no live Postgres is available here. The behavior under test is the
pool-creation ordering bug documented in AGENTS.md: a pooled connection's
`init` hook (which calls pgvector's `register_vector`) fails on a fresh
database unless `CREATE EXTENSION IF NOT EXISTS vector` has already run on a
separate, standalone connection first.
"""

from pathlib import Path
from typing import Any

import pytest

import scripts.ingest as ingest_script
from config import AppConfig, RagConfig
from tests.test_rag.conftest import FakePool


class FakeBootstrapConnection:
    """Records execute()/close() calls made on the standalone bootstrap connection."""

    def __init__(self, call_order: list[str]) -> None:
        self._call_order = call_order
        self.executed: list[str] = []

    async def execute(self, sql: str) -> None:
        self._call_order.append("bootstrap_execute")
        self.executed.append(sql)

    async def close(self) -> None:
        self._call_order.append("bootstrap_close")


class ClosableFakePool(FakePool):
    """FakePool plus the close() that scripts.ingest.main() calls in its finally block."""

    async def close(self) -> None:
        pass


@pytest.fixture
def call_order() -> list[str]:
    return []


@pytest.fixture
def patched_asyncpg(monkeypatch: pytest.MonkeyPatch, call_order: list[str]) -> ClosableFakePool:
    """Patch asyncpg.connect/create_pool to fakes and record the order they're invoked in."""
    bootstrap_conn = FakeBootstrapConnection(call_order)
    pool = ClosableFakePool()

    async def fake_connect(dsn: str) -> FakeBootstrapConnection:
        call_order.append("bootstrap_connect")
        return bootstrap_conn

    async def fake_create_pool(**kwargs: Any) -> ClosableFakePool:
        call_order.append("pool_created")
        return pool

    monkeypatch.setattr(ingest_script.asyncpg, "connect", fake_connect)
    monkeypatch.setattr(ingest_script.asyncpg, "create_pool", fake_create_pool)
    return pool


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    (data / "one.txt").write_text("CPF LIFE provides a monthly retirement payout for life.")
    (data / "two.txt").write_text("The Aedes mosquito breeds in stagnant water.")
    return data


@pytest.fixture
def patched_config(monkeypatch: pytest.MonkeyPatch, data_dir: Path) -> None:
    config = AppConfig(
        rag=RagConfig(embedding_model="all-MiniLM-L6-v2", embedding_dim=384),
        database_url="postgresql://fake:fake@fake/fake",
        data_dir=str(data_dir),
    )
    monkeypatch.setattr(ingest_script, "load_config", lambda: config)


@pytest.mark.asyncio
async def test_main_bootstraps_extension_before_creating_pool(
    patched_asyncpg: ClosableFakePool, patched_config: None, call_order: list[str]
) -> None:
    """The extension is created and the bootstrap connection closed before pool creation.

    This is a regression test for the bug fixed in this change: creating the
    pool first (with register_vector as its init hook) fails against a fresh
    database because that hook needs the vector type to already exist.
    """
    await ingest_script.main()

    assert call_order == [
        "bootstrap_connect",
        "bootstrap_execute",
        "bootstrap_close",
        "pool_created",
    ]


@pytest.mark.asyncio
async def test_main_ingests_every_document_via_the_pool(
    patched_asyncpg: ClosableFakePool, patched_config: None
) -> None:
    """main() ensures the table and upserts one row per chunk via the post-bootstrap pool."""
    await ingest_script.main()

    ddl_statements = [sql for sql, _ in patched_asyncpg.connection.executed]
    assert any("CREATE EXTENSION IF NOT EXISTS vector" in s for s in ddl_statements)
    assert any("CREATE TABLE IF NOT EXISTS" in s for s in ddl_statements)

    assert len(patched_asyncpg.connection.executed_many) == 1
    _, rows = patched_asyncpg.connection.executed_many[0]
    assert len(rows) == 2  # one.txt and two.txt each fit in a single chunk
