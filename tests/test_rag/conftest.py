"""Fake asyncpg pool/connection for RAG tests, since no live Postgres is available."""

from collections.abc import Sequence
from types import TracebackType
from typing import Any, Self


class FakeTransaction:
    """No-op async context manager standing in for asyncpg's transaction()."""

    async def __aenter__(self) -> Self:
        """Enter the fake transaction."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Exit the fake transaction, never suppressing exceptions."""
        return None


class FakeConnection:
    """Records every call made against it instead of touching a real database.

    Attributes:
        executed: (sql, args) pairs passed to execute().
        executed_many: (sql, rows) pairs passed to executemany().
        fetch_rows: Canned rows returned by the next fetch() call.
        fetch_calls: (sql, args) pairs passed to fetch().
    """

    def __init__(self, fetch_rows: Sequence[dict[str, Any]] | None = None) -> None:
        """Initialize the fake connection.

        Args:
            fetch_rows: Rows to return from any fetch() call.
        """
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.executed_many: list[tuple[str, list[tuple[Any, ...]]]] = []
        self.fetch_rows = list(fetch_rows or [])
        self.fetch_calls: list[tuple[str, tuple[Any, ...]]] = []

    def transaction(self) -> FakeTransaction:
        """Return a no-op transaction context manager."""
        return FakeTransaction()

    async def execute(self, sql: str, *args: Any) -> None:
        """Record a DDL/DML statement."""
        self.executed.append((sql, args))

    async def executemany(self, sql: str, rows: list[tuple[Any, ...]]) -> None:
        """Record a batch of rows passed to an upsert statement."""
        self.executed_many.append((sql, rows))

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        """Record the query and return the canned rows."""
        self.fetch_calls.append((sql, args))
        return self.fetch_rows


class FakeAcquireContext:
    """Async context manager returned by FakePool.acquire()."""

    def __init__(self, connection: FakeConnection) -> None:
        """Wrap a fake connection for use as an async context manager."""
        self._connection = connection

    async def __aenter__(self) -> FakeConnection:
        """Enter the context, yielding the fake connection."""
        return self._connection

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Exit the context, never suppressing exceptions."""
        return None


class FakePool:
    """Stands in for asyncpg.Pool, handing out a single FakeConnection."""

    def __init__(self, fetch_rows: Sequence[dict[str, Any]] | None = None) -> None:
        """Initialize the fake pool.

        Args:
            fetch_rows: Rows the underlying connection's fetch() will return.
        """
        self.connection = FakeConnection(fetch_rows=fetch_rows)

    def acquire(self) -> FakeAcquireContext:
        """Return an async context manager yielding the fake connection."""
        return FakeAcquireContext(self.connection)
