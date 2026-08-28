"""Pure computation helpers used by the agent graph's tool-calling nodes."""

from exceptions import DataGovSgError
from generation.prompt import QueryPlan

_MAX_LIST_ROWS = 20


def compute_structured_result(
    records: list[dict[str, str]], total: int, plan: QueryPlan
) -> tuple[list[dict[str, str]], int]:
    """Compute the plan's operation over fetched rows.

    Args:
        records: Rows fetched from datastore_search matching the plan's filters.
        total: The server-reported total matching row count (may exceed
            len(records) when fetch_all_matching hit its safety cap).
        plan: The query plan describing what to compute.

    Returns:
        A tuple of (result_records, total) suitable for building the answer
        prompt: a single summary row for aggregate operations, or up to
        _MAX_LIST_ROWS raw rows for "list".

    Raises:
        DataGovSgError: If no rows matched, or an aggregate operation names
            a numeric_field that is missing or non-numeric on every row.
    """
    if not records:
        raise DataGovSgError(f"No rows matched filters {plan.filters!r}")

    if plan.operation == "count":
        return [{"count": str(len(records))}], total

    if plan.operation == "list":
        return records[:_MAX_LIST_ROWS], total

    if not plan.numeric_field:
        raise DataGovSgError(f"Operation {plan.operation!r} requires a numeric_field")

    values = _numeric_values(records, plan.numeric_field)
    if not values:
        raise DataGovSgError(
            f"Column {plan.numeric_field!r} has no numeric values in the matched rows"
        )

    aggregate = {
        "average": sum(values) / len(values),
        "sum": sum(values),
        "min": min(values),
        "max": max(values),
    }[plan.operation]

    return [{plan.operation: str(round(aggregate, 2))}], total


def _numeric_values(records: list[dict[str, str]], field: str) -> list[float]:
    """Parse a column's values to floats, skipping rows where it is missing or invalid.

    Args:
        records: Rows to read from.
        field: The column name to parse.

    Returns:
        Successfully parsed float values, in row order.
    """
    values: list[float] = []
    for record in records:
        raw = record.get(field)
        if raw is None:
            continue
        try:
            values.append(float(raw))
        except ValueError:
            continue
    return values
