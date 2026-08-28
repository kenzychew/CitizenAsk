"""Tests for structured-query result computation."""

import pytest

from agent.tools import compute_structured_result
from exceptions import DataGovSgError
from generation.prompt import QueryPlan


class TestComputeStructuredResult:
    """Tests for compute_structured_result across each operation."""

    def test_count_returns_row_count(self) -> None:
        """Count operation returns the number of matched records."""
        records = [{"town": "BISHAN"} for _ in range(7)]
        plan = QueryPlan(operation="count", filters={"town": "BISHAN"})

        result, total = compute_structured_result(records, total=7, plan=plan)

        assert result == [{"count": "7"}]
        assert total == 7

    def test_average_computes_mean_of_numeric_field(self) -> None:
        """Average operation computes the mean of the numeric_field."""
        records = [{"resale_price": "100000"}, {"resale_price": "200000"}]
        plan = QueryPlan(operation="average", numeric_field="resale_price")

        result, _ = compute_structured_result(records, total=2, plan=plan)

        assert result == [{"average": "150000.0"}]

    def test_sum_computes_total_of_numeric_field(self) -> None:
        """Sum operation adds up the numeric_field across rows."""
        records = [{"resale_price": "100000"}, {"resale_price": "200000"}]
        plan = QueryPlan(operation="sum", numeric_field="resale_price")

        result, _ = compute_structured_result(records, total=2, plan=plan)

        assert result == [{"sum": "300000.0"}]

    def test_min_and_max(self) -> None:
        """Min and max operations find the extreme values of the numeric_field."""
        records = [
            {"resale_price": "300000"},
            {"resale_price": "100000"},
            {"resale_price": "200000"},
        ]

        min_result, _ = compute_structured_result(
            records, total=3, plan=QueryPlan(operation="min", numeric_field="resale_price")
        )
        max_result, _ = compute_structured_result(
            records, total=3, plan=QueryPlan(operation="max", numeric_field="resale_price")
        )

        assert min_result == [{"min": "100000.0"}]
        assert max_result == [{"max": "300000.0"}]

    def test_list_caps_at_twenty_rows(self) -> None:
        """List operation returns at most 20 raw rows even with more matches."""
        records = [{"town": "BISHAN", "_id": str(i)} for i in range(50)]
        plan = QueryPlan(operation="list")

        result, total = compute_structured_result(records, total=50, plan=plan)

        assert len(result) == 20
        assert total == 50

    def test_skips_non_numeric_values_when_averaging(self) -> None:
        """Rows with a non-numeric or missing numeric_field are excluded, not fatal."""
        records = [
            {"resale_price": "100000"},
            {"resale_price": "N/A"},
            {"resale_price": "300000"},
        ]
        plan = QueryPlan(operation="average", numeric_field="resale_price")

        result, _ = compute_structured_result(records, total=3, plan=plan)

        assert result == [{"average": "200000.0"}]

    def test_no_matching_rows_raises(self) -> None:
        """An empty record list raises rather than silently returning nothing."""
        with pytest.raises(DataGovSgError, match="No rows matched"):
            compute_structured_result([], total=0, plan=QueryPlan(operation="count"))

    def test_aggregate_without_numeric_field_raises(self) -> None:
        """An aggregate operation with no numeric_field set is rejected."""
        with pytest.raises(DataGovSgError, match="requires a numeric_field"):
            compute_structured_result(
                [{"town": "BISHAN"}], total=1, plan=QueryPlan(operation="average")
            )

    def test_all_values_non_numeric_raises(self) -> None:
        """An aggregate over a column with zero parseable values raises."""
        records = [{"resale_price": "unknown"}, {"resale_price": "n/a"}]
        plan = QueryPlan(operation="average", numeric_field="resale_price")

        with pytest.raises(DataGovSgError, match="no numeric values"):
            compute_structured_result(records, total=2, plan=plan)
