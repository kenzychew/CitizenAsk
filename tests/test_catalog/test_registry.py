"""Tests for the curated dataset registry."""

import pytest

from catalog.registry import REGISTRY
from config import DataGovSgConfig
from datagovsg.client import DataGovSgClient
from schemas import DatasetKind


class TestRegistryShape:
    """Structural invariants of the curated registry, no network required."""

    def test_registry_size_within_curated_range(self) -> None:
        """The registry has between 20 and 50 entries, per the curation brief."""
        assert 20 <= len(REGISTRY) <= 50

    def test_structured_entries_have_dataset_ids(self) -> None:
        """Every structured entry carries a non-empty data.gov.sg dataset_id."""
        for entry in REGISTRY:
            if entry.kind == DatasetKind.STRUCTURED:
                assert entry.dataset_id.startswith("d_")

    def test_structured_entries_have_fields(self) -> None:
        """Every structured entry lists at least one real column name."""
        for entry in REGISTRY:
            if entry.kind == DatasetKind.STRUCTURED:
                assert len(entry.fields) > 0

    def test_dataset_ids_are_unique(self) -> None:
        """No two structured entries share a dataset_id."""
        ids = [e.dataset_id for e in REGISTRY if e.dataset_id]
        assert len(ids) == len(set(ids))

    def test_covers_multiple_agencies(self) -> None:
        """The registry spans more than a handful of distinct agencies."""
        agencies = {entry.agency for entry in REGISTRY}
        assert len(agencies) >= 8

    def test_includes_both_kinds(self) -> None:
        """The registry has both structured and document entries."""
        kinds = {entry.kind for entry in REGISTRY}
        assert kinds == {DatasetKind.STRUCTURED, DatasetKind.DOCUMENT}

    def test_every_entry_has_title_description_and_tags(self) -> None:
        """No entry is missing basic discovery metadata."""
        for entry in REGISTRY:
            assert entry.title
            assert entry.description
            assert entry.tags


@pytest.mark.integration
class TestRegistryLiveDatasetIds:
    """Confirms a sample of curated dataset_ids resolve on the real, live API."""

    @pytest.mark.asyncio
    async def test_sample_of_structured_dataset_ids_are_live(self) -> None:
        """A spread of structured entries all resolve to real datastore resources."""
        client = DataGovSgClient(DataGovSgConfig())
        structured = [e for e in REGISTRY if e.kind == DatasetKind.STRUCTURED]
        sample = structured[::4]  # every 4th entry, spread across the whole list

        for entry in sample:
            result = await client.datastore_search(entry.dataset_id, limit=1)
            assert result.total > 0
