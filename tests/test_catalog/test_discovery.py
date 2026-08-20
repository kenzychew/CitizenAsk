"""Tests for BM25-based catalog discovery."""

import pytest

from src.catalog.discovery import CatalogDiscovery
from src.schemas import DatasetEntry, DatasetKind


def _entry(dataset_id: str, title: str, description: str, tags: list[str]) -> DatasetEntry:
    """Build a minimal DatasetEntry for discovery tests."""
    return DatasetEntry(
        dataset_id=dataset_id,
        title=title,
        agency="Test Agency",
        description=description,
        tags=tags,
        kind=DatasetKind.STRUCTURED,
    )


@pytest.fixture
def registry() -> list[DatasetEntry]:
    """A small, topically diverse registry for discovery tests."""
    return [
        _entry(
            "d_hdb",
            "HDB Resale Flat Prices",
            "Resale transaction prices of HDB flats including town and flat type.",
            ["housing", "hdb", "resale", "property"],
        ),
        _entry(
            "d_coe",
            "COE Bidding Results",
            "Certificate of Entitlement bidding results by vehicle category.",
            ["transport", "coe", "vehicle", "lta"],
        ),
        _entry(
            "d_dengue",
            "Dengue Clusters",
            "Active dengue cluster locations and case counts from NEA.",
            ["health", "dengue", "nea", "environment"],
        ),
    ]


class TestCatalogDiscovery:
    """Tests for CatalogDiscovery.discover."""

    def test_empty_registry_raises(self) -> None:
        """Building a discovery index over an empty registry is rejected."""
        with pytest.raises(ValueError, match="empty registry"):
            CatalogDiscovery([])

    def test_matches_correct_dataset_for_housing_question(
        self, registry: list[DatasetEntry]
    ) -> None:
        """A clearly housing-flavoured question ranks the HDB dataset first."""
        discovery = CatalogDiscovery(registry)

        matches = discovery.discover("What is the average HDB resale price in Bishan?")

        assert matches[0].dataset.dataset_id == "d_hdb"

    def test_matches_correct_dataset_for_transport_question(
        self, registry: list[DatasetEntry]
    ) -> None:
        """A COE-flavoured question ranks the COE dataset first."""
        discovery = CatalogDiscovery(registry)

        matches = discovery.discover("How much did Category A COE premiums cost last month?")

        assert matches[0].dataset.dataset_id == "d_coe"

    def test_matches_correct_dataset_for_health_question(
        self, registry: list[DatasetEntry]
    ) -> None:
        """A dengue-flavoured question ranks the dengue dataset first."""
        discovery = CatalogDiscovery(registry)

        matches = discovery.discover("Are there any active dengue clusters near me?")

        assert matches[0].dataset.dataset_id == "d_dengue"

    def test_respects_top_k(self, registry: list[DatasetEntry]) -> None:
        """No more than top_k matches are returned."""
        discovery = CatalogDiscovery(registry)

        matches = discovery.discover("Singapore government data", top_k=2)

        assert len(matches) == 2

    def test_scores_are_descending(self, registry: list[DatasetEntry]) -> None:
        """Matches are sorted by descending score."""
        discovery = CatalogDiscovery(registry)

        matches = discovery.discover("HDB flat resale price", top_k=3)

        scores = [m.score for m in matches]
        assert scores == sorted(scores, reverse=True)

    def test_unrelated_question_scores_near_zero(self, registry: list[DatasetEntry]) -> None:
        """A question with no lexical overlap gets a low top score."""
        discovery = CatalogDiscovery(registry)

        matches = discovery.discover("What is the capital of France?")

        assert matches[0].score < 1.0

    def test_empty_question_returns_no_matches(self, registry: list[DatasetEntry]) -> None:
        """A question with no tokens (blank/punctuation) returns no matches."""
        discovery = CatalogDiscovery(registry)

        matches = discovery.discover("???")

        assert matches == []
