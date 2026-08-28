"""Lexical dataset discovery over the curated registry.

Uses BM25 rather than embeddings so the discovery tool works fully offline
and needs no LLM or API key: the registry is small and hand-curated, and
matching a question to the right dataset's title/description/tags is a
lexical retrieval problem, not one that needs semantic embeddings.
"""

import re

from rank_bm25 import BM25Okapi

from schemas import DatasetEntry, DiscoveryMatch

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase and split text into alphanumeric tokens.

    Args:
        text: Text to tokenize.

    Returns:
        List of lowercase tokens.
    """
    return _TOKEN_RE.findall(text.lower())


def _entry_document(entry: DatasetEntry) -> str:
    """Build the searchable text blob for one registry entry.

    Args:
        entry: The dataset entry.

    Returns:
        Title, agency, description, and tags concatenated, with the title
        repeated to weight it more heavily than the rest of the metadata.
    """
    return " ".join(
        [entry.title, entry.title, entry.agency, entry.description, " ".join(entry.tags)]
    )


class CatalogDiscovery:
    """Scores curated datasets against a question via BM25 over their metadata."""

    def __init__(self, registry: list[DatasetEntry]) -> None:
        """Build the BM25 index over the registry.

        Args:
            registry: The curated list of dataset entries to search over.

        Raises:
            ValueError: If the registry is empty.
        """
        if not registry:
            raise ValueError("Cannot build discovery over an empty registry")

        self._registry = registry
        self._corpus = [_tokenize(_entry_document(entry)) for entry in registry]
        self._bm25 = BM25Okapi(self._corpus)

    def __len__(self) -> int:
        """Return the number of datasets in the underlying registry."""
        return len(self._registry)

    def discover(self, question: str, top_k: int = 3) -> list[DiscoveryMatch]:
        """Find the top-k datasets whose metadata best matches a question.

        Args:
            question: The user's natural-language question.
            top_k: Maximum number of matches to return.

        Returns:
            Matches sorted by descending score. May be shorter than top_k
            if the registry itself is smaller, but never empty unless
            top_k is 0 -- callers decide confidence by score, not by
            whether this list is empty.
        """
        tokens = _tokenize(question)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            zip(self._registry, scores, strict=True), key=lambda pair: pair[1], reverse=True
        )
        return [
            DiscoveryMatch(dataset=entry, score=float(score)) for entry, score in ranked[:top_k]
        ]
