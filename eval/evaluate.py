"""Evaluation harness: dataset-selection Hit@K, structured-answer correctness,
RAG retrieval Hit@K/MRR, and abstention accuracy, all against real data.

Run with `uv run python eval/evaluate.py`. The catalog discovery, data.gov.sg
client, and local embedder are all real; only the final LLM answer-synthesis
step is out of scope here since it needs a live OPENAI_API_KEY that this
environment does not have (see README for what that leaves untested).
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from agent.tools import compute_structured_result
from catalog.discovery import CatalogDiscovery
from catalog.registry import REGISTRY
from config import AppConfig, load_config
from datagovsg.client import DataGovSgClient
from exceptions import DataGovSgError
from generation.prompt import QueryPlan
from rag.ingest import Embedder, chunk_text, load_documents

logging.basicConfig(level=logging.WARNING)

QUESTIONS_PATH = Path(__file__).parent / "questions.json"
RESULTS_PATH = Path(__file__).parent / "results.json"
DATA_DIR = Path(__file__).parent.parent / "data"


def load_questions() -> dict[str, list[dict[str, Any]]]:
    """Load the benchmark question set.

    Returns:
        The parsed questions.json content.
    """
    return dict(json.loads(QUESTIONS_PATH.read_text(encoding="utf-8")))


def evaluate_dataset_selection(
    discovery: CatalogDiscovery, questions: list[dict[str, Any]]
) -> dict[str, Any]:
    """Score catalog discovery against expected dataset titles.

    Args:
        discovery: The catalog discovery instance to evaluate.
        questions: Dataset-selection questions, each with expected_title.

    Returns:
        Hit@1, Hit@3, question count, and per-question detail.
    """
    hit_at_1 = 0
    hit_at_3 = 0
    details = []

    for q in questions:
        matches = discovery.discover(q["question"], top_k=3)
        titles = [m.dataset.title for m in matches]
        is_hit1 = bool(titles) and titles[0] == q["expected_title"]
        is_hit3 = q["expected_title"] in titles
        hit_at_1 += int(is_hit1)
        hit_at_3 += int(is_hit3)
        details.append(
            {
                "id": q["id"],
                "question": q["question"],
                "expected_title": q["expected_title"],
                "top_matches": titles,
                "hit_at_1": is_hit1,
                "hit_at_3": is_hit3,
            }
        )

    n = len(questions)
    return {"hit_at_1": hit_at_1 / n, "hit_at_3": hit_at_3 / n, "n": n, "details": details}


def _answer_matches(actual: dict[str, str], expected: Any, operation: str) -> bool:
    """Compare a computed structured answer to the benchmark's expected value.

    Args:
        actual: The single summary/list row computed by compute_structured_result.
        expected: The expected value from questions.json.
        operation: The query plan operation, which decides comparison rules.

    Returns:
        True if the computed answer matches, within a small float tolerance.
    """
    if operation == "count":
        return int(actual["count"]) == int(expected)
    if operation == "list":
        return isinstance(expected, dict) and all(
            str(actual.get(key)) == str(value) for key, value in expected.items()
        )
    return abs(float(actual[operation]) - float(expected)) < 0.05


async def evaluate_structured_answers(
    client: DataGovSgClient, questions: list[dict[str, Any]]
) -> dict[str, Any]:
    """Run each structured question against the real datastore_search API.

    Args:
        client: The data.gov.sg client to query with.
        questions: Structured-answer questions with dataset_id, filters,
            operation, numeric_field, and a manually-verified expected_answer.

    Returns:
        Accuracy, question count, and per-question detail.
    """
    correct = 0
    details = []

    for q in questions:
        plan = QueryPlan(
            operation=q["operation"], filters=q["filters"], numeric_field=q.get("numeric_field")
        )
        try:
            fetched = await client.fetch_all_matching(q["dataset_id"], filters=q["filters"])
            records, _total = compute_structured_result(fetched.records, fetched.total, plan)
            actual = records[0]
            is_correct = _answer_matches(actual, q["expected_answer"], q["operation"])
            error = None
        except DataGovSgError as exc:
            actual = None
            is_correct = False
            error = str(exc)

        correct += int(is_correct)
        details.append(
            {
                "id": q["id"],
                "question": q["question"],
                "expected_answer": q["expected_answer"],
                "actual": actual,
                "correct": is_correct,
                "error": error,
            }
        )

    n = len(questions)
    return {"accuracy": correct / n, "n": n, "details": details}


def evaluate_rag(rag_config: Any, questions: list[dict[str, Any]]) -> dict[str, Any]:
    """Score real local-embedding retrieval over the ingested document corpus.

    Runs the actual chunker and sentence-transformers embedder from
    src/rag/ingest.py, then brute-force cosine similarity search (no
    Postgres needed for evaluation purposes -- the pgvector storage layer
    itself is covered by tests/test_rag's mocked-pool tests).

    Args:
        rag_config: RAG configuration (embedding model, chunk size, top_k).
        questions: RAG questions with an expected_source filename.

    Returns:
        Hit@3, MRR, question count, per-question detail, and a note on
        what faithfulness scoring would additionally require.
    """
    embedder = Embedder(rag_config)
    documents = load_documents(DATA_DIR)
    all_chunks = []
    for source, text in documents:
        all_chunks.extend(chunk_text(text, source, rag_config.chunk_size, rag_config.chunk_overlap))
    embedder.embed_chunks(all_chunks)

    chunk_vectors = np.array([c.embedding for c in all_chunks])
    chunk_norms = np.linalg.norm(chunk_vectors, axis=1)

    hits_at_3 = 0
    reciprocal_ranks = []
    details = []

    for q in questions:
        query_vector = np.array(embedder.embed_query(q["question"]))
        similarities = (chunk_vectors @ query_vector) / (
            chunk_norms * np.linalg.norm(query_vector) + 1e-9
        )
        ranked_indices = np.argsort(-similarities)

        seen_sources: list[str] = []
        for idx in ranked_indices:
            source = all_chunks[idx].source
            if source not in seen_sources:
                seen_sources.append(source)

        expected = q["expected_source"]
        top3 = seen_sources[:3]
        is_hit3 = expected in top3
        hits_at_3 += int(is_hit3)

        rank = seen_sources.index(expected) + 1 if expected in seen_sources else None
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        details.append(
            {
                "id": q["id"],
                "question": q["question"],
                "expected_source": expected,
                "top_sources": top3,
                "hit_at_3": is_hit3,
                "rank": rank,
            }
        )

    n = len(questions)
    return {
        "hit_at_3": hits_at_3 / n,
        "mrr": sum(reciprocal_ranks) / n,
        "n": n,
        "details": details,
        "note": (
            "Faithfulness (does the generated answer actually reflect the "
            "retrieved chunks) is not scored here: it requires a live LLM "
            "generation call, which needs OPENAI_API_KEY and is not available "
            "in this environment. Retrieval quality above uses the real local "
            "sentence-transformers embedder, not a mock."
        ),
    }


def evaluate_abstention(
    discovery: CatalogDiscovery,
    config: AppConfig,
    in_scope_questions: list[dict[str, Any]],
    out_of_scope_questions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score abstention decisions across both in-scope and out-of-scope questions.

    Args:
        discovery: The catalog discovery instance to evaluate.
        config: Application configuration, for the confidence threshold.
        in_scope_questions: Dataset-selection questions, all expected to
            NOT trigger abstention.
        out_of_scope_questions: The benchmark's abstention question set,
            all expected to trigger abstention.

    Returns:
        Accuracy, question count, and per-question detail.
    """
    correct = 0
    details = []

    def _check(question: str, expected_abstain: bool, source: str) -> None:
        nonlocal correct
        matches = discovery.discover(question, top_k=1)
        score = matches[0].score if matches else 0.0
        abstained = not matches or score < config.discovery.confidence_threshold
        is_correct = abstained == expected_abstain
        correct += int(is_correct)
        details.append(
            {
                "question": question,
                "source": source,
                "score": score,
                "expected_abstain": expected_abstain,
                "actual_abstain": abstained,
                "correct": is_correct,
            }
        )

    for q in out_of_scope_questions:
        _check(q["question"], expected_abstain=True, source="abstention_set")
    for q in in_scope_questions:
        _check(q["question"], expected_abstain=False, source="dataset_selection_set")

    n = len(details)
    return {"accuracy": correct / n, "n": n, "details": details}


async def main() -> None:
    """Run every evaluation and print/save a summary report."""
    config = load_config()
    questions = load_questions()
    discovery = CatalogDiscovery(REGISTRY)
    client = DataGovSgClient(config.datagovsg)

    dataset_selection = evaluate_dataset_selection(discovery, questions["dataset_selection"])
    structured = await evaluate_structured_answers(client, questions["structured_answers"])
    rag = evaluate_rag(config.rag, questions["rag"])
    abstention = evaluate_abstention(
        discovery, config, questions["dataset_selection"], questions["abstention"]
    )

    summary = {
        "dataset_selection": {
            "hit_at_1": dataset_selection["hit_at_1"],
            "hit_at_3": dataset_selection["hit_at_3"],
            "n": dataset_selection["n"],
        },
        "structured_answers": {"accuracy": structured["accuracy"], "n": structured["n"]},
        "rag_retrieval": {"hit_at_3": rag["hit_at_3"], "mrr": rag["mrr"], "n": rag["n"]},
        "abstention": {"accuracy": abstention["accuracy"], "n": abstention["n"]},
    }

    full_report = {
        "summary": summary,
        "dataset_selection_details": dataset_selection["details"],
        "structured_answers_details": structured["details"],
        "rag_retrieval_details": rag["details"],
        "rag_retrieval_note": rag["note"],
        "abstention_details": abstention["details"],
    }
    RESULTS_PATH.write_text(json.dumps(full_report, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nFull details written to {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
