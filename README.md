# CitizenAsk

A general-purpose agentic assistant over Singapore government open data (data.gov.sg), built to be excellent at the mechanism rather than one fixed topic: finding the right dataset, querying it correctly, and knowing when it can't answer.

## Motivation

data.gov.sg holds 4,000+ real datasets across 69+ agencies, from HDB resale prices to dengue cluster counts. Most "agentic RAG" demos pick one narrow vertical and get good at it. This project asks a different question: can an agent be genuinely reliable across an arbitrary slice of that catalog, by being disciplined about three things: discovering the right dataset from a curated registry, querying it correctly through the real `datastore_search` API, and abstaining plainly when nothing in the registry actually matches?

The agent is a LangGraph `StateGraph`, not a single prompt-and-hope call. Routing is genuinely conditional: discover a candidate dataset, then branch into a structured API query, a document RAG lookup, or an abstention, depending on what was actually found.

## Architecture

```
question
   |
   v
discover (BM25 over curated registry metadata: title, agency, description, tags)
   |
   +-- no confident match --------------------------> abstain (no LLM call)
   |
   +-- matched a STRUCTURED dataset -----> plan query (LLM) -> datastore_search (real API)
   |                                                              -> compute (count/avg/sum/min/max/list)
   |                                                              -> synthesize answer (LLM)
   |
   +-- matched a DOCUMENT dataset --------> pgvector retrieval -> synthesize cited answer (LLM)
                                             (abstains if nothing relevant retrieves)
```

### Dataset discovery: BM25, not embeddings

The curated registry is 32 hand-picked, individually verified data.gov.sg datasets. Discovery scores a question against each entry's title/agency/description/tags with BM25 (`catalog/discovery.py`), not an embedding model. The registry is small and the matching problem is lexical (does this question's vocabulary overlap with this dataset's metadata), so BM25 gets the job done without an API key or a vector index, and it's what makes discovery accuracy fully testable offline.

The tradeoff: BM25 can false-positive on short, generic phrasing that happens to share a word with a dataset title. In manual calibration, "What is the meaning of life?" scored 5.98 against `CPF LIFE Payout Guide` (because of the word "life") and "Recommend a good science fiction movie" scored 8.3 against the PSLE Science dataset (because of "science"), both above a naive threshold. The confidence threshold (4.5, in `configs/config.yaml`) was picked from a wider calibration sweep where genuinely out-of-scope questions scored under 3.5 and genuinely in-scope questions scored above 5.1, but adversarial short queries that lexically collide with a dataset's title are a known, real limitation of this approach rather than a solved problem.

### Structured queries against the real API

Once discovery finds a `STRUCTURED` dataset, an LLM call turns the question into a `QueryPlan` (operation + column filters + numeric field, `generation/prompt.py`), which `datagovsg/client.py` executes against the real `https://data.gov.sg/api/action/datastore_search` endpoint, no API key required. The client paginates through matching rows (`fetch_all_matching`, capped at 10,000 rows as a safety limit) and retries HTTP 429s with backoff, which the live API does return under sustained unauthenticated traffic. `agent/tools.py` then computes count/average/sum/min/max over the fetched rows, or returns a capped row list, before a final LLM call synthesizes the answer.

### RAG fallback for document-shaped agencies

Some registry entries are guide-shaped rather than table-shaped (CPF LIFE payouts, HDB's BTO process, NEA's dengue programme, LTA's COE system, PUB's NEWater, MOH's MediSave). Their content lives as plain text under `data/` and is chunked, embedded locally with `sentence-transformers` (`all-MiniLM-L6-v2`, no API key), and stored in Postgres via `pgvector` (`rag/ingest.py`, `rag/retriever.py`). If retrieval turns up nothing relevant, the RAG node abstains rather than answering from thin air; if Postgres was unreachable at startup (no retriever configured), it abstains with a distinct "document search is temporarily unavailable" message instead of crashing.

### Abstention

If discovery's top score is below the confidence threshold, the graph routes straight to an `abstain` node that never touches the LLM: the agent says plainly that it doesn't have a matching dataset (`generation/prompt.py::ABSTENTION_MESSAGE`). A structured query that finds a matching dataset but then fails at the data.gov.sg call (unknown resource_id, no matching rows) is treated differently: that's a real answer ("I found the dataset but couldn't compute this"), not a silent guess and not a blanket abstention.

## Evaluation

Run with `uv run python eval/evaluate.py`. Every number below comes from that script running against the real, live data.gov.sg API and the real local embedding model, not a mock, and was captured on 2026-08-21.

| Metric | Result | n |
| --- | --- | --- |
| Dataset selection Hit@1 | 1.000 | 20 |
| Dataset selection Hit@3 | 1.000 | 20 |
| Structured-answer accuracy | 1.000 | 8 |
| RAG retrieval Hit@3 | 1.000 | 6 |
| RAG retrieval MRR | 1.000 | 6 |
| Abstention accuracy | 1.000 | 30 |

Full per-question detail (which dataset each question resolved to, the exact computed vs. expected structured answer, retrieval ranks, discovery scores) is written to `eval/results.json` on every run.

**Structured-answer correctness** is checked against 8 questions across 8 different datasets (HDB resale, median income, PSLE math/science, NEWater volume/tariff, manufacturing output, labour force), each with an expected value I verified by hand against the live API while writing the benchmark. All 8 use historically-closed filters (a specific past year, a specific past month) rather than "latest"/"this month" questions, since several of these datasets update on a rolling basis and a hardcoded expectation for a moving window would go stale.

**RAG retrieval** measures Hit@3/MRR for the real chunker + real `sentence-transformers` embedder against the 6-document corpus under `data/`, using brute-force cosine similarity rather than a live pgvector call (see "What isn't tested live" below). It does not measure faithfulness: judging whether a generated answer stays grounded in retrieved chunks needs an actual LLM call, which needs `OPENAI_API_KEY`.

**Abstention accuracy** is scored across two directions in the same run: the 10 out-of-scope questions in `eval/questions.json` (`abstention`, all correctly abstained) and the 20 in-scope dataset-selection questions checked for the opposite (correctly did *not* abstain).

### What isn't tested live in this environment

On 2026-08-25 the running app (`uv run uvicorn api.main:app --port 8000`) was smoke-tested against a real `OPENAI_API_KEY` by hitting `/query` for real, and `eval/evaluate.py` was re-run with that key present. Results:

- **Abstention** (`agent/graph.py`'s `abstain` node): confirmed live, no LLM call. A genuinely out-of-scope question ("What's a good recipe for chicken curry?") correctly returned `ABSTENTION_MESSAGE` with `route: "abstain"`. A short, lexically-colliding question ("What is the meaning of life?") reproduced live the exact false positive the discovery section above documents from manual calibration: it matched the CPF LIFE Payout Guide dataset and routed to `rag` instead of abstaining.
- **Structured query planning** (`generation/llm.py::make_plan_query`, called from `structured_node` in `agent/graph.py`): on 2026-08-25 this failed on every structured question with a real error from the OpenAI API, not a mock or environment artifact — `llm.with_structured_output(QueryPlan)`'s generated JSON schema was rejected by OpenAI's strict structured-output validation because `QueryPlan.filters` is an open-ended `dict[str, str]` (arbitrary keys, no fixed `properties` list), which strict mode does not support. This was fixed on 2026-08-26: the LLM call now targets a new `QueryPlanLLM` schema (`filters: list[QueryFilter]`, i.e. a list of `{"column", "value"}` objects) instead of `QueryPlan` directly, converted to the existing `QueryPlan.filters` dict shape immediately after the structured-output call returns. `QueryPlan`'s public shape, and every downstream caller/test, are unchanged. Re-verified live against the running app with a real `OPENAI_API_KEY`:
  - The exact original failure reproduction, "What is the average HDB resale price in Bishan?", now returns a real computed answer instead of a 400: "The average HDB resale price in Bishan is 344,194.18."
  - Two `eval/questions.json` structured-answer fixtures were run end-to-end and matched their hand-verified expected values exactly: "What percentage of Chinese PSLE students scored A* to C in Standard Mathematics in 2020?" → 89.2%, and "How much NEWater was sold in Singapore in 2020?" → 141.1 million cubic meters. **Final answer synthesis** (`generate_answer`) is now confirmed live on this path.
  - Fixing the 400 error surfaced a second, separate, pre-existing bug: at temperature 0.0, gpt-4o-mini didn't reliably follow the prompt's casing guidance for filter values, so the Bishan question initially planned `{"town": "Bishan"}` instead of `{"town": "BISHAN"}` — since `datastore_search` does exact-string matching, that returned zero rows instead of an error. Strengthening `_PLAN_SYSTEM_PROMPT`'s casing guidance fixed this for town names (confirmed live, both above). A related but distinct value-format mismatch remained open and was *not* fixed at the time: `eval/questions.json`'s `sa01` ("How many HDB resale transactions were there in Ang Mo Kio in January 1990?") planned a `month` filter of `"January 1990"` rather than the dataset's `"1990-01"` format, so that question still returned "no rows matched" live — a general column-value-formatting gap beyond town-name casing.
  - That value-format gap was fixed on 2026-08-26: column names alone don't tell the planning LLM what literal format a column's values take, so `build_plan_messages` (`generation/prompt.py`) now takes an optional `sample_rows` argument and includes it in the prompt as "Example rows", and `structured_node` (`agent/graph.py`) fetches 2-3 real rows via a new `DataGovSgClient.sample_rows` (`datagovsg/client.py`, cached per dataset_id on the client instance) before calling `plan_query`, so the LLM can mirror the exact value format it sees rather than guessing from the question's phrasing. A sample-fetch failure (network hiccup, no live resource) is caught and falls back to planning without a sample rather than failing the whole request. Re-verified live against the running app with a real `OPENAI_API_KEY`:
    - "average HDB resale price in Bishan in January 1990" (a variant of the `sa01` reproduction, using `average` instead of `count` so it also exercises `numeric_field`) now returns a real computed answer instead of "no rows matched": "66,525.0. Source: HDB Resale Flat Prices (Housing & Development Board)." This was independently cross-checked by calling the live `datastore_search` API directly with `{"town": "BISHAN", "month": "1990-01"}`: 8 matching rows, average 66,525.0 — an exact match.
    - "average HDB resale price in Bishan" (no date filter, the already-fixed path from the casing fix above) still returns "The average HDB resale price in Bishan is 344,194.18." — unchanged from the earlier verification, confirming the new sample-fetch step didn't regress it.
    - **Tradeoff, not fully eliminated**: fetching a sample adds one extra live `datastore_search` call the first time `structured_node` sees a given dataset in a process's lifetime — not on every question, since `sample_rows` caches per dataset_id on the `DataGovSgClient` instance, and that instance is a process-wide singleton (`api/dependencies.py`). For the 32-dataset curated registry this is at most 32 extra calls total over the life of a running app, not one per request.
  - Abstention was re-confirmed live and is unchanged from 2026-08-25 (see bullet above).
- **pgvector storage / RAG retrieval** (`rag/ingest.py::VectorIndexer`, `rag/retriever.py::DocRetriever`): still not tested live. No Postgres was reachable in this environment (port 5432 closed, no `psql`/`pg_isready` binary installed, no `docker-compose` file in the repo, and Docker itself isn't wired up in this WSL distro to stand one up locally). `/health` correctly reported `rag_available: false` instead of crashing, and the RAG node's graceful-degradation message ("document search is temporarily unavailable") was confirmed live on both queries above that routed to `rag`. Unit tests continue to cover the real SQL (extension/table/index DDL, upsert, cosine similarity search) against a fake `asyncpg.Pool`/`Connection` (`tests/test_rag/conftest.py`).
- **`eval/evaluate.py`**: re-run live with a real key present on 2026-08-25; output was numerically identical to the table above (all metrics 1.0). This is expected, not a live-LLM confirmation — as already noted below, this script's RAG/structured checks don't call the LLM at all. Not re-run for the 2026-08-26 fix, since it wouldn't exercise `make_plan_query` either; the direct `/query` calls above are the live confirmation for that path instead.

Everything that doesn't need Postgres or a live LLM call continues to be exercised for real: the data.gov.sg client hits the live API in both unit tests (`@pytest.mark.integration`, `tests/test_datagovsg/test_client.py`) and the eval script; the embedder downloads and runs a real `sentence-transformers` model; BM25 discovery runs against the real 32-entry registry.

## Tech stack

- **Orchestration**: LangGraph `StateGraph`
- **LLM**: OpenAI direct via `langchain-openai` (no Vertex AI/Azure)
- **Dataset discovery**: BM25 (`rank-bm25`) over curated metadata
- **Structured data**: real `data.gov.sg` `datastore_search` API, `httpx` + `tenacity` retries
- **RAG**: PostgreSQL + pgvector, local `sentence-transformers` embeddings
- **API**: FastAPI, SSE streaming (`sse-starlette`)
- **Config**: YAML under `configs/`, loaded through `config.py::load_config`
- **Testing**: pytest, mocked LLM/DB boundaries, live data.gov.sg integration tests

<details>
<summary>Getting started</summary>

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- PostgreSQL with the `pgvector` extension available (only needed for the RAG fallback; the structured-query path and discovery work without it)
- An OpenAI API key (only needed for actually generating answers; discovery, structured queries, and abstention logic all work and are tested without one)

### Quick start

```bash
cp .env.example .env
# Edit .env: set OPENAI_API_KEY and (if using RAG) DATABASE_URL

uv sync
uv run uvicorn api.main:app --reload --port 8000
```

Then query it:

```bash
curl "http://localhost:8000/query?q=What+is+the+average+HDB+resale+price+in+Bishan%3F"
curl "http://localhost:8000/datasets"
curl "http://localhost:8000/health"
```

### Ingesting the RAG document corpus

Against a running Postgres with `pgvector`, from a Python shell or a short script:

```python
import asyncio
from config import load_config
from rag.ingest import Embedder, VectorIndexer, chunk_text, load_documents
import asyncpg


async def main():
    config = load_config()
    pool = await asyncpg.create_pool(config.database_url)
    indexer = VectorIndexer(pool, config.rag)
    await indexer.ensure_table()

    embedder = Embedder(config.rag)
    chunks = []
    for source, text in load_documents(config.data_dir):
        chunks.extend(chunk_text(text, source, config.rag.chunk_size, config.rag.chunk_overlap))
    embedder.embed_chunks(chunks)
    await indexer.upsert_chunks(chunks)


asyncio.run(main())
```

### Local development

```bash
# Install dependencies
uv sync

# Run tests (unit only, no live services required)
uv run pytest -m "not integration"

# Run tests including live data.gov.sg calls (no API key needed)
uv run pytest

# Run the evaluation harness
uv run python eval/evaluate.py

# Lint, format, typecheck
uv run ruff check .
uv run ruff format .
uv run mypy --strict agent api catalog datagovsg generation rag config.py schemas.py exceptions.py app_logging.py
```

### API

- `GET /query?q=your+question` SSE stream: a `token` event with the answer, then a `metadata` event with route/citations/abstained/dataset_id
- `GET /datasets` list every dataset in the curated registry
- `GET /health` reports RAG (pgvector) availability and registry size

</details>

<details>
<summary>Project structure</summary>

```
agent/
  graph.py              # LangGraph StateGraph: discover -> structured/rag/abstain
  tools.py               # Structured-query result computation (count/avg/sum/min/max)
catalog/
  registry.py            # 32 curated, individually verified data.gov.sg datasets
  discovery.py            # BM25 discovery over registry metadata
datagovsg/
  client.py               # Real datastore_search client (pagination, 429 retry)
rag/
  ingest.py                # Document loading, chunking, local embedding, pgvector indexing
  retriever.py              # pgvector cosine similarity retrieval
generation/
  llm.py                    # OpenAI client factory + plan_query/generate_answer wiring
  prompt.py                  # Prompt templates, QueryPlan schema, abstention message
api/
  main.py                     # FastAPI app: /query (SSE), /datasets, /health
  dependencies.py              # Dependency container, graceful Postgres degradation
config.py                      # AppConfig dataclasses, load_config helper
schemas.py                      # Shared dataclasses (DatasetEntry, DocChunk, AgentAnswer, ...)
exceptions.py                    # Domain exception hierarchy
app_logging.py                    # setup_logging() YAML config with basicConfig fallback (named app_logging to avoid shadowing the stdlib logging module)
configs/                          # config.yaml, logging.yaml
eval/
  questions.json                  # 20 dataset-selection + 8 structured + 6 RAG + 10 abstention
  evaluate.py                      # Runs all four against real data, writes results.json
data/                              # RAG document corpus (6 agency guides)
tests/                             # pytest, one test package per top-level package
```

</details>
