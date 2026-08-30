# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Add durable project-specific notes here as they are discovered through real work.

## Structured-output schemas and OpenAI strict mode

`llm.with_structured_output(SomeModel)` runs OpenAI's strict structured-output
mode, which rejects any field typed as an open-ended `dict[str, str]`
(arbitrary keys, no fixed `properties` list) — a 400 on the *first* call, not
a lint-time or unit-test failure, since the boundary-mocked tests stub the
LLM call rather than exercising `with_structured_output` for real. See
`generation/prompt.py`'s `QueryPlanLLM`/`QueryFilter` for the fix
pattern: give the LLM a `list[{key, value}]`-shaped field instead of a dict,
then convert to the dict shape downstream code expects immediately after the
call returns. Apply the same pattern to any new structured-output schema
that would otherwise need an arbitrary-keys dict field.

## Grounding query-planning filter values with real sample rows

`DatasetEntry.fields` (`schemas.py`) only carries column *names*, not the
literal format values take (e.g. a `month` column stored as `"1990-01"`, not
"January 1990"). The planning LLM (`build_plan_messages`,
`generation/prompt.py`) has no way to guess that format from metadata
alone, so `structured_node` (`agent/graph.py`) fetches a few real rows
via `DataGovSgClient.sample_rows` (`datagovsg/client.py`, cached per
`dataset_id` on the client instance) and passes them into the prompt as
"Example rows" before planning. Apply the same pattern — show the LLM real
data rather than hand-encoding a formatting rule — for any new column whose
value format isn't self-evident from its name.

## Flat layout: no `src/` nesting, and the `app_logging.py` naming exception

Packages live directly at the repo root (`agent/`, `api/`, `catalog/`,
`datagovsg/`, `generation/`, `rag/`) alongside root-level modules
(`config.py`, `schemas.py`, `exceptions.py`, `app_logging.py`) — there is no
`src/` layer. One module was deliberately named `app_logging.py` rather than
`logging.py`: a root-level `logging.py` shadows Python's stdlib `logging`
package (since the project root sits on `sys.path` via the editable
install), and which module wins is invocation-dependent — `uvicorn` picked
stdlib `logging` and failed to find `setup_logging`, while `python -c`/pytest
picked the local file and broke every third-party import of stdlib `logging`
(e.g. `python-dotenv`). When adding or renaming any top-level module, check
its name against `sys.stdlib_module_names` first and pick a non-colliding
name rather than reproducing this.

## pgvector pool creation must bootstrap the extension first

`asyncpg.create_pool(..., init=<hook that calls pgvector's register_vector>)`
fails on a fresh database: `register_vector` looks up the `vector` type on
the first connection the pool opens, but that type doesn't exist until
`CREATE EXTENSION IF NOT EXISTS vector` has run, and the pool's `init` hook
runs before any application code gets a chance to run that DDL. Open a
standalone bootstrap connection to create the extension, close it, and only
then call `create_pool` — see `api/dependencies.py::_try_create_pool` and
`scripts/ingest.py::main` for the pattern. Apply it to any new code path
that creates its own pool against this database rather than reusing the
process-wide one.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
