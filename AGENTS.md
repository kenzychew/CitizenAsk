# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Add durable project-specific notes here as they are discovered through real work.

## Structured-output schemas and OpenAI strict mode

`llm.with_structured_output(SomeModel)` runs OpenAI's strict structured-output
mode, which rejects any field typed as an open-ended `dict[str, str]`
(arbitrary keys, no fixed `properties` list) — a 400 on the *first* call, not
a lint-time or unit-test failure, since the boundary-mocked tests stub the
LLM call rather than exercising `with_structured_output` for real. See
`src/generation/prompt.py`'s `QueryPlanLLM`/`QueryFilter` for the fix
pattern: give the LLM a `list[{key, value}]`-shaped field instead of a dict,
then convert to the dict shape downstream code expects immediately after the
call returns. Apply the same pattern to any new structured-output schema
that would otherwise need an arbitrary-keys dict field.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
