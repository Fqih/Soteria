# Changelog

All notable changes to Hernness are recorded here. Versions follow
[Semantic Versioning](https://semver.org/). The first public release
is **0.1.0**.

## [Unreleased]

### Added

- `src/hernness/config.py` — single-entry `build_provider_from_env`
  factory that dispatches `HERNNESS_PROVIDER` to the right provider
  config (`ollama`, | `minimax`, | `anthropic`, | `openai`).
- `src/hernness/providers/ollama.py` — native `/api/chat` adapter.
- `src/hernness/providers/anthropic.py` — Messages API adapter
  with `tool_use` block parsing.
- `src/hernness/providers/openai_compatible.py` — OpenAI
  Chat-Completions adapter that also covers self-hosted gateways.
- `src/hernness/providers/minimax.py` — new `from_soteria_env`
  constructor alongside the legacy benchmark `from_env` path.
- `src/hernness/app_tools/workspace.py` — `Workspace` class and
  `validate_path` helper. Rejects `..`, absolute escapes, symlink
  leaves, null bytes, and empty strings.
- `src/hernness/app_tools/approval.py` — `build_approval_callback`
  reading `HERNNESS_TOOLS_REQUIRE_APPROVAL`.
- `src/hernness/app_tools/file_tools.py` — `read_file_tool` and
  `write_file_tool` bound through `bind_workspace`.
- `src/hernness/chat.py` — interactive REPL that drives one
  `AgentRuntime.run(...)` invocation per user line. Slash commands:
  `/provider`, / `/inspect RUN_ID`, / `/resume RUN_ID`, / `/quit`.
- `hernness chat` — new CLI subcommand (slice 0.4 of the
  roadmap). Delegates to existing `build_provider_from_env`,
  `Workspace`, `bind_workspace`, file tools, and `SQLiteEventStore`.
- `.env.example` — empty placeholder template.
- `examples/app_tools_demo.py` — offline walk-through of the workspace
  + approval tools.

### Quality

- 219 offline tests across `tests/` (workspace traversal, approval,
  provider HTTP-mock, resume, state, SQLite parity).
- mypy strict, ruff lint + format, coverage gate `fail_under = 90`.

## [0.1.0] — 2026-07-20

Initial alpha foundation. Provider-agnostic async state machine,
append-only event history, deterministic `FakeProvider`, SQLite-backed
event store with resume, stop-reason taxonomy (13 enum values).
See the README for the user-facing overview.