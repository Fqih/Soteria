# Changelog

All notable changes to Soteria are recorded here. Versions follow
[Semantic Versioning](https://semver.org/). The first public release
is **0.1.0**.

## [Unreleased]

### Added

- `src/soteria_loop/config.py` — single-entry `build_provider_from_env`
  factory that dispatches `SOTERIA_PROVIDER` to the right provider
  config (`ollama`, | `minimax`, | `anthropic`, | `openai`).
- `src/soteria_loop/providers/ollama.py` — native `/api/chat` adapter.
- `src/soteria_loop/providers/anthropic.py` — Messages API adapter
  with `tool_use` block parsing.
- `src/soteria_loop/providers/openai_compatible.py` — OpenAI
  Chat-Completions adapter that also covers self-hosted gateways.
- `src/soteria_loop/providers/minimax.py` — new `from_soteria_env`
  constructor alongside the legacy benchmark `from_env` path.
- `src/soteria_loop/app_tools/workspace.py` — `Workspace` class and
  `validate_path` helper. Rejects `..`, absolute escapes, symlink
  leaves, null bytes, and empty strings.
- `src/soteria_loop/app_tools/approval.py` — `build_approval_callback`
  reading `SOTERIA_TOOLS_REQUIRE_APPROVAL`.
- `src/soteria_loop/app_tools/file_tools.py` — `read_file_tool` and
  `write_file_tool` bound through `bind_workspace`.
- `src/soteria_loop/chat.py` — interactive REPL that drives one
  `AgentRuntime.run(...)` invocation per user line. Slash commands:
  `/provider`, / `/inspect RUN_ID`, / `/resume RUN_ID`, / `/quit`.
- `soteria-loop chat` — new CLI subcommand (slice 0.4 of the
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