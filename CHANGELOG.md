# Changelog

All notable changes to avo are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- OpenAPI-style tool schema helpers (JSON Schema 2020-12 export from
  Pydantic argument models) for downstream interop.
- OpenAI strict function-calling mode (`strict: true`) on
  `ChatCompletions` tool payloads.
- Anthropic `input_schema` block on `tool_use` payloads.
- OpenTelemetry tracing support behind the `[otel]` extra; emits
  spans per turn with `gen_ai.*` semantic-convention attributes.
- `docs/semver.md` — public SemVer commitment and deprecation policy.
- `docs/api-stability.md` — frozen public API surface declaration.
- `.github/dependabot.yml` — weekly dependency update PRs.
- CycloneDX SBOM generation per release.
- Sigstore signing for release wheels and source distributions.

### Changed

- `ci.yml` extended with `bandit`, `pip-audit`, and SBOM steps.
- `pyproject.toml` `[build]` target switched to `reproducible = true`
  to honor `SOURCE_DATE_EPOCH`.

## [0.1.2] — 2026-09-01

### Added

- `avo init` — scaffolds `.avo/skills/repo-overview/SKILL.md` and
  `AGENTS.md` in the current directory. Idempotent; respects existing
  files. Detects repo kind (python, node, rust, go, java, make, git,
  generic) and emits it in the output.
- Background jobs: a trailing `&` on a REPL line submits a turn as a
  background task. New slash commands: `/jobs`, `/job ID`, `/cancel ID`.
- `[jobs: N running]` indicator appended to the prompt when at least
  one background job is active.
- Typed `ContentBlock` model (`TextBlock`, `ImageBlock`) with
  per-provider translators for Anthropic, OpenAI, Ollama, and
  MiniMax.
- `/image <path>` slash command for one-off image input in the REPL.

### Changed

- `ChatContext` gained a `background: BackgroundJobManager` field.
- `run_shell` path uses `SandboxExecutor` exclusively; no direct
  `subprocess` host calls remain.
- Slash-command banner rendered as a multi-line box.
- Picker numbering aligned between `/sessions`, `/model`, `/skills`.

### Fixed

- `chat_shell_rc.py` now uses atomic write (`.tmp` + `os.replace`).
- Skill markdown frontmatter parser strips a leading UTF-8 BOM.
- `cli mcp add --env NAME=value` masks values whose key names suggest
  a secret (`*KEY`, `*TOKEN`, `*SECRET`, `*PASSWORD`, `*AUTH`,
  `*CREDENTIAL`).
- Destructive remove subcommands (`plugin remove`, `skill remove`,
  `mcp remove`) prompt for confirmation and accept `--yes` to skip.
- `provider` and `permission_mode` doctor output uses lowercase
  `yes` / `no` instead of capitalized booleans.
- README no longer duplicates the wordmark (SVG already shows it).
- Sweep of leftover `soteria` / `hernness` strings in code, docs,
  and benchmark fixtures.

## [0.1.1] — 2026-07-21

### Added

- Re-tag of 0.1.0 line with corrected metadata (no code changes).

## [0.1.0] — 2026-07-21

### Added

- Initial alpha foundation.
- Provider-agnostic async state machine with strict `StopReason`
  taxonomy (13 enum values) and a finite set of `AgentState`
  transitions.
- Append-only event log persisted in SQLite.
- Deterministic `FakeProvider` for offline tests and replays.
- `SQLiteEventStore` with checkpoint snapshots and resume helpers.
- Provider adapters: Ollama (`/api/chat`), Anthropic Messages,
  OpenAI-compatible Chat Completions, MiniMax.
- Application tools: `read_file`, `write_file`, `edit_file`,
  `glob`, `grep`, `web_fetch`, `web_search`, `git_status`,
  `workspace_map`, `plan_tasks`, `task` (sub-agent dispatch).
- `Workspace` path validator; rejects `..`, absolute escapes,
  symlink leaves, null bytes, and empty strings.
- `AVO_TOOLS_REQUIRE_APPROVAL` env var drives per-tool approval
  callbacks.
- Permission modes: `default`, `accept_edits`, `plan`, `bypass`.
- Hook registry: `PreToolUse`, `PostToolUse`, `Stop`,
  `Notification`.
- MCP adapter: JSON-RPC 2.0 client + `FunctionTool` wrapper.
- Markdown skill loader (`<workspace>/.avo/skills/<name>/SKILL.md`).
- Cost tracking: `UsageTracker` + USD estimator + `TokenLedger`.
- Budget enforcement: `BudgetConfig` + `BudgetChecker`.
- Rate limiting: async token-bucket `RateLimiter`.
- Audit log: JSONL with deep secret redaction (suffix variants,
  JWT, AWS, GitLab, Stripe, query strings, cycle-safe, thread-safe,
  symlink-guarded).
- Eval harness: `EvalCase` + `EvalReport` with substring and
  tool-call assertions.
- Plugin discovery via `avo.tools`, `avo.providers`,
  `avo.notifiers` entry-point groups.
- Metrics: counter, gauge, histogram with label cardinality cap.
- Concurrency limiter: async semaphore + in-flight gauge.
- Schema registry for keyed Pydantic validation.
- Conversation store with persistent append-only turns.
- Retry policy with exponential backoff.
- Notification dispatcher (webhook + desktop).
- CLI subcommands: `avo runs list|inspect|resume`, `avo chat`,
  `avo doctor`.
- `project.md` deep technical reference; `README.md` user guide;
  `Makefile` for common dev tasks.

### Quality

- 219 offline tests in `tests/`.
- `mypy --strict` clean across `src/avo`.
- `ruff check` + `ruff format --check` clean.
- Coverage gate: `fail_under = 90`.

[Unreleased]: https://github.com/Fqih/avo/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/Fqih/avo/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/Fqih/avo/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Fqih/avo/releases/tag/v0.1.0
