# Hernness — project reference

A Python runtime for safe, observable AI agents. This document is the deep
technical reference for contributors and operators. The README stays short
and product-oriented; this file covers every module, every public type,
every state-machine invariant, and every CLI / API surface.

---

## 1. Mission and scope

Hernness answers the six questions every agent operator eventually asks:

1. What is the agent doing *right now*?
2. Why did this run stop?
3. Did it repeat itself without making progress?
4. Can an interrupted run continue safely?
5. Which tool calls actually executed?
6. Can the run be reproduced without calling a paid model again?

It does so by wrapping a tool-using agent loop in:

- An **explicit state machine** with eight states and four terminal states.
- An **append-only event log** with sequence invariants.
- A **policy engine** for step, runtime, token, repetition, error, and
  no-progress limits.
- A **provider-neutral `ModelProvider` Protocol** with built-in adapters for
  `FakeProvider`, `OllamaProvider`, `MiniMaxProvider`, `AnthropicProvider`,
  and `OpenAICompatibleProvider`.
- A **`HERNNESS_`-prefixed env-var factory** (`config.build_provider_from_env`)
  that dispatches a single env block to the right provider without touching
  agent code.
- A **pluggable event store** with in-memory and SQLite implementations.
- A **checkpoint + resume** path that uses completed tool-call IDs to make
  duplicates impossible.
- A **`LetheMemoryAdapter`** for long-term context recall outside the
  operational event log.
- A **`app_tools/` package** of safe-by-construction file and shell tools
  bound to a fixed workspace, plus an `HERNNESS_TOOLS_REQUIRE_APPROVAL`
  approval policy read from the environment.

The core runtime depends only on Pydantic. `httpx` is an optional
dependency for live providers and is loaded lazily inside each provider
module; `matplotlib` is an optional dependency for the live-benchmark
renderer; `docker` is an optional dependency reserved for the next
`sandbox.py` iteration.

### 1.1 Direction: model-agnostic CLI assistant

Hernness is positioned to grow into a **terminal-based AI assistant** that
plugs into any provider the operator chooses — local Ollama for offline
work, Anthropic or OpenAI for hosted quality, MiniMax for vendor
flexibility. The pieces that make that direction viable are already
in place:

- Provider selection is one env var (`HERNNESS_PROVIDER`), not a code edit.
- Every provider shares the same `ModelProvider` Protocol contract, so
  the CLI / runtime code is identical regardless of backend.
- `app_tools/` gives the assistant a safe-by-construction toolbox
  (read_file, write_file, future run_shell through a sandbox).
- The SQLite event store means a long assistant session can be paused
  and resumed, and the operator can inspect its history with
  `hernness runs inspect RUN_ID`.

The CLI is currently a database inspector (`hernness runs list /
inspect / resume`) plus a `hernness chat` REPL that wires a
runtime + `app_tools/` + the env-driven provider into an interactive
session, and `hernness doctor` that verifies the env without an
HTTP call. The roadmap below (§26) tracks the remaining slices.

---

## 2. Public API surface (`src/hernness/__init__.py`)

```python
from hernness import (
    AgentEvent,
    AgentRuntime,
    Checkpoint,
    EventType,
    FakeProvider,
    FunctionTool,
    LetheMemoryAdapter,
    LoopPolicy,
    MemoryProvider,
    ModelRequest,
    ModelResponse,
    ProgressDetector,
    RunRecord,
    RunResult,
    RunState,
    RunTrace,
    ScriptItem,
    StopReason,
    TokenUsage,
    Tool,
    ToolCall,
    ToolMetadata,
    ToolRegistry,
    ToolResult,
    TraceEntry,
    TraceInspector,
)

__version__ = "0.1.0"
```

The distribution name is `hernness` (PyPI-friendly hyphen); the Python
package name is `hernness` (hyphens are illegal in module names).
The CLI entry point is `hernness`.

---

## 3. Core runtime (`src/hernness/runtime.py`)

`AgentRuntime` owns the lifecycle of one run at a time. A single runtime
instance serializes its own runs because a provider may maintain cursor
state; different runtime instances operate independently. Serialization
is enforced by `self._execution_lock = asyncio.Lock()`.

### 3.1 Construction

```python
AgentRuntime(
    provider: ModelProvider,                     # required
    tools: Iterable[Tool] = (),                  # default empty
    policy: LoopPolicy | None = None,            # default LoopPolicy()
    event_store: EventStore | None = None,       # default InMemoryEventStore()
    clock: Callable[[], datetime] = utc_now,
    approval_callback: Callable[[ToolCall], bool | Awaitable[bool]] | None = None,
    memory: LetheMemoryAdapter | None = None,
)
```

`clock` must return a timezone-aware `datetime` (UTC preferred); the
constructor raises `ValueError` from `_now()` otherwise.

### 3.2 Public methods

| Method | Returns | Effect |
|---|---|---|
| `await runtime.run(task, *, user_state=None, run_id=None)` | `RunResult` | Create and execute a run to terminal state. |
| `await runtime.resume(run_id)` | `RunResult` | Reconcile a persisted non-terminal run and continue it. |
| `await runtime.inspect(run_id)` | `RunTrace` | Build a chronological trace for a stored run. |

`run()` and `resume()` hold `_execution_lock` for the entire run.

### 3.3 Internal state machine

The runtime is dispatched via a flat `handlers` dict:

```python
handlers = {
    RunState.CREATED:              self._handle_created,
    RunState.MODEL_PENDING:        self._handle_model_pending,
    RunState.DECISION_RECEIVED:    self._handle_decision_received,
    RunState.TOOL_PENDING:         self._handle_tool_pending,
    RunState.APPROVAL_PENDING:     self._handle_approval_pending,
    RunState.TOOL_EXECUTING:       self._handle_tool_executing,
    RunState.OBSERVATION_RECORDED: self._handle_observation_recorded,
    RunState.PAUSED:               self._handle_paused,
}
```

`_drive` loops until `is_terminal(state)` is true, then returns
`self._result(run)`.

#### Boundary checks between operations

`_handle_model_pending` and `_handle_tool_executing` call
`_operation_boundary_reason` *before* issuing the next I/O. The runtime
**does not preempt a tool already in flight**. If the user supplies a
provider timeout via `LoopPolicy.provider_timeout_seconds`, it is enforced
via `asyncio.wait_for` around the provider call. If the user supplies a
tool timeout via `LoopPolicy.tool_timeout_seconds`, the same applies to
tool calls.

#### Token-budget policy

After every `MODEL_RESPONDED` event, the runtime asks
`policy.token_budget_reason(token_usage, accounting_available=...)`:

- If `accounting_available=False`, the policy **never** raises
  `TOKEN_BUDGET_EXCEEDED` (missing usage is never fabricated as zero).
- If both halves are present, the sum is compared against `max_total_tokens`.

#### Repeated-action containment

After `MODEL_RESPONDED` and the `TOOL_REQUESTED` event,
`_handle_decision_received` calls `detector.repeated_action(limit)` with
`policy.repeated_action_limit`. If true, `StopReason.REPEATED_ACTION`
triggers. Fingerprints include the tool name and canonical JSON arguments
but exclude the tool-call ID.

#### Error policy

Each provider exception increments `consecutive_errors`. If the limit is
hit, `StopReason.CONSECUTIVE_ERRORS` triggers. `FakeProviderExhaustedError`
or a non-retryable `ProviderError` terminates immediately with
`StopReason.PROVIDER_ERROR`.

### 3.4 Checkpoint and resume

`_checkpoint(context)` snapshots:

- The full `messages` list.
- `next_step`, `token_usage`, `token_accounting_available`,
  `consecutive_errors`.
- The progress detector's `action_history`, `observation_history`,
  `model_history`, and `progress_markers`.
- The set of `completed_tool_call_ids`.
- The `user_state` dict.
- The `provider_metadata` snapshot (empty unless the provider implements
  `StatefulModelProvider`).
- The `pending_response` (if any).
- The `policy` (full JSON dump).
- A generated `checkpoint_id`.

Default checkpoints are written when `policy.checkpoint_every_step` is true
or after every tool call. `resume(run_id)`:

1. Loads the `RunRecord` and refuses if terminal (`RunAlreadyTerminalError`).
2. Loads the latest `Checkpoint` (`CheckpointNotFoundError` if none).
3. Restores the policy, detector, completed tool-call IDs, user state, and
   provider state.
4. Calls `_reconcile_after_checkpoint` to walk the trailing event tail:
   - `TOOL_STARTED` IDs are collected.
   - For every `TOOL_COMPLETED` / `TOOL_FAILED` event after the
     checkpoint, the result is replayed into the live message log and
     `consecutive_errors` is updated.
   - If any `TOOL_STARTED` has no matching durable result, **resume
     refuses** with `UnsafeResumeError` — the external side effect is
     uncertain.
5. Appends a `RUN_RESUMED` event and continues with `_drive`.

---

## 4. Domain models (`src/hernness/models.py`)

`SoteriaModel` is the strict base:

```python
class SoteriaModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
```

| Model | Key fields | Validation |
|---|---|---|
| `TokenUsage` | `input_tokens`, `output_tokens` (`>= 0`) | `.plus(other)` returns the element-wise sum; `.total_tokens` returns the sum. |
| `ToolMetadata` | `name`, `description`, `input_schema` | Both strings must be non-empty. |
| `ToolCall` | `tool_call_id` (default UUID), `name`, `arguments` | `tool_call_id` and `name` must be non-empty. |
| `ToolResult` | `tool_call_id`, `tool_name`, `success`, `output`, `error`, `started_at`, `finished_at`, `duration_ms` | `started_at` / `finished_at` must be timezone-aware; success and error are mutually exclusive; `finished_at >= started_at`. |
| `ModelRequest` | `request_id`, `run_id`, `step`, `messages`, `tools` | `step >= 1`. |
| `ModelResponse` | `content` / `tool_call` / `usage` | Exactly one of `content` or `tool_call` must be set; `is_final` returns `content is not None`. |
| `RunRecord` | `run_id`, `task`, `state`, `stop_reason`, `output`, `error`, `steps`, `token_usage`, `token_accounting_available`, `user_state`, `created_at`, `updated_at`, `duration_seconds` | Terminal state requires a `stop_reason`; non-terminal forbids one; `updated_at >= created_at`. |
| `Checkpoint` | `checkpoint_id`, `run_id`, `state`, `messages`, `next_step`, `token_usage`, `token_accounting_available`, `consecutive_errors`, `*_history`, `completed_tool_call_ids`, `user_state`, `policy`, `provider_metadata`, `pending_response`, `last_event_sequence` | `next_step >= 1`. |
| `RunResult` | `run_id`, `status`, `stop_reason`, `output`, `error`, `steps`, `token_usage`, `token_accounting_available` | Always terminal: `validate_terminal_outcome(status, stop_reason)` is enforced. |

`utc_now()` and `new_id()` (UUID4) are module-level helpers used by every
model.

---

## 5. State machine (`src/hernness/state.py`)

Eight states, four terminal:

| State | Terminal? | Description |
|---|:---:|---|
| `CREATED` | no | Persisted but not yet driving. |
| `MODEL_PENDING` | no | About to call the provider. |
| `DECISION_RECEIVED` | no | Provider returned; deciding content vs. tool call. |
| `TOOL_PENDING` | no | Tool call identified; about to log `TOOL_REQUESTED`. |
| `APPROVAL_PENDING` | no | Awaiting `approval_callback`. |
| `TOOL_EXECUTING` | no | Tool call in flight. |
| `OBSERVATION_RECORDED` | no | Tool returned; about to write back to the message log. |
| `PAUSED` | no | Reserved for future checkpoint-then-pause flows. |
| `COMPLETED` | **yes** | Successful final answer. |
| `FAILED` | **yes** | Operational failure or non-retryable provider error. |
| `STOPPED` | **yes** | Policy containment (`REPEATED_ACTION`, `MAX_STEPS`, `MAX_RUNTIME`, `TOKEN_BUDGET_EXCEEDED`, `NO_PROGRESS`, `POLICY_DENIED`, `CONSECUTIVE_ERRORS`, ...). |
| `CANCELLED` | **yes** | `asyncio.CancelledError` propagated from the caller. |

### 5.1 Stop reasons (13 enum values)

```python
class StopReason(StrEnum):
    COMPLETED = "completed"
    MAX_STEPS = "max_steps"
    MAX_RUNTIME = "max_runtime"
    TOKEN_BUDGET_EXCEEDED = "token_budget_exceeded"
    REPEATED_ACTION = "repeated_action"
    NO_PROGRESS = "no_progress"
    CONSECUTIVE_ERRORS = "consecutive_errors"
    POLICY_DENIED = "policy_denied"
    PROVIDER_ERROR = "provider_error"
    TOOL_ERROR = "tool_error"
    INVALID_MODEL_RESPONSE = "invalid_model_response"
    INTERNAL_ERROR = "internal_error"
    USER_CANCELLED = "user_cancelled"
```

### 5.2 Transition table (`STOP_REASONS_BY_STATE`)

`validate_transition(from_state, to_state)` rejects illegal edges.
`validate_terminal_outcome(state, stop_reason)` enforces that every
`StopReason` is legal in the matching terminal state. `MAX_STEPS`,
`MAX_RUNTIME`, `TOKEN_BUDGET_EXCEEDED`, `REPEATED_ACTION`, `NO_PROGRESS`,
`CONSECUTIVE_ERRORS`, `POLICY_DENIED` all map to `STOPPED`.
`PROVIDER_ERROR`, `TOOL_ERROR`, `INVALID_MODEL_RESPONSE`, `INTERNAL_ERROR`
map to `FAILED`. `COMPLETED` maps to `COMPLETED`. `USER_CANCELLED` maps to
`CANCELLED`.

---

## 6. Event log (`src/hernness/events.py`)

| EventType | When | Key payload |
|---|---|---|
| `RUN_CREATED` | `runtime.run` starts | task, policy |
| `RUN_RESUMED` | `runtime.resume` after reconciliation | checkpoint_id, checkpoint_sequence, state |
| `MODEL_REQUESTED` | before provider call | step, request dump |
| `MODEL_RESPONDED` | after provider call | step, response dump, duration_ms, `token_accounting_available` |
| `MODEL_FAILED` | provider exception | step, error, error_type, duration_ms, consecutive_errors |
| `STATE_CHANGED` | every transition | from_state, to_state, stop_reason?, error? |
| `TOOL_REQUESTED` | tool call identified | tool_call_id, idempotency_key, name, arguments |
| `TOOL_APPROVAL_REQUESTED` | awaiting approval | tool_call_id, idempotency_key, name, mode |
| `TOOL_APPROVED` / `TOOL_DENIED` | approval callback result | tool_call_id, name |
| `TOOL_STARTED` | tool invocation begins | tool_call_id, idempotency_key, name, arguments |
| `TOOL_COMPLETED` / `TOOL_FAILED` | tool invocation ends | tool_call_id, name, result dump, duration_ms, error |
| `POLICY_TRIGGERED` | a policy containment fired | stop_reason, error |
| `RUN_FINALIZED` | terminalization | final state, stop_reason |

`validate_event_append(existing, new)` enforces:

- Sequence is monotonic per run.
- `STATE_CHANGED.from_state` matches the previous event's recorded state.
- `TOOL_COMPLETED.tool_call_id` and `TOOL_FAILED.tool_call_id` correspond
  to a prior `TOOL_STARTED`.

---

## 7. Policies (`src/hernness/policies.py`)

```python
@dataclass(frozen=True)
class LoopPolicy:
    max_steps: int = 12
    max_total_tokens: int = 50_000
    max_runtime_seconds: float = 120.0
    repeated_action_limit: int = 3
    no_progress_window: int = 10
    consecutive_error_limit: int = 3
    checkpoint_every_step: bool = False
    provider_timeout_seconds: float | None = None
    tool_timeout_seconds: float | None = None
```

`runtime_reason(elapsed)` returns a `StopReason` if `elapsed >=
max_runtime_seconds`. `token_budget_reason(usage, accounting_available)`
returns `TOKEN_BUDGET_EXCEEDED` if the sum is over `max_total_tokens` and
accounting is available; otherwise `None`.

---

## 8. Progress detection (`src/hernness/progress.py`)

`ProgressDetector` records three histories:

- `action_history` — canonical JSON of `{"name", "arguments"}` (no
  tool_call_id).
- `observation_history` — canonical JSON of `ToolResult` fingerprints.
- `model_history` — canonical JSON of assistant message fingerprints.

`repeated_action(limit)` returns `True` when the latest action matches the
immediately preceding `limit` actions. `no_progress(window)` returns
`True` when the last `window` observations are identical and non-empty.

---

## 9. Tools (`src/hernness/tools.py`)

`ToolRegistry` rejects duplicate names (`DuplicateToolError`) and unknown
tools (`ToolNotFoundError`). `FunctionTool` wraps a `BaseModel`
arguments schema and an `async` callable:

```python
FunctionTool(
    name="add",
    description="Add two integers.",
    arguments_model=AddArguments,
    function=add,
)
```

`invoke(call, completed_tool_call_ids=...)` raises
`ToolAlreadyCompletedError` if the id is already in the set. Successful
calls return `ToolResult(success=True, output=...)`. Exceptions are
wrapped as `ToolResult(success=False, error="...")` so the runtime can
classify via `StopReason.TOOL_ERROR`.

`tool_call_fingerprint(call)` returns the canonical JSON used by the
detector and the `TOOL_REQUESTED` event.

---

## 10. Event stores (`src/hernness/storage/`)

`EventStore` is a Protocol with:

```python
async def create_run(run, event) -> AgentEvent
async def append_event(event) -> AgentEvent
async def append_event_and_update_run(event, run) -> AgentEvent
async def save_checkpoint(checkpoint, event) -> tuple[Checkpoint, AgentEvent]
async def finalize_run(run, state_event, terminal_event) -> tuple[AgentEvent, AgentEvent]
async def get_run(run_id) -> RunRecord
async def get_events(run_id) -> list[AgentEvent]
async def get_latest_checkpoint(run_id) -> Checkpoint | None
async def close() -> None
```

### 10.1 `InMemoryEventStore`

`StorageBase`-style implementation backed by dicts. Holds `runs`, `events`,
and `checkpoints`. Operations are atomic under a single `asyncio.Lock`.

### 10.2 `SQLiteEventStore`

Single-connection SQLite. The schema:

```sql
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    state TEXT NOT NULL,
    record_json TEXT NOT NULL
);

CREATE TABLE events (
    run_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    event_id TEXT NOT NULL UNIQUE,
    event_type TEXT NOT NULL,
    created_at TEXT NOT NULL,
    event_json TEXT NOT NULL,
    PRIMARY KEY (run_id, sequence),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    event_sequence INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    checkpoint_json TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX idx_events_run_sequence ON events(run_id, sequence);
CREATE INDEX idx_checkpoints_run_sequence ON checkpoints(run_id, event_sequence DESC);
```

Every mutation runs inside `BEGIN IMMEDIATE`/`COMMIT`/`ROLLBACK` under an
`asyncio.Lock`. `finalize_run` checks that:

- The current run is not already terminal.
- `state_event.from_state` matches the stored state.
- `state_event.to_state` matches the new run metadata's state.

The CLI rejects resume for generic application tool callables because they
cannot be reconstructed from JSON; persisted `FakeProvider` runs without a
pending application tool are recoverable.

---

## 11. Tracing (`src/hernness/tracing.py`)

`TraceInspector` walks a stored run's events and emits one `TraceEntry`
per row:

```python
class TraceEntry:
    sequence: int
    event_type: EventType
    created_at: datetime
    summary: str
    detail: dict[str, Any]
```

`RunTrace.to_text()` returns a deterministic, multi-line plain-text
view suitable for `runs inspect`. The CLI prints it after the summary.

---

## 12. Provider adapters (`src/hernness/providers/`)

### 12.1 Protocol

```python
@runtime_checkable
class ModelProvider(Protocol):
    async def generate(self, request: ModelRequest) -> ModelResponse: ...

@runtime_checkable
class StatefulModelProvider(Protocol):
    def snapshot_state(self) -> dict[str, JsonValue]: ...
    def restore_state(self, state: dict[str, JsonValue]) -> None: ...
```

### 12.2 `FakeProvider` (`providers/fake.py`)

Deterministic replay of `ModelResponse` instances, dicts, or
`Exception` instances. Supports `repeat_last=True` (replay the last script
item forever) and a `cursor` reset by `reset()`. Implements
`StatefulModelProvider` so checkpoints survive across resume.

### 12.3 `OllamaProvider` (`providers/ollama.py`)

Native `/api/chat` endpoint (default `http://localhost:11434`). Tool
schema follows the OpenAI-compat shape (`type=function` blocks). Usage
fields `prompt_eval_count` and `eval_count` are returned only when the
response is finished; when absent, `usage` is left as `None`. The
optional `HERNNESS_OLLAMA_API_KEY` adds a Bearer header for setups that
front the server with a reverse proxy.

### 12.4 `MiniMaxProvider` (`providers/minimax.py`)

Supports both `openai` and `anthropic` API styles. Two env conventions:

- **Live-benchmark (legacy)**: `MODEL_MINIMAX`, `BASE_URL`,
  `MINIMAX_API_STYLE`, and a style-dependent `OPENAI_AUTH_TOKEN` (openai)
  or `AUTH_TOKEN` (anthropic). Read via `MiniMaxConfig.from_env`.
- **`HERNNESS_` (official)**: `HERNNESS_MINIMAX_API_KEY` (required),
  `HERNNESS_MINIMAX_BASE_URL` (default `https://api.minimax.io`),
  `HERNNESS_MINIMAX_MODEL` (fallback to `HERNNESS_MODEL`),
  `HERNNESS_MINIMAX_API_STYLE` (default `anthropic`). Read via
  `MiniMaxConfig.from_soteria_env`.

The private token attribute `_soteria_api_key` takes precedence over the
legacy `_auth_token` / `_openai_auth_token` when present.

### 12.5 `AnthropicProvider` (`providers/anthropic.py`)

`POST /v1/messages` with `x-api-key` and `anthropic-version:
2023-06-01` headers. System messages are stripped from `messages` and
joined into a top-level `system` field. `tool_use` blocks become
`ToolCall`. Missing `usage` returns `usage=None`.

### 12.6 `OpenAICompatibleProvider` (`providers/openai_compatible.py`)

`POST <base_url>/chat/completions` with Bearer auth. Default base URL
`https://api.openai.com/v1`. Override `HERNNESS_OPENAI_BASE_URL` for
self-hosted gateways (vLLM, llama.cpp server, Together, ...). Parses via
the shared `parse_openai_response` helper.

### 12.7 `OpenAIProvider` (`providers/openai.py`)

Older module kept for the live-benchmark CLI. Reads `OPENAI_MODEL`,
`OPENAI_API_KEY`, optional `OPENAI_BASE_URL`. Identical behavior to
`OpenAICompatibleProvider`; both exist so reproduction commands in the
live benchmark stay verbatim.

### 12.8 Shared HTTP helpers (`providers/http_common.py`)

- `build_openai_payload(model, request, max_completion_tokens)`: produces
  the OpenAI-compatible JSON body, including JSON-stringified function
  arguments.
- `parse_openai_response(payload)`: returns `ModelResponse` with
  `usage=None` when the API omits it.
- `redact_text(value)`: removes `Authorization` / `x-api-key` /
  `sk-…` patterns from error messages so logs never carry secrets.
- `json_safe_content(value)`: deterministic JSON encoding for tool
  results that may contain non-string values.

---

## 13. Config loader (`src/hernness/config.py`)

```python
build_provider_from_env(environ=None, *, max_completion_tokens=1024,
                        request_timeout_seconds=30.0) -> Any
```

Reads `HERNNESS_PROVIDER` (`ollama | minimax | anthropic | openai`) and
`HERNNESS_MODEL`, then delegates to the matching
`<Provider>Config.from_soteria_env`. Raises `ConfigError(ValueError)`
with one actionable message per missing required variable.

| Variable | Required | Default |
|---|:---:|---|
| `HERNNESS_PROVIDER` | yes | – |
| `HERNNESS_MODEL` | yes | – |
| `HERNNESS_<PROVIDER>_API_KEY` | per provider | – |
| `HERNNESS_<PROVIDER>_BASE_URL` | no | provider-specific |
| `HERNNESS_<PROVIDER>_MODEL` | no | falls back to `HERNNESS_MODEL` |
| `HERNNESS_DATABASE_PATH` | no | in-memory |
| `HERNNESS_MAX_TOTAL_TOKENS` | no | `50000` |
| `HERNNESS_MAX_RUNTIME_SECONDS` | no | `120.0` |
| `HERNNESS_REPEATED_ACTION_LIMIT` | no | `3` |

`apply_runtime_overrides(policy_kwargs, environ)` populates
`max_total_tokens`, `max_runtime_seconds`, and `repeated_action_limit`
from `HERNNESS_MAX_TOTAL_TOKENS`, `HERNNESS_MAX_RUNTIME_SECONDS`, and
`HERNNESS_REPEATED_ACTION_LIMIT`. Garbage raises immediately.

`database_path_from_env(environ)` returns the `HERNNESS_DATABASE_PATH`
value (stripped) or `None` for in-memory storage.

This is the single entry point the CLI uses to build its provider. The
contract — never read provider config in two places — keeps the
runtime, the live benchmark, and the upcoming `hernness chat`
subcommand all pointing at the same auth surface.

---

## 13.b Application tools (`src/hernness/app_tools/`)

The `app_tools/` package is the safe-by-construction toolbox that the
runtime hands to the model. It plugs into the existing
`FunctionTool` / `ToolRegistry` / `approval_callback` contracts — no
changes to `AgentRuntime` or the state machine.

### 13.b.1 `Workspace` and `validate_path` (`app_tools/workspace.py`)

```python
workspace = Workspace(root, create=False)            # rejects missing root
resolved = workspace.validate_path("sub/file.txt")  # must_exist=True default
write_target = workspace.validate_for_write("new")   # refuses symlinks in chain
```

Resolution happens with `Path.resolve(strict=False)` so lexical `..`
segments and existing-link symlinks normalize **before** the
containment check. The containment check fires before any existence
test, so `../file.txt` is rejected as an escape even when the leaf is
missing on disk.

Rejected inputs:

- Null bytes (`"\x00"` anywhere in the path).
- Empty strings.
- Absolute paths outside the root.
- `..` traversal (any depth).
- Symlinks whose target — or any ancestor — is outside the root
  (for reads) or anywhere in the chain (for writes — `validate_for_write`
  refuses to follow a symlink even when the target is inside).
- Brand-new files whose parent directory does not exist.

The TOCTOU window between `validate_path` and the actual `open()` is
not closed by this module; `file_tools.py` uses `O_NOFOLLOW` on POSIX.

### 13.b.2 Approval policy (`app_tools/approval.py`)

```python
from hernness.app_tools.approval import build_approval_callback

callback = build_approval_callback()   # reads HERNNESS_TOOLS_REQUIRE_APPROVAL
runtime = AgentRuntime(..., approval_callback=callback)
```

Reads `HERNNESS_TOOLS_REQUIRE_APPROVAL` (comma- or whitespace-separated
list of tool names). Tools in the list → `False` (the runtime stops
with `StopReason.POLICY_DENIED`). Tools not in the list → `True`
(auto-approve, no callback invoked). Optional `on_require` hook lets
operators escalate to a real interactive prompt before the deny
decision.

### 13.b.3 File tools (`app_tools/file_tools.py`)

```python
from hernness.app_tools.file_tools import (
    bind_workspace, read_file_tool, write_file_tool,
)

workspace = Workspace(root)
with bind_workspace(workspace):
    runtime = AgentRuntime(..., tools=[read_file_tool(), write_file_tool()])
    await runtime.run("Edit src/foo.py to fix the bug.")
```

`read_file_tool()` reads inside the workspace through `validate_path`;
`write_file_tool()` writes inside the workspace through
`validate_for_write` (refuses leaf symlinks) and opens with
`O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW` when available. If a file
tool is invoked outside a `bind_workspace` block, the call raises
`WorkspaceNotBoundError`, which the registry surfaces as
`ToolExecutionError`.

---

## 14. Lethe integration (`src/hernness/integrations/lethe.py`)

```python
class MemoryProvider(Protocol):
    def recall(self, query: str, k: int = 5) -> Sequence[str]: ...
    def remember(self, text: str, *, session_id: str | None = None) -> None: ...
```

`LetheMemoryAdapter` adapts a `lethe.MemoryStore`:

- Before the first `MODEL_RESPONDED` event, `recall_text(task)` injects
  bounded memories as a system message.
- On `StopReason.COMPLETED`, the final assistant content is persisted via
  `remember_output(content, session_id=run_id)`.

If `memory=None`, the runtime runs with no recall and no answer
persistence. The shipped `tests/test_lethe_integration.py` covers both
shapes with a `MemoryProvider` fake.

---

## 15. CLI (`src/hernness/cli.py`)

```bash
hernness --database hernness.db runs list
hernness --database hernness.db runs inspect RUN_ID
hernness --database hernness.db runs resume RUN_ID

# Interactive REPL — one AgentRuntime turn per line
hernness chat --workspace-root $(pwd)

# Verify HERNNESS_ provider config without HTTP
hernness doctor
```

The CLI uses only the Python standard library (`argparse`, `sqlite3`,
`asyncio`). It reads the persisted `record_json` for each run, prints a
one-line summary per row for `list`, prints `trace.to_text()` for
`inspect`, and tries to recreate a `FakeProvider` from a serialized
cursor for `resume` (only for runs without a pending application tool).

`chat` drives `AgentRuntime` interactively. On startup, when no
`HERNNESS_PROVIDER` is set, it launches an interactive first-run wizard
(numbered menu, hidden API-key prompt via `getpass`) and, on consent,
persists the resulting `HERNNESS_*` exports to `~/.zshrc` or `~/.bashrc`
(delimited by `# >>> soteria setup >>>` markers, `chmod 0o600` on POSIX).
Slash commands inside the REPL: `/provider`, `/inspect RUN_ID`,
`/resume RUN_ID`, `/quit`.

`doctor` reads `os.environ` (or an injected mapping) and prints which
`HERNNESS_*` variables are present, which are missing, the resolved
endpoint URL, and whether `ConfigError` would fire at provider
construction time. It is purely synchronous and dependency-free, so it
works on any machine where `hernness` is installed, even without
network access.

### 15.1 `doctor.py` (`src/hernness/doctor.py`)

Public surface:

- `run_doctor(environ=None) -> DoctorReport` — pure, never raises;
  missing required vars and `ConfigError` both surface as report fields.
- `render_report(report, *, out)` — print a human-readable summary that
  shows provider/model/endpoint/api-style but **never the API key**.
- `main(argv=None)` — argparse wrapper for the `hernness doctor`
  entry point. Exits non-zero when the report is not OK so the command
  is usable in CI.

`DoctorReport.ok` is `True` only when every required variable for the
resolved provider is set **and** `build_provider_from_env(env)` does
not raise. The endpoint string is produced by the same `endpoint`
property each provider uses at runtime, so what `doctor` prints matches
what `chat` would actually hit.

---

## 16. Exception hierarchy (`src/hernness/exceptions.py`)

```
SoteriaError
├── InvalidStateTransitionError
├── RunNotFoundError
├── RunAlreadyTerminalError
├── CheckpointNotFoundError
├── UnsafeResumeError
├── DuplicateToolError
├── ToolNotFoundError
├── ToolValidationError
├── ToolExecutionError
├── ToolAlreadyCompletedError
├── StorageError
│   └── EventInvariantError
├── ProviderError(retryable=True)
│   └── FakeProviderExhaustedError(retryable=False)
```

`ProviderError.retryable` is the signal the runtime uses to decide
between `CONSECUTIVE_ERRORS` (retryable, count toward limit) and
`PROVIDER_ERROR` (non-retryable, terminate immediately).

---

## 17. Live benchmark (`benchmark/live/`)

The live benchmark is opt-in and **spends real money**. It exists to
exercise Hernness's policy machinery against a real model, not as a
performance claim.

### 17.1 Scenarios (`benchmark/live/scenarios.py`)

Three scenarios, each with metadata about whether raw / Hernness / resume
are supported:

| Scenario | Raw | Hernness | Resume |
|---|:---:|:---:|:---:|
| `normal_completion` | ✓ | ✓ | – |
| `repetition_prone` | ✓ | ✓ | – |
| `interrupted_resume` | – | ✓ | ✓ |

### 17.2 Cost-consent gate (`benchmark/live/consent.py`)

Two equivalent channels grant consent:

- CLI flag `--i-understand-this-costs-money`.
- Env var `HERNNESS_I_UNDERSTAND_THIS_COSTS_MONEY` set to `1`, `true`, or
  `yes` (case-insensitive).

Missing both → exit code `2`, no provider module imported, no HTTP call.

### 17.3 Pricing (`benchmark/live/pricing.py`)

- **MiniMax**: baked-in `0.30 / M` input, `1.20 / M` output. Source URL
  referenced as `MINIMAX_PRICING_SOURCE_URL`.
- **OpenAI**: operator-supplied only. Both `OPENAI_INPUT_USD_PER_MILLION`
  and `OPENAI_OUTPUT_USD_PER_MILLION` are required.

`estimate_upper_bound()` deliberately **over**-estimates by assuming both
raw and Hernness approaches run for every (scenario, run_index) pair at
the configured token caps. The CLI prints it as
`"~$X.XXXX USD across n=Y run(s) (Z steps total) — upper-bound
estimate, not a bill"`.

### 17.4 Runners

- `run_raw_loop(provider, scenario, manual_step_cap, max_completion_tokens)`
  — minimal `while True` loop, no policies, capped externally.
- `run_hernness(provider, scenario, run_index)` — full runtime with
  repeated-action and token-budget policies.
- `run_hernness_interrupted(provider_factory, scenario, run_index_inner)` —
  splits a run across two SQLite sessions to exercise the resume path.

### 17.5 Renderer (`benchmark/live/render.py`)

Produces two PNGs deterministically from a `LiveResults` JSON:

- `repetition_containment.png` — grouped bars per approach.
- `normal_completion_comparison.png` — side-by-side mean steps and
  wall-clock.

The renderer reads only `LiveResults` records; titles are derived from
the bundle so the same renderer can chart any provider's JSON output.

### 17.6 Checked-in artifacts (`benchmark/live/example_output/`)

- `example_results.json` — real MiniMax M3 anthropic-style run, n=3 per
  applicable (scenario, approach), 15 records total, all reporting
  `token_accounting_available=True`.
- `repetition_containment.png` and `normal_completion_comparison.png` —
  rendered from the JSON above.

`tests/test_example_output.py` guards the JSON's shape and asserts the
two PNGs exist with non-trivial size and a real PNG signature.

---

## 18. Deterministic benchmark (`benchmark/run_benchmark.py`)

Eight scenarios compare a minimal raw loop with Hernness using
`FakeProvider`. Latest run (Linux, local):

| Metric | Raw loop | Hernness |
|---|---:|---:|
| Loop containment rate | 0.0% | 100.0% |
| Resume success rate | 0.0% | 100.0% |
| Duplicate side-effect count | 6 | 0 |
| Terminal completeness | 0.0% | 100.0% |
| Mean steps | 5.00 | 1.88 |

These measure **runtime behavior**, not model intelligence. An external
six-step harness stops runaway raw-loop scenarios and is not counted as
containment. Wall-clock timings vary by machine. See
`benchmark/RESULTS.md` for the complete results and methodology.

---

## 19. Tests (offline, no network)

`tests/` directory, 163 cases, runs in ~0.6 s without any API key:

| File | Coverage |
|---|---|
| `test_models.py` | All Pydantic invariants, including terminal/non-terminal rules and `ToolResult` mutual exclusion. |
| `test_state.py` | Every transition edge and the `STOP_REASONS_BY_STATE` table. |
| `test_event_invariants.py` | Sequence monotonicity, `STATE_CHANGED` matching, `TOOL_COMPLETED` linkage. |
| `test_fake_provider.py` | Replay, cursor, snapshot round-trip, error injection. |
| `test_policies.py` | `LoopPolicy` defaults, runtime reason, token-budget reason. |
| `test_progress_detection.py` | Repeated-action and no-progress fingerprinting. |
| `test_tools.py` | Registry, duplicate detection, `ToolAlreadyCompletedError`. |
| `test_runtime.py` | End-to-end behavior across every state handler. |
| `test_resume.py` | `test_inter中断_after_tool_result_resumes_without_duplicate_side_effect`, checkpoint round-trip, `UnsafeResumeError` on missing durable result. |
| `test_sqlite_parity.py` | `InMemoryEventStore` and `SQLiteEventStore` produce identical event sequences for the same scripted run. |
| `test_tracing.py` | `RunTrace.to_text()` ordering and content. |
| `test_lethe_integration.py` | Recall injection, answer persistence, no-memory default. |
| `test_cli.py` | argparse, SQLite path handling, trace printing. |
| `test_config.py` | `ConfigError` on missing required vars; per-provider HERNNESS_ wiring; `apply_runtime_overrides` integer/float parsing; `database_path_from_env` defaults. |
| `test_providers_http.py` | httpx-shaped fake client covering normal text, tool calls, HTTP 5xx, transport errors, missing usage, and `aclose` ownership. |

`benchmark/live/tests/` adds offline coverage for the opt-in CLI:
`test_consent`, `test_cli`, `test_example_output`, `test_minimax_provider`,
`test_openai_provider`, `test_pricing`, `test_provider_config`,
`test_provider_conversion`, `test_raw_loop`, `test_render`,
`test_scaffold`, `test_scenarios`, `test_soteria_run`.

---

## 20. Configuration reference

| Variable | Required | Default | Notes |
|---|:---:|---|---|
| `HERNNESS_PROVIDER` | yes | – | `ollama` \| `minimax` \| `anthropic` \| `openai` |
| `HERNNESS_MODEL` | yes | – | Default model name |
| `HERNNESS_OLLAMA_BASE_URL` | no | `http://localhost:11434` | Native `/api/chat` |
| `HERNNESS_OLLAMA_MODEL` | no | `HERNNESS_MODEL` | Override |
| `HERNNESS_OLLAMA_API_KEY` | no | empty | Bearer header for reverse-proxy setups |
| `HERNNESS_MINIMAX_API_KEY` | yes (minimax) | – | One key for both API styles |
| `HERNNESS_MINIMAX_BASE_URL` | no | `https://api.minimax.io` | Bare host; suffix added per style |
| `HERNNESS_MINIMAX_MODEL` | no | `HERNNESS_MODEL` | Override |
| `HERNNESS_MINIMAX_API_STYLE` | no | `anthropic` | `anthropic` \| `openai` |
| `HERNNESS_ANTHROPIC_API_KEY` | yes (anthropic) | – | |
| `HERNNESS_ANTHROPIC_MODEL` | no | `claude-sonnet-4-6` | |
| `HERNNESS_ANTHROPIC_BASE_URL` | no | `https://api.anthropic.com` | |
| `HERNNESS_OPENAI_API_KEY` | yes (openai) | – | |
| `HERNNESS_OPENAI_MODEL` | no | `HERNNESS_MODEL` | |
| `HERNNESS_OPENAI_BASE_URL` | no | `https://api.openai.com/v1` | Override for self-hosted gateways |
| `HERNNESS_DATABASE_PATH` | no | in-memory | Empty = `InMemoryEventStore` |
| `HERNNESS_MAX_TOTAL_TOKENS` | no | `50000` | `LoopPolicy.max_total_tokens` |
| `HERNNESS_MAX_RUNTIME_SECONDS` | no | `120.0` | `LoopPolicy.max_runtime_seconds` |
| `HERNNESS_REPEATED_ACTION_LIMIT` | no | `3` | `LoopPolicy.repeated_action_limit` |
| `HERNNESS_I_UNDERSTAND_THIS_COSTS_MONEY` | for live bench | – | `1` \| `true` \| `yes` (case-insensitive) |
| `MODEL_MINIMAX` / `BASE_URL` / `AUTH_TOKEN` / `OPENAI_AUTH_TOKEN` | live bench (legacy) | – | Kept verbatim for reproduction |
| `OPENAI_MODEL` / `OPENAI_API_KEY` / `OPENAI_BASE_URL` | live bench (legacy) | – | Kept verbatim for reproduction |
| `OPENAI_INPUT_USD_PER_MILLION` / `OPENAI_OUTPUT_USD_PER_MILLION` | live bench (openai) | – | Required for OpenAI cost estimate |

`.env.example` lists every variable with empty placeholders. `.env` stays
in `.gitignore`.

---

## 21. Performance characteristics

Local microbenchmarks (single-thread, no provider calls):

| Workload | Latency |
|---|---:|
| 100 trivial in-memory runs (no tool) | ~2 ms / run |
| 20 SQLite-backed runs × 5 tool calls + `checkpoint_every_step=True` | ~7 ms / run |

Per-step overhead is dominated by Pydantic validation, event-store
serialization, and JSON dumping for checkpoint snapshots. The runtime is
single-threaded per `AgentRuntime` instance; multiple runtime instances can
operate independently.

Live-benchmark case study (MiniMax M3, anthropic style, n=3 per
applicable scenario):

| Quantity | Value |
| --- | ---:|
| Input tokens (all 15 records) | 7,327 |
| Output tokens (all 15 records) | 1,830 |
| Cost at $0.30 / M input, $1.20 / M output | $0.0044 USD |
| Pre-flight upper-bound estimate | $0.3318 USD |
| Records with `token_accounting_available=False` | 0 / 15 |

The actual spend landed ~75× below the upper-bound estimate because model
responses were short and `--input-tokens-per-step 2048` was conservative.

---

## 22. Quality gates (run before any commit / push)

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy src/hernness
pytest
python -m build
```

Current state (offline):

| Gate | Result |
|---|---|
| `ruff check src/hernness tests` | All checks passed |
| `ruff format --check src/hernness tests` | 43 files already formatted |
| `mypy src/hernness` (strict) | 27 files, no issues |
| `pytest` | 163 passed in ~0.6 s |
| `python -m build` | `hernness-0.1.0-py3-none-any.whl` + `.tar.gz` |

---

## 23. Known limitations (deferred from 0.1)

- Repetition and no-progress detection are exact deterministic heuristics,
  not semantic loop detection.
- Runtime limits are checked between operations; Hernness does **not**
  preempt a tool already in flight.
- Missing provider usage leaves `token_accounting_available=False`; Hernness
  never fabricates zeros.
- SQLite v0.1 assumes normal single-process use. No distributed lease or
  multi-process scheduler.
- Tool calls execute serially. Parallel calls, MCP, OpenTelemetry, approval
  UIs, and replay are deferred.
- A `TOOL_STARTED` event without a durable result is intentionally treated
  as **unsafe to resume automatically**.
- The event schema has no migration system yet.

---

## 24. Distribution and packaging

`pyproject.toml`:

- Distribution name: `hernness` (PyPI-friendly hyphen).
- Python package: `hernness` (hyphens are illegal in module names).
- Version: `0.1.0`.
- Console script: `hernness = hernness.cli:main`.
- Core dependency: `pydantic >= 2.8, < 3`.
- Optional `[live-benchmark,providers]` extras pull `httpx >= 0.27`.
- Optional `[live-benchmark]` extra additionally pulls `matplotlib >= 3.8`.
- Hatchling build backend, wheel packages `src/hernness`.
- mypy strict with `pydantic.mypy` plugin.
- Coverage gate `fail_under = 90` on `hernness`.

---

## 25. Glossary

| Term | Meaning |
|---|---|
| **Run** | One execution of an agent against a single task. Identified by `run_id` (UUID4). |
| **Event** | An immutable entry in the append-only log. Every state transition, model call, tool call, and policy trigger is one event. |
| **Checkpoint** | A snapshot of the runtime context (messages, detector history, completed tool-call IDs, user state, policy, provider metadata). The latest checkpoint is sufficient to continue a non-terminal run safely. |
| **StopReason** | A 13-value enum describing why a run terminated. Always set on terminal `RunRecord`. |
| **Fingerprint** | A canonical-JSON hash of the tool name + arguments (no tool_call_id) for repeated-action detection; or of the tool result for no-progress detection. |
| **Token accounting** | Whether `ModelResponse.usage` was present. Hernness never invents zeros; `token_accounting_available=False` propagates to `RunResult`. |
| **Containment** | A run that stopped due to a Hernness policy (e.g. `REPEATED_ACTION`). Distinct from "external safety cap" (a non-runtime fence). |
| **Idempotency key** | The tool-call fingerprint at the time of request. Used to skip already-completed tool-call IDs on resume. |

---

## 26. Roadmap — terminal-based, model-agnostic CLI assistant

The mid-term goal is a **terminal-based AI assistant** that the
operator can point at any model the runtime supports. The pieces are
already in place; the roadmap below lists the remaining work in the
order that keeps each slice shippable and testable.

### 26.1 Why this direction

- Operator's choice of model is one env var (`HERNNESS_PROVIDER`), not
  a code change. The assistant binary stays identical whether the
  backend is Ollama on a developer laptop, Anthropic in a CI runner, or
  MiniMax behind a vendor proxy.
- The state machine, event log, and `app_tools/` already give the
  assistant safe-by-construction file I/O and resumable sessions. The
  CLI assistant is mostly an interactive shell on top of that
  machinery.
- Reusing the SQLite event store means a long chat session is a `run_id`
  the operator can inspect, replay, or hand off to another tool.

### 26.2 Slice plan

| Slice | Scope | Status |
|---|---|---|
| **0.1** (released) | Provider-agnostic runtime, deterministic benchmark, live MiniMax M3 case study, 13-stop-reason state machine. | done |
| **0.2** (released) | `HERNNESS_` env factory + Ollama / Anthropic / OpenAI-compatible adapters; `app_tools/` workspace + approval + read/write; project.md, CHANGELOG, Makefile, .github templates. | done |
| **0.3** | `sandbox.py` + `shell_tool.py` using ephemeral Docker containers (`network_mode="none"`, `mem_limit`, `remove=True`). `HERNNESS_SANDBOX_*` env for image / cpu / mem overrides. | next |
| **0.4** (released) | `hernness chat` REPL: reads `HERNNESS_*` env, builds the runtime + `app_tools/`, accepts user input, prints streaming tool calls and final answers, persists every turn to SQLite. Interactive first-run wizard with hidden API-key input and opt-in shell-rc persistence (`~/.zshrc` / `~/.bashrc`, `chmod 0o600`). | done |
| **0.4.1** (released) | `hernness doctor` subcommand: verify `HERNNESS_*` env and resolved endpoint without sending an HTTP call. Exits non-zero when configuration is incomplete, so it doubles as a CI gate. | done |
| **0.5** | `hernness chat --resume RUN_ID` and `--resume-last` to pick up an interrupted session. The SQLite event store makes this almost free. | planned |
| **0.6** | MCP adapter (`mcp_tool.py`): bridge between the runtime and any MCP server. Operator drops MCP server URLs into `HERNNESS_MCP_SERVERS`. | planned |
| **0.7** | Multi-process scheduler for the SQLite store: a `LEASE_ID` column plus a background reaper so two operators can share a workspace without duplicate side effects. | deferred |

### 26.3 `hernness chat` — target shape

```bash
# 0.4 prototype: pick provider from env, open a chat run, persist every turn.
HERNNESS_PROVIDER=ollama HERNNESS_MODEL=llama3.1 \
HERNNESS_DATABASE_PATH=~/.soteria/chat.db \
HERNNESS_WORKSPACE_ROOT=$(pwd) \
hernness chat
```

```
> Add a docstring to src/hernness/foo.py.
   ⟶ tool_call read_file(path="src/hernness/foo.py")
   ⟶ tool_call write_file(path="src/hernness/foo.py", content="...")
   ⟶ stop_reason=completed
> /inspect
   run_id=01HK...  steps=3  duration=2.1s  tokens=482  status=COMPLETED
> /resume 01HK...
> /quit
```

The REPL accepts slash-commands (`/inspect`, `/resume`, `/quit`,
`/provider`) and plain prose. Every turn is one `AgentRuntime.run(...)`
invocation so the operator can always answer the six questions for any
previous turn with `runs inspect RUN_ID`.

### 26.4 Decisions already locked in

- **Provider neutrality stays at the runtime layer.** The CLI assistant
  is a thin shell; it never branches on `HERNNESS_PROVIDER`. New
  providers join the same `ModelProvider` Protocol without code in
  `chat.py`.
- **Tools are bound at runtime construction, not per turn.** The
  `bind_workspace` context manager ensures the workspace is set up
  before the first model call and torn down after the last.
- **Approval is a synchronous boolean callback.** The CLI assistant
  prints `tool_call` details to the terminal and asks the operator to
  approve / deny. `HERNNESS_TOOLS_REQUIRE_APPROVAL` sets which tools
  always ask; everything else auto-approves (default) or runs through
  a registered interactive handler.
- **The event store is the source of truth.** The REPL never holds
  per-turn state that isn't in SQLite, so `--resume` is a different
  way to attach to the same `run_id` rather than a separate code
  path.

### 26.5 What this is not

- Not a hosted product. The CLI runs entirely on the operator's
  machine; provider keys stay in the operator's env.
- Not a replacement for project-local CL tools. The assistant can use
  them (via `run_shell` through the sandbox), but the local CL stays
  the authoritative tool for project-shaped operations.
- Not a multi-agent orchestrator. Each chat turn is a single run; a
  future iteration may layer an orchestrator on top, but that's out
  of scope for the 0.4 prototype.