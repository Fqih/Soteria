# Soteria

> **Reliable execution infrastructure for long-running AI agents.**
>
> *Bounded. Observable. Resumable. Honest about why it stopped.*

Soteria is a provider-agnostic Python runtime that wraps your tool-using agent loop in a strict state machine, an append-only event history, and a configurable set of safety policies. It answers the six questions every agent operator eventually asks:

1. What is the agent doing *right now*?
2. Why did this run stop?
3. Did it repeat itself without making progress?
4. Can an interrupted run continue safely?
5. Which tool calls actually executed?
6. Can the run be reproduced without calling a paid model again?

> ⚠️ Soteria 0.1 is an **alpha foundation**. It is suitable for evaluation, deterministic testing, and local prototypes; it is **not production-ready**.

---

## Why does this exist?

A minimal agent loop is short:

```python
while True:
    response = model.generate(messages, tools)
    if response.is_final:
        return response
    result = execute_tool(response.tool_call)
    messages.append(result)
```

It works for one happy path. It breaks in five painful ways once the loop runs for more than a handful of steps:

| Pain | What goes wrong | What Soteria does |
|---|---|---|
| **Repeated tool calls** | Model asks `get_weather("Tokyo")` five times. | `repeated_action_limit=3` stops the run before the third duplicate, citing `StopReason.REPEATED_ACTION`. |
| **Runaway token usage** | Loop spins into oblivion and the bill surprises you. | `max_total_tokens` + `max_runtime_seconds` enforce a hard upper bound. |
| **Process restart loses state** | You restart, the model re-asks, the external tool fires twice. | `SQLiteEventStore` + `resume(run_id)` re-uses completed tool-call IDs so duplicates are impossible. |
| **No audit trail** | "What did the agent actually do?" is unanswerable. | Every state transition, tool call, and policy trigger is an immutable event. |
| **No explicit stop reason** | "Why did it stop?" is a guess. | Every terminal run records one `StopReason` from a 13-value enum. |

A real-world incident timeline that motivated Soteria:

```mermaid
flowchart LR
    A[Tool call sent] --> B[Process killed<br/>mid-flight]
    B --> C[Operator restarts<br/>& runs again]
    C --> D[Tool fires<br/>a SECOND time]
    D --> E[Billing double-charge<br/>+ customer impact]
```

Soteria turns that into:

```mermaid
flowchart LR
    A[Tool call sent] --> B[TOOL_COMPLETED event<br/>persisted to SQLite]
    B --> C[Process killed<br/>mid-flight]
    C --> D[Operator resumes<br/>via runtime.resume&#40;run_id&#41;]
    D --> E[Already-completed<br/>tool-call ID skipped]
    E --> F[Tool fires<br/>exactly ONCE]
```

The integrity tests prove this — see `tests/test_resume.py::test_interrupt_after_tool_result_resumes_without_duplicate_side_effect`.

---

## What you get

```mermaid
flowchart LR
    Task[User task] --> Runtime[AgentRuntime state machine]
    Runtime --> Provider[ModelProvider]
    Runtime --> Registry[ToolRegistry]
    Runtime --> Policy[LoopPolicy]
    Runtime --> Progress[ProgressDetector]
    Runtime --> Store[EventStore]
    Store --> Memory[In-memory]
    Store --> SQLite[SQLite]
    Store --> Trace[TraceInspector]
```

- An explicit, validated execution state machine (8 states, 4 terminal).
- An append-only, per-run event history with sequence invariants.
- Step, runtime, token, repetition, error, and no-progress policies.
- Configurable provider and tool operation timeouts.
- In-memory and durable SQLite event stores.
- Checkpoints and `resume(run_id)` with completed tool-call ID tracking.
- Deterministic fake-provider scripts so tests never need an API key.
- Chronological text and structured traces.
- One explicit `StopReason` per terminal run.

---

## Install

Python 3.11 or newer. For development from this repository:

```bash
python -m pip install -e ".[dev]"
```

For the optional live benchmark extras (`httpx`, `matplotlib`):

```bash
python -m pip install -e ".[live-benchmark]"
```

When the 0.1 package is published, the runtime-only installation will be:

```bash
python -m pip install soteria
```

### Optional Lethe context management

Soteria can use the companion Lethe memory store to retrieve a bounded set of
relevant memories before a model call and persist final answers after
completion. Lethe is intentionally not a core dependency:

```python
from lethe import MemoryStore
from soteria import AgentRuntime, LetheMemoryAdapter

memory = LetheMemoryAdapter(MemoryStore(), recall_k=5)
runtime = AgentRuntime(provider=provider, memory=memory)
```

Install or expose Lethe separately in the application environment. The
adapter uses Lethe's public `MemoryStore.recall` and `MemoryStore.remember`
methods.

The core runtime depends only on **Pydantic**. The CLI uses the Python standard library, so Typer and Rich are not runtime dependencies.

---

## Quickstart

This example makes one typed tool call and then completes without an API key:

```python
import asyncio
from pydantic import BaseModel
from soteria import AgentRuntime, FunctionTool, ModelResponse, ToolCall
from soteria.providers import FakeProvider


class AddArguments(BaseModel):
    left: int
    right: int


async def add(arguments: AddArguments) -> object:
    return {"sum": arguments.left + arguments.right}


async def main() -> None:
    runtime = AgentRuntime(
        provider=FakeProvider(
            [
                ModelResponse(
                    tool_call=ToolCall(
                        tool_call_id="add-1",
                        name="add",
                        arguments={"left": 2, "right": 3},
                    )
                ),
                ModelResponse(content="The sum is 5."),
            ]
        ),
        tools=[
            FunctionTool(
                name="add",
                description="Add two integers.",
                arguments_model=AddArguments,
                function=add,
            )
        ],
    )
    result = await runtime.run("Add 2 and 3.")
    trace = await runtime.inspect(result.run_id)
    print(result.status, result.stop_reason, result.output)
    print(trace.to_text())


asyncio.run(main())
```

The complete runnable version is [examples/basic_agent.py](examples/basic_agent.py).

---

## Deterministic benchmark

The included benchmark compares a minimal raw loop with Soteria across **eight scripted scenarios**. On the latest local run:

| Metric | Minimal raw loop | Soteria |
|---|---:|---:|
| Loop containment rate | 0.0% | 100.0% |
| Resume success rate | 0.0% | 100.0% |
| Duplicate side-effect count | 6 | 0 |
| Terminal completeness | 0.0% | 100.0% |
| Mean steps | 5.00 | 1.88 |

These fake-provider results measure **runtime behavior**, not model intelligence. An external six-step harness stops runaway raw-loop scenarios and is not counted as containment. Wall-clock timings vary by machine. See [benchmark/RESULTS.md](benchmark/RESULTS.md) for the complete results and methodology.

Regenerate them with:

```bash
python benchmark/run_benchmark.py
```

### What the benchmark proves

```mermaid
flowchart TB
    subgraph "Raw loop"
        R1[Tool call] --> R2{No policy}
        R2 -->|spins| R3[Duplicate side effects]
        R2 -->|runs out| R4[External cap stops it]
    end
    subgraph "Soteria"
        L1[Tool call] --> L2{Policy fingerprint check}
        L2 -->|new| L3[Execute]
        L2 -->|duplicate x3| L4[Stop: REPEATED_ACTION]
        L3 --> L5[Checkpoint + persist]
    end
    R3 -.is NOT.-> X[Runtime containment]
    L4 --> X
    X --> Y[100% Soteria containment, 0% raw]
```

---

## Live agent case study (MiniMax M3)

> **Small, non-reproducible, illustrative run against a real model — not a benchmark claim.**

The checked-in artifacts come from a **single real run** against `MiniMax-M3` (provider `minimax`, api_style `anthropic`, endpoint `https://api.minimax.io/anthropic/v1/messages`). The JSON source for the charts is [`benchmark/live/example_output/example_results.json`](benchmark/live/example_output/example_results.json); numbers below are derived from that file, not hand-entered.

### Why bother running this at all?

The deterministic benchmark uses `FakeProvider` — it measures the runtime, not the model. The live case study answers a complementary question: **does Soteria's policy machinery still fire when a real model is making real mistakes?** Three scenarios, three runs each, two approaches (raw vs. Soteria), one model. Snapshot, not statistic.

### Repetition containment (n=3 runs per approach)

![Repetition containment — minimax / MiniMax-M3 (n=3 runs per approach)](benchmark/live/example_output/repetition_containment.png)

| Approach | Contained runs (n=3) | Stop reason | Outcome |
|---|---:|---|---|
| **Raw loop** | 0/3 | manual cap (external fence, not Soteria containment) | tool fired multiple times until manual safety cap |
| **Soteria** | **3/3** | `REPEATED_ACTION` | policy stopped before the duplicate became a side effect |

### Normal completion comparison (n=3 runs per approach)

![Normal completion comparison — minimax / MiniMax-M3 (n=3 runs per approach)](benchmark/live/example_output/normal_completion_comparison.png)

| Approach | Mean steps (n=3) | Mean wall-clock (n=3) | Token accounting |
|---|---:|---:|---|
| Raw loop | 1.67 | 4.09 s | available |
| Soteria | 1.67 | 5.08 s | available |

### Cost vs. estimate

| Quantity | Value |
|---|---:|
| Pre-flight upper-bound estimate (CLI) | **$0.3318 USD** for 108 steps |
| Actual input tokens (all 15 records) | 7,327 |
| Actual output tokens (all 15 records) | 1,830 |
| Actual cost at MiniMax M3 standard rates ($0.30 / M input, $1.20 / M output) | **$0.0044 USD** |
| Records with `token_accounting_available=False` | **0 / 15** |

Real spend landed ~75× below the upper bound because model responses were short and `--input-tokens-per-step 2048` was conservative. The CLI explicitly labels the estimate as "upper-bound estimate, not a bill."

### Why this isn't a benchmark

- **n=3** is a snapshot, not statistical evidence.
- One model, one style, one timestamp.
- Real provider behaviour changes; the JSON in this repo is from the run captured at commit `0d8b984`.
- The raw loop's manual safety cap is **not** runtime containment.

A second, genuinely OpenAI API run with `--provider openai` is also supported. See [benchmark/live/README.md](benchmark/live/README.md) for provider-specific environment variables, explicit cost-consent flag, pricing requirements, and reproduction commands.

---

## Repeated-action containment

Tool fingerprints include the normalized tool name and canonical JSON arguments, but exclude the tool-call ID. With `repeated_action_limit=3`, the third consecutive identical request triggers `POLICY_TRIGGERED` and stops before that third invocation:

```python
policy = LoopPolicy(
    repeated_action_limit=3,
    no_progress_window=10,
)
```

Run [examples/repeated_action.py](examples/repeated_action.py) to see the full trace and side-effect count.

---

## Durable resume

Use `SQLiteEventStore` when a run must survive process restart:

```python
store = SQLiteEventStore("soteria.db")
runtime = AgentRuntime(
    provider=provider,
    tools=[tool],
    event_store=store,
)

result = await runtime.resume("existing-run-id")
await store.close()
```

If interruption occurs after a `TOOL_COMPLETED` event but before its next checkpoint, resume reconciles the event tail and does **not** execute that completed tool-call ID again. See [examples/resume_after_interrupt.py](examples/resume_after_interrupt.py).

---

## Architecture in one picture

```mermaid
flowchart LR
    Task[User task] --> Runtime[AgentRuntime state machine]
    Runtime --> Provider[ModelProvider]
    Runtime --> Registry[ToolRegistry]
    Runtime --> Policy[LoopPolicy]
    Runtime --> Progress[ProgressDetector]
    Runtime --> Store[EventStore]
    Store --> Memory[In-memory]
    Store --> SQLite[SQLite]
    Store --> Trace[TraceInspector]
```

The runtime dispatches one handler per state. State changes pass through a central validator and are persisted. SQLite transactions group run creation, state metadata updates, checkpoints, and terminalization with their associated events.

---

## Stop reasons

`StopReason` distinguishes successful completion, policy containment, caller cancellation, and operational failure:

- **Limits:** `MAX_STEPS`, `MAX_RUNTIME`, `TOKEN_BUDGET_EXCEEDED`
- **Heuristics:** `REPEATED_ACTION`, `NO_PROGRESS`
- **Errors / policy:** `CONSECUTIVE_ERRORS`, `POLICY_DENIED`, `PROVIDER_ERROR`, `TOOL_ERROR`, `INVALID_MODEL_RESPONSE`, `INTERNAL_ERROR`
- **Lifecycle:** `COMPLETED`, `USER_CANCELLED`

Exact enum values are lowercase when serialized.

---

## CLI

The CLI reads a SQLite database path:

```bash
soteria --database soteria.db runs list
soteria --database soteria.db runs inspect RUN_ID
soteria --database soteria.db runs resume RUN_ID
```

Generic provider and tool callables cannot be reconstructed from a database. CLI resume therefore supports persisted `FakeProvider` runs that do not have a pending application tool. Application runs should resume through Python with their provider and tool registry configured.

---

## Important limitations

- Repetition and no-progress detection are exact deterministic heuristics, not semantic loop detection.
- Runtime limits are checked between model and tool operations. Soteria does **not** preempt a tool already in flight.
- If any provider response omits usage, token accounting is marked unavailable; Soteria never treats missing usage as zero.
- SQLite v0.1 assumes normal single-process use. There is no distributed lease or multi-process scheduler.
- Tool calls execute serially. Parallel calls, real provider adapters, MCP, OpenTelemetry, approval UIs, and replay are deferred.
- A `TOOL_STARTED` event without a durable result is intentionally treated as **unsafe** to resume automatically because the external side effect is uncertain.
- The event schema has no migration system yet.

---

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy src/soteria
pytest
python -m build
```

Run the offline examples and benchmark:

```bash
python examples/basic_agent.py
python examples/repeated_action.py
python examples/resume_after_interrupt.py
python benchmark/run_benchmark.py
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [DESIGN.md](DESIGN.md) for workflow and architecture details.

---

## Project status

Version 0.1.0 is under active development. The state and event schemas should be treated as unstable until a compatibility and migration policy is published. Production provider adapters and multi-process safety are deliberately out of scope for this release.

---

## License

Soteria is available under the [MIT License](LICENSE).
