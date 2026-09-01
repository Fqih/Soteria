"""``avo bench`` — lightweight cross-provider benchmark harness.

MVP scope:

- Use :class:`avo.providers.fake.FakeProvider` as the deterministic
  baseline. The provider's scripted responses let the harness run
  reproducibly without any HTTP traffic or API keys.
- Capture per-turn metrics: latency, token usage (input + output),
  step count, stop reason.
- Emit a JSON report consumable by ``avo diff`` and external
  dashboard tooling.

Roadmap (not in this MVP):

- Live provider adapters (ollama, anthropic, openai, minimax).
- Task-matrix sweep (``--tasks-file``).
- Cost-normalized scoring.
- Regression detection against a stored baseline.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from avo.exceptions import AvoError
from avo.models import ModelResponse, TokenUsage
from avo.providers.base import ModelProvider
from avo.providers.fake import FakeProvider
from avo.runtime import AgentRuntime


class BenchError(AvoError):
    """User-facing failure from ``avo bench``."""


@dataclass
class TurnRecord:
    """One turn's measurement under :func:`run_benchmark`."""

    turn: int
    latency_ms: float
    input_tokens: int
    output_tokens: int
    steps: int
    stop_reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "steps": self.steps,
            "stop_reason": self.stop_reason,
        }


@dataclass
class BenchReport:
    """Aggregate benchmark report; serialises to JSON."""

    provider: str
    model: str
    turns: int
    task: str
    records: list[TurnRecord] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "turns_requested": self.turns,
            "task": self.task,
            "turns_completed": len(self.records),
            "summary": _summary(self.records),
            "turns": [record.as_dict() for record in self.records],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, sort_keys=True)


def _summary(records: Sequence[TurnRecord]) -> dict[str, Any]:
    if not records:
        return {"empty": True}
    latencies = [r.latency_ms for r in records]
    inputs = [r.input_tokens for r in records]
    outputs = [r.output_tokens for r in records]
    return {
        "latency_ms": {
            "min": min(latencies),
            "max": max(latencies),
            "mean": statistics.fmean(latencies),
            "stdev": statistics.pstdev(latencies) if len(latencies) > 1 else 0.0,
        },
        "tokens": {
            "input_total": sum(inputs),
            "output_total": sum(outputs),
            "input_mean": statistics.fmean(inputs),
            "output_mean": statistics.fmean(outputs),
        },
        "steps_total": sum(r.steps for r in records),
    }


def _script_for_turns(turns: int) -> list[ModelResponse]:
    """Return a deterministic response script that completes ``turns`` runs."""

    if turns <= 0:
        raise BenchError(f"turns must be > 0; got {turns}.")
    return [ModelResponse(content=f"reply-{n}") for n in range(turns)]


async def _one_turn(
    runtime: AgentRuntime,
    *,
    task: str,
    run_id: str,
) -> TurnRecord:
    started = time.perf_counter()
    result = await runtime.run(task, run_id=run_id)
    elapsed = (time.perf_counter() - started) * 1000
    return TurnRecord(
        turn=1,
        latency_ms=elapsed,
        input_tokens=result.token_usage.input_tokens,
        output_tokens=result.token_usage.output_tokens,
        steps=result.steps,
        stop_reason=result.stop_reason.value,
    )


async def run_benchmark(
    *,
    provider: ModelProvider,
    task: str = "Hello, world.",
    turns: int = 1,
) -> BenchReport:
    """Run ``turns`` single-turn agent runs and aggregate metrics."""

    provider_name = getattr(provider, "name", "fake")
    model_name = getattr(provider, "model", "fake")
    report = BenchReport(provider=provider_name, model=model_name, turns=turns, task=task)
    for index in range(turns):
        runtime = AgentRuntime(provider=provider)
        record = await _one_turn(runtime, task=task, run_id=f"bench-{index}")
        report.records.append(record)
    return report


def _scripted_provider(turns: int) -> FakeProvider:
    responses = _script_for_turns(turns)
    return FakeProvider(responses)


def main(argv: Sequence[str] | None = None) -> int:
    """``avo bench`` entry point. Parses argv and emits JSON to stdout."""

    args = list(argv if argv is not None else sys.argv[1:])
    turns = 1
    task = "Hello, world."
    output: Path | None = None
    while args:
        head = args[0]
        if head in {"--turns", "-n"} and len(args) > 1:
            turns = int(args[1])
            args = args[2:]
            continue
        if head == "--task" and len(args) > 1:
            task = args[1]
            args = args[2:]
            continue
        if head in {"--output", "-o"} and len(args) > 1:
            output = Path(args[1])
            args = args[2:]
            continue
        if head in {"--help", "-h"}:
            print(
                "Usage: avo bench [--turns N] [--task TEXT] [--output FILE]\n"
                "\n"
                "Run a deterministic benchmark against the FakeProvider. The\n"
                "report is emitted as JSON. Combine with `avo diff` to spot\n"
                "regressions between reports."
            )
            return 0
        raise BenchError(f"Unknown argument: {head!r}")

    if turns <= 0:
        raise BenchError("--turns must be > 0")

    report = asyncio.run(
        run_benchmark(provider=_scripted_provider(turns), task=task, turns=turns)
    )
    text = report.to_json()
    if output is None:
        sys.stdout.write(text + "\n")
    else:
        output.write_text(text + "\n", encoding="utf-8")
    return 0


__all__ = [
    "BenchError",
    "BenchReport",
    "TurnRecord",
    "main",
    "run_benchmark",
]


def _ensure_token_usage_typed() -> TokenUsage:
    """Type-hint anchor for the ``TokenUsage`` import above."""

    return TokenUsage()
