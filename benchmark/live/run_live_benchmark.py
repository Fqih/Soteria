"""Opt-in CLI runner that orchestrates the live benchmark scenarios.

The CLI is the user-facing entry point for spending real money against a paid
LLM provider.  Three responsibilities live here:

1. ``build_parser`` + ``preflight`` validate consent, provider-specific
   environment, pricing catalog, and positive limits before any provider is
   constructed.
2. ``run_all`` orchestrates the three registered scenarios (raw + Hernness
   for the completion scenarios, Hernness with interruption for the resume
   scenario) sequentially, building fresh config + provider objects per run.
3. ``main`` writes the JSON source of truth, prints a short aggregate summary,
   and renders one saved ``RunTrace.to_text()`` for the user.

Tests inject stub runners via keyword arguments; ``build_provider_factory`` is
the only seam the CLI exposes for replacing the real provider construction.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import traceback
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from benchmark.live.consent import (
    COST_CONSENT_ENV,
    COST_CONSENT_FLAG,
    require_cost_consent,
)
from benchmark.live.hernness_run import (
    run_hernness,
    run_hernness_interrupted,
)
from benchmark.live.models import Approach, LiveResults, LiveRunRecord
from benchmark.live.pricing import estimate_upper_bound, resolve_pricing
from benchmark.live.raw_loop import run_raw_loop
from benchmark.live.scenarios import LIVE_SCENARIOS, LiveScenario
from hernness.providers.base import ModelProvider

SleepFn = Callable[[float], Any]
ProviderFactory = Callable[[], ModelProvider]
RawRunner = Callable[..., Awaitable[LiveRunRecord]]
HernnessRunner = Callable[..., Awaitable[LiveRunRecord]]
InterruptedRunner = Callable[..., Awaitable[LiveRunRecord]]
InterruptedKwargs = Callable[..., Awaitable[LiveRunRecord]]

# Provider modules imported lazily so the tests stay offline and the CLI
# fails closed with a clear error when optional dependencies are absent.
_PROVIDER_MODULES: dict[str, tuple[str, str, str]] = {
    # provider -> (module path, "Config" attr, "Provider" attr)
    "minimax": (
        "examples.live_providers.minimax_provider",
        "MiniMaxConfig",
        "MiniMaxProvider",
    ),
    "openai": (
        "examples.live_providers.openai_provider",
        "OpenAIConfig",
        "OpenAIProvider",
    ),
}

_REQUIRED_ENV: dict[str, tuple[str, ...]] = {
    "minimax": ("MODEL_MINIMAX", "BASE_URL"),
    "openai": ("OPENAI_MODEL", "OPENAI_API_KEY"),
}

_INCOMPLETE_OUTCOMES = frozenset({"INTERNAL_ERROR"})


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the live benchmark CLI."""

    parser = argparse.ArgumentParser(
        prog="benchmark.live.run_live_benchmark",
        description=(
            "Run the opt-in live benchmark against a paid LLM provider. "
            "Refuses to run without --"
            f"{COST_CONSENT_FLAG} or {COST_CONSENT_ENV}=1|true|yes."
        ),
    )
    parser.add_argument(
        "--provider",
        choices=sorted(_PROVIDER_MODULES),
        default="minimax",
        help="Live provider to target (default: minimax).",
    )
    parser.add_argument(
        "--runs",
        type=_positive_int,
        default=3,
        help="Number of repetitions per (scenario, approach) pair (default: 3).",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.0,
        help="Seconds to sleep between model calls (default: 1.0).",
    )
    parser.add_argument(
        "--manual-step-cap",
        type=_positive_int,
        default=6,
        help="Maximum model steps for the raw baseline (default: 6).",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=_positive_int,
        default=1024,
        help="Max completion tokens surfaced to providers (default: 1024).",
    )
    parser.add_argument(
        "--input-tokens-per-step",
        type=_positive_int,
        default=2048,
        help="Token cap per step used by the upper-bound estimate (default: 2048).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=300.0,
        help="Per-call timeout budget used for asyncio.wait_for (default: 300).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark/live/results"),
        help="Directory that receives live_results_<UTC>.json (default: benchmark/live/results).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Override the resolved provider model. Defaults to the value of "
            "MODEL_MINIMAX / OPENAI_MODEL discovered from the environment."
        ),
    )
    parser.add_argument(
        "--api-style",
        choices=("openai", "anthropic"),
        default=None,
        help="Force a MiniMax API style; ignored for other providers.",
    )
    parser.add_argument(
        f"--{COST_CONSENT_FLAG}",
        dest=COST_CONSENT_FLAG.replace("-", "_"),
        action="store_true",
        default=False,
        help="Explicitly acknowledge the live benchmark will spend real money.",
    )
    return parser


def _positive_int(value: str) -> int:
    """argparse type that rejects zero and negatives."""

    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _resolve_flag_name() -> str:
    return COST_CONSENT_FLAG.replace("-", "_")


def preflight(
    args: argparse.Namespace,
    environ: Mapping[str, str],
    *,
    stdout: Any = None,
    stderr: Any = None,
) -> None:
    """Validate consent, provider env, pricing, and limits.

    Exits the process via ``SystemExit`` on any failure so the CLI fails closed
    without constructing any provider.
    """

    flag_name = _resolve_flag_name()
    flag = bool(getattr(args, flag_name, False))
    try:
        require_cost_consent(flag, environ)
    except Exception as exc:
        _emit(stderr or sys.stderr, f"error: {exc}")
        raise SystemExit(2) from exc

    provider = args.provider
    required = _REQUIRED_ENV.get(provider, ())
    missing = [name for name in required if name not in environ]
    if missing:
        _emit(
            stderr or sys.stderr,
            f"error: missing required environment for {provider}: {', '.join(missing)}",
        )
        raise SystemExit(2)

    try:
        model = _resolve_model(args, environ)
        pricing = resolve_pricing(provider, model, environ)
        estimate = estimate_upper_bound(
            pricing,
            max_steps=args.manual_step_cap,
            scenario_count=len(LIVE_SCENARIOS),
            runs=args.runs,
            input_tokens_per_step=args.input_tokens_per_step,
            output_tokens_per_step=args.input_tokens_per_step,
        )
    except ValueError as exc:
        _emit(stderr or sys.stderr, f"error: {exc}")
        raise SystemExit(2) from exc

    if args.manual_step_cap < 1:
        _emit(stderr or sys.stderr, "error: --manual-step-cap must be positive")
        raise SystemExit(2)
    if args.max_completion_tokens < 1:
        _emit(stderr or sys.stderr, "error: --max-completion-tokens must be positive")
        raise SystemExit(2)
    if args.input_tokens_per_step < 1:
        _emit(stderr or sys.stderr, "error: --input-tokens-per-step must be positive")
        raise SystemExit(2)
    if args.timeout_seconds <= 0:
        _emit(stderr or sys.stderr, "error: --timeout-seconds must be positive")
        raise SystemExit(2)
    if args.runs < 1:
        _emit(stderr or sys.stderr, "error: --runs must be positive")
        raise SystemExit(2)

    _emit(
        stdout or sys.stdout,
        f"Pre-flight estimate for provider={provider} model={model}: "
        f"~${estimate.cost_usd:.4f} USD across "
        f"n={args.runs} run(s) ({estimate.total_steps} steps total) - "
        f"{estimate.label}",
    )


def _resolve_model(args: argparse.Namespace, environ: Mapping[str, str]) -> str:
    explicit = getattr(args, "model", None)
    if explicit:
        return explicit
    if args.provider == "minimax":
        return environ.get("MODEL_MINIMAX", "MiniMax-M3")
    return environ.get("OPENAI_MODEL", "openai-model")


def _emit(stream: Any, message: str) -> None:
    """Write a line to ``stream``, falling back to ``print`` for non-stream args."""

    write = getattr(stream, "write", None)
    if callable(write):
        write(message + "\n")
        flush = getattr(stream, "flush", None)
        if callable(flush):
            flush()
    else:
        print(message)


def _load_provider_module(provider: str) -> ModuleType:
    module_path, _, _ = _PROVIDER_MODULES[provider]
    try:
        return _load_module(module_path)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Provider {provider!r} requires optional live-benchmark dependencies. "
            f"Install with `pip install hernness[live-benchmark]`. ({exc})"
        ) from exc


def _load_module(module_path: str) -> ModuleType:
    import importlib

    return importlib.import_module(module_path)


def _build_config(provider: str, args: argparse.Namespace, environ: Mapping[str, str]) -> Any:
    module = _load_provider_module(provider)
    config_cls = getattr(module, _PROVIDER_MODULES[provider][1])
    if provider == "minimax" and args.api_style is not None:
        # The MiniMax config reads MINIMAX_API_STYLE from environ; let users
        # override it on the CLI without mutating os.environ globally.
        merged = dict(environ)
        merged["MINIMAX_API_STYLE"] = args.api_style
        return config_cls.from_env(merged)
    return config_cls.from_env(environ)


def build_provider_factory(
    args: argparse.Namespace,
    environ: Mapping[str, str] | None = None,
) -> ProviderFactory:
    """Return a callable that builds a fresh provider/config pair per run."""

    env = os.environ if environ is None else environ

    def _factory() -> ModelProvider:
        config = _build_config(args.provider, args, env)
        module = _load_provider_module(args.provider)
        provider_cls = getattr(module, _PROVIDER_MODULES[args.provider][2])
        provider = provider_cls(
            config,
            max_completion_tokens=args.max_completion_tokens,
            request_timeout_seconds=args.timeout_seconds,
        )
        return provider

    return _factory


async def run_all(
    args: argparse.Namespace,
    provider_factory: ProviderFactory,
    raw_runner: RawRunner | None = None,
    hernness_runner: HernnessRunner | None = None,
    interrupted_runner: InterruptedRunner | None = None,
    sleep: SleepFn = time.sleep,
) -> LiveResults:
    """Execute the three scenarios sequentially and persist a :class:`LiveResults`.

    Each (scenario, approach) pair is repeated ``args.runs`` times.  Fresh
    provider objects are constructed per run via ``provider_factory`` so the
    raw baseline never shares state with the Hernness-managed runner.
    """

    raw_runner = raw_runner or _default_raw_runner
    hernness_runner = hernness_runner or _default_hernness_runner
    interrupted_runner = interrupted_runner or _default_interrupted_runner

    api_style, model = _describe_provider(args)
    records: list[LiveRunRecord] = []

    for scenario in LIVE_SCENARIOS:
        if scenario.supports_raw:
            for run_index in range(args.runs):
                records.extend(
                    await _run_with_sleep(
                        args,
                        provider_factory,
                        scenario,
                        run_index,
                        raw_runner=raw_runner,
                        hernness_runner=hernness_runner,
                        sleep=sleep,
                    )
                )

        if scenario.supports_resume:
            for run_index in range(args.runs):
                record = await _invoke_with_timeout(
                    args,
                    interrupted_runner,
                    scenario=scenario,
                    run_index=run_index,
                    approach="hernness",
                    provider_factory=provider_factory,
                    run_index_inner=run_index,
                )
                records.append(_mark_complete(record))
                _sleep(sleep, args.sleep_seconds)

    return LiveResults(
        provider=args.provider,
        api_style=api_style,
        model=model,
        runs=len(records),
        records=records,
    )


async def _run_with_sleep(
    args: argparse.Namespace,
    provider_factory: ProviderFactory,
    scenario: LiveScenario,
    run_index: int,
    *,
    raw_runner: RawRunner,
    hernness_runner: HernnessRunner,
    sleep: SleepFn,
) -> list[LiveRunRecord]:
    records: list[LiveRunRecord] = []
    if scenario.supports_raw:
        provider = provider_factory()
        record = await _invoke_with_timeout(
            args,
            raw_runner,
            scenario=scenario,
            run_index=run_index,
            approach="raw",
            provider=provider,
            manual_step_cap=args.manual_step_cap,
            max_completion_tokens=args.max_completion_tokens,
        )
        records.append(_mark_complete(record))
        _sleep(sleep, args.sleep_seconds)

        provider = provider_factory()
        record = await _invoke_with_timeout(
            args,
            _wrap_hernness(hernness_runner, scenario, run_index),
            scenario=scenario,
            run_index=run_index,
            approach="hernness",
            provider=provider,
        )
        records.append(_mark_complete(record))
        _sleep(sleep, args.sleep_seconds)
    return records


def _wrap_hernness(
    hernness_runner: HernnessRunner, scenario: LiveScenario, run_index: int
) -> Callable[..., Awaitable[LiveRunRecord]]:
    async def _call(
        provider: ModelProvider,
        **_unused: Any,
    ) -> LiveRunRecord:
        return await hernness_runner(provider, scenario=scenario, run_index=run_index)

    return _call


def _describe_provider(args: argparse.Namespace) -> tuple[str | None, str]:
    model = _resolve_model(args, os.environ)
    return (getattr(args, "api_style", None), model)


async def _invoke_with_timeout(
    args: argparse.Namespace,
    runner: Callable[..., Awaitable[LiveRunRecord]],
    *,
    scenario: LiveScenario,
    run_index: int,
    approach: Approach,
    **call_kwargs: Any,
) -> LiveRunRecord:
    def _record_failure(exc: BaseException, kind: str, message: str) -> LiveRunRecord:
        return _incomplete_record(
            exc,
            kind,
            message,
            scenario=scenario,
            run_index=run_index,
            approach=approach,
        )

    forwarded_kwargs: dict[str, Any] = dict(call_kwargs)
    forwarded_kwargs.setdefault("scenario", scenario)
    forwarded_kwargs.setdefault("run_index", run_index)

    try:
        coro = runner(**forwarded_kwargs)
    except TypeError as exc:
        return _record_failure(exc, type(exc).__name__, str(exc))
    try:
        return await asyncio.wait_for(coro, timeout=args.timeout_seconds)
    except TimeoutError as exc:
        return _record_failure(exc, "TimeoutError", "asyncio.TimeoutError")
    except Exception as exc:
        return _record_failure(exc, type(exc).__name__, str(exc))


def _incomplete_record(
    exc: BaseException,
    kind: str,
    message: str,
    *,
    scenario: LiveScenario | str,
    run_index: int,
    approach: Approach,
) -> LiveRunRecord:
    scenario_name = scenario.name if isinstance(scenario, LiveScenario) else str(scenario)
    common = {
        "scenario": scenario_name,
        "approach": approach,
        "run_index": run_index,
        "steps": 0,
        "duration_seconds": 0.0,
        "token_accounting_available": False,
    }
    if kind in ("ProviderError", "ToolExecutionError"):
        return LiveRunRecord(
            **common,
            expected_error_type=kind,
            unexpected_error_message=message,
        )
    return LiveRunRecord(
        **common,
        unexpected_error_type=kind,
        unexpected_error_message=message,
    )


def _mark_complete(record: LiveRunRecord) -> LiveRunRecord:
    return record


def _sleep(sleep: SleepFn, seconds: float) -> None:
    if seconds > 0:
        sleep(seconds)


async def _default_raw_runner(
    provider: ModelProvider,
    *,
    scenario: LiveScenario,
    manual_step_cap: int,
    max_completion_tokens: int,
    **_unused: Any,
) -> LiveRunRecord:
    return await run_raw_loop(provider, scenario, manual_step_cap, max_completion_tokens)


async def _default_hernness_runner(
    provider: ModelProvider,
    *,
    scenario: LiveScenario,
    run_index: int,
    **_unused: Any,
) -> LiveRunRecord:
    return await run_hernness(provider, scenario, run_index)


async def _default_interrupted_runner(
    provider_factory: ProviderFactory,
    *,
    scenario: LiveScenario,
    run_index_inner: int = 0,
    **_unused: Any,
) -> LiveRunRecord:
    return await run_hernness_interrupted(provider_factory, scenario, run_index_inner)


def _print_summary(results: LiveResults, output_path: Path, stdout: Any = None) -> None:
    completed = sum(1 for r in results.records if _is_cleanly_complete(r))
    errors = sum(1 for r in results.records if r.unexpected_error_type)
    expected_errors = sum(1 for r in results.records if r.expected_error_type)
    manual_caps = sum(1 for r in results.records if r.manual_step_cap_hit)
    stream = stdout or sys.stdout
    _emit(
        stream,
        "Summary: "
        f"{completed}/{len(results.records)} cleanly completed, "
        f"{expected_errors} expected errors, {errors} unexpected errors, "
        f"{manual_caps} manual caps hit.",
    )
    _emit(stream, f"JSON: {output_path}")
    example = next((r for r in results.records if r.trace_text), None)
    if example is not None:
        _emit(stream, f"Saved trace example ({example.scenario}, approach={example.approach}):")
        _emit(stream, example.trace_text)


def _is_cleanly_complete(record: LiveRunRecord) -> bool:
    if record.unexpected_error_type:
        return False
    if record.approach == "raw":
        return record.outcome == "completed"
    return record.status is not None and str(record.status) == "completed"


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: parse args, run preflight, execute, persist, summarize."""

    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        preflight(args, os.environ)
    except SystemExit as exc:
        return int(exc.code or 2)

    try:
        results = asyncio.run(run_all(args, build_provider_factory(args, os.environ)))
    except Exception as exc:
        traceback.print_exc()
        print(f"error: live benchmark crashed: {exc}", file=sys.stderr)
        return 1

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = results.write_json(output_dir)
    _print_summary(results, output_path)

    if any(record.unexpected_error_type in _INCOMPLETE_OUTCOMES for record in results.records):
        return 1
    if any(record.unexpected_error_type for record in results.records):
        return 1
    return 0


__all__ = [
    "build_parser",
    "build_provider_factory",
    "main",
    "preflight",
    "run_all",
]
