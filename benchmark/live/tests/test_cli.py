"""Offline tests for the live benchmark CLI runner and preflight."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from avo import TokenUsage
from avo.providers.base import ModelProvider
from benchmark.live.consent import COST_CONSENT_ENV
from benchmark.live.models import LiveRunRecord
from benchmark.live.run_live_benchmark import (
    build_parser,
    main,
    preflight,
    run_all,
)
from benchmark.live.scenarios import LiveScenario, scenario_by_name

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _scenario(name: str) -> LiveScenario:
    return scenario_by_name(name)


def _provider_factory_stub(records: list[LiveRunRecord]) -> MagicMock:
    """Return a callable that builds a fresh fake provider each invocation."""

    factory = MagicMock(side_effect=lambda *a, **kw: SimpleProvider(records))

    def _build() -> ModelProvider:
        factory()
        return SimpleProvider(records)

    _build.factory = factory  # expose for assertions
    return _build  # type: ignore[return-value]


class SimpleProvider(ModelProvider):
    """Minimal provider stub used by the CLI orchestration tests."""

    def __init__(self, records: list[LiveRunRecord]) -> None:
        self.records = records
        self.generate_calls = 0

    async def generate(self, request):  # type: ignore[override]
        self.generate_calls += 1
        raise NotImplementedError("CLI tests inject run_all directly")


# ---------------------------------------------------------------------------
# argparse + preflight
# ---------------------------------------------------------------------------


def test_parser_defaults_match_brief() -> None:
    parser = build_parser()
    args = parser.parse_args([])

    assert args.provider == "minimax"
    assert args.runs == 3
    assert args.sleep_seconds == pytest.approx(1.0)
    assert args.manual_step_cap == 6
    assert args.max_completion_tokens == 1024
    assert args.input_tokens_per_step == 2048
    assert args.timeout_seconds == pytest.approx(300.0)
    assert args.i_understand_this_costs_money is False


def test_parser_accepts_overrides() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--provider",
            "openai",
            "--runs",
            "5",
            "--sleep-seconds",
            "0",
            "--manual-step-cap",
            "8",
            "--max-completion-tokens",
            "256",
            "--input-tokens-per-step",
            "512",
            "--timeout-seconds",
            "60",
            "--i-understand-this-costs-money",
        ]
    )

    assert args.provider == "openai"
    assert args.runs == 5
    assert args.sleep_seconds == pytest.approx(0.0)
    assert args.manual_step_cap == 8
    assert args.max_completion_tokens == 256
    assert args.input_tokens_per_step == 512
    assert args.timeout_seconds == pytest.approx(60.0)
    assert args.i_understand_this_costs_money is True


def test_parser_rejects_negative_runs() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--runs", "0"])


def test_parser_rejects_unknown_provider() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--provider", "anthropic"])


def test_preflight_refuses_without_consent() -> None:
    args = argparse.Namespace(
        provider="minimax",
        runs=1,
        sleep_seconds=0.0,
        manual_step_cap=2,
        max_completion_tokens=64,
        input_tokens_per_step=128,
        timeout_seconds=10.0,
        i_understand_this_costs_money=False,
    )

    with pytest.raises(SystemExit):
        preflight(args, environ={})


def test_preflight_refuses_openai_without_api_key() -> None:
    args = argparse.Namespace(
        provider="openai",
        runs=1,
        sleep_seconds=0.0,
        manual_step_cap=2,
        max_completion_tokens=64,
        input_tokens_per_step=128,
        timeout_seconds=10.0,
        i_understand_this_costs_money=True,
    )

    with pytest.raises(SystemExit):
        preflight(args, environ={COST_CONSENT_ENV: "1"})


def test_preflight_passes_for_minimax_with_flag() -> None:
    args = argparse.Namespace(
        provider="minimax",
        runs=1,
        sleep_seconds=0.0,
        manual_step_cap=2,
        max_completion_tokens=64,
        input_tokens_per_step=128,
        timeout_seconds=10.0,
        i_understand_this_costs_money=True,
    )

    # Preflight enforces consent + provider env + positive limits; passing
    # MODEL_MINIMAX + BASE_URL is sufficient for the catalog resolution path.
    preflight(args, environ={"MODEL_MINIMAX": "MiniMax-M3", "BASE_URL": "https://api.minimax.io"})


def test_preflight_rejects_non_positive_limits() -> None:
    args = argparse.Namespace(
        provider="minimax",
        runs=1,
        sleep_seconds=0.0,
        manual_step_cap=0,
        max_completion_tokens=64,
        input_tokens_per_step=128,
        timeout_seconds=10.0,
        i_understand_this_costs_money=True,
    )
    with pytest.raises(SystemExit):
        preflight(args, environ={COST_CONSENT_ENV: "1"})


# ---------------------------------------------------------------------------
# run_all orchestration
# ---------------------------------------------------------------------------


def _completed_record(scenario_name: str, approach: str, run_index: int) -> LiveRunRecord:
    if approach == "raw":
        return LiveRunRecord(
            scenario=scenario_name,
            approach="raw",
            run_index=run_index,
            outcome="completed",
            steps=2,
            duration_seconds=0.01,
            token_usage=TokenUsage(input_tokens=10, output_tokens=4),
            token_accounting_available=True,
        )
    return LiveRunRecord(
        scenario=scenario_name,
        approach="avo",
        run_index=run_index,
        status="completed",
        stop_reason="completed",
        steps=2,
        duration_seconds=0.01,
        token_usage=TokenUsage(input_tokens=10, output_tokens=4),
        token_accounting_available=True,
    )


@pytest.mark.asyncio
async def test_run_all_orchestrates_three_scenarios_raw_and_avo(tmp_path: Path) -> None:
    args = SimpleNamespace(
        provider="minimax",
        runs=2,
        sleep_seconds=0.0,
        manual_step_cap=6,
        max_completion_tokens=128,
        input_tokens_per_step=2048,
        timeout_seconds=10.0,
        output_dir=tmp_path,
    )

    call_log: list[str] = []

    def provider_factory_stub() -> ModelProvider:
        call_log.append("provider_factory")
        return SimpleProvider([])

    async def raw_runner(provider, *, scenario, manual_step_cap, max_completion_tokens, **_unused):
        scenario = scenario_by_name(scenario) if isinstance(scenario, str) else scenario
        call_log.append(f"raw:{scenario.name}")
        return _completed_record(scenario.name, "raw", run_index=0)

    async def avo_runner(provider, *, scenario, run_index, **_unused):
        scenario = scenario_by_name(scenario) if isinstance(scenario, str) else scenario
        call_log.append(f"avo:{scenario.name}:{run_index}")
        return _completed_record(scenario.name, "avo", run_index=run_index)

    async def interrupted_runner(provider_factory, *, scenario, run_index_inner=0, **_unused):
        scenario = scenario_by_name(scenario) if isinstance(scenario, str) else scenario
        call_log.append(f"avo-interrupted:{scenario.name}:{run_index_inner}")
        return _completed_record(scenario.name, "avo", run_index=run_index_inner)

    results = await run_all(
        args,
        provider_factory=provider_factory_stub,
        raw_runner=raw_runner,
        avo_runner=avo_runner,
        interrupted_runner=interrupted_runner,
        sleep=lambda _seconds: None,
    )

    # Two scenarios are raw-and-avo capable (3 raw + 3 avo per
    # scenario * 2 scenarios * 2 runs), and one scenario is
    # avo-interrupted-only.
    raw_calls = [entry for entry in call_log if entry.startswith("raw:")]
    avo_calls = [entry for entry in call_log if entry.startswith("avo:")]
    interrupted_calls = [entry for entry in call_log if entry.startswith("avo-interrupted:")]

    assert len(raw_calls) == 2 * 2  # 2 raw-capable scenarios * 2 runs
    assert len(avo_calls) == 2 * 2  # 2 raw-capable scenarios * 2 runs
    assert len(interrupted_calls) == 2  # 1 resume scenario * 2 runs

    assert results.provider == "minimax"
    assert results.runs == 10  # total records
    assert len(results.records) == 10


@pytest.mark.asyncio
async def test_run_all_persists_utc_timestamped_json(tmp_path: Path) -> None:
    args = SimpleNamespace(
        provider="minimax",
        runs=1,
        sleep_seconds=0.0,
        manual_step_cap=2,
        max_completion_tokens=64,
        input_tokens_per_step=128,
        timeout_seconds=10.0,
        output_dir=tmp_path,
    )

    async def provider_factory_stub_async() -> ModelProvider:
        return SimpleProvider([])

    def provider_factory_stub() -> ModelProvider:
        return SimpleProvider([])

    async def raw_runner(provider, *, scenario, manual_step_cap, max_completion_tokens, **_unused):
        scenario = scenario_by_name(scenario) if isinstance(scenario, str) else scenario
        return _completed_record(scenario.name, "raw", run_index=0)

    async def avo_runner(provider, *, scenario, run_index, **_unused):
        scenario = scenario_by_name(scenario) if isinstance(scenario, str) else scenario
        return _completed_record(scenario.name, "avo", run_index=run_index)

    async def interrupted_runner(provider_factory, *, scenario, run_index_inner=0, **_unused):
        scenario = scenario_by_name(scenario) if isinstance(scenario, str) else scenario
        return _completed_record(scenario.name, "avo", run_index=run_index_inner)

    results = await run_all(
        args,
        provider_factory=provider_factory_stub,
        raw_runner=raw_runner,
        avo_runner=avo_runner,
        interrupted_runner=interrupted_runner,
        sleep=lambda _seconds: None,
    )

    output = results.write_json(tmp_path)
    assert output.parent == tmp_path
    assert re.match(r"live_results_\d{8}T\d{6}Z\.json$", output.name) is not None
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["provider"] == "minimax"
    assert payload["runs"] == len(results.records)


@pytest.mark.asyncio
async def test_run_all_marks_internal_errors_incomplete() -> None:
    args = SimpleNamespace(
        provider="minimax",
        runs=1,
        sleep_seconds=0.0,
        manual_step_cap=2,
        max_completion_tokens=64,
        input_tokens_per_step=128,
        timeout_seconds=0.05,
        output_dir=Path("/tmp/live-bench-tests-noop"),
    )

    async def provider_factory_stub_async() -> ModelProvider:
        return SimpleProvider([])

    def provider_factory_stub() -> ModelProvider:
        return SimpleProvider([])

    async def raw_runner(provider, *, scenario, manual_step_cap, max_completion_tokens, **_unused):
        scenario = scenario_by_name(scenario) if isinstance(scenario, str) else scenario
        raise RuntimeError("intentional raw failure")

    async def avo_runner(provider, *, scenario, run_index, **_unused):
        scenario = scenario_by_name(scenario) if isinstance(scenario, str) else scenario
        return _completed_record(scenario.name, "avo", run_index=run_index)

    async def interrupted_runner(provider_factory, *, scenario, run_index_inner=0, **_unused):
        scenario = scenario_by_name(scenario) if isinstance(scenario, str) else scenario
        return _completed_record(scenario.name, "avo", run_index=run_index_inner)

    results = await run_all(
        args,
        provider_factory=provider_factory_stub,
        raw_runner=raw_runner,
        avo_runner=avo_runner,
        interrupted_runner=interrupted_runner,
        sleep=lambda _seconds: None,
    )

    errored = [r for r in results.records if r.unexpected_error_type == "RuntimeError"]
    assert errored, "expected at least one record to capture the raw failure"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_refuses_without_consent(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--provider", "minimax", "--runs", "1"])
    captured = capsys.readouterr()
    assert code != 0
    assert "i-understand-this-costs-money" in captured.err or "consent" in captured.err.lower()


def test_main_prints_summary_and_one_trace_example(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    async def provider_factory_stub_async() -> ModelProvider:
        return SimpleProvider([])

    def provider_factory_stub() -> ModelProvider:
        return SimpleProvider([])

    async def raw_runner(provider, scenario, manual_step_cap, max_completion_tokens, **_unused):
        scenario = scenario_by_name(scenario) if isinstance(scenario, str) else scenario
        return _completed_record(scenario.name, "raw", run_index=0)

    async def avo_runner(provider, scenario, run_index, **_unused):
        scenario = scenario_by_name(scenario) if isinstance(scenario, str) else scenario
        record = _completed_record(scenario.name, "avo", run_index=run_index)
        # Ensure the summary trace render has a non-empty trace to print.
        record.trace_text = f"Run: {scenario.name}\nStop reason: completed\n"
        return record

    async def interrupted_runner(provider_factory, scenario, run_index=0, **_unused):
        scenario = scenario_by_name(scenario) if isinstance(scenario, str) else scenario
        record = _completed_record(scenario.name, "avo", run_index=run_index)
        record.trace_text = f"Run: {scenario.name}\nStop reason: completed\n"
        return record

    monkeypatch.setattr(
        "benchmark.live.run_live_benchmark.build_provider_factory",
        lambda *_a, **_kw: provider_factory_stub,
    )
    monkeypatch.setattr("benchmark.live.run_live_benchmark.time.sleep", lambda _s: None)
    monkeypatch.setenv("MODEL_MINIMAX", "MiniMax-M3")
    monkeypatch.setenv("BASE_URL", "https://api.minimax.io")

    # Replace the real runners inside the CLI module directly so that the
    # production ``run_all`` still owns orchestration.  We do not patch
    # ``run_all`` itself because it would re-enter through ``main``'s
    # import-time reference and recurse.
    import benchmark.live.run_live_benchmark as cli_module

    original_raw = cli_module.run_raw_loop
    original_avo = cli_module.run_avo
    original_interrupted = cli_module.run_avo_interrupted
    cli_module.run_raw_loop = raw_runner  # type: ignore[assignment]
    cli_module.run_avo = avo_runner  # type: ignore[assignment]
    cli_module.run_avo_interrupted = interrupted_runner  # type: ignore[assignment]
    try:
        code = main(
            [
                "--provider",
                "minimax",
                "--runs",
                "1",
                "--sleep-seconds",
                "0",
                "--i-understand-this-costs-money",
                "--output-dir",
                str(tmp_path),
            ]
        )
    finally:
        cli_module.run_raw_loop = original_raw
        cli_module.run_avo = original_avo
        cli_module.run_avo_interrupted = original_interrupted

    assert code == 0
    captured = capsys.readouterr()
    assert "Summary" in captured.out or "summary" in captured.out
    assert "Stop reason: completed" in captured.out
    assert "live_results_" in captured.out
