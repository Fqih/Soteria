"""Tests for ``avo.bench``."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from avo.bench import (
    BenchError,
    BenchReport,
    TurnRecord,
    _script_for_turns,
    run_benchmark,
)
from avo.bench import (
    main as bench_main,
)
from avo.models import ModelResponse
from avo.providers.fake import FakeProvider


def test_script_for_turns_rejects_zero() -> None:
    with pytest.raises(BenchError, match="turns must be > 0"):
        _script_for_turns(0)


def test_script_for_turns_returns_n_responses() -> None:
    responses = _script_for_turns(3)
    assert len(responses) == 3
    assert all(r.content for r in responses)


@pytest.mark.asyncio
async def test_run_benchmark_records_turns() -> None:
    provider = FakeProvider([ModelResponse(content="hi"), ModelResponse(content="hey")])
    report = await run_benchmark(provider=provider, task="say hi", turns=2)
    assert isinstance(report, BenchReport)
    assert report.turns == 2
    assert len(report.records) == 2
    assert all(isinstance(r, TurnRecord) for r in report.records)
    assert all(r.latency_ms >= 0 for r in report.records)


def test_turn_record_as_dict_has_expected_keys() -> None:
    record = TurnRecord(
        turn=1, latency_ms=1.5, input_tokens=10, output_tokens=5, steps=1, stop_reason="completed"
    )
    payload = record.as_dict()
    assert payload["turn"] == 1
    assert payload["latency_ms"] == 1.5
    assert payload["stop_reason"] == "completed"


def test_bench_report_to_json_is_valid_json() -> None:
    report = BenchReport(
        provider="fake",
        model="m",
        turns=1,
        task="t",
        records=[
            TurnRecord(
                turn=1,
                latency_ms=2.0,
                input_tokens=3,
                output_tokens=4,
                steps=1,
                stop_reason="completed",
            )
        ],
    )
    parsed = json.loads(report.to_json())
    assert parsed["provider"] == "fake"
    assert parsed["turns_completed"] == 1
    assert "summary" in parsed
    assert parsed["summary"]["tokens"]["input_total"] == 3


def test_bench_report_summary_empty() -> None:
    report = BenchReport(provider="p", model="m", turns=0, task="t")
    payload = report.to_json()
    parsed = json.loads(payload)
    assert parsed["summary"]["empty"] is True


def test_main_emits_json_to_stdout() -> None:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = bench_main(["--turns", "1"])
    assert code == 0
    parsed = json.loads(buffer.getvalue())
    assert parsed["turns_requested"] == 1


def test_main_writes_to_output_file(tmp_path: Path) -> None:
    target = tmp_path / "report.json"
    code = bench_main(["--turns", "1", "--output", str(target)])
    assert code == 0
    assert target.is_file()
    parsed = json.loads(target.read_text())
    assert parsed["turns_requested"] == 1


def test_main_rejects_zero_turns() -> None:
    with pytest.raises(BenchError):
        bench_main(["--turns", "0"])


def test_main_rejects_unknown_arg() -> None:
    with pytest.raises(BenchError, match="Unknown argument"):
        bench_main(["--bogus"])


def test_main_help_returns_zero() -> None:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = bench_main(["--help"])
    assert code == 0
    assert "Usage:" in buffer.getvalue()
