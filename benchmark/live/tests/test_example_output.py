"""Guard the checked-in real live-benchmark outputs.

The example bundle under ``benchmark/live/example_output/`` is a real
``MiniMax-M3`` anthropic-style run (provider=``minimax``, api_style=``anthropic``)
recorded against MiniMax's API.  These tests verify the JSON conforms to
:class:`LiveResults` and describes the expected shape (provider=``minimax``,
api_style=``anthropic``, model=``MiniMax-M3``, three runs per applicable
approach, and a Avo interruption/resume record), and that the two stable
PNGs exist with non-trivial size.  The PNGs must always be produced by the
renderer, never hand-authored.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

pytest.importorskip("matplotlib")

from avo import RunState, StopReason
from benchmark.live.models import LiveResults
from benchmark.live.render import load_results

_EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "example_output"
_JSON_PATH = _EXAMPLE_DIR / "example_results.json"
_REPETITION_PNG = _EXAMPLE_DIR / "repetition_containment.png"
_NORMAL_PNG = _EXAMPLE_DIR / "normal_completion_comparison.png"

_SAMPLE_SIZE = 3


def _load() -> LiveResults:
    return load_results(_JSON_PATH)


def test_example_json_conforms_to_live_results() -> None:
    results = _load()

    assert isinstance(results, LiveResults)
    assert results.provider == "minimax"
    assert results.api_style == "anthropic"
    assert results.model == "MiniMax-M3"
    # LiveResults enforces runs == len(records); the load above validated it.
    assert results.runs == len(results.records)
    assert results.runs > 0


def test_example_json_has_three_runs_per_applicable_approach() -> None:
    results = _load()

    grouped: Counter[tuple[str, str]] = Counter(
        (record.scenario, record.approach) for record in results.records
    )

    assert grouped[("normal_completion", "raw")] == _SAMPLE_SIZE
    assert grouped[("normal_completion", "avo")] == _SAMPLE_SIZE
    assert grouped[("repetition_prone", "raw")] == _SAMPLE_SIZE
    assert grouped[("repetition_prone", "avo")] == _SAMPLE_SIZE
    assert grouped[("interrupted_resume", "avo")] == _SAMPLE_SIZE


def test_example_records_carry_token_duration_and_containment_fields() -> None:
    results = _load()

    for record in results.records:
        assert record.token_accounting_available is True
        assert record.duration_seconds > 0
        assert record.token_usage.input_tokens > 0
        assert record.token_usage.output_tokens > 0

    repetition_soteria = [
        record
        for record in results.records
        if record.scenario == "repetition_prone" and record.approach == "avo"
    ]
    assert repetition_soteria
    for record in repetition_soteria:
        assert record.status is RunState.STOPPED
        assert record.stop_reason is StopReason.REPEATED_ACTION
        assert record.repeated_action_detected is True


def test_example_interruption_record_resumes_exactly_once() -> None:
    results = _load()

    interruptions = [
        record for record in results.records if record.scenario == "interrupted_resume"
    ]
    assert len(interruptions) == _SAMPLE_SIZE
    for record in interruptions:
        assert record.approach == "avo"
        assert record.resume_tool_executed_exactly_once is True


def test_example_pngs_exist_and_are_non_trivial() -> None:
    for path in (_REPETITION_PNG, _NORMAL_PNG):
        assert path.exists(), f"missing rendered PNG: {path}"
        assert path.stat().st_size > 1000
        assert path.read_bytes()[:8].startswith(b"\x89PNG\r\n\x1a\n")
