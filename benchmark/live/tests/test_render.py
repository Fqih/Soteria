"""Offline tests for the live benchmark JSON-to-chart renderer.

The renderer is intentionally matplotlib-optional: ``pytest.importorskip`` keeps
the suite green when matplotlib is not installed while still exercising the
public interfaces (aggregation, titles, and file layout) end-to-end.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

matplotlib = pytest.importorskip("matplotlib")

from benchmark.live.models import LiveResults, LiveRunRecord  # noqa: E402
from benchmark.live.render import (  # noqa: E402
    chart_titles,
    load_results,
    render_results,
    render_results_data,
)
from soteria import RunState, StopReason  # noqa: E402

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def _normal_raw(steps: int, duration: float, run_index: int) -> LiveRunRecord:
    return LiveRunRecord(
        scenario="normal_completion",
        approach="raw",
        run_index=run_index,
        outcome="completed",
        steps=steps,
        duration_seconds=duration,
    )


def _normal_soteria(steps: int, duration: float, run_index: int) -> LiveRunRecord:
    return LiveRunRecord(
        scenario="normal_completion",
        approach="soteria",
        run_index=run_index,
        status=RunState.COMPLETED,
        stop_reason=StopReason.COMPLETED,
        steps=steps,
        duration_seconds=duration,
    )


def _repetition_raw(contained: bool, run_index: int) -> LiveRunRecord:
    return LiveRunRecord(
        scenario="repetition_prone",
        approach="raw",
        run_index=run_index,
        outcome="hit_manual_step_cap" if contained else "completed",
        steps=6 if contained else 4,
        duration_seconds=1.2 if contained else 0.8,
        manual_step_cap_hit=contained,
    )


def _repetition_soteria(contained: bool, run_index: int) -> LiveRunRecord:
    return LiveRunRecord(
        scenario="repetition_prone",
        approach="soteria",
        run_index=run_index,
        status=RunState.STOPPED if contained else RunState.COMPLETED,
        stop_reason=StopReason.REPEATED_ACTION if contained else StopReason.COMPLETED,
        steps=3,
        duration_seconds=0.9,
        repeated_action_detected=contained,
    )


def _build_fixture() -> LiveResults:
    records: list[LiveRunRecord] = [
        _normal_raw(steps=4, duration=0.6, run_index=0),
        _normal_raw(steps=5, duration=0.7, run_index=1),
        _normal_soteria(steps=3, duration=0.5, run_index=0),
        _normal_soteria(steps=4, duration=0.6, run_index=1),
        _repetition_raw(contained=True, run_index=0),
        _repetition_raw(contained=True, run_index=1),
        _repetition_soteria(contained=True, run_index=0),
        _repetition_soteria(contained=True, run_index=1),
    ]
    return LiveResults(
        provider="fixture",
        api_style=None,
        model="Fixture-Model",
        runs=len(records),
        records=records,
    )


# ---------------------------------------------------------------------------
# Aggregation / titles
# ---------------------------------------------------------------------------


def test_chart_titles_embed_provider_model_and_sample_size() -> None:
    fixture = _build_fixture()

    repetition_title, normal_title = chart_titles(fixture)

    assert "Fixture-Model" in repetition_title
    assert "fixture" in repetition_title
    assert "n=2" in repetition_title
    assert "Fixture-Model" in normal_title
    assert "fixture" in normal_title
    assert "n=2" in normal_title


def test_load_results_round_trips_json(tmp_path: Path) -> None:
    fixture = _build_fixture()

    json_path = fixture.write_json(tmp_path)
    loaded = load_results(json_path)

    assert loaded.provider == fixture.provider
    assert loaded.model == fixture.model
    assert loaded.runs == fixture.runs
    assert len(loaded.records) == len(fixture.records)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_render_results_data_writes_stable_png_files(tmp_path: Path) -> None:
    fixture = _build_fixture()

    repetition_path, normal_path = render_results_data(fixture, tmp_path)

    assert repetition_path.name == "repetition_containment.png"
    assert normal_path.name == "normal_completion_comparison.png"
    assert repetition_path.exists()
    assert normal_path.exists()
    assert repetition_path.stat().st_size > 1000
    assert normal_path.stat().st_size > 1000


def test_render_results_data_copies_into_example_output_dir(tmp_path: Path) -> None:
    fixture = _build_fixture()
    example_dir = tmp_path / "examples"

    repetition_path, normal_path = render_results_data(
        fixture, tmp_path, example_output_dir=example_dir
    )

    copied_repetition = example_dir / "repetition_containment.png"
    copied_normal = example_dir / "normal_completion_comparison.png"
    assert copied_repetition.exists()
    assert copied_normal.exists()
    assert copied_repetition.stat().st_size == repetition_path.stat().st_size
    assert copied_normal.stat().st_size == normal_path.stat().st_size


def test_render_results_loads_json_and_renders(tmp_path: Path) -> None:
    fixture = _build_fixture()
    json_path = fixture.write_json(tmp_path)
    output_dir = tmp_path / "charts"

    repetition_path, normal_path = render_results(json_path, output_dir)

    assert repetition_path.parent == output_dir
    assert normal_path.parent == output_dir
    assert repetition_path.exists()
    assert normal_path.exists()


def test_render_results_data_rejects_missing_required_groups(tmp_path: Path) -> None:
    raw_only = LiveResults(
        provider="fixture",
        model="Fixture-Model",
        runs=1,
        records=[_normal_raw(steps=4, duration=0.6, run_index=0)],
    )

    with pytest.raises(ValueError):
        render_results_data(raw_only, tmp_path)


def test_chart_pngs_have_valid_headers(tmp_path: Path) -> None:
    fixture = _build_fixture()

    repetition_path, normal_path = render_results_data(fixture, tmp_path)
    for path in (repetition_path, normal_path):
        header = path.read_bytes()[:8]
        assert header.startswith(b"\x89PNG\r\n\x1a\n")


# ---------------------------------------------------------------------------
# JSON shape sanity check (offline)
# ---------------------------------------------------------------------------


def test_written_json_matches_loaded_results(tmp_path: Path) -> None:
    fixture = _build_fixture()
    json_path = fixture.write_json(tmp_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["provider"] == "fixture"
    assert payload["model"] == "Fixture-Model"
    assert payload["runs"] == len(fixture.records)
    assert {record["scenario"] for record in payload["records"]} == {
        "normal_completion",
        "repetition_prone",
    }
    assert {record["approach"] for record in payload["records"]} == {"raw", "soteria"}
