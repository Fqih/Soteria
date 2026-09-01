"""Render live benchmark ``LiveResults`` bundles into matplotlib PNG charts.

The renderer is deliberately JSON-driven: provider, model, and sample size are
read from the bundle rather than hardcoded so the same code can chart any
provider's results.  Matplotlib is an optional dependency of the live benchmark
extras (``pip install avo[live-benchmark]``); tests gate on
``pytest.importorskip("matplotlib")`` so the suite stays green without it.

Public surface:

- :func:`load_results` parses a JSON bundle written by the CLI.
- :func:`chart_titles` returns the (repetition, normal) chart titles.
- :func:`render_results_data` charts an in-memory ``LiveResults``.
- :func:`render_results` loads a JSON bundle and charts it.

Stable filenames (``repetition_containment.png`` and
``normal_completion_comparison.png``) make downstream tooling and example-output
fixtures trivially reproducible.
"""

from __future__ import annotations

import contextlib
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib
from benchmark.live.avo_run import avo_contained

from benchmark.live.models import LiveResults, LiveRunRecord

matplotlib.use("Agg")  # headless backend, must precede pyplot

import matplotlib.pyplot as plt

_REPETITION_SCENARIO = "repetition_prone"
_NORMAL_SCENARIO = "normal_completion"

_REPETITION_PNG = "repetition_containment.png"
_NORMAL_PNG = "normal_completion_comparison.png"

_APPROACH_ORDER: tuple[str, ...] = ("raw", "avo")

_CHART_STYLE = "seaborn-v0_8-whitegrid"

_CONTAINED_COLOR = "#1f77b4"
_ESCAPED_COLOR = "#aec7e8"
_STEPS_COLOR = "#2ca02c"
_DURATION_COLOR = "#d62728"


def load_results(path: Path) -> LiveResults:
    """Read a :class:`LiveResults` JSON bundle from ``path`` and validate it."""

    return LiveResults.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _record_contained(record: LiveRunRecord) -> bool:
    """Return whether ``record``'s loop was contained by a safety fence.

    For the raw baseline, containment is signalled by hitting the manual step
    cap because the raw loop has no policy-driven fences.  For the Avo
    approach, containment is decided by the policy stop reasons recorded by
    :func:`benchmark.live.avo_run.avo_contained`.
    """

    if record.approach == "raw":
        return bool(record.manual_step_cap_hit)
    return avo_contained(record)


def _aggregate_repetition(
    results: LiveResults,
) -> tuple[dict[str, int], dict[str, int]]:
    """Group repetition records by approach into (contained_counts, sample_size)."""

    contained: dict[str, int] = defaultdict(int)
    sample: dict[str, int] = defaultdict(int)
    for record in results.records:
        if record.scenario != _REPETITION_SCENARIO:
            continue
        sample[record.approach] += 1
        if _record_contained(record):
            contained[record.approach] += 1
    return dict(contained), dict(sample)


def _aggregate_normal(results: LiveResults) -> dict[str, tuple[float, float]]:
    """Return per-approach ``(mean_steps, mean_duration)`` for the normal scenario."""

    steps: dict[str, list[int]] = defaultdict(list)
    durations: dict[str, list[float]] = defaultdict(list)
    for record in results.records:
        if record.scenario != _NORMAL_SCENARIO:
            continue
        steps[record.approach].append(record.steps)
        durations[record.approach].append(record.duration_seconds)
    means: dict[str, tuple[float, float]] = {}
    for approach, step_values in steps.items():
        means[approach] = (
            sum(step_values) / len(step_values),
            sum(durations[approach]) / len(durations[approach]),
        )
    return means


def _aggregate_normal_sample(results: LiveResults) -> int:
    """Return the per-approach sample size for the normal scenario, or 0."""

    for approach in _APPROACH_ORDER:
        values = [
            record
            for record in results.records
            if record.scenario == _NORMAL_SCENARIO and record.approach == approach
        ]
        if values:
            return len(values)
    return 0


def _require_complete(sample: dict[str, int], label: str) -> None:
    """Refuse to chart incomplete data: every approach must have at least one record."""

    for approach in _APPROACH_ORDER:
        if sample.get(approach, 0) == 0:
            raise ValueError(
                f"Missing {label} records for approach={approach!r}; "
                "refusing to fabricate zero values."
            )


def chart_titles(results: LiveResults) -> tuple[str, str]:
    """Return ``(repetition_title, normal_title)`` derived from the bundle."""

    _, repetition_sample = _aggregate_repetition(results)
    repetition_size = next(iter(repetition_sample.values()), 0) if repetition_sample else 0
    repetition_title = (
        f"Repetition containment — {results.provider} / {results.model} "
        f"(n={repetition_size} runs per approach)"
    )
    normal_size = _aggregate_normal_sample(results)
    normal_title = (
        f"Normal completion comparison — {results.provider} / {results.model} "
        f"(n={normal_size} runs per approach)"
    )
    return repetition_title, normal_title


def _apply_style() -> None:
    """Apply the seaborn whitegrid style once, falling back to the default style."""

    with contextlib.suppress(OSError):
        plt.style.use(_CHART_STYLE)


def _render_repetition_chart(
    results: LiveResults,
    contained: dict[str, int],
    sample: dict[str, int],
    title: str,
    target: Path,
) -> Path:
    """Write the grouped repetition-containment bar chart to ``target``."""

    approaches = list(_APPROACH_ORDER)
    contained_counts = [contained.get(a, 0) for a in approaches]
    escaped_counts = [sample.get(a, 0) - contained.get(a, 0) for a in approaches]
    width = 0.4
    positions = list(range(len(approaches)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        [position - width / 2 for position in positions],
        contained_counts,
        width=width,
        label="contained",
        color=_CONTAINED_COLOR,
    )
    ax.bar(
        [position + width / 2 for position in positions],
        escaped_counts,
        width=width,
        label="escaped",
        color=_ESCAPED_COLOR,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{approach}\n(n={sample.get(approach, 0)})" for approach in approaches])
    ax.set_ylabel(f"runs (provider={results.provider}, model={results.model})")
    ax.set_title(title)
    ax.set_xlabel("approach")
    ax.legend()
    fig.tight_layout()
    fig.savefig(target)
    plt.close(fig)
    return target


def _render_normal_chart(
    results: LiveResults,
    means: dict[str, tuple[float, float]],
    title: str,
    target: Path,
) -> Path:
    """Write the two-panel normal-completion comparison chart to ``target``."""

    approaches = list(_APPROACH_ORDER)
    steps_values = [means.get(approach, (0.0, 0.0))[0] for approach in approaches]
    duration_values = [means.get(approach, (0.0, 0.0))[1] for approach in approaches]

    fig, (ax_steps, ax_seconds) = plt.subplots(1, 2, figsize=(12, 5))
    ax_steps.bar(approaches, steps_values, color=_STEPS_COLOR)
    ax_steps.set_title("Mean steps per approach")
    ax_steps.set_ylabel(f"steps (provider={results.provider}, model={results.model})")
    ax_steps.set_xlabel("approach")

    ax_seconds.bar(approaches, duration_values, color=_DURATION_COLOR)
    ax_seconds.set_title("Mean duration per approach")
    ax_seconds.set_ylabel(f"seconds (provider={results.provider}, model={results.model})")
    ax_seconds.set_xlabel("approach")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(target)
    plt.close(fig)
    return target


def _copy_to_example_dir(outputs: tuple[Path, Path], example_output_dir: Path) -> None:
    """Copy the stable PNG outputs into ``example_output_dir`` via ``shutil.copy2``."""

    example_output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(outputs[0], example_output_dir / _REPETITION_PNG)
    shutil.copy2(outputs[1], example_output_dir / _NORMAL_PNG)


def render_results_data(
    results: LiveResults,
    output_dir: Path,
    example_output_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Render both PNG charts from an in-memory :class:`LiveResults` bundle.

    Returns the (repetition_path, normal_path) of the saved charts.  When
    ``example_output_dir`` is supplied, the same PNGs are copied there under
    their stable filenames so example-output fixtures stay in sync.
    """

    _apply_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    contained, sample = _aggregate_repetition(results)
    _require_complete(sample, "repetition_prone")

    means = _aggregate_normal(results)
    _require_complete(means, "normal_completion")

    repetition_title, normal_title = chart_titles(results)

    repetition_path = _render_repetition_chart(
        results, contained, sample, repetition_title, output_dir / _REPETITION_PNG
    )
    normal_path = _render_normal_chart(results, means, normal_title, output_dir / _NORMAL_PNG)

    outputs = (repetition_path, normal_path)
    if example_output_dir is not None:
        _copy_to_example_dir(outputs, Path(example_output_dir))
    return outputs


def render_results(
    results_path: Path,
    output_dir: Path,
    example_output_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Load a JSON bundle from ``results_path`` and render both PNG charts."""

    results = load_results(results_path)
    return render_results_data(results, output_dir, example_output_dir)


__all__ = [
    "chart_titles",
    "load_results",
    "render_results",
    "render_results_data",
]
