"""pytest plugin entry point that registers the ``--run-live`` flag.

Registered via ``pyproject.toml`` ``entry_points`` so ``pytest --run-live``
becomes a first-class CLI option without needing a ``conftest.py`` at
the repo root. Without the flag the integration tests stay skipped by
the ``--ignore`` default in ``[tool.pytest.ini_options]`` — this hook
re-enables them when the operator opts in.
"""

from __future__ import annotations

from pathlib import Path

# Locate the integration test directory *relative to this plugin* so the
# plugin works regardless of pytest's rootdir resolution.
INTEGRATION_DIR = Path(__file__).resolve().parent.parent.parent / "tests" / "integration"


def pytest_addoption(parser: object) -> None:
    """Register ``--run-live`` so opt-in integration tests can be enabled."""

    group = getattr(parser, "getgroup", lambda _name: parser)("hernness")
    add_option = getattr(group, "addoption", None)
    if add_option is None:  # pragma: no cover - argparse fallback path
        add_option = parser.add_option  # type: ignore[attr-defined]
    add_option(
        "--run-live",
        action="store_true",
        default=False,
        help="Run live HTTP integration tests against real provider endpoints.",
    )


def pytest_collection_modifyitems(config: object, items: list[object]) -> None:
    """Skip live tests unless ``--run-live`` was passed."""

    run_live = bool(getattr(config, "getoption", lambda _n: False)("--run-live"))
    if run_live:
        return
    skip = _build_skipper()
    for item in items:
        path = getattr(item, "fspath", None) or getattr(item, "path", None)
        if path is None:
            continue
        try:
            is_integration = INTEGRATION_DIR in Path(str(path)).resolve().parents
        except OSError:  # pragma: no cover
            is_integration = False
        if is_integration:
            mark = getattr(item, "add_marker", None)
            if mark is not None:
                mark(skip)


def _build_skipper() -> object:
    import pytest

    return pytest.mark.skip(reason="--run-live not enabled; offline suite skips live tests")
