# Contributing to Soteria

Soteria welcomes focused changes that improve boundedness, observability,
recovery, or safety without expanding the project into a general agent
framework.

## Environment setup

Use Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

No API key or network access is needed after development dependencies are
installed.

## Local checks

Run the same gates used by CI:

```bash
ruff check .
ruff format --check .
mypy src/soteria_loop
pytest
python -m build
```

To measure core coverage:

```bash
coverage run -m pytest
coverage report
```

Run all offline examples and regenerate the benchmark when runtime behavior
changes:

```bash
python examples/basic_agent.py
python examples/repeated_action.py
python examples/resume_after_interrupt.py
python benchmark/run_benchmark.py
```

## Contribution workflow

1. Open an issue for substantial API or event-schema changes.
2. Create a focused branch and keep commits reviewable.
3. Preserve append-only history, state validation, and terminal-event
   invariants.
4. Add deterministic behavioral tests for every behavior change or bug fix.
5. Update `README.md`, `DESIGN.md`, examples, and benchmark results
   when user-visible contracts change.
6. Run every local check before opening a pull request.

Avoid real API calls in the default test suite. Error messages should name the
run, tool, provider operation, or database involved and explain what the user
can do next.

## Pull requests

Describe the execution behavior before and after the change, the failure modes
considered, and the checks run. If the change affects persistence or resume,
include a close/reopen test and a failure-injection test at the relevant
checkpoint boundary.

By participating, you agree to follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
