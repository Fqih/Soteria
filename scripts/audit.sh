#!/usr/bin/env bash
# Run the bandit and pip-audit gates locally.
#
# Mirrors the CI steps in `.github/workflows/ci.yml`. Use this script
# before opening a pull request to catch security regressions early.

set -euo pipefail

echo "Running bandit on src/avo (excluding mcp_servers)..."
bandit -r src/avo -c pyproject.toml --severity-level medium

echo
echo "Running pip-audit on the current environment..."
# ``pip-audit --strict`` fails on any known vulnerability; we exclude
# the noise about packages that are not on PyPI by listing only our
# project tree.
pip-audit --strict --requirement <(pip-compile pyproject.toml 2>/dev/null || true)
