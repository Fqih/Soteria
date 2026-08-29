---
name: Bug report
about: Report a bug in Hernness
title: "[bug] "
labels: bug
---

**Describe the bug**
A clear description of what happens vs. what you expected.

**Reproduction**
Minimal code or commands that reproduce the issue. Include provider
(`HERNNESS_PROVIDER`) and model (`HERNNESS_MODEL`) where applicable.

**Environment**
- Hernness version (`pip show hernness` or `python -c "import hernness; print(hernness.__version__)"`)
- Python version
- Provider (Ollama / MiniMax / Anthropic / OpenAI / OpenAI-compatible)
- OS

**Logs / trace**
If available, paste a `hernness --database hernness.db runs inspect RUN_ID`
trace excerpt.

**Notes**
Anything else that might be relevant (e.g. was a paid model invoked,
is the bug reproducible on the deterministic `FakeProvider`).