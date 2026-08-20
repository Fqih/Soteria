---
name: Bug report
about: Report a bug in Soteria
title: "[bug] "
labels: bug
---

**Describe the bug**
A clear description of what happens vs. what you expected.

**Reproduction**
Minimal code or commands that reproduce the issue. Include provider
(`SOTERIA_PROVIDER`) and model (`SOTERIA_MODEL`) where applicable.

**Environment**
- Soteria version (`pip show soteria-loop` or `python -c "import soteria_loop; print(soteria_loop.__version__)"`)
- Python version
- Provider (Ollama / MiniMax / Anthropic / OpenAI / OpenAI-compatible)
- OS

**Logs / trace**
If available, paste a `soteria-loop --database soteria_loop.db runs inspect RUN_ID`
trace excerpt.

**Notes**
Anything else that might be relevant (e.g. was a paid model invoked,
is the bug reproducible on the deterministic `FakeProvider`).