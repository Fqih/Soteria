---
name: Bug report
about: Report a bug in Avo
title: "[bug] "
labels: bug
---

**Describe the bug**
A clear description of what happens vs. what you expected.

**Reproduction**
Minimal code or commands that reproduce the issue. Include provider
(`AVO_PROVIDER`) and model (`AVO_MODEL`) where applicable.

**Environment**
- Avo version (`pip show avo` or `python -c "import avo; print(avo.__version__)"`)
- Python version
- Provider (Ollama / MiniMax / Anthropic / OpenAI / OpenAI-compatible)
- OS

**Logs / trace**
If available, paste a `avo --database avo.db runs inspect RUN_ID`
trace excerpt.

**Notes**
Anything else that might be relevant (e.g. was a paid model invoked,
is the bug reproducible on the deterministic `FakeProvider`).