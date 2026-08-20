---
name: Pull request
about: Propose changes to the Soteria runtime or its tooling
title: ""
labels:---

## Summary

<!-- One-paragraph description of what this PR does and why. -->

## Linked issues

<!-- Link any issue this closes, e.g. "Closes #42". -->

## Quality gates

- [ ] `make lint`
- [ ] `make format-check`
- [ ] `make typecheck`
- [ ] `make test` — 219+ tests passing
- [ ] No new runtime dependency unless justified in the PR description
- [ ] Offline tests added for any new public surface

## Provider or state-machine changes

If your PR touches `src/soteria_loop/providers/` or
`src/soteria_loop/runtime.py`, confirm:

- [ ] No change to the public `ModelProvider` Protocol contract
- [ ] No change to the `STOP_REASONS_BY_STATE` mapping without
  updating `src/soteria_loop/state.py` and the README stop-reasons list
- [ ] No change to event ordering or sequence invariants

## Docs

- [ ] README updated if the public API changed
- [ ] `project.md` updated if a new module was added
- [ ] `CHANGELOG.md` entry under `[Unreleased]`