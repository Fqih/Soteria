# Deterministic Benchmark Results

This benchmark uses only scripted FakeProvider decisions. It measures runtime 
containment, persistence, and recovery behavior—not model intelligence.

## Aggregate metrics

| Metric | Minimal raw loop | Avo |
|---|---:|---:|
| Task Completion Rate | 25.0% | 25.0% |
| Loop Containment Rate | 0.0% | 100.0% |
| Resume Success Rate | 0.0% | 100.0% |
| Duplicate Side-Effect Count | 6 | 0 |
| Terminal Completeness | 0.0% | 100.0% |
| Trace Integrity Rate | 0.0% | 100.0% |
| Mean Steps | 5.00 | 1.88 |
| Mean Runtime | 0.17 ms | 6.48 ms |

## Scenario outcomes

| Scenario | Raw completed | Avo completed | Avo contained | Avo resumed |
|---|---:|---:|---:|---:|
| normal_completion | true | true | false | false |
| repeated_tool_calls | false | false | true | false |
| repeated_observations | false | false | true | false |
| provider_failures | false | false | true | false |
| tool_failures | false | false | true | false |
| interruption_after_side_effect | true | true | false | true |
| malformed_arguments | false | false | true | false |
| budget_exhaustion | false | false | true | false |

The raw loop is stopped by an external six-step benchmark harness so runaway 
scenarios terminate during measurement; this external cap is not counted as 
runtime containment. Terminal completeness requires a persisted terminal state 
and explicit reason. Trace integrity requires JSON serialization and gap-free 
event ordering. Wall-clock timings vary by machine.

Reproduce with:

    python benchmark/run_benchmark.py
