# Live benchmark

The live benchmark is an **opt-in** harness that spends real money against a
paid LLM provider. It exercises the same three scenarios the deterministic
benchmark drives offline, but against a real chat-completion endpoint, so the
results reflect actual token usage, latency, and policy-stop behaviour rather
than synthetic traces. It is a **case study**, not a benchmark claim: a real
provider, a real key, and a small number of runs.

The CLI refuses to run without an explicit cost-consent signal. Pricing is
projected up-front from a published catalog (or operator-supplied overrides)
so you see an upper bound before any HTTP traffic is generated.

## What it produces

For every `--runs N` invocation the CLI writes one JSON bundle named
`live_results_<UTC>.json` under `--output-dir` (default
`benchmark/live/results/`). That bundle is the **source of truth**:

- `provider`, `api_style`, `model`, `recorded_at`, `runs`, `records[]`
- One record per `(scenario, approach, run_index)` row with steps, duration,
  token usage, safety-fence flags, status/stop reason, and any captured
  unexpected error.

The two charts in the case study are rendered **from this JSON**; nothing is
hand-entered. Re-rendering is deterministic for a given bundle:

| Chart file | What it shows | Derivation |
| --- | --- | --- |
| `repetition_containment.png` | Grouped bars of contained vs escaped counts per approach on the `repetition_prone` scenario. | `_aggregate_repetition` groups records by approach and counts `loop_contained` (Soteria policy stops) versus manual `step_cap_hit` (raw baseline). |
| `normal_completion_comparison.png` | Side-by-side mean steps and mean wall-clock duration on the `normal_completion` scenario. | `_aggregate_normal` averages `steps` and `duration_seconds` per approach across the n-run slice. |

The titles are derived from the bundle (`"Repetition containment — {provider}
/ {model} (n={size} runs per approach)"` and the analogue for the normal
scenario) so the same renderer can chart any provider's JSON output.

## Raw outcome classification

The raw baseline records exactly one of three `outcome` values:

- `completed` — the loop finished under the manual step cap.
- `hit_manual_step_cap` — the loop was bounded by `--manual-step-cap` (the
  raw baseline has no policy-driven fences).
- `error` — the run raised an exception (expected provider/tool errors stay in
  `expected_error_type`; unexpected ones land in `unexpected_error_type`).

The Soteria approach records `status` and `stop_reason` instead:
`completed`, `repeated_action` (policy containment), or any other RunState
value. The renderer's containment rule is `approach=raw -> manual_step_cap_hit`,
and `approach=soteria_loop -> soteria_loop_contained(record)`, so the raw loop's
manual safety cap is treated as containment for *charting* but is never
attributed to Soteria.

## Cost-consent gate

Two equivalent channels grant consent before the CLI builds any provider:

- The CLI flag `--i-understand-this-costs-money`.
- The environment variable `SOTERIA_I_UNDERSTAND_THIS_COSTS_MONEY` set to one
  of `1`, `true`, or `yes` (case-insensitive).

If neither is present, the CLI exits with status `2` and the message names the
exact flag. No provider module is imported, no HTTP call is made, and no JSON
bundle is written.

## Pricing behavior

The CLI prints one pre-flight line per invocation:

```
Pre-flight estimate for provider=<p> model=<m>: ~$<cost> USD across n=<runs>
run(s) (<steps> steps total) - upper-bound estimate, not a bill
```

The estimator (`benchmark.live.pricing.estimate_upper_bound`) deliberately
**over**-estimates by assuming both raw and Soteria approaches run for every
`(scenario, run_index)` pair at the configured token caps. The default three
runs therefore prints an upper bound, **not** an actual bill. Treat the number
as a planning aid, not a settled invoice.

| Provider | Catalog source | Override env vars |
| --- | --- | --- |
| `minimax` (MiniMax-M3) | Baked into the CLI from the public pricing page: 0.30 USD per million input tokens, 1.20 USD per million output tokens. See `MINIMAX_PRICING_SOURCE_URL` in `benchmark/live/pricing.py`. | None needed. |
| `openai` | Operator-supplied only. The CLI ships **no** baked-in OpenAI rates. | `OPENAI_INPUT_USD_PER_MILLION`, `OPENAI_OUTPUT_USD_PER_MILLION`. Both must be set; the CLI fails closed otherwise. |

API-key acquisition and pricing pages (no credential values are reproduced
here):

- MiniMax-M3: see the pricing page linked from
  `MINIMAX_PRICING_SOURCE_URL` in `benchmark/live/pricing.py`. Generate an
  API key through the provider's normal developer-console flow.
- OpenAI: pricing and key issuance are surfaced from the OpenAI dashboard;
  the OpenAI provider module expects `OPENAI_API_KEY`.

## Environment variables

### MiniMax (default)

The MiniMax provider supports both the **OpenAI-compatible** endpoint and the
**Anthropic-compatible** endpoint. The CLI picks the variant from
`MINIMAX_API_STYLE` (`openai` or `anthropic`) or from the `--api-style`
flag, which overrides the env var for one invocation.

OpenAI-compatible:

```bash
MODEL_MINIMAX=MiniMax-M3 \
BASE_URL=https://api.minimax.io/ \
MINIMAX_API_STYLE=openai \
OPENAI_AUTH_TOKEN="$OPENAI_AUTH_TOKEN" \
python -m benchmark.live.run_live_benchmark --provider minimax --runs 3 --i-understand-this-costs-money
```

Anthropic-compatible (note the `AUTH_TOKEN` variable, not `OPENAI_AUTH_TOKEN`):

```bash
MODEL_MINIMAX=MiniMax-M3 \
BASE_URL=https://api.minimax.io/ \
MINIMAX_API_STYLE=anthropic \
AUTH_TOKEN="$AUTH_TOKEN" \
python -m benchmark.live.run_live_benchmark --provider minimax --runs 3 --i-understand-this-costs-money
```

Required env vars for `--provider minimax`: `MODEL_MINIMAX`, `BASE_URL`. The
auth token name is style-dependent: `OPENAI_AUTH_TOKEN` for
`MINIMAX_API_STYLE=openai`, `AUTH_TOKEN` for `MINIMAX_API_STYLE=anthropic`.

### OpenAI

```bash
OPENAI_MODEL=gpt-4o-mini \
OPENAI_API_KEY="$OPENAI_API_KEY" \
OPENAI_BASE_URL=https://api.openai.com/v1 \
OPENAI_INPUT_USD_PER_MILLION=0.15 \
OPENAI_OUTPUT_USD_PER_MILLION=0.60 \
python -m benchmark.live.run_live_benchmark --provider openai --runs 3 --i-understand-this-costs-money
```

Required env vars for `--provider openai`: `OPENAI_MODEL`, `OPENAI_API_KEY`,
plus both USD-per-million overrides; the CLI fails closed without them.

## Full command

For convenience, the complete flags the CLI accepts:

```text
--provider {minimax,openai}    default: minimax
--runs INT                     default: 3 (positive)
--sleep-seconds FLOAT          default: 1.0
--manual-step-cap INT          default: 6
--max-completion-tokens INT    default: 1024
--input-tokens-per-step INT    default: 2048
--timeout-seconds FLOAT        default: 300.0
--output-dir PATH              default: benchmark/live/results
--model TEXT                   override the resolved provider model
--api-style {openai,anthropic} default: from MINIMAX_API_STYLE (minimax only)
--i-understand-this-costs-money
                               explicit opt-in flag
```

## Optional dependency

The live benchmark is gated behind an optional extra so the core package and
the deterministic benchmark stay install-free:

```bash
pip install soteria-loop[live-benchmark]
```

The optional extra pulls in the live provider modules under
`examples.live_providers` (`minimax_provider`, `openai_provider`) and the
matplotlib dependency used by the renderer. The CLI fails closed with a
clear error if those modules are missing.

## Illustrative fixture

`benchmark/live/example_output/` contains a checked-in example_results.json
plus the two rendered charts. The provider is `illustrative`, the model is
`MiniMax-M3`, and the bundle covers 15 records (n=3 runs per applicable
scenario and approach). See the case-study section in
[`/README.md`](../../README.md#live-agent-case-study-minimax-m3) for the prose
summary and per-chart link.
