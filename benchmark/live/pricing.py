"""Pricing catalog and upper-bound estimator for the live benchmark.

Two pricing regimes are supported today:

* **MiniMax** - the only provider with currently-verified rates.  MiniMax-M3
  ships at ``0.30`` USD per million input tokens and ``1.20`` USD per million
  output tokens; the rates are public and stable enough to bake into the
  catalog directly.

* **OpenAI** - we deliberately ship **no** baked-in catalog entries.  Any
  OpenAI model request without overrides must surface a preflight error so the
  caller either confirms the rates themselves or supplies
  ``OPENAI_INPUT_USD_PER_MILLION`` and ``OPENAI_OUTPUT_USD_PER_MILLION``
  explicitly.

The estimator multiplies ``max_steps * scenario_count * 2 * runs`` to model
both the raw baseline and the Soteria-managed run per scenario/run pair.  It
deliberately over-estimates by also using the configured token caps, never the
observed usage; the result is labeled ``upper-bound estimate, not a bill`` so
the JSON dump cannot be mistaken for a settled invoice.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

MINIMAX_M3_INPUT_USD_PER_MILLION: float = 0.30
MINIMAX_M3_OUTPUT_USD_PER_MILLION: float = 1.20

MINIMAX_PRICING_SOURCE_URL = "https://docs.minimax.io/guides/pricing/"

_UPPER_BOUND_LABEL = "upper-bound estimate, not a bill"


class Pricing(BaseModel):
    """Immutable per-million-token pricing for a single provider/model pair."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    input_usd_per_million: float = Field(ge=0)
    output_usd_per_million: float = Field(ge=0)
    currency: str = "USD"
    source_url: str = Field(min_length=1)


class CostEstimate(BaseModel):
    """An upper-bound cost projection for a full live benchmark sweep."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    currency: str = "USD"
    runs: int = Field(ge=1)
    scenario_count: int = Field(ge=1)
    max_steps: int = Field(ge=1)
    total_steps: int = Field(ge=1)
    input_tokens_per_step: int = Field(ge=1)
    output_tokens_per_step: int = Field(ge=1)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0)
    label: str = _UPPER_BOUND_LABEL


def resolve_pricing(provider: str, model: str, environ: Mapping[str, str]) -> Pricing:
    """Return the :class:`Pricing` for ``provider``/``model``.

    MiniMax rates are baked in.  OpenAI rates require explicit overrides via
    ``OPENAI_INPUT_USD_PER_MILLION`` and ``OPENAI_OUTPUT_USD_PER_MILLION``.
    The overrides are not echoed back through the serialized pricing.
    """

    if provider == "minimax":
        if model != "MiniMax-M3":
            raise ValueError(
                f"Unsupported MiniMax model {model!r}; only MiniMax-M3 has verified rates."
            )
        return Pricing(
            provider="minimax",
            model="MiniMax-M3",
            input_usd_per_million=MINIMAX_M3_INPUT_USD_PER_MILLION,
            output_usd_per_million=MINIMAX_M3_OUTPUT_USD_PER_MILLION,
            currency="USD",
            source_url=MINIMAX_PRICING_SOURCE_URL,
        )
    if provider == "openai":
        try:
            input_rate = float(environ["OPENAI_INPUT_USD_PER_MILLION"])
            output_rate = float(environ["OPENAI_OUTPUT_USD_PER_MILLION"])
        except KeyError as exc:
            raise ValueError(
                "OpenAI pricing is not in the verified catalog; supply "
                "OPENAI_INPUT_USD_PER_MILLION and OPENAI_OUTPUT_USD_PER_MILLION "
                "before running the live benchmark."
            ) from exc
        if input_rate < 0 or output_rate < 0:
            raise ValueError("OpenAI USD-per-million overrides must be non-negative.")
        return Pricing(
            provider="openai",
            model=model,
            input_usd_per_million=input_rate,
            output_usd_per_million=output_rate,
            currency="USD",
            source_url="env-overrides",
        )
    raise ValueError(f"Unknown live-benchmark provider {provider!r}")


def estimate_upper_bound(
    pricing: Pricing,
    max_steps: int,
    scenario_count: int,
    runs: int,
    input_tokens_per_step: int,
    output_tokens_per_step: int,
) -> CostEstimate:
    """Project the upper-bound USD cost of the configured live benchmark sweep.

    The estimate assumes both the raw and the Soteria approach run for every
    scenario/run pair, yielding ``max_steps * scenario_count * 2 * runs`` total
    model steps.  Each step is charged the configured token cap so the result
    is strictly an upper bound.
    """

    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if scenario_count < 1:
        raise ValueError("scenario_count must be at least 1")
    if runs < 1:
        raise ValueError("runs must be at least 1")
    if input_tokens_per_step < 1:
        raise ValueError("input_tokens_per_step must be at least 1")
    if output_tokens_per_step < 1:
        raise ValueError("output_tokens_per_step must be at least 1")

    total_steps = max_steps * scenario_count * 2 * runs
    input_tokens = total_steps * input_tokens_per_step
    output_tokens = total_steps * output_tokens_per_step

    cost = Decimal(input_tokens) / Decimal(1_000_000) * Decimal(
        str(pricing.input_usd_per_million)
    ) + Decimal(output_tokens) / Decimal(1_000_000) * Decimal(str(pricing.output_usd_per_million))

    return CostEstimate(
        provider=pricing.provider,
        model=pricing.model,
        currency=pricing.currency,
        runs=runs,
        scenario_count=scenario_count,
        max_steps=max_steps,
        total_steps=total_steps,
        input_tokens_per_step=input_tokens_per_step,
        output_tokens_per_step=output_tokens_per_step,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=float(cost),
        label=_UPPER_BOUND_LABEL,
    )


__all__ = [
    "MINIMAX_M3_INPUT_USD_PER_MILLION",
    "MINIMAX_M3_OUTPUT_USD_PER_MILLION",
    "MINIMAX_PRICING_SOURCE_URL",
    "CostEstimate",
    "Pricing",
    "estimate_upper_bound",
    "resolve_pricing",
]
