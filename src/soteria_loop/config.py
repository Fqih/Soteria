"""SOTERIA_-prefixed environment loader and provider factory.

This module is the single entry point that turns the operator-facing
environment into a ready-to-use ``ModelProvider``. It owns the
``SOTERIA_PROVIDER`` / ``SOTERIA_MODEL`` dispatch table and validates the
required per-provider credentials before any HTTP client is constructed.

The live-benchmark CLI under ``benchmark/live/`` keeps its own legacy env
names (``MODEL_MINIMAX``, ``AUTH_TOKEN``, ``OPENAI_AUTH_TOKEN``, ...) so that
the published reproduction commands stay stable. The new SOTERIA_ names
documented in this module are the official Soteria runtime API.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Literal

ProviderName = Literal["ollama", "minimax", "anthropic", "openai"]
_PROVIDER_NAMES: tuple[ProviderName, ...] = ("ollama", "minimax", "anthropic", "openai")


class ConfigError(ValueError):
    """Raised when the SOTERIA_-prefixed environment is missing or invalid."""


def build_provider_from_env(
    environ: Mapping[str, str] | None = None,
    *,
    max_completion_tokens: int = 1024,
    request_timeout_seconds: float = 30.0,
) -> Any:
    """Build the configured ``ModelProvider`` from the SOTERIA_ environment.

    Raises ``ConfigError`` for every missing required variable so callers can
    surface a single, actionable error message instead of a deferred
    ``KeyError`` deep inside a provider module.
    """

    env: Mapping[str, str] = os.environ if environ is None else environ

    name = env.get("SOTERIA_PROVIDER", "").strip().lower()
    if name not in _PROVIDER_NAMES:
        allowed = ", ".join(_PROVIDER_NAMES)
        raise ConfigError(f"SOTERIA_PROVIDER must be one of {allowed!s}; got {name!r}")

    model = env.get("SOTERIA_MODEL", "").strip()
    if not model:
        raise ConfigError("SOTERIA_MODEL is required")

    if name == "ollama":
        from soteria_loop.providers.ollama import OllamaConfig, OllamaProvider

        ollama_config = OllamaConfig.from_soteria_env(env, fallback_model=model)
        return OllamaProvider(
            ollama_config,
            max_completion_tokens=max_completion_tokens,
            request_timeout_seconds=request_timeout_seconds,
        )

    if name == "minimax":
        from soteria_loop.providers.minimax import MiniMaxConfig, MiniMaxProvider

        minimax_config = MiniMaxConfig.from_soteria_env(env, fallback_model=model)
        return MiniMaxProvider(
            minimax_config,
            max_completion_tokens=max_completion_tokens,
            request_timeout_seconds=request_timeout_seconds,
        )

    if name == "anthropic":
        from soteria_loop.providers.anthropic import AnthropicConfig, AnthropicProvider

        anthropic_config = AnthropicConfig.from_soteria_env(env, fallback_model=model)
        return AnthropicProvider(
            anthropic_config,
            max_completion_tokens=max_completion_tokens,
            request_timeout_seconds=request_timeout_seconds,
        )

    # name == "openai"
    from soteria_loop.providers.openai_compatible import (
        OpenAICompatibleConfig,
        OpenAICompatibleProvider,
    )

    openai_config = OpenAICompatibleConfig.from_soteria_env(env, fallback_model=model)
    return OpenAICompatibleProvider(
        openai_config,
        max_completion_tokens=max_completion_tokens,
        request_timeout_seconds=request_timeout_seconds,
    )


def apply_runtime_overrides(policy_kwargs: dict[str, Any], environ: Mapping[str, str]) -> None:
    """Apply SOTERIA_MAX_TOTAL_TOKENS / SOTERIA_MAX_RUNTIME_SECONDS / SOTERIA_REPEATED_ACTION_LIMIT.

    Mutates ``policy_kwargs`` in place; missing keys are no-ops so the loader
    never blocks startup.
    """

    tokens = environ.get("SOTERIA_MAX_TOTAL_TOKENS", "").strip()
    if tokens:
        try:
            policy_kwargs["max_total_tokens"] = int(tokens)
        except ValueError as exc:
            raise ConfigError(
                f"SOTERIA_MAX_TOTAL_TOKENS must be an integer; got {tokens!r}"
            ) from exc

    runtime = environ.get("SOTERIA_MAX_RUNTIME_SECONDS", "").strip()
    if runtime:
        try:
            policy_kwargs["max_runtime_seconds"] = float(runtime)
        except ValueError as exc:
            raise ConfigError(
                f"SOTERIA_MAX_RUNTIME_SECONDS must be a number; got {runtime!r}"
            ) from exc

    repeats = environ.get("SOTERIA_REPEATED_ACTION_LIMIT", "").strip()
    if repeats:
        try:
            policy_kwargs["repeated_action_limit"] = int(repeats)
        except ValueError as exc:
            raise ConfigError(
                f"SOTERIA_REPEATED_ACTION_LIMIT must be an integer; got {repeats!r}"
            ) from exc


def database_path_from_env(environ: Mapping[str, str] | None = None) -> str | None:
    """Return the configured SQLite path or ``None`` for in-memory storage."""

    env: Mapping[str, str] = os.environ if environ is None else environ
    value = env.get("SOTERIA_DATABASE_PATH", "").strip()
    return value or None


__all__ = [
    "ConfigError",
    "ProviderName",
    "apply_runtime_overrides",
    "build_provider_from_env",
    "database_path_from_env",
]
