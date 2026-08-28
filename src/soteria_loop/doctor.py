"""``soteria-loop doctor`` — verify SOTERIA_ setup without an HTTP call.

Reads ``os.environ`` (or an injected mapping) and reports which
SOTERIA_* variables are present, which are missing, and what the
resulting provider endpoint would be. Never sends a request to the
provider; use :mod:`soteria_loop.chat` for an actual smoke test.

The doctor is intentionally synchronous and dependency-free so it works
in any shell where ``soteria-loop`` is installed, even with no network
access.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import IO, Any

from soteria_loop.config import _PROVIDER_NAMES, ConfigError, build_provider_from_env

_PROVIDER_LABELS = {
    "ollama": "Ollama",
    "openai": "OpenAI-compatible",
    "anthropic": "Anthropic",
    "minimax": "MiniMax",
}

_REQUIRED_BY_PROVIDER: dict[str, tuple[str, ...]] = {
    "ollama": ("SOTERIA_PROVIDER", "SOTERIA_MODEL"),
    "openai": ("SOTERIA_PROVIDER", "SOTERIA_MODEL", "SOTERIA_OPENAI_API_KEY"),
    "anthropic": ("SOTERIA_PROVIDER", "SOTERIA_MODEL", "SOTERIA_ANTHROPIC_API_KEY"),
    "minimax": ("SOTERIA_PROVIDER", "SOTERIA_MODEL", "SOTERIA_MINIMAX_API_KEY"),
}


@dataclass(frozen=True)
class DoctorReport:
    """Structured result of one ``soteria-loop doctor`` invocation."""

    provider: str | None
    model: str | None
    base_url: str | None
    endpoint: str | None
    api_style: str | None
    has_api_key: bool
    missing_vars: tuple[str, ...]
    config_error: str | None
    extra_vars: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """True when every required variable is set and provider config is buildable."""

        return not self.missing_vars and self.config_error is None


def _provider_for(env: Mapping[str, str]) -> str | None:
    raw = env.get("SOTERIA_PROVIDER", "").strip().lower()
    if raw in _PROVIDER_NAMES:
        return raw
    return None


def _endpoint_for(env: Mapping[str, str], provider: str) -> tuple[str | None, str | None]:
    """Return ``(endpoint, api_style)`` by introspecting each provider's config class."""

    fallback_model = env.get("SOTERIA_MODEL", "x")

    if provider == "ollama":
        from soteria_loop.providers.ollama import OllamaConfig

        try:
            ollama_cfg: Any = OllamaConfig.from_soteria_env(env, fallback_model=fallback_model)
        except (ValueError, KeyError):
            return None, None
        return ollama_cfg.endpoint, None

    if provider == "minimax":
        from soteria_loop.providers.minimax import MiniMaxConfig

        try:
            minimax_cfg: Any = MiniMaxConfig.from_soteria_env(env, fallback_model=fallback_model)
        except (ValueError, KeyError):
            return None, env.get("SOTERIA_MINIMAX_API_STYLE", "anthropic")
        return minimax_cfg.endpoint, minimax_cfg.api_style

    if provider == "anthropic":
        from soteria_loop.providers.anthropic import AnthropicConfig

        try:
            anthropic_cfg: Any = AnthropicConfig.from_soteria_env(
                env, fallback_model=fallback_model
            )
        except (ValueError, KeyError):
            return None, None
        return anthropic_cfg.endpoint, None

    if provider == "openai":
        from soteria_loop.providers.openai_compatible import OpenAICompatibleConfig

        try:
            openai_cfg: Any = OpenAICompatibleConfig.from_soteria_env(
                env, fallback_model=fallback_model
            )
        except (ValueError, KeyError):
            return None, None
        return openai_cfg.endpoint, None

    return None, None


def run_doctor(environ: Mapping[str, str] | None = None) -> DoctorReport:
    """Inspect ``environ`` (defaults to ``os.environ``) and return a report.

    Never raises; surfaces every failure as a field on :class:`DoctorReport`.
    """

    env: Mapping[str, str] = os.environ if environ is None else environ

    provider = _provider_for(env)
    model = env.get("SOTERIA_MODEL", "").strip() or None
    base_url: str | None = None
    endpoint: str | None = None
    api_style: str | None = None
    has_api_key = False
    missing: list[str] = []
    config_error: str | None = None

    if provider is None:
        missing.append("SOTERIA_PROVIDER")
        missing.extend(_REQUIRED_BY_PROVIDER["ollama"][1:])
    else:
        required = _REQUIRED_BY_PROVIDER[provider]
        for var in required:
            if not env.get(var, "").strip():
                missing.append(var)

        api_key_var = f"SOTERIA_{provider.upper()}_API_KEY"
        has_api_key = bool(env.get(api_key_var, "").strip())

        base_url_key = f"SOTERIA_{provider.upper()}_BASE_URL"
        base_url = env.get(base_url_key, "").strip() or None

        endpoint, api_style = _endpoint_for(env, provider)

        if not missing:
            try:
                build_provider_from_env(env)
            except ConfigError as exc:
                config_error = str(exc)

    extra = tuple(sorted(k for k in env if k.startswith("SOTERIA_") and k not in set(missing)))

    return DoctorReport(
        provider=provider,
        model=model,
        base_url=base_url,
        endpoint=endpoint,
        api_style=api_style,
        has_api_key=has_api_key,
        missing_vars=tuple(missing),
        config_error=config_error,
        extra_vars=extra,
    )


def render_report(report: DoctorReport, *, out: IO[str]) -> None:
    """Print a human-readable report to ``out`` without leaking secrets."""

    if report.provider is None:
        out.write("SOTERIA provider: (unset)\n")
        out.write("  set SOTERIA_PROVIDER to one of: " + ", ".join(_PROVIDER_NAMES) + "\n")
    else:
        label = _PROVIDER_LABELS.get(report.provider, report.provider)
        out.write(f"SOTERIA provider: {label}\n")

    if report.model:
        out.write(f"SOTERIA model: {report.model}\n")
    else:
        out.write("SOTERIA model: (unset)\n")

    if report.base_url:
        out.write(f"base URL: {report.base_url}\n")
    elif report.provider is not None:
        out.write("base URL: (provider default)\n")

    if report.api_style:
        out.write(f"API style: {report.api_style}\n")

    if report.endpoint:
        out.write(f"endpoint: {report.endpoint}\n")

    out.write(f"API key configured: {report.has_api_key}\n")

    if report.missing_vars:
        out.write("missing variables:\n")
        for var in report.missing_vars:
            out.write(f"  - {var}\n")

    if report.config_error:
        out.write(f"config error: {report.config_error}\n")

    if report.ok:
        out.write("\nResult: OK. Provider config is buildable.\n")
        out.write(
            "Note: this only checks the configuration; it does not call "
            "the provider. Run `soteria-loop chat` for an end-to-end smoke test.\n"
        )
    else:
        out.write("\nResult: NOT READY. Fix the issues above, then retry.\n")

    out.flush()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: parse args, run doctor, return exit status.

    When called from another entry point (e.g. the ``soteria-loop`` CLI
    dispatcher) pass an empty ``argv`` so the parser does not re-read
    ``sys.argv`` and reject the parent command's tail.
    """

    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="soteria-loop doctor",
        description="Verify SOTERIA_ provider configuration without HTTP.",
    )
    parser.parse_args(argv if argv is not None else sys.argv[1:])
    report = run_doctor()
    render_report(report, out=sys.stdout)
    return 0 if report.ok else 1


__all__ = [
    "DoctorReport",
    "main",
    "render_report",
    "run_doctor",
]
