"""Tests for ``hernness doctor``."""

from __future__ import annotations

import io

from hernness.doctor import render_report, run_doctor


def test_doctor_empty_env_reports_missing_provider() -> None:
    report = run_doctor({})
    assert report.provider is None
    assert "HERNNESS_PROVIDER" in report.missing_vars
    assert report.ok is False
    assert report.config_error is None


def test_doctor_unknown_provider_reports_missing() -> None:
    report = run_doctor({"HERNNESS_PROVIDER": "azure", "HERNNESS_MODEL": "x"})
    assert report.provider is None
    assert "HERNNESS_PROVIDER" in report.missing_vars
    assert report.ok is False


def test_doctor_ollama_only_requires_provider_and_model() -> None:
    report = run_doctor(
        {
            "HERNNESS_PROVIDER": "ollama",
            "HERNNESS_MODEL": "llama3.1",
        }
    )
    assert report.provider == "ollama"
    assert report.model == "llama3.1"
    assert report.has_api_key is False
    assert report.missing_vars == ()
    assert report.ok is True
    assert report.endpoint is not None
    assert report.endpoint.endswith("/api/chat")


def test_doctor_minimax_requires_api_key() -> None:
    report = run_doctor(
        {
            "HERNNESS_PROVIDER": "minimax",
            "HERNNESS_MODEL": "MiniMax-M3",
        }
    )
    assert report.provider == "minimax"
    assert report.missing_vars == ("HERNNESS_MINIMAX_API_KEY",)
    assert report.ok is False


def test_doctor_minimax_full_url_is_not_double_suffixed() -> None:
    report = run_doctor(
        {
            "HERNNESS_PROVIDER": "minimax",
            "HERNNESS_MODEL": "MiniMax-M3",
            "HERNNESS_MINIMAX_API_KEY": "k",
            "HERNNESS_MINIMAX_BASE_URL": "https://api.minimax.io/anthropic",
            "HERNNESS_MINIMAX_API_STYLE": "anthropic",
        }
    )
    assert report.ok is True
    assert report.endpoint == "https://api.minimax.io/anthropic/v1/messages"
    assert report.api_style == "anthropic"
    assert report.has_api_key is True


def test_doctor_anthropic_requires_api_key() -> None:
    report = run_doctor(
        {
            "HERNNESS_PROVIDER": "anthropic",
            "HERNNESS_MODEL": "claude-sonnet-4-6",
        }
    )
    assert report.missing_vars == ("HERNNESS_ANTHROPIC_API_KEY",)
    assert report.ok is False


def test_doctor_openai_requires_api_key() -> None:
    report = run_doctor(
        {
            "HERNNESS_PROVIDER": "openai",
            "HERNNESS_MODEL": "gpt-5.6",
        }
    )
    assert report.missing_vars == ("HERNNESS_OPENAI_API_KEY",)
    assert report.ok is False


def test_doctor_full_anthropic_setup_ok() -> None:
    report = run_doctor(
        {
            "HERNNESS_PROVIDER": "anthropic",
            "HERNNESS_MODEL": "claude-sonnet-4-6",
            "HERNNESS_ANTHROPIC_API_KEY": "ant-key",
        }
    )
    assert report.ok is True
    assert report.endpoint is not None


def test_render_report_ok_prints_endpoint_and_no_missing() -> None:
    report = run_doctor(
        {
            "HERNNESS_PROVIDER": "ollama",
            "HERNNESS_MODEL": "llama3.1",
        }
    )
    out = io.StringIO()
    render_report(report, out=out)
    text = out.getvalue()
    assert "Result: OK." in text
    assert "missing variables" not in text


def test_render_report_not_ok_lists_missing() -> None:
    report = run_doctor({"HERNNESS_PROVIDER": "minimax", "HERNNESS_MODEL": "x"})
    out = io.StringIO()
    render_report(report, out=out)
    text = out.getvalue()
    assert "HERNNESS_MINIMAX_API_KEY" in text
    assert "Result: NOT READY." in text


def test_render_report_redacts_api_key() -> None:
    secret = "sk-THISISTOPSECRET-1234"
    report = run_doctor(
        {
            "HERNNESS_PROVIDER": "openai",
            "HERNNESS_MODEL": "gpt-5.6",
            "HERNNESS_OPENAI_API_KEY": secret,
        }
    )
    out = io.StringIO()
    render_report(report, out=out)
    text = out.getvalue()
    assert secret not in text
