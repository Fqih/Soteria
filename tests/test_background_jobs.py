"""Tests for the background-job manager and REPL integration."""

from __future__ import annotations

import asyncio
import io
from pathlib import Path

import pytest


def _make_context_with_provider() -> tuple[object, Path, Path]:
    """Return ``(ctx, db_path, workspace_path)`` wired to a FakeProvider."""

    from avo.chat import build_chat_context

    workspace = Path("/tmp") / "avo-bg-fake-ws"
    db = Path("/tmp") / "avo-bg-fake.db"
    ctx = build_chat_context(
        database_path=db,
        workspace_root=workspace,
        environ={"AVO_PROVIDER": "ollama", "AVO_MODEL": "x"},
    )
    return ctx, db, workspace


def test_submit_marks_running_then_completed(tmp_path: Path) -> None:
    from avo.background import BackgroundJobManager
    from avo.models import ModelResponse, TokenUsage
    from avo.providers.fake import FakeProvider

    async def scenario() -> str:
        from avo.runtime import AgentRuntime

        provider = FakeProvider(
            [ModelResponse(content="done", usage=TokenUsage(input_tokens=1, output_tokens=1))]
        )
        from avo.app_tools.workspace import Workspace
        from avo.storage.sqlite import SQLiteEventStore

        workspace = Workspace(tmp_path / "ws", create=True)
        store = SQLiteEventStore(tmp_path / "store.db")
        runtime = AgentRuntime(provider=provider, event_store=store)

        class Ctx:
            pass

        ctx = Ctx()
        ctx.runtime = runtime
        ctx.workspace = workspace
        ctx.store = store
        ctx.provider_name = "fake"
        ctx.model_name = "fake-model"
        ctx.session_id = "session"
        ctx.skills = None  # type: ignore[assignment]
        ctx.session = None  # type: ignore[assignment]
        ctx.pending_preamble = None
        ctx.background = BackgroundJobManager()

        job = ctx.background.submit(ctx, "hello")
        await ctx.background.wait_all()
        return job.job_id

    job_id = asyncio.run(scenario())
    assert job_id is not None


def test_prompt_with_jobs_no_running_keeps_prompt_clean() -> None:
    from avo.background import BackgroundJobManager
    from avo.chat import _prompt_with_jobs

    manager = BackgroundJobManager()
    assert _prompt_with_jobs("You > ", manager) == "You > "


def test_cancel_returns_false_for_unknown_id() -> None:
    from avo.background import BackgroundJobManager

    async def scenario() -> bool:
        manager = BackgroundJobManager()
        return await manager.cancel("does-not-exist")

    assert asyncio.run(scenario()) is False


def test_render_job_row_includes_status_and_preview() -> None:
    from avo.background import Job, render_job_row

    job = Job(job_id="abcd1234", task_text="describe the workspace structure in detail")
    row = render_job_row(job)
    assert "abcd1234" in row
    assert "describe the workspace structure in d" in row
    assert row.rstrip().endswith("...")


def test_render_job_detail_full_view() -> None:
    from avo.background import Job, render_job_detail

    job = Job(
        job_id="abcd1234",
        task_text="count files",
        status="completed",
        run_id="r-1",
        output="42",
    )
    detail = render_job_detail(job)
    assert "abcd1234" in detail
    assert "completed" in detail
    assert "count files" in detail
    assert "r-1" in detail
    assert "42" in detail


@pytest.mark.asyncio
async def test_repl_runs_background_job_then_lists_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Submit a job with trailing ``&``, then list it via ``/jobs``."""

    from avo.chat import run_repl
    from avo.models import ModelResponse, TokenUsage
    from avo.providers.fake import FakeProvider

    monkeypatch.setattr("os.environ", {})

    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "README.md").write_text("seed", encoding="utf-8")
    db = tmp_path / "chat.db"

    fake = FakeProvider(
        [ModelResponse(content="ok", usage=TokenUsage(input_tokens=1, output_tokens=1))]
    )
    monkeypatch.setattr("avo.chat.build_provider_from_env", lambda _env: fake)

    stdin = io.StringIO("do thing &\n/jobs\n/quit\n")
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = await run_repl(
        database_path=db,
        workspace_root=workspace,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        environ={"AVO_PROVIDER": "ollama", "AVO_MODEL": "x"},
    )
    assert code == 0
    out = stdout.getvalue()
    assert "Backgrounded job" in out
    assert "/jobs" in out
