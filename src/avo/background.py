"""Background task manager for the chat REPL.

A long-running user prompt can be submitted with a trailing ``&`` and
continue executing while the operator types the next message. The
manager owns the asyncio tasks, tracks their state, and exposes a
narrow surface for ``/jobs``, ``/job``, and ``/cancel`` slash commands.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from avo.chat import ChatContext


_JOB_STATUS_PENDING = "pending"
_JOB_STATUS_RUNNING = "running"
_JOB_STATUS_COMPLETED = "completed"
_JOB_STATUS_FAILED = "failed"
_JOB_STATUS_CANCELLED = "cancelled"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _new_job_id() -> str:
    return uuid.uuid4().hex[:8]


@dataclass
class Job:
    """One tracked background task."""

    job_id: str
    task_text: str
    status: str = _JOB_STATUS_PENDING
    created_at: datetime = field(default_factory=_utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    run_id: str | None = None
    output: str | None = None
    error: str | None = None

    @property
    def is_terminal(self) -> bool:
        """Whether the job has settled into a non-running state."""

        return self.status in (
            _JOB_STATUS_COMPLETED,
            _JOB_STATUS_FAILED,
            _JOB_STATUS_CANCELLED,
        )


class BackgroundJobManager:
    """Owns asyncio tasks for background chat turns."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()

    @property
    def running_count(self) -> int:
        """Number of jobs currently executing."""

        return sum(1 for job in self._jobs.values() if not job.is_terminal)

    def submit(self, ctx: ChatContext, task_text: str) -> Job:
        """Create a new job, spawn its asyncio task, return the Job."""

        job = Job(job_id=_new_job_id(), task_text=task_text)
        self._jobs[job.job_id] = job
        async_task = asyncio.create_task(self._drive(job, ctx), name=f"avo-job-{job.job_id}")
        self._tasks[job.job_id] = async_task
        async_task.add_done_callback(lambda _t: self._tasks.pop(job.job_id, None))
        return job

    async def cancel(self, job_id: str) -> bool:
        """Cancel a running job. Returns False if the id is unknown."""

        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.is_terminal:
                return False
            task = self._tasks.get(job_id)
        if task is not None:
            task.cancel()
        return True

    def list_jobs(self) -> tuple[Job, ...]:
        """Return a snapshot of all jobs sorted by creation time."""

        return tuple(sorted(self._jobs.values(), key=lambda job: job.created_at))

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    async def wait_all(self) -> None:
        """Await all still-running tasks (used at REPL shutdown)."""

        pending = [task for task in self._tasks.values() if not task.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def _drive(self, job: Job, ctx: ChatContext) -> None:
        from avo.app_tools.file_tools import bind_workspace

        job.status = _JOB_STATUS_RUNNING
        job.started_at = _utc_now()
        try:
            with bind_workspace(ctx.workspace):
                result = await ctx.runtime.run(job.task_text)
            job.run_id = result.run_id
            if result.status.value == "completed":
                job.status = _JOB_STATUS_COMPLETED
                job.output = result.output
            elif result.status.value == "cancelled":
                job.status = _JOB_STATUS_CANCELLED
                job.error = result.error
            else:
                job.status = _JOB_STATUS_FAILED
                job.error = result.error or result.stop_reason.value
        except asyncio.CancelledError:
            job.status = _JOB_STATUS_CANCELLED
            job.error = "cancelled by operator"
            raise
        except Exception as exc:
            job.status = _JOB_STATUS_FAILED
            job.error = f"{type(exc).__name__}: {exc}"
        finally:
            job.finished_at = _utc_now()


def render_job_row(job: Job) -> str:
    """Return a one-line, fixed-width representation of ``job``."""

    status = f"{job.status:<10}"
    started = job.started_at.strftime("%H:%M:%S") if job.started_at else "-"
    preview = job.task_text if len(job.task_text) <= 40 else job.task_text[:37] + "..."
    return f"{job.job_id}  {status} {started}  {preview}"


def render_job_detail(job: Job) -> str:
    """Return a multi-line detailed view of ``job``."""

    lines = [f"job     : {job.job_id}", f"status  : {job.status}"]
    if job.run_id:
        lines.append(f"run_id  : {job.run_id}")
    lines.append(f"task    : {job.task_text}")
    if job.started_at:
        lines.append(f"started : {job.started_at.isoformat()}")
    if job.finished_at:
        lines.append(f"finished: {job.finished_at.isoformat()}")
    if job.output:
        lines.append("output  :")
        lines.append(job.output.rstrip())
    if job.error:
        lines.append(f"error   : {job.error}")
    return "\n".join(lines)


__all__ = [
    "BackgroundJobManager",
    "Job",
    "render_job_detail",
    "render_job_row",
]
