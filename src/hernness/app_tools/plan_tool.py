"""``submit_plan`` tool: mark the active run as having a plan.

The tool is intentionally simple — it stores ``plan_text`` in the
permission module's plan-submitted tracker so that subsequent
mutating tool calls under :class:`PermissionMode.PLAN` are allowed
without re-asking the operator. The tool itself always returns
``True`` from the approval callback (see ``build_approval_callback``)
because the act of submitting a plan is the plan.

The plan text is also returned in the tool result so the runtime
records it in the event log and the operator can later inspect what
was authorised.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, JsonValue

from hernness import FunctionTool as PublicFunctionTool

from ..permissions import active_run_id, is_plan_submitted, mark_plan_submitted


class SubmitPlanArguments(BaseModel):
    """Arguments for the ``submit_plan`` tool."""

    plan_text: str = Field(min_length=1, description="Human-readable plan summary")


async def _submit_plan(arguments: SubmitPlanArguments) -> JsonValue:
    run_id = active_run_id()
    if run_id is not None:
        mark_plan_submitted(run_id)
    payload: dict[str, JsonValue] = {
        "run_id": run_id,
        "plan_text": arguments.plan_text,
        "plan_submitted": is_plan_submitted(run_id),
    }
    return payload


def submit_plan_tool() -> PublicFunctionTool[SubmitPlanArguments]:
    """Return a :class:`FunctionTool` that records the active run's plan."""

    return PublicFunctionTool(
        name="submit_plan",
        description=(
            "Record a plan for the current run so subsequent edit tools can "
            "proceed under permission_mode=plan. The plan text is stored in "
            "the run's event log; the operator can review it via /inspect."
        ),
        arguments_model=SubmitPlanArguments,
        function=_submit_plan,
    )


__all__ = ["SubmitPlanArguments", "submit_plan_tool"]
