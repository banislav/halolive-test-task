from __future__ import annotations

from typing import Any

from langchain_core.runnables import Runnable

from deep_agents.models import (
    ExecutionPlan,
    ExecutionPlannerInput,
    PlanState,
    PlanStatus,
    RuntimeCommandResult,
    RuntimeReplanResult,
    RuntimeReplanStatus,
    TaskStatus,
)
from deep_agents.runtime.plan_tracker import PlanTracker
from deep_agents.runtime.results import TaskRunResult


class RuntimeReplanner:
    """Replace an active execution plan from a runtime replan command."""

    def __init__(
        self,
        planner: Runnable[ExecutionPlannerInput, ExecutionPlan | dict[str, Any]],
        *,
        available_tools: list[str] | None = None,
        available_skills: list[str] | None = None,
    ) -> None:
        """Create a replanner around an execution-planner runnable."""
        self.planner = planner
        self.available_tools = available_tools or []
        self.available_skills = available_skills or []

    def replan(
        self,
        *,
        trigger: RuntimeCommandResult,
        execution_plan: ExecutionPlan,
        plan_state: PlanState,
        results: dict[str, TaskRunResult],
        memory_context: dict[str, Any] | None = None,
    ) -> tuple[ExecutionPlan, RuntimeReplanResult]:
        """Invoke the planner and reconcile runtime state against its replacement plan."""
        if plan_state.discovery_plan is None:
            return execution_plan, RuntimeReplanResult(
                trigger=trigger,
                status=RuntimeReplanStatus.SKIPPED,
                reason="Cannot replan without a discovery plan.",
                previous_execution_plan_id=execution_plan.id,
            )

        previous_statuses = dict(plan_state.task_statuses)
        previous_results = dict(results)
        planner_input = self._build_input(
            trigger=trigger,
            execution_plan=execution_plan,
            plan_state=plan_state,
            results=results,
            memory_context=memory_context,
        )

        try:
            replacement_plan = self._coerce_plan(self.planner.invoke(planner_input))
        except Exception as exc:  # noqa: BLE001
            return execution_plan, RuntimeReplanResult(
                trigger=trigger,
                status=RuntimeReplanStatus.FAILED,
                reason=f"Runtime replanner failed: {exc}",
                previous_execution_plan_id=execution_plan.id,
            )

        self._reconcile_state(
            plan_state=plan_state,
            replacement_plan=replacement_plan,
            previous_statuses=previous_statuses,
            previous_results=previous_results,
            results=results,
        )
        return replacement_plan, RuntimeReplanResult(
            trigger=trigger,
            status=RuntimeReplanStatus.APPLIED,
            reason="Execution plan replaced by runtime replanner.",
            previous_execution_plan_id=execution_plan.id,
            new_execution_plan_id=replacement_plan.id,
        )

    def _build_input(
        self,
        *,
        trigger: RuntimeCommandResult,
        execution_plan: ExecutionPlan,
        plan_state: PlanState,
        results: dict[str, TaskRunResult],
        memory_context: dict[str, Any] | None,
    ) -> ExecutionPlannerInput:
        completed_results = {}
        for task_id, result in results.items():
            status = plan_state.task_statuses.get(task_id)
            if status is not None and TaskStatus(status) == TaskStatus.COMPLETED:
                completed_results[task_id] = result.model_dump(mode="json")
        return ExecutionPlannerInput(
            discovery_plan=plan_state.discovery_plan,
            available_tools=self.available_tools,
            available_skills=self.available_skills,
            context={
                "trigger": trigger.model_dump(mode="json"),
                "current_execution_plan": execution_plan.model_dump(mode="json"),
                "plan_state": plan_state.model_dump(mode="json"),
                "completed_results": completed_results,
                "memory_context": memory_context or {},
            },
        )

    def _coerce_plan(self, value: ExecutionPlan | dict[str, Any]) -> ExecutionPlan:
        if isinstance(value, ExecutionPlan):
            return value
        return ExecutionPlan(**value)

    def _reconcile_state(
        self,
        *,
        plan_state: PlanState,
        replacement_plan: ExecutionPlan,
        previous_statuses: dict[str, TaskStatus],
        previous_results: dict[str, TaskRunResult],
        results: dict[str, TaskRunResult],
    ) -> None:
        replacement_task_ids = {task.id for task in replacement_plan.task_cards}
        preserved_completed = {
            task_id
            for task_id, status in previous_statuses.items()
            if task_id in replacement_task_ids and TaskStatus(status) == TaskStatus.COMPLETED
        }

        plan_state.execution_plan_id = replacement_plan.id
        plan_state.status = PlanStatus.REFINING
        plan_state.task_statuses = {
            task.id: TaskStatus.COMPLETED if task.id in preserved_completed else TaskStatus.PENDING
            for task in replacement_plan.task_cards
        }

        results.clear()
        results.update(
            {
                task_id: result
                for task_id, result in previous_results.items()
                if task_id in preserved_completed
            }
        )
        PlanTracker(plan_state, replacement_plan).refresh_readiness()
