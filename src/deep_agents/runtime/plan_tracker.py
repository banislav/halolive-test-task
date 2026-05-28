from __future__ import annotations

from deep_agents.models import (
    ExecutionPlan,
    JudgeRecommendation,
    JudgeVerdict,
    JudgeVerdictValue,
    PlanState,
    PlanStatus,
    TaskStatus,
)
from deep_agents.runtime.dispatcher import Dispatcher


class PlanTracker:
    """Single owner for plan task state transitions."""

    def __init__(self, state: PlanState, execution_plan: ExecutionPlan | None = None) -> None:
        """Attach plan state to an optional execution plan and initialize task statuses."""
        self.state = state
        self.execution_plan = execution_plan
        self.dispatcher = Dispatcher(execution_plan) if execution_plan else None

        if execution_plan and not self.state.execution_plan_id:
            self.state.execution_plan_id = execution_plan.id
        if execution_plan:
            for task in execution_plan.task_cards:
                self.state.task_statuses.setdefault(task.id, TaskStatus.PENDING)
            self.refresh_readiness()

    def refresh_readiness(self) -> list[str]:
        """Mark newly unblocked tasks as ready and still-blocked tasks as blocked."""
        if not self.dispatcher:
            return []

        ready_ids: list[str] = []
        for task in self.dispatcher.ready_tasks(self.state.task_statuses):
            self.state.task_statuses[task.id] = TaskStatus.READY
            ready_ids.append(task.id)
        for task in self.dispatcher.blocked_tasks(self.state.task_statuses):
            self.state.task_statuses[task.id] = TaskStatus.BLOCKED
        return ready_ids

    def mark_running(self, task_id: str) -> None:
        """Move a known task into the running state and mark the plan executing."""
        self._ensure_known_task(task_id)
        self.state.status = PlanStatus.EXECUTING
        self.state.task_statuses[task_id] = TaskStatus.RUNNING

    def apply_task_completion(self, task_id: str) -> list[str]:
        """Mark a task complete, refresh dependencies, and return newly ready task ids."""
        self._ensure_known_task(task_id)
        self.state.task_statuses[task_id] = TaskStatus.COMPLETED
        ready_ids = self.refresh_readiness()
        if self._all_tasks_completed():
            self.state.status = PlanStatus.COMPLETED
        return ready_ids

    def apply_task_failure(self, task_id: str, *, recoverable: bool = True) -> None:
        """Record task failure as refining when recoverable or failed when terminal."""
        self._ensure_known_task(task_id)
        self.state.task_statuses[task_id] = TaskStatus.PAUSED if recoverable else TaskStatus.FAILED
        self.state.status = PlanStatus.REFINING if recoverable else PlanStatus.FAILED

    def apply_judge_verdict(self, verdict: JudgeVerdict) -> list[str]:
        """Apply a judge verdict and return task ids that should be dispatched next."""
        self._ensure_known_task(verdict.task_id)

        if verdict.verdict == JudgeVerdictValue.PASS:
            return self.apply_task_completion(verdict.task_id)

        if verdict.recommendation == JudgeRecommendation.RETRY:
            self.state.task_statuses[verdict.task_id] = TaskStatus.READY
            self.state.status = PlanStatus.REFINING
            return [verdict.task_id]

        if verdict.recommendation == JudgeRecommendation.REPLAN:
            self.state.task_statuses[verdict.task_id] = TaskStatus.PAUSED
            self.state.status = PlanStatus.REFINING
            return []

        self.state.task_statuses[verdict.task_id] = TaskStatus.FAILED
        self.state.status = PlanStatus.FAILED
        return []

    def pause_all_running(self) -> list[str]:
        """Pause all running tasks and return the task ids that were paused."""
        paused: list[str] = []
        for task_id, status in self.state.task_statuses.items():
            if TaskStatus(status) == TaskStatus.RUNNING:
                self.state.task_statuses[task_id] = TaskStatus.PAUSED
                paused.append(task_id)
        if paused:
            self.state.status = PlanStatus.PAUSED
        return paused

    def _all_tasks_completed(self) -> bool:
        """Return whether every tracked task is complete."""
        return bool(self.state.task_statuses) and all(
            TaskStatus(status) == TaskStatus.COMPLETED
            for status in self.state.task_statuses.values()
        )

    def _ensure_known_task(self, task_id: str) -> None:
        """Raise when a transition targets a task outside the tracked plan state."""
        if task_id not in self.state.task_statuses:
            raise KeyError(f"unknown task id: {task_id}")
