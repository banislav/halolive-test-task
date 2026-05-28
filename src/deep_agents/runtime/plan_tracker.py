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
from deep_agents.observability import get_logger
from deep_agents.runtime.dispatcher import Dispatcher

logger = get_logger(__name__)


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
            previous_status = self.state.task_statuses.get(task.id)
            self.state.task_statuses[task.id] = TaskStatus.READY
            ready_ids.append(task.id)
            self._log_task_change(task.id, previous_status, TaskStatus.READY, "refresh_readiness")
        for task in self.dispatcher.blocked_tasks(self.state.task_statuses):
            previous_status = self.state.task_statuses.get(task.id)
            self.state.task_statuses[task.id] = TaskStatus.BLOCKED
            self._log_task_change(task.id, previous_status, TaskStatus.BLOCKED, "refresh_readiness")
        return ready_ids

    def mark_running(self, task_id: str) -> None:
        """Move a known task into the running state and mark the plan executing."""
        self._ensure_known_task(task_id)
        previous_plan_status = self.state.status
        previous_task_status = self.state.task_statuses[task_id]
        self.state.status = PlanStatus.EXECUTING
        self.state.task_statuses[task_id] = TaskStatus.RUNNING
        self._log_plan_change(previous_plan_status, PlanStatus.EXECUTING, "mark_running")
        self._log_task_change(task_id, previous_task_status, TaskStatus.RUNNING, "mark_running")

    def apply_task_completion(self, task_id: str) -> list[str]:
        """Mark a task complete, refresh dependencies, and return newly ready task ids."""
        self._ensure_known_task(task_id)
        previous_task_status = self.state.task_statuses[task_id]
        self.state.task_statuses[task_id] = TaskStatus.COMPLETED
        self._log_task_change(
            task_id,
            previous_task_status,
            TaskStatus.COMPLETED,
            "apply_task_completion",
        )
        ready_ids = self.refresh_readiness()
        if self._all_tasks_completed():
            previous_plan_status = self.state.status
            self.state.status = PlanStatus.COMPLETED
            self._log_plan_change(
                previous_plan_status,
                PlanStatus.COMPLETED,
                "apply_task_completion",
            )
        return ready_ids

    def apply_task_failure(self, task_id: str, *, recoverable: bool = True) -> None:
        """Record task failure as refining when recoverable or failed when terminal."""
        self._ensure_known_task(task_id)
        previous_task_status = self.state.task_statuses[task_id]
        previous_plan_status = self.state.status
        next_task_status = TaskStatus.PAUSED if recoverable else TaskStatus.FAILED
        next_plan_status = PlanStatus.REFINING if recoverable else PlanStatus.FAILED
        self.state.task_statuses[task_id] = next_task_status
        self.state.status = next_plan_status
        self._log_task_change(task_id, previous_task_status, next_task_status, "apply_task_failure")
        self._log_plan_change(previous_plan_status, next_plan_status, "apply_task_failure")

    def apply_judge_verdict(self, verdict: JudgeVerdict) -> list[str]:
        """Apply a judge verdict and return task ids that should be dispatched next."""
        self._ensure_known_task(verdict.task_id)
        logger.info(
            "applying judge verdict",
            extra={
                "task_id": verdict.task_id,
                "verdict": verdict.verdict,
                "recommendation": verdict.recommendation,
                "confidence": verdict.overall_confidence,
            },
        )

        if verdict.verdict == JudgeVerdictValue.PASS:
            return self.apply_task_completion(verdict.task_id)

        if verdict.recommendation == JudgeRecommendation.RETRY:
            previous_task_status = self.state.task_statuses[verdict.task_id]
            previous_plan_status = self.state.status
            self.state.task_statuses[verdict.task_id] = TaskStatus.READY
            self.state.status = PlanStatus.REFINING
            self._log_task_change(
                verdict.task_id,
                previous_task_status,
                TaskStatus.READY,
                "apply_judge_verdict.retry",
            )
            self._log_plan_change(
                previous_plan_status,
                PlanStatus.REFINING,
                "apply_judge_verdict.retry",
            )
            return [verdict.task_id]

        if verdict.recommendation == JudgeRecommendation.REPLAN:
            previous_task_status = self.state.task_statuses[verdict.task_id]
            previous_plan_status = self.state.status
            self.state.task_statuses[verdict.task_id] = TaskStatus.PAUSED
            self.state.status = PlanStatus.REFINING
            self._log_task_change(
                verdict.task_id,
                previous_task_status,
                TaskStatus.PAUSED,
                "apply_judge_verdict.replan",
            )
            self._log_plan_change(
                previous_plan_status,
                PlanStatus.REFINING,
                "apply_judge_verdict.replan",
            )
            return []

        if verdict.recommendation in {JudgeRecommendation.HOLD, JudgeRecommendation.BLOCK}:
            previous_task_status = self.state.task_statuses[verdict.task_id]
            previous_plan_status = self.state.status
            self.state.task_statuses[verdict.task_id] = TaskStatus.PAUSED
            self.state.status = PlanStatus.REFINING
            self._log_task_change(
                verdict.task_id,
                previous_task_status,
                TaskStatus.PAUSED,
                f"apply_judge_verdict.{verdict.recommendation}",
            )
            self._log_plan_change(
                previous_plan_status,
                PlanStatus.REFINING,
                f"apply_judge_verdict.{verdict.recommendation}",
            )
            return []

        previous_task_status = self.state.task_statuses[verdict.task_id]
        previous_plan_status = self.state.status
        self.state.task_statuses[verdict.task_id] = TaskStatus.FAILED
        self.state.status = PlanStatus.FAILED
        self._log_task_change(
            verdict.task_id,
            previous_task_status,
            TaskStatus.FAILED,
            "apply_judge_verdict.escalate",
        )
        self._log_plan_change(
            previous_plan_status,
            PlanStatus.FAILED,
            "apply_judge_verdict.escalate",
        )
        return []

    def pause_all_running(self) -> list[str]:
        """Pause all running tasks and return the task ids that were paused."""
        paused: list[str] = []
        for task_id, status in self.state.task_statuses.items():
            if TaskStatus(status) == TaskStatus.RUNNING:
                previous_task_status = self.state.task_statuses[task_id]
                self.state.task_statuses[task_id] = TaskStatus.PAUSED
                paused.append(task_id)
                self._log_task_change(
                    task_id,
                    previous_task_status,
                    TaskStatus.PAUSED,
                    "pause_all_running",
                )
        if paused:
            previous_plan_status = self.state.status
            self.state.status = PlanStatus.PAUSED
            self._log_plan_change(previous_plan_status, PlanStatus.PAUSED, "pause_all_running")
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
            logger.error("unknown task id", extra={"task_id": task_id})
            raise KeyError(f"unknown task id: {task_id}")

    def _log_task_change(
        self,
        task_id: str,
        previous: TaskStatus | str | None,
        current: TaskStatus | str,
        reason: str,
    ) -> None:
        if previous == current:
            return
        logger.info(
            "task status changed",
            extra={
                "task_id": task_id,
                "previous_status": previous,
                "current_status": current,
                "reason": reason,
            },
        )

    def _log_plan_change(
        self,
        previous: PlanStatus | str,
        current: PlanStatus | str,
        reason: str,
    ) -> None:
        if previous == current:
            return
        logger.info(
            "plan status changed",
            extra={
                "previous_status": previous,
                "current_status": current,
                "reason": reason,
            },
        )
