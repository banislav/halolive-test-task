from __future__ import annotations

from collections.abc import Iterable

from deep_agents.models import ExecutionPlan, TaskCard, TaskStatus


class Dispatcher:
    """Dependency helper for selecting dispatchable task cards."""

    def __init__(self, execution_plan: ExecutionPlan) -> None:
        """Index task cards from an execution plan for runtime lookup."""
        self.execution_plan = execution_plan
        self._task_cards = {task.id: task for task in execution_plan.task_cards}

    def get_task(self, task_id: str) -> TaskCard:
        """Return the task card with the given id."""
        return self._task_cards[task_id]

    def ready_tasks(
        self,
        task_statuses: dict[str, TaskStatus | str],
        *,
        wave: int | None = None,
    ) -> list[TaskCard]:
        """Return pending or blocked tasks whose dependencies have completed.

        When ``wave`` is provided, only task cards assigned to that wave are
        considered. Missing task statuses are treated as pending.
        """
        candidates: Iterable[TaskCard] = self.execution_plan.task_cards
        if wave is not None:
            candidates = (task for task in candidates if task.wave == wave)

        ready: list[TaskCard] = []
        for task in candidates:
            status = TaskStatus(task_statuses.get(task.id, TaskStatus.PENDING))
            dependencies_met = all(
                TaskStatus(task_statuses.get(blocker)) == TaskStatus.COMPLETED
                for blocker in task.blocked_by
            )
            if status in {TaskStatus.PENDING, TaskStatus.BLOCKED} and dependencies_met:
                ready.append(task)
        return ready

    def blocked_tasks(self, task_statuses: dict[str, TaskStatus | str]) -> list[TaskCard]:
        """Return pending or blocked tasks that are still waiting on dependencies."""
        blocked: list[TaskCard] = []
        for task in self.execution_plan.task_cards:
            status = TaskStatus(task_statuses.get(task.id, TaskStatus.PENDING))
            dependencies_met = all(
                TaskStatus(task_statuses.get(blocker)) == TaskStatus.COMPLETED
                for blocker in task.blocked_by
            )
            if status in {TaskStatus.PENDING, TaskStatus.BLOCKED} and not dependencies_met:
                blocked.append(task)
        return blocked
