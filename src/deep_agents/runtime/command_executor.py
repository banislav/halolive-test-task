from __future__ import annotations

from deep_agents.models import (
    ExecutionPlan,
    PlanState,
    PlanStatus,
    RuntimeCommand,
    RuntimeCommandResult,
    RuntimeCommandStatus,
    RuntimeCommandType,
    TaskStatus,
)
from deep_agents.runtime.long_running import LongRunningRunRegistry
from deep_agents.runtime.plan_tracker import PlanTracker


class RuntimeCommandExecutor:
    """Apply deterministic runtime commands to plan state."""

    def __init__(self, *, long_running_registry: LongRunningRunRegistry | None = None) -> None:
        self.long_running_registry = long_running_registry

    def execute(
        self,
        command: RuntimeCommand,
        *,
        plan_state: PlanState,
        execution_plan: ExecutionPlan,
    ) -> RuntimeCommandResult:
        """Execute one command and return its application result."""
        tracker = PlanTracker(plan_state, execution_plan)

        if command.type == RuntimeCommandType.HALT:
            return self._halt(command, tracker)
        if command.type == RuntimeCommandType.PAUSE_TASK:
            return self._pause_task(command, tracker)
        if command.type == RuntimeCommandType.RESUME_TASK:
            return self._resume_task(command, tracker)
        if command.type == RuntimeCommandType.TERMINATE_TASK:
            return self._terminate_task(command, tracker)
        if command.type == RuntimeCommandType.ADJUST_TIMEOUT:
            return self._adjust_timeout(command)
        if command.type == RuntimeCommandType.MARK_EARLY_COMPLETE:
            return RuntimeCommandResult(
                command=command,
                status=RuntimeCommandStatus.APPLIED,
                reason="Task completion was already applied when the process judgment was handled.",
                affected_task_ids=[command.task_id] if command.task_id else [],
            )
        return RuntimeCommandResult(
            command=command,
            status=RuntimeCommandStatus.IGNORED,
            reason=f"{command.type} is advisory in RuntimeCommandExecutor v1.",
        )

    def execute_all(
        self,
        commands: list[RuntimeCommand],
        *,
        plan_state: PlanState,
        execution_plan: ExecutionPlan,
    ) -> list[RuntimeCommandResult]:
        """Execute commands in order and return per-command results."""
        return [
            self.execute(command, plan_state=plan_state, execution_plan=execution_plan)
            for command in commands
        ]

    def _halt(
        self,
        command: RuntimeCommand,
        tracker: PlanTracker,
    ) -> RuntimeCommandResult:
        previous = tracker.state.status
        tracker.state.status = PlanStatus.PAUSED
        tracker._log_plan_change(previous, PlanStatus.PAUSED, "runtime_command.halt")
        affected = (
            self.long_running_registry.request_cancel(None, reason=command.reason)
            if self.long_running_registry is not None
            else []
        )
        return RuntimeCommandResult(
            command=command,
            status=RuntimeCommandStatus.APPLIED,
            reason="Runtime halt command paused the plan.",
            affected_task_ids=affected,
        )

    def _pause_task(
        self,
        command: RuntimeCommand,
        tracker: PlanTracker,
    ) -> RuntimeCommandResult:
        if command.task_id:
            if command.task_id not in tracker.state.task_statuses:
                return self._failed(command, f"Unknown task id: {command.task_id}")
            status = TaskStatus(tracker.state.task_statuses[command.task_id])
            if status != TaskStatus.RUNNING:
                return RuntimeCommandResult(
                    command=command,
                    status=RuntimeCommandStatus.IGNORED,
                    reason=f"Task {command.task_id} is {status}, not running.",
                )
            previous = tracker.state.task_statuses[command.task_id]
            tracker.state.task_statuses[command.task_id] = TaskStatus.PAUSED
            affected = (
                self.long_running_registry.request_cancel(command.task_id, reason=command.reason)
                if self.long_running_registry is not None
                else [command.task_id]
            )
            tracker._log_task_change(
                command.task_id,
                previous,
                TaskStatus.PAUSED,
                "runtime_command.pause_task",
            )
            return RuntimeCommandResult(
                command=command,
                status=RuntimeCommandStatus.APPLIED,
                reason=f"Paused running task {command.task_id}.",
                affected_task_ids=affected,
            )

        paused = tracker.pause_all_running()
        if self.long_running_registry is not None:
            self.long_running_registry.request_cancel(None, reason=command.reason)
        status = RuntimeCommandStatus.APPLIED if paused else RuntimeCommandStatus.IGNORED
        reason = "Paused all running tasks." if paused else "No running tasks to pause."
        return RuntimeCommandResult(
            command=command,
            status=status,
            reason=reason,
            affected_task_ids=paused,
        )

    def _resume_task(
        self,
        command: RuntimeCommand,
        tracker: PlanTracker,
    ) -> RuntimeCommandResult:
        target_ids = (
            [command.task_id]
            if command.task_id
            else [
                task_id
                for task_id, status in tracker.state.task_statuses.items()
                if TaskStatus(status) == TaskStatus.PAUSED
            ]
        )
        affected: list[str] = []
        for task_id in target_ids:
            if task_id not in tracker.state.task_statuses:
                continue
            if TaskStatus(tracker.state.task_statuses[task_id]) != TaskStatus.PAUSED:
                continue
            task = tracker.dispatcher.get_task(task_id) if tracker.dispatcher else None
            blockers_done = task is None or all(
                TaskStatus(tracker.state.task_statuses.get(blocker)) == TaskStatus.COMPLETED
                for blocker in task.blocked_by
            )
            if not blockers_done:
                continue
            previous = tracker.state.task_statuses[task_id]
            tracker.state.task_statuses[task_id] = TaskStatus.READY
            tracker._log_task_change(
                task_id,
                previous,
                TaskStatus.READY,
                "runtime_command.resume_task",
            )
            affected.append(task_id)

        if affected:
            previous_plan_status = tracker.state.status
            tracker.state.status = PlanStatus.EXECUTING
            tracker._log_plan_change(
                previous_plan_status,
                PlanStatus.EXECUTING,
                "runtime_command.resume_task",
            )
        return RuntimeCommandResult(
            command=command,
            status=RuntimeCommandStatus.APPLIED if affected else RuntimeCommandStatus.IGNORED,
            reason="Resumed paused tasks whose blockers are complete."
            if affected
            else "No paused unblocked tasks to resume.",
            affected_task_ids=affected,
        )

    def _terminate_task(
        self,
        command: RuntimeCommand,
        tracker: PlanTracker,
    ) -> RuntimeCommandResult:
        if not command.task_id:
            return self._failed(command, "terminate_task requires a task_id.")
        if command.task_id not in tracker.state.task_statuses:
            return self._failed(command, f"Unknown task id: {command.task_id}")

        previous = tracker.state.task_statuses[command.task_id]
        tracker.state.task_statuses[command.task_id] = TaskStatus.TERMINATED
        if self.long_running_registry is not None:
            self.long_running_registry.request_cancel(command.task_id, reason=command.reason)
        tracker._log_task_change(
            command.task_id,
            previous,
            TaskStatus.TERMINATED,
            "runtime_command.terminate_task",
        )
        previous_plan_status = tracker.state.status
        tracker.state.status = PlanStatus.REFINING
        tracker._log_plan_change(
            previous_plan_status,
            PlanStatus.REFINING,
            "runtime_command.terminate_task",
        )
        return RuntimeCommandResult(
            command=command,
            status=RuntimeCommandStatus.APPLIED,
            reason="Terminated task and marked plan refining for follow-up replanning.",
            affected_task_ids=[command.task_id],
        )

    def _adjust_timeout(self, command: RuntimeCommand) -> RuntimeCommandResult:
        if self.long_running_registry is None or command.task_id is None:
            return RuntimeCommandResult(
                command=command,
                status=RuntimeCommandStatus.IGNORED,
                reason="No active long-running registry or task id for timeout adjustment.",
            )
        seconds = command.payload.get(
            "seconds",
            command.payload.get("timeout_seconds", command.payload.get("value", 0)),
        )
        if not isinstance(seconds, int) or seconds <= 0:
            return self._failed(command, "adjust_timeout requires positive integer seconds.")
        affected = self.long_running_registry.extend_timeout(command.task_id, seconds)
        return RuntimeCommandResult(
            command=command,
            status=RuntimeCommandStatus.APPLIED if affected else RuntimeCommandStatus.IGNORED,
            reason="Recorded cooperative timeout extension."
            if affected
            else "No active long-running attempt matched timeout adjustment.",
            affected_task_ids=affected,
        )

    def _failed(self, command: RuntimeCommand, reason: str) -> RuntimeCommandResult:
        return RuntimeCommandResult(
            command=command,
            status=RuntimeCommandStatus.FAILED,
            reason=reason,
        )
