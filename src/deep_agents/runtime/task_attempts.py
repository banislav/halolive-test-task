from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any
from uuid import uuid4

from deep_agents.models import (
    AgentLifecycleEvent,
    AgentLifecycleState,
    TaskAttemptRecord,
    TaskAttemptStatus,
    TaskCard,
)
from deep_agents.models.base import utc_now
from deep_agents.runtime.context import TaskExecutionContext
from deep_agents.runtime.memory import MemoryRecorder
from deep_agents.runtime.results import TaskRunResult

TaskInvoker = Callable[[TaskCard, TaskCard | TaskExecutionContext], TaskRunResult | dict[str, Any]]


class TaskAttemptRunError(RuntimeError):
    """Raised after a task exhausts all retry attempts."""

    def __init__(
        self,
        *,
        task_id: str,
        attempts: list[TaskAttemptRecord],
        last_exception: BaseException,
    ) -> None:
        super().__init__(f"Task {task_id} failed after {len(attempts)} attempt(s).")
        self.task_id = task_id
        self.attempts = attempts
        self.last_exception = last_exception


class TaskAttemptRunner:
    """Run a task through lifecycle, timeout, retry, and attempt audit handling."""

    def __init__(
        self,
        *,
        invoker: TaskInvoker,
        memory_recorder: MemoryRecorder,
        plan_id: str | None,
    ) -> None:
        self.invoker = invoker
        self.memory_recorder = memory_recorder
        self.plan_id = plan_id

    def invoke(
        self,
        task: TaskCard,
        worker_input: TaskCard | TaskExecutionContext,
    ) -> tuple[TaskRunResult, list[TaskAttemptRecord]]:
        """Invoke a task, retrying according to its invocation policy."""
        max_retries = task.invocation.retry_policy.max_retries
        attempts: list[TaskAttemptRecord] = []
        last_exception: BaseException | None = None

        for retry_index in range(max_retries + 1):
            attempt = TaskAttemptRecord(
                id=f"attempt-{task.id}-{retry_index}-{uuid4().hex}",
                task_id=task.id,
                agent=task.assigned_to,
                retry_index=retry_index,
                max_retries=max_retries,
                timeout_seconds=task.invocation.timeout_seconds,
            )
            attempts.append(attempt)

            try:
                self._emit_lifecycle(
                    attempt,
                    AgentLifecycleState.SPAWNED,
                    detail="Agent attempt spawned.",
                )
                self._emit_lifecycle(
                    attempt,
                    AgentLifecycleState.SKILLS_LOADED,
                    detail="Assigned skills loaded.",
                    metadata={
                        "skill_ids": [skill.id for skill in task.assigned_to.skills],
                    },
                )
                self._emit_lifecycle(
                    attempt,
                    AgentLifecycleState.CONTEXT_LOADED,
                    detail="Task context loaded.",
                    metadata=self._context_metadata(worker_input),
                )
                self._emit_lifecycle(
                    attempt,
                    AgentLifecycleState.EXECUTING,
                    detail="Task execution started.",
                )
                raw_result = self._invoke_with_timeout(task, worker_input)
                result = self._coerce_result(task.id, raw_result)
                self._emit_lifecycle(
                    attempt,
                    AgentLifecycleState.REPORTING,
                    detail="Task result reported.",
                )
                attempt.status = TaskAttemptStatus.SUCCEEDED
                attempt.result = result.model_dump(mode="json")
                attempt.completed_at = utc_now().isoformat()
                self._emit_lifecycle(
                    attempt,
                    AgentLifecycleState.TERMINATED,
                    detail="Agent attempt terminated.",
                )
                self.memory_recorder.record_task_attempt(attempt, plan_id=self.plan_id)
                return result, attempts
            except Exception as exc:
                last_exception = exc
                if isinstance(exc, TimeoutError):
                    exhausted_status = TaskAttemptStatus.TIMED_OUT
                    error_type = "timeout"
                else:
                    exhausted_status = TaskAttemptStatus.FAILED
                    error_type = exc.__class__.__name__

                attempt.error_type = error_type
                attempt.error_message = str(exc)
                attempt.completed_at = utc_now().isoformat()

                if retry_index < max_retries:
                    attempt.status = TaskAttemptStatus.RETRYING
                    self._emit_lifecycle(
                        attempt,
                        AgentLifecycleState.RETRYING,
                        detail="Retry budget remains; scheduling another attempt.",
                        metadata={"next_retry_index": retry_index + 1},
                    )
                    self._emit_lifecycle(
                        attempt,
                        AgentLifecycleState.TERMINATED,
                        detail="Agent attempt terminated before retry.",
                    )
                    self.memory_recorder.record_task_attempt(attempt, plan_id=self.plan_id)
                    continue

                attempt.status = exhausted_status
                self._emit_lifecycle(
                    attempt,
                    AgentLifecycleState.TERMINATED,
                    detail="Agent attempt terminated after failure.",
                )
                self.memory_recorder.record_task_attempt(attempt, plan_id=self.plan_id)
                raise TaskAttemptRunError(
                    task_id=task.id,
                    attempts=attempts,
                    last_exception=last_exception,
                ) from exc

        msg = f"Task {task.id} exhausted retry handling without a terminal attempt."
        raise RuntimeError(msg)

    def _invoke_with_timeout(
        self,
        task: TaskCard,
        worker_input: TaskCard | TaskExecutionContext,
    ) -> TaskRunResult | dict[str, Any]:
        timeout_seconds = task.invocation.timeout_seconds
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self.invoker, task, worker_input)
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeoutError as exc:
            future.cancel()
            raise TimeoutError(
                f"Task {task.id} exceeded timeout of {timeout_seconds} seconds."
            ) from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _emit_lifecycle(
        self,
        attempt: TaskAttemptRecord,
        state: AgentLifecycleState,
        *,
        detail: str,
        metadata: dict[str, Any] | None = None,
    ) -> AgentLifecycleEvent:
        event = AgentLifecycleEvent(
            task_id=attempt.task_id,
            attempt_id=attempt.id,
            state=state,
            detail=detail,
            metadata=metadata or {},
        )
        attempt.lifecycle_events.append(event)
        self.memory_recorder.record_lifecycle_event(event, plan_id=self.plan_id)
        return event

    def _context_metadata(self, worker_input: TaskCard | TaskExecutionContext) -> dict[str, Any]:
        if isinstance(worker_input, TaskExecutionContext):
            return {
                "input_type": "task_execution_context",
                "dependency_count": len(worker_input.dependency_results),
                "artifact_count": len(worker_input.artifacts),
                "loaded_skill_count": len(worker_input.loaded_skill_ids),
            }
        return {"input_type": "task_card"}

    def _coerce_result(self, task_id: str, value: TaskRunResult | dict[str, Any]) -> TaskRunResult:
        if isinstance(value, TaskRunResult):
            return value
        data = dict(value)
        data.setdefault("task_id", task_id)
        return TaskRunResult(**data)
