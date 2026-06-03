from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any
from uuid import uuid4

from deep_agents.models import (
    AgentLifecycleEvent,
    AgentLifecycleState,
    MemoryQuery,
    TaskAttemptRecord,
    TaskAttemptStatus,
    TaskCard,
)
from deep_agents.models.base import utc_now
from deep_agents.runtime.context import TaskExecutionContext
from deep_agents.runtime.long_running import (
    LongRunningContext,
    LongRunningRunRegistry,
    long_running_context,
)
from deep_agents.runtime.memory import MemoryRecorder
from deep_agents.runtime.observability import ProgressSignalBus
from deep_agents.runtime.results import TaskRunResult
from deep_agents.runtime.tools import tool_attempt_context

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
        progress_bus: ProgressSignalBus | None = None,
        long_running_registry: LongRunningRunRegistry | None = None,
    ) -> None:
        self.invoker = invoker
        self.memory_recorder = memory_recorder
        self.plan_id = plan_id
        self.progress_bus = progress_bus or ProgressSignalBus()
        self.long_running_registry = long_running_registry or LongRunningRunRegistry()

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
            long_context = self._long_running_context(task, worker_input, attempt.id)

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
                raw_result = self._invoke_with_timeout(
                    task,
                    worker_input,
                    attempt.id,
                    long_context,
                )
                result = self._coerce_result(task.id, raw_result)
                if long_context is not None:
                    long_context.complete()
                self._emit_lifecycle(
                    attempt,
                    AgentLifecycleState.REPORTING,
                    detail="Task result reported.",
                )
                attempt.status = TaskAttemptStatus.SUCCEEDED
                attempt.result = {
                    "result": result.model_dump(mode="json"),
                    "tool_calls": self._tool_call_summaries(task.id, attempt.id),
                }
                if long_context is not None:
                    attempt.result["long_running"] = long_context.summary()
                attempt.completed_at = utc_now().isoformat()
                self._emit_lifecycle(
                    attempt,
                    AgentLifecycleState.TERMINATED,
                    detail="Agent attempt terminated.",
                )
                self.memory_recorder.record_task_attempt(attempt, plan_id=self.plan_id)
                return result, attempts
            except Exception as exc:
                if long_context is not None:
                    long_context.fail(exc)
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
        attempt_id: str,
        long_context: LongRunningContext | None,
    ) -> TaskRunResult | dict[str, Any]:
        timeout_seconds = task.invocation.timeout_seconds
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            self._invoke_with_attempt_context,
            task,
            worker_input,
            attempt_id,
            long_context,
        )
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeoutError as exc:
            future.cancel()
            raise TimeoutError(
                f"Task {task.id} exceeded timeout of {timeout_seconds} seconds."
            ) from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _invoke_with_attempt_context(
        self,
        task: TaskCard,
        worker_input: TaskCard | TaskExecutionContext,
        attempt_id: str,
        long_context: LongRunningContext | None,
    ) -> TaskRunResult | dict[str, Any]:
        with tool_attempt_context(attempt_id):
            with long_running_context(long_context):
                return self.invoker(task, worker_input)

    def _long_running_context(
        self,
        task: TaskCard,
        worker_input: TaskCard | TaskExecutionContext,
        attempt_id: str,
    ) -> LongRunningContext | None:
        config = task.invocation.long_running
        if config is None:
            return None
        resume_from = (
            self.long_running_registry.latest_checkpoint_for_task(task.id)
            if config.resumable
            else None
        )
        context = LongRunningContext(
            task_id=task.id,
            attempt_id=attempt_id,
            config=config,
            registry=self.long_running_registry,
            memory_recorder=self.memory_recorder,
            progress_bus=self.progress_bus,
            plan_id=self.plan_id,
            resume_from=resume_from,
        )
        if isinstance(worker_input, TaskExecutionContext):
            worker_input.long_running = context
        return context

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

    def _tool_call_summaries(self, task_id: str, attempt_id: str) -> list[dict[str, Any]]:
        records = self.memory_recorder.store.query(
            MemoryQuery(task_ids=[task_id], tags=["tool_result"])
        )
        summaries: list[dict[str, Any]] = []
        for record in records:
            result = record.payload.get("result", {})
            if result.get("attempt_id") != attempt_id:
                continue
            summaries.append(
                {
                    "tool_id": result.get("tool_id"),
                    "status": result.get("status"),
                    "duration_seconds": result.get("duration_seconds"),
                    "error_type": result.get("error_type"),
                }
            )
        return summaries
