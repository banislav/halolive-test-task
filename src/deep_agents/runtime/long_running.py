from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any
from uuid import uuid4

from deep_agents.models import (
    LongRunningCheckpoint,
    LongRunningRunState,
    LongRunningStatus,
    LongRunningTaskConfig,
    ProgressSignal,
    ProgressSignalPayload,
    ProgressSignalType,
)
from deep_agents.models.base import JsonObject
from deep_agents.runtime.memory import MemoryRecorder
from deep_agents.runtime.observability import ProgressSignalBus

_current_long_running_context: ContextVar[LongRunningContext | None] = ContextVar(
    "current_long_running_context",
    default=None,
)


class LongRunningRunRegistry:
    """In-memory cooperative state registry for active and recent long-running attempts."""

    def __init__(self) -> None:
        self._states: dict[str, LongRunningRunState] = {}
        self._task_attempts: dict[str, list[str]] = {}
        self._checkpoints: dict[str, LongRunningCheckpoint] = {}

    def start(self, *, task_id: str, attempt_id: str) -> LongRunningRunState:
        """Register a running long-running attempt."""
        state = LongRunningRunState(task_id=task_id, attempt_id=attempt_id)
        self._states[attempt_id] = state
        self._task_attempts.setdefault(task_id, []).append(attempt_id)
        return state

    def state(self, attempt_id: str) -> LongRunningRunState | None:
        """Return run state by attempt id."""
        return self._states.get(attempt_id)

    def latest_state_for_task(self, task_id: str) -> LongRunningRunState | None:
        """Return the latest run state for a task, if any."""
        attempt_ids = self._task_attempts.get(task_id, [])
        if not attempt_ids:
            return None
        return self._states.get(attempt_ids[-1])

    def latest_checkpoint_for_task(self, task_id: str) -> LongRunningCheckpoint | None:
        """Return the latest checkpoint for a task across attempts."""
        checkpoints: list[LongRunningCheckpoint] = []
        for attempt_id in self._task_attempts.get(task_id, []):
            state = self._states.get(attempt_id)
            if state is None:
                continue
            for checkpoint_id in state.checkpoint_ids:
                checkpoint = self._checkpoints.get(checkpoint_id)
                if checkpoint is not None:
                    checkpoints.append(checkpoint)
        return checkpoints[-1] if checkpoints else None

    def add_checkpoint(
        self,
        checkpoint: LongRunningCheckpoint,
    ) -> str:
        """Store a checkpoint and link it to its run state."""
        checkpoint_id = f"checkpoint-{checkpoint.task_id}-{checkpoint.sequence}-{uuid4().hex}"
        self._checkpoints[checkpoint_id] = checkpoint
        state = self._states[checkpoint.attempt_id]
        state.status = LongRunningStatus.CHECKPOINTED
        state.last_checkpoint_at = checkpoint.timestamp
        state.checkpoint_ids.append(checkpoint_id)
        return checkpoint_id

    def request_cancel(self, task_id: str | None, *, reason: str) -> list[str]:
        """Mark matching active runs as cooperatively cancelled."""
        affected: list[str] = []
        states = (
            [state for state in self._states.values() if state.task_id == task_id]
            if task_id
            else list(self._states.values())
        )
        for state in states:
            if state.status not in {LongRunningStatus.RUNNING, LongRunningStatus.CHECKPOINTED}:
                continue
            state.cancel_requested = True
            state.cancel_reason = reason
            affected.append(state.task_id)
        return affected

    def extend_timeout(self, task_id: str, seconds: int) -> list[str]:
        """Record a cooperative timeout extension for matching active runs."""
        affected: list[str] = []
        for state in self._states.values():
            if state.task_id != task_id:
                continue
            if state.status not in {LongRunningStatus.RUNNING, LongRunningStatus.CHECKPOINTED}:
                continue
            state.timeout_extension_seconds = seconds
            affected.append(state.task_id)
        return affected


class LongRunningCheckpointRecorder:
    """Create and persist checkpoints for one long-running run."""

    def __init__(
        self,
        *,
        registry: LongRunningRunRegistry,
        memory_recorder: MemoryRecorder,
        plan_id: str | None,
    ) -> None:
        self.registry = registry
        self.memory_recorder = memory_recorder
        self.plan_id = plan_id

    def record(
        self,
        *,
        task_id: str,
        attempt_id: str,
        sequence: int,
        payload: JsonObject,
        cursor: JsonObject | None,
        percent_complete: float | None,
    ) -> LongRunningCheckpoint:
        """Record one checkpoint in registry and memory."""
        checkpoint = LongRunningCheckpoint(
            task_id=task_id,
            attempt_id=attempt_id,
            sequence=sequence,
            payload=payload,
            cursor=cursor,
            percent_complete=percent_complete,
        )
        checkpoint_id = self.registry.add_checkpoint(checkpoint)
        self.memory_recorder.record_long_running_checkpoint(
            checkpoint,
            checkpoint_id=checkpoint_id,
            plan_id=self.plan_id,
        )
        return checkpoint


class LongRunningContext:
    """Cooperative API exposed to long-running workers."""

    def __init__(
        self,
        *,
        task_id: str,
        attempt_id: str,
        config: LongRunningTaskConfig,
        registry: LongRunningRunRegistry,
        memory_recorder: MemoryRecorder,
        progress_bus: ProgressSignalBus,
        plan_id: str | None,
        resume_from: LongRunningCheckpoint | None = None,
    ) -> None:
        self.task_id = task_id
        self.attempt_id = attempt_id
        self.config = config
        self.registry = registry
        self.memory_recorder = memory_recorder
        self.progress_bus = progress_bus
        self.plan_id = plan_id
        self.resume_from = resume_from
        self.resume_cursor = resume_from.cursor if resume_from is not None else None
        self.state = registry.start(task_id=task_id, attempt_id=attempt_id)
        self._checkpoint_recorder = LongRunningCheckpointRecorder(
            registry=registry,
            memory_recorder=memory_recorder,
            plan_id=plan_id,
        )
        self._checkpoint_sequence = 0

    def heartbeat(
        self,
        status: str | None = None,
        percent_complete: float | None = None,
        data: JsonObject | None = None,
    ) -> ProgressSignal:
        """Emit and record a heartbeat signal."""
        signal = self._publish(
            signal_type=ProgressSignalType.HEARTBEAT,
            payload=ProgressSignalPayload(
                status=status or "long_running_heartbeat",
                percent_complete=percent_complete,
                data=data or {},
            ),
        )
        self.state.last_heartbeat_at = signal.timestamp
        return signal

    def progress(
        self,
        status: str | None = None,
        percent_complete: float | None = None,
        data: JsonObject | None = None,
        items_processed: int | None = None,
        estimated_remaining_seconds: int | None = None,
    ) -> ProgressSignal | None:
        """Emit and record a progress signal when progress reporting is enabled."""
        if not self.config.progress_reporting:
            return None
        return self._publish(
            signal_type=ProgressSignalType.PROGRESS,
            payload=ProgressSignalPayload(
                status=status or "long_running_progress",
                percent_complete=percent_complete,
                items_processed=items_processed,
                estimated_remaining_seconds=estimated_remaining_seconds,
                data=data or {},
            ),
        )

    def finding(
        self,
        data: JsonObject,
        *,
        status: str | None = None,
        relevance_score: float | None = None,
        actionable: bool | None = True,
    ) -> ProgressSignal | None:
        """Emit and record an early finding signal when enabled."""
        if not self.config.early_findings_enabled:
            return None
        return self._publish(
            signal_type=ProgressSignalType.FINDING,
            payload=ProgressSignalPayload(
                status=status or "long_running_finding",
                data=data,
                relevance_score=relevance_score,
                actionable=actionable,
            ),
        )

    def checkpoint(
        self,
        payload: JsonObject,
        *,
        cursor: JsonObject | None = None,
        percent_complete: float | None = None,
    ) -> LongRunningCheckpoint:
        """Record a resumable checkpoint."""
        self._checkpoint_sequence += 1
        checkpoint = self._checkpoint_recorder.record(
            task_id=self.task_id,
            attempt_id=self.attempt_id,
            sequence=self._checkpoint_sequence,
            payload=payload,
            cursor=cursor,
            percent_complete=percent_complete,
        )
        self.progress(
            status="long_running_checkpointed",
            percent_complete=percent_complete,
            data={"checkpoint_sequence": checkpoint.sequence, "cursor": cursor or {}},
        )
        return checkpoint

    def observe_resources(
        self,
        *,
        elapsed_seconds: float | None = None,
        memory_mb: float | None = None,
        cpu_time_seconds: float | None = None,
        data: JsonObject | None = None,
    ) -> ProgressSignal | None:
        """Record resource observations and emit warnings when configured limits are exceeded."""
        observation: JsonObject = {
            "elapsed_seconds": elapsed_seconds,
            "memory_mb": memory_mb,
            "cpu_time_seconds": cpu_time_seconds,
            "data": data or {},
        }
        self.state.resource_observations.append(observation)
        self.memory_recorder.record_long_running_resource_observation(
            task_id=self.task_id,
            attempt_id=self.attempt_id,
            observation=observation,
            plan_id=self.plan_id,
        )
        warnings = self._resource_warnings(
            elapsed_seconds=elapsed_seconds,
            memory_mb=memory_mb,
            cpu_time_seconds=cpu_time_seconds,
        )
        if not warnings:
            return None
        return self.progress(
            status="long_running_resource_warning",
            data={"warnings": warnings, "observation": observation},
        )

    def should_cancel(self) -> bool:
        """Return whether runtime control has requested cooperative cancellation."""
        return self.state.cancel_requested

    def complete(self) -> None:
        """Mark the long-running run complete and record final state."""
        self.state.status = LongRunningStatus.COMPLETED
        self.memory_recorder.record_long_running_state(self.state, plan_id=self.plan_id)

    def fail(self, error: BaseException) -> None:
        """Mark the long-running run failed and record final state."""
        self.state.status = (
            LongRunningStatus.CANCELLED if self.should_cancel() else LongRunningStatus.FAILED
        )
        self.memory_recorder.record_long_running_state(
            self.state,
            plan_id=self.plan_id,
            error={"type": error.__class__.__name__, "message": str(error)},
        )

    def summary(self) -> JsonObject:
        """Return compact long-running run details for task attempt summaries."""
        return {
            "state": self.state.model_dump(mode="json"),
            "resume_from": self.resume_from.model_dump(mode="json")
            if self.resume_from is not None
            else None,
        }

    def _publish(
        self,
        *,
        signal_type: ProgressSignalType,
        payload: ProgressSignalPayload,
    ) -> ProgressSignal:
        signal = ProgressSignal(task_id=self.task_id, signal_type=signal_type, payload=payload)
        self.progress_bus.publish(signal)
        self.memory_recorder.record_progress_signal(signal, plan_id=self.plan_id)
        return signal

    def _resource_warnings(
        self,
        *,
        elapsed_seconds: float | None,
        memory_mb: float | None,
        cpu_time_seconds: float | None,
    ) -> list[str]:
        warnings: list[str] = []
        if (
            elapsed_seconds is not None
            and self.config.max_elapsed_seconds is not None
            and elapsed_seconds > self.config.max_elapsed_seconds
        ):
            warnings.append("elapsed_time_exceeded")
        if (
            memory_mb is not None
            and self.config.max_memory_mb is not None
            and memory_mb > self.config.max_memory_mb
        ):
            warnings.append("memory_exceeded")
        if (
            cpu_time_seconds is not None
            and self.config.max_cpu_time_seconds is not None
            and cpu_time_seconds > self.config.max_cpu_time_seconds
        ):
            warnings.append("cpu_time_exceeded")
        return warnings


@contextmanager
def long_running_context(context: LongRunningContext | None) -> Any:
    """Set the current long-running context for worker code in this thread."""
    token = _current_long_running_context.set(context)
    try:
        yield
    finally:
        _current_long_running_context.reset(token)


def current_long_running_context() -> LongRunningContext | None:
    """Return the current long-running context, if the active worker has one."""
    return _current_long_running_context.get()
