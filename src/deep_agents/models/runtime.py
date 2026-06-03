from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from deep_agents.models.agents import AgentAssignment, AgentLifecycleState
from deep_agents.models.base import DeepAgentsModel, JsonObject, utc_now


class RuntimeCommandType(StrEnum):
    ADJUST_TIMEOUT = "adjust_timeout"
    HALT = "halt"
    HOLD_GATE = "hold_gate"
    TERMINATE_TASK = "terminate_task"
    MARK_EARLY_COMPLETE = "mark_early_complete"
    ESCALATE_HITL = "escalate_hitl"
    PAUSE_TASK = "pause_task"
    RESUME_TASK = "resume_task"
    REQUEST_REPLAN = "request_replan"


class RuntimeCommandStatus(StrEnum):
    PENDING = "pending"
    APPLIED = "applied"
    IGNORED = "ignored"
    FAILED = "failed"


class RuntimeReplanStatus(StrEnum):
    APPLIED = "applied"
    SKIPPED = "skipped"
    FAILED = "failed"


class TaskAttemptStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class RuntimeCommand(DeepAgentsModel):
    type: RuntimeCommandType
    task_id: str | None = None
    reason: str
    payload: JsonObject = Field(default_factory=dict)
    source: str
    created_at: str = Field(default_factory=lambda: utc_now().isoformat())


class RuntimeCommandResult(DeepAgentsModel):
    command: RuntimeCommand
    status: RuntimeCommandStatus = RuntimeCommandStatus.PENDING
    reason: str
    affected_task_ids: list[str] = Field(default_factory=list)


class RuntimeReplanResult(DeepAgentsModel):
    trigger: RuntimeCommandResult
    status: RuntimeReplanStatus
    reason: str
    previous_execution_plan_id: str
    new_execution_plan_id: str | None = None


class AgentLifecycleEvent(DeepAgentsModel):
    task_id: str
    attempt_id: str
    state: AgentLifecycleState
    timestamp: str = Field(default_factory=lambda: utc_now().isoformat())
    detail: str | None = None
    metadata: JsonObject = Field(default_factory=dict)


class TaskAttemptRecord(DeepAgentsModel):
    id: str
    task_id: str
    agent: AgentAssignment
    retry_index: int = Field(ge=0)
    max_retries: int = Field(ge=0)
    timeout_seconds: float | None = Field(default=None, gt=0)
    status: TaskAttemptStatus = TaskAttemptStatus.RUNNING
    lifecycle_events: list[AgentLifecycleEvent] = Field(default_factory=list)
    started_at: str = Field(default_factory=lambda: utc_now().isoformat())
    completed_at: str | None = None
    result: JsonObject = Field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None
