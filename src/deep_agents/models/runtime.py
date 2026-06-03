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


class RuntimeMessageType(StrEnum):
    RESULT = "result"
    VERDICT = "verdict"
    SIGNAL = "signal"
    REQUEST = "request"
    PROGRESS = "progress"
    PROMPT = "prompt"
    COMMAND = "command"
    ERROR = "error"


class RuntimeSessionStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class LongRunningStatus(StrEnum):
    RUNNING = "running"
    CHECKPOINTED = "checkpointed"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


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


class RuntimeMessage(DeepAgentsModel):
    from_agent: str = Field(alias="from")
    to_agent: str = Field(alias="to")
    type: RuntimeMessageType
    payload: JsonObject = Field(default_factory=dict)
    correlation_id: str
    timestamp: str = Field(default_factory=lambda: utc_now().isoformat())


class RuntimeSessionSnapshot(DeepAgentsModel):
    session_id: str
    status: RuntimeSessionStatus
    execution_plan_id: str | None = None
    current_task_id: str | None = None
    plan_state: JsonObject = Field(default_factory=dict)
    results: dict[str, JsonObject] = Field(default_factory=dict)
    runtime_commands: list[JsonObject] = Field(default_factory=list)
    command_results: list[JsonObject] = Field(default_factory=list)
    prompt_results: list[JsonObject] = Field(default_factory=list)
    pending_prompt_ids: list[str] = Field(default_factory=list)
    memory_record_count: int = 0


class LongRunningCheckpoint(DeepAgentsModel):
    task_id: str
    attempt_id: str
    sequence: int = Field(ge=1)
    payload: JsonObject = Field(default_factory=dict)
    percent_complete: float | None = Field(default=None, ge=0, le=100)
    cursor: JsonObject | None = None
    timestamp: str = Field(default_factory=lambda: utc_now().isoformat())


class LongRunningRunState(DeepAgentsModel):
    task_id: str
    attempt_id: str
    status: LongRunningStatus = LongRunningStatus.RUNNING
    last_heartbeat_at: str | None = None
    last_checkpoint_at: str | None = None
    checkpoint_ids: list[str] = Field(default_factory=list)
    resource_observations: list[JsonObject] = Field(default_factory=list)
    cancel_requested: bool = False
    cancel_reason: str | None = None
    timeout_extension_seconds: int | None = Field(default=None, ge=0)


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
