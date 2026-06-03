from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from deep_agents.models.agents import AgentAssignment
from deep_agents.models.base import DeepAgentsModel, JsonObject, utc_now


class ToolSafetyLevel(StrEnum):
    SAFE = "safe"
    SENSITIVE = "sensitive"
    DESTRUCTIVE = "destructive"
    HITL_REQUIRED = "hitl_required"


class ToolCallStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DENIED = "denied"
    VALIDATION_FAILED = "validation_failed"
    RATE_LIMITED = "rate_limited"
    SAFETY_BLOCKED = "safety_blocked"


class ToolDefinition(DeepAgentsModel):
    id: str
    name: str
    description: str | None = None
    input_schema: JsonObject = Field(default_factory=dict)
    output_schema: JsonObject = Field(default_factory=dict)
    allowed_task_ids: list[str] = Field(default_factory=list)
    allowed_agent_ids: list[str] = Field(default_factory=list)
    allowed_agent_names: list[str] = Field(default_factory=list)
    allowed_agent_types: list[str] = Field(default_factory=list)
    safety_level: ToolSafetyLevel = ToolSafetyLevel.SAFE


class ToolCallRequest(DeepAgentsModel):
    tool_id: str
    task_id: str
    attempt_id: str | None = None
    input: JsonObject = Field(default_factory=dict)
    caller_agent: AgentAssignment | None = None
    metadata: JsonObject = Field(default_factory=dict)


class ToolCallResult(DeepAgentsModel):
    tool_id: str
    task_id: str
    attempt_id: str | None = None
    status: ToolCallStatus
    output: JsonObject = Field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None
    started_at: str = Field(default_factory=lambda: utc_now().isoformat())
    completed_at: str | None = None
    duration_seconds: float | None = Field(default=None, ge=0)
    metadata: JsonObject = Field(default_factory=dict)
