from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from deep_agents.models.base import DeepAgentsModel, JsonObject, utc_now


class RuntimeCommandType(StrEnum):
    ADJUST_TIMEOUT = "adjust_timeout"
    TERMINATE_TASK = "terminate_task"
    MARK_EARLY_COMPLETE = "mark_early_complete"
    ESCALATE_HITL = "escalate_hitl"
    PAUSE_TASK = "pause_task"
    RESUME_TASK = "resume_task"
    REQUEST_REPLAN = "request_replan"


class RuntimeCommand(DeepAgentsModel):
    type: RuntimeCommandType
    task_id: str | None = None
    reason: str
    payload: JsonObject = Field(default_factory=dict)
    source: str
    created_at: str = Field(default_factory=lambda: utc_now().isoformat())
