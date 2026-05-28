from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from deep_agents.models.base import DeepAgentsModel, JsonObject, utc_now


class ProgressSignalType(StrEnum):
    HEARTBEAT = "heartbeat"
    PROGRESS = "progress"
    FINDING = "finding"
    ERROR = "error"
    ESCALATION = "escalation"


class ProgressSignalPayload(DeepAgentsModel):
    status: str | None = None
    percent_complete: float | None = Field(default=None, ge=0, le=100)
    items_processed: int | None = Field(default=None, ge=0)
    estimated_remaining_seconds: int | None = Field(default=None, ge=0)
    data: JsonObject = Field(default_factory=dict)
    relevance_score: float | None = Field(default=None, ge=0, le=1)
    actionable: bool | None = None
    error_type: str | None = None
    detail: str | None = None
    self_recovering: bool | None = None
    urgency: str | None = None
    reason: str | None = None


class ProgressSignal(DeepAgentsModel):
    task_id: str
    signal_type: ProgressSignalType
    payload: ProgressSignalPayload = Field(default_factory=ProgressSignalPayload)
    timestamp: str = Field(default_factory=lambda: utc_now().isoformat())
