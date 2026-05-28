from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from deep_agents.models import ArtifactRef, DeepAgentsModel, JudgeVerdict, ProgressSignal
from deep_agents.models.base import JsonObject, utc_now


class RuntimeEventType(StrEnum):
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    JUDGE_VERDICT_RECEIVED = "judge_verdict_received"
    PROGRESS_SIGNAL_RECEIVED = "progress_signal_received"


class RuntimeEvent(DeepAgentsModel):
    type: RuntimeEventType
    task_id: str | None = None
    timestamp: str = Field(default_factory=lambda: utc_now().isoformat())


class TaskCompleted(RuntimeEvent):
    type: RuntimeEventType = RuntimeEventType.TASK_COMPLETED
    task_id: str
    output: JsonObject = Field(default_factory=dict)
    artifacts: list[ArtifactRef] = Field(default_factory=list)


class TaskFailed(RuntimeEvent):
    type: RuntimeEventType = RuntimeEventType.TASK_FAILED
    task_id: str
    error: str
    recoverable: bool = True


class JudgeVerdictReceived(RuntimeEvent):
    type: RuntimeEventType = RuntimeEventType.JUDGE_VERDICT_RECEIVED
    task_id: str
    verdict: JudgeVerdict


class ProgressSignalReceived(RuntimeEvent):
    type: RuntimeEventType = RuntimeEventType.PROGRESS_SIGNAL_RECEIVED
    task_id: str
    signal: ProgressSignal
