from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from deep_agents.models.base import DeepAgentsModel


class JudgeRecommendation(StrEnum):
    ADVANCE = "advance"
    RETRY = "retry"
    REPLAN = "replan"
    ESCALATE = "escalate"


class JudgeVerdictValue(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"


class CriteriaResult(DeepAgentsModel):
    criterion: str
    met: bool
    evidence: str | None = None


class JudgeVerdict(DeepAgentsModel):
    task_id: str
    verdict: JudgeVerdictValue
    criteria_results: list[CriteriaResult] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0, le=1)
    recommendation: JudgeRecommendation


class ProcessAssessment(StrEnum):
    NEEDS_MORE_TIME = "needs_more_time"
    HEALTHY = "healthy"
    EARLY_TERMINATE = "early_terminate"
    ESCALATE_HITL = "escalate_hitl"


class ProcessAction(DeepAgentsModel):
    type: str
    value: str | int | float | bool | None = None


class ProcessJudgment(DeepAgentsModel):
    task_id: str
    assessment: ProcessAssessment
    reasoning: str
    actions: list[ProcessAction] = Field(default_factory=list)
