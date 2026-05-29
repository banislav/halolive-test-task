from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field, field_validator

from deep_agents.models.base import DeepAgentsModel


class JudgeRecommendation(StrEnum):
    ADVANCE = "advance"
    HOLD = "hold"
    BLOCK = "block"
    RETRY = "retry"
    REPLAN = "replan"
    ESCALATE = "escalate"


class JudgeVerdictValue(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"


class GateDecision(StrEnum):
    OPEN = "open"
    HOLD = "hold"
    REJECT = "reject"
    ESCALATE = "escalate"


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

    @field_validator("criteria_results", mode="before")
    @classmethod
    def normalize_criteria_results(cls, value: Any) -> Any:
        """Accept provider responses that return criteria results as strings."""
        if not isinstance(value, list):
            return value

        normalized: list[Any] = []
        for item in value:
            if isinstance(item, str):
                normalized.append(_criteria_result_from_text(item))
            else:
                normalized.append(item)
        return normalized


class GateJudgment(DeepAgentsModel):
    gate_id: str
    milestone_id: str | None = None
    decision: GateDecision
    criteria_results: list[CriteriaResult] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0, le=1)
    reasoning: str
    actions: list[ProcessAction] = Field(default_factory=list)

    @field_validator("criteria_results", mode="before")
    @classmethod
    def normalize_criteria_results(cls, value: Any) -> Any:
        """Accept provider responses that return criteria results as strings."""
        if not isinstance(value, list):
            return value

        normalized: list[Any] = []
        for item in value:
            if isinstance(item, str):
                normalized.append(_criteria_result_from_text(item))
            else:
                normalized.append(item)
        return normalized


def _criteria_result_from_text(text: str) -> dict[str, Any]:
    criterion, separator, status = text.partition(":")
    status_text = status.strip().lower() if separator else text.strip().lower()
    met = any(token in status_text for token in ("met", "pass", "passed", "true", "yes"))
    unmet = any(token in status_text for token in ("not met", "fail", "failed", "false", "no"))
    return {
        "criterion": criterion.strip() or text,
        "met": False if unmet else met,
        "evidence": text,
    }


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


class ObserverHealth(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    STUCK = "stuck"
    DIVERGING = "diverging"


class ObserverJudgment(DeepAgentsModel):
    health: ObserverHealth
    reasoning: str
    actions: list[ProcessAction] = Field(default_factory=list)
