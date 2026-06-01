from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from deep_agents.models.base import DeepAgentsModel, TimestampedModel


class PlanStatus(StrEnum):
    INITIALIZING = "initializing"
    DISCOVERY = "discovery"
    PLANNING = "planning"
    EXECUTING = "executing"
    REFINING = "refining"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class TaskStatus(StrEnum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    TERMINATED = "terminated"
    ROLLED_BACK = "rolled_back"


class GateType(StrEnum):
    QUALITY = "quality_gate"
    SAFETY = "safety_gate"
    HUMAN_APPROVAL = "human_approval"
    DEPENDENCY = "dependency_gate"


class Objective(DeepAgentsModel):
    raw: str
    normalized: str | None = None
    constraints: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)


class AcceptanceCriterion(DeepAgentsModel):
    description: str
    measurable: bool = True


class Risk(DeepAgentsModel):
    description: str
    fallback: str | None = None
    severity: str = "medium"


class Clarification(DeepAgentsModel):
    question: str
    resolution: str | None = None


class Gate(DeepAgentsModel):
    id: str
    type: GateType = GateType.QUALITY
    condition: str
    action_on_fail: str = "replan"


class Task(DeepAgentsModel):
    id: str
    name: str
    description: str | None = None
    acceptance_criteria: list[AcceptanceCriterion] = Field(default_factory=list)
    tools_needed: list[str] = Field(default_factory=list)
    skills_needed: list[str] = Field(default_factory=list)
    estimated_complexity: str = "medium"
    risks: list[Risk] = Field(default_factory=list)
    blocked_by: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING


class Milestone(DeepAgentsModel):
    id: str
    name: str
    gates: list[str] = Field(default_factory=list)
    tasks: list[Task] = Field(default_factory=list)


class DiscoveryPlan(TimestampedModel):
    objective: Objective
    clarifications: list[Clarification] = Field(default_factory=list)
    milestones: list[Milestone] = Field(default_factory=list)
    gates: list[Gate] = Field(default_factory=list)
    capability_map: dict[str, list[str]] = Field(default_factory=dict)
    skill_assignments: dict[str, list[str]] = Field(default_factory=dict)
    risk_register: list[Risk] = Field(default_factory=list)
    dependency_graph: dict[str, list[str]] = Field(default_factory=dict)


class PlanState(TimestampedModel):
    objective: Objective
    status: PlanStatus = PlanStatus.INITIALIZING
    discovery_plan: DiscoveryPlan | None = None
    execution_plan_id: str | None = None
    task_statuses: dict[str, TaskStatus] = Field(default_factory=dict)
    checkpoint_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def sync_discovery_objective(self) -> PlanState:
        if self.discovery_plan and self.discovery_plan.objective.raw != self.objective.raw:
            msg = "plan objective must match discovery plan objective"
            raise ValueError(msg)
        return self
