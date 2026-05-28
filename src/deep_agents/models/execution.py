from __future__ import annotations

from pydantic import Field, model_validator

from deep_agents.models.agents import AgentAssignment
from deep_agents.models.base import DeepAgentsModel, JsonObject, TimestampedModel
from deep_agents.models.context import ArtifactRef, ContextBudget
from deep_agents.models.planning import AcceptanceCriterion, Risk


class RetryPolicy(DeepAgentsModel):
    max_retries: int = Field(default=2, ge=0)
    backoff: str = "exponential"
    on_exhaust: str = "escalate_to_replanner"


class TaskInvocation(DeepAgentsModel):
    method: str = "async_dispatch"
    input: JsonObject = Field(default_factory=dict)
    input_schema: JsonObject = Field(default_factory=dict)
    expected_output_schema: JsonObject = Field(default_factory=dict)
    timeout_seconds: int = Field(default=120, gt=0)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)


class TaskResponsiveness(DeepAgentsModel):
    heartbeat_interval_seconds: int = Field(default=15, gt=0)
    progress_events: bool = True
    early_findings_enabled: bool = True


class TaskCard(DeepAgentsModel):
    id: str
    name: str
    wave: int = Field(ge=0)
    blocked_by: list[str] = Field(default_factory=list)
    blocks: list[str] = Field(default_factory=list)
    assigned_to: AgentAssignment
    invocation: TaskInvocation = Field(default_factory=TaskInvocation)
    acceptance_criteria: list[AcceptanceCriterion] = Field(default_factory=list)
    responsiveness: TaskResponsiveness = Field(default_factory=TaskResponsiveness)
    context_budget: ContextBudget = Field(default_factory=ContextBudget)
    input_artifacts: list[ArtifactRef] = Field(default_factory=list)
    estimated_complexity: str = "medium"
    risks: list[Risk] = Field(default_factory=list)


class Wave(DeepAgentsModel):
    index: int = Field(ge=0)
    name: str | None = None
    blocked_by: list[str] = Field(default_factory=list)
    task_ids: list[str] = Field(default_factory=list)


class DependencyGraph(DeepAgentsModel):
    blocked_by: dict[str, list[str]] = Field(default_factory=dict)
    blocks: dict[str, list[str]] = Field(default_factory=dict)


class ExecutionPlan(TimestampedModel):
    id: str
    objective: str
    waves: list[Wave] = Field(default_factory=list)
    task_cards: list[TaskCard] = Field(default_factory=list)
    dependency_graph: DependencyGraph = Field(default_factory=DependencyGraph)
    data_flow: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def task_ids_must_be_unique_and_referenced(self) -> ExecutionPlan:
        task_ids = [task.id for task in self.task_cards]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("task card ids must be unique")

        known = set(task_ids)
        referenced = {task_id for wave in self.waves for task_id in wave.task_ids}
        missing = referenced - known
        if missing:
            raise ValueError(f"wave references unknown task ids: {sorted(missing)}")
        return self
