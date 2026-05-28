from __future__ import annotations

from pydantic import Field

from deep_agents.models.base import DeepAgentsModel, JsonObject
from deep_agents.models.planning import DiscoveryPlan


class PlannerInput(DeepAgentsModel):
    """Input for the initial discovery planner."""

    objective: str
    constraints: list[str] = Field(default_factory=list)
    available_tools: list[str] = Field(default_factory=list)
    available_skills: list[str] = Field(default_factory=list)
    context: JsonObject = Field(default_factory=dict)


class ExecutionPlannerInput(DeepAgentsModel):
    """Input for converting a discovery plan into an execution plan."""

    discovery_plan: DiscoveryPlan
    available_tools: list[str] = Field(default_factory=list)
    available_skills: list[str] = Field(default_factory=list)
    context: JsonObject = Field(default_factory=dict)
