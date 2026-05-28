from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from deep_agents.models.base import DeepAgentsModel
from deep_agents.models.skills import SkillAssignment


class AgentKind(StrEnum):
    WORKER = "worker"
    SPECIALIST = "specialist"
    ORCHESTRATOR = "orchestrator"
    JUDGE = "judge"
    ROUTER = "router"


class AgentLifecycleState(StrEnum):
    SPAWNED = "spawned"
    SKILLS_LOADED = "skills_loaded"
    CONTEXT_LOADED = "context_loaded"
    EXECUTING = "executing"
    REPORTING = "reporting"
    RETRYING = "retrying"
    TERMINATED = "terminated"


class AgentAssignment(DeepAgentsModel):
    type: AgentKind
    name: str
    skills: list[SkillAssignment] = Field(default_factory=list)
