from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from deep_agents.models.base import DeepAgentsModel, JsonObject


class ArtifactKind(StrEnum):
    FILE = "file"
    URL = "url"
    DATASET = "dataset"
    MESSAGE = "message"
    STRUCTURED = "structured"


class ArtifactRef(DeepAgentsModel):
    id: str
    kind: ArtifactKind
    uri: str
    summary: str | None = None
    metadata: JsonObject = Field(default_factory=dict)


class ContextBudget(DeepAgentsModel):
    max_tokens: int = Field(default=4000, gt=0)
    reserved_response_tokens: int = Field(default=500, ge=0)


class LayeredContext(DeepAgentsModel):
    global_objective: JsonObject | None = None
    plan_state: JsonObject | None = None
    execution_state: JsonObject | None = None
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    skill_state: JsonObject = Field(default_factory=dict)
    agent_state: JsonObject = Field(default_factory=dict)
