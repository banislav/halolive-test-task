from __future__ import annotations

from pydantic import Field

from deep_agents.models import ArtifactRef, DeepAgentsModel
from deep_agents.models.base import JsonObject


class TaskRunResult(DeepAgentsModel):
    """Structured output from a worker task invocation."""

    task_id: str
    output: JsonObject = Field(default_factory=dict)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
