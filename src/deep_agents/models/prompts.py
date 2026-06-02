from __future__ import annotations

from enum import IntEnum, StrEnum

from pydantic import Field

from deep_agents.models.base import DeepAgentsModel, JsonObject, utc_now
from deep_agents.models.planning import PlanState
from deep_agents.models.runtime import RuntimeCommand


class PromptCategory(StrEnum):
    CONTENT_REASONING = "content_reasoning"
    PLAN_UPDATE = "plan_update"


class InterruptPriority(IntEnum):
    P0_HALT = 0
    P1_PAUSE = 1
    P2_REDIRECT = 2
    P3_FEEDBACK = 3
    P4_INFORMATIONAL = 4


class PromptQueueItem(DeepAgentsModel):
    id: str
    content: str
    priority: InterruptPriority = InterruptPriority.P3_FEEDBACK
    category: PromptCategory | None = None
    metadata: JsonObject = Field(default_factory=dict)
    queued_at: str = Field(default_factory=lambda: utc_now().isoformat())

    @property
    def is_lifo(self) -> bool:
        return self.priority in {InterruptPriority.P0_HALT, InterruptPriority.P1_PAUSE}


class PromptClassification(DeepAgentsModel):
    prompt_id: str
    category: PromptCategory
    priority: InterruptPriority
    reasoning: str


class PromptResponse(DeepAgentsModel):
    prompt_id: str
    answer: str
    referenced_task_ids: list[str] = Field(default_factory=list)
    referenced_artifact_ids: list[str] = Field(default_factory=list)


class PromptReasoningInput(DeepAgentsModel):
    prompt: PromptQueueItem
    plan_state: PlanState
    results: dict[str, JsonObject] = Field(default_factory=dict)
    context: JsonObject = Field(default_factory=dict)


class PromptHandlingResult(DeepAgentsModel):
    prompt: PromptQueueItem
    classification: PromptClassification
    response: PromptResponse | None = None
    commands: list[RuntimeCommand] = Field(default_factory=list)
