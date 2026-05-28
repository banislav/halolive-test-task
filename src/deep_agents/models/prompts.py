from __future__ import annotations

from enum import IntEnum, StrEnum

from pydantic import Field

from deep_agents.models.base import DeepAgentsModel, JsonObject, utc_now


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
