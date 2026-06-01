from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field, field_validator

from deep_agents.models.base import DeepAgentsModel, JsonObject


class SkillLoadMode(StrEnum):
    FULL = "full"
    PROGRESSIVE = "progressive"
    ON_DEMAND = "on_demand"


class SkillPriority(StrEnum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    OPTIONAL = "optional"


class SkillAssignment(DeepAgentsModel):
    id: str
    load_mode: SkillLoadMode = SkillLoadMode.PROGRESSIVE
    priority: SkillPriority = SkillPriority.PRIMARY
    context_budget_tokens: int | None = Field(default=None, ge=0)


class SkillTool(DeepAgentsModel):
    id: str
    schema_: JsonObject = Field(default_factory=dict, alias="schema")


class SkillSubSkill(DeepAgentsModel):
    id: str
    trigger: str


class SkillResource(DeepAgentsModel):
    type: str
    path: str
    load_trigger: str | None = None


class SkillContextCost(DeepAgentsModel):
    base_prompt_tokens: int = Field(default=0, ge=0, alias="base_prompt")
    sub_skill_tokens_each: int = Field(default=0, ge=0, alias="sub_skills")

    @field_validator("base_prompt_tokens", "sub_skill_tokens_each", mode="before")
    @classmethod
    def parse_token_count(cls, value: Any) -> Any:
        if isinstance(value, str):
            digits = "".join(char for char in value if char.isdigit())
            return int(digits) if digits else 0
        return value


class SkillDefinition(DeepAgentsModel):
    id: str
    name: str
    version: str = "0.1.0"
    prompt: str
    sub_skills: list[SkillSubSkill] = Field(default_factory=list)
    tools: list[SkillTool] = Field(default_factory=list)
    resources: list[SkillResource] = Field(default_factory=list)
    compatible_agent_types: list[str] = Field(default_factory=list)
    context_cost: SkillContextCost = Field(default_factory=SkillContextCost)
