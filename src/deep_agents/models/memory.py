from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from deep_agents.models.base import DeepAgentsModel, JsonObject, utc_now


class MemoryKind(StrEnum):
    """Architecture-native memory categories."""

    WORKING = "working"
    SESSION = "session"
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"
    SKILL = "skill"


class MemoryRecord(DeepAgentsModel):
    """One auditable memory entry."""

    id: str
    kind: MemoryKind
    scope: JsonObject = Field(default_factory=dict)
    payload: JsonObject = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    task_id: str | None = None
    plan_id: str | None = None
    source: str
    timestamp: str = Field(default_factory=lambda: utc_now().isoformat())


class MemoryQuery(DeepAgentsModel):
    """Filter for retrieving memory records."""

    kinds: list[MemoryKind] = Field(default_factory=list)
    task_ids: list[str] = Field(default_factory=list)
    plan_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    text_query: str | None = None
    limit: int | None = Field(default=None, gt=0)
