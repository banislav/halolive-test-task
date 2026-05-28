from __future__ import annotations

from collections.abc import Iterable

from deep_agents.models import SkillDefinition


class SkillRegistry:
    """In-memory registry for skill definitions."""

    def __init__(self, skills: Iterable[SkillDefinition] | None = None) -> None:
        """Create a registry with optional initial skill definitions."""
        self._skills: dict[str, SkillDefinition] = {}
        for skill in skills or ():
            self.register(skill)

    def register(self, skill: SkillDefinition) -> None:
        """Register or replace a skill definition by id."""
        self._skills[skill.id] = skill

    def get(self, skill_id: str) -> SkillDefinition | None:
        """Return a skill definition by id, if present."""
        return self._skills.get(skill_id)

    def require(self, skill_id: str) -> SkillDefinition:
        """Return a skill definition or raise KeyError when missing."""
        skill = self.get(skill_id)
        if skill is None:
            raise KeyError(f"unknown skill id: {skill_id}")
        return skill

    def list(self) -> list[SkillDefinition]:
        """Return all registered skills."""
        return list(self._skills.values())
