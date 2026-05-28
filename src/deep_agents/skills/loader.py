from __future__ import annotations

import json

from pydantic import Field

from deep_agents.models import DeepAgentsModel, SkillAssignment, SkillDefinition, SkillLoadMode
from deep_agents.skills.registry import SkillRegistry


class LoadedSkill(DeepAgentsModel):
    """Skill definition resolved for a specific task assignment."""

    assignment: SkillAssignment
    definition: SkillDefinition
    prompt: str


class SkillLoader:
    """Resolve assigned skills into compact prompt context."""

    def __init__(self, registry: SkillRegistry, *, strict: bool = True) -> None:
        """Create a loader bound to a registry."""
        self.registry = registry
        self.strict = strict

    def load(self, assignments: list[SkillAssignment]) -> list[LoadedSkill]:
        """Resolve assignments into loaded skill prompts."""
        loaded: list[LoadedSkill] = []
        for assignment in assignments:
            definition = self.registry.get(assignment.id)
            if definition is None:
                if self.strict:
                    raise KeyError(f"unknown skill id: {assignment.id}")
                continue
            loaded.append(
                LoadedSkill(
                    assignment=assignment,
                    definition=definition,
                    prompt=self._prompt_for_assignment(assignment, definition),
                )
            )
        return loaded

    def render_context(self, assignments: list[SkillAssignment]) -> str:
        """Render loaded skills as text suitable for worker prompt injection."""
        loaded = self.load(assignments)
        if not loaded:
            return ""

        sections = ["Loaded skills:"]
        for skill in loaded:
            sections.append(
                "\n".join(
                    [
                        f"- {skill.definition.name} ({skill.definition.id})",
                        f"  version: {skill.definition.version}",
                        f"  load_mode: {skill.assignment.load_mode}",
                        f"  priority: {skill.assignment.priority}",
                        _context_cost_text(skill.definition),
                        _compatible_agent_text(skill.definition),
                        "  instructions:",
                        _indent(skill.prompt, spaces=4),
                        _sub_skills_text(skill.definition),
                        _tools_text(skill.definition),
                        _resources_text(skill.definition),
                    ]
                )
            )
        return "\n".join(sections)

    def _prompt_for_assignment(
        self,
        assignment: SkillAssignment,
        definition: SkillDefinition,
    ) -> str:
        if assignment.load_mode == SkillLoadMode.ON_DEMAND:
            return (
                "This skill is available on demand. Request it only if the task needs "
                f"{definition.name} guidance."
            )
        return definition.prompt


class SkillContext(DeepAgentsModel):
    """Rendered skill context attached to a worker invocation."""

    text: str = ""
    loaded: list[LoadedSkill] = Field(default_factory=list)


def _indent(text: str, *, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(f"{prefix}{line}" if line else line for line in text.splitlines())


def _context_cost_text(definition: SkillDefinition) -> str:
    cost = definition.context_cost
    if cost.base_prompt_tokens == 0 and cost.sub_skill_tokens_each == 0:
        return "  context_cost: not specified"
    return (
        "  context_cost: "
        f"base_prompt={cost.base_prompt_tokens}, "
        f"sub_skills={cost.sub_skill_tokens_each}"
    )


def _compatible_agent_text(definition: SkillDefinition) -> str:
    if not definition.compatible_agent_types:
        return "  compatible_agent_types: any"
    return f"  compatible_agent_types: {', '.join(definition.compatible_agent_types)}"


def _sub_skills_text(definition: SkillDefinition) -> str:
    if not definition.sub_skills:
        return "  sub_skills: none"
    lines = ["  sub_skills:"]
    for sub_skill in definition.sub_skills:
        lines.append(f"    - {sub_skill.id}: {sub_skill.trigger}")
    return "\n".join(lines)


def _tools_text(definition: SkillDefinition) -> str:
    if not definition.tools:
        return "  tools: none"
    lines = ["  tools:"]
    for tool in definition.tools:
        schema = json.dumps(tool.schema_, sort_keys=True)
        lines.append(f"    - {tool.id}: {schema}")
    return "\n".join(lines)


def _resources_text(definition: SkillDefinition) -> str:
    if not definition.resources:
        return "  resources: none"
    lines = ["  resources:"]
    for resource in definition.resources:
        trigger = f" when {resource.load_trigger}" if resource.load_trigger else ""
        lines.append(f"    - {resource.type}: {resource.path}{trigger}")
    return "\n".join(lines)
