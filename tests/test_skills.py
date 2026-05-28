import pytest

from deep_agents.models import (
    SkillAssignment,
    SkillDefinition,
    SkillLoadMode,
    SkillResource,
    SkillSubSkill,
    SkillTool,
)
from deep_agents.skills import SkillLoader, SkillRegistry


def build_skill() -> SkillDefinition:
    return SkillDefinition(
        id="technical_writing",
        name="Technical Writing",
        version="1.0.0",
        prompt="Write clearly, prefer concrete nouns, and keep output concise.",
    )


def test_skill_registry_registers_and_requires_skills() -> None:
    registry = SkillRegistry([build_skill()])

    assert registry.require("technical_writing").name == "Technical Writing"
    assert [skill.id for skill in registry.list()] == ["technical_writing"]


def test_skill_registry_raises_for_missing_required_skill() -> None:
    registry = SkillRegistry()

    with pytest.raises(KeyError, match="unknown skill id"):
        registry.require("missing")


def test_skill_loader_renders_assigned_skill_context() -> None:
    loader = SkillLoader(SkillRegistry([build_skill()]))

    context = loader.render_context([SkillAssignment(id="technical_writing")])

    assert "Loaded skills:" in context
    assert "Technical Writing" in context
    assert "Write clearly" in context


def test_skill_loader_can_ignore_missing_skills_when_not_strict() -> None:
    loader = SkillLoader(SkillRegistry(), strict=False)

    assert loader.render_context([SkillAssignment(id="missing")]) == ""


def test_skill_loader_marks_on_demand_skills_as_available() -> None:
    loader = SkillLoader(SkillRegistry([build_skill()]))

    loaded = loader.load(
        [SkillAssignment(id="technical_writing", load_mode=SkillLoadMode.ON_DEMAND)]
    )

    assert "available on demand" in loaded[0].prompt


def test_skill_loader_renders_full_architecture_skill_schema() -> None:
    skill = SkillDefinition(
        id="academic_research",
        name="Academic Research Specialist",
        version="2.1",
        prompt=(
            "You are an expert academic researcher. Prioritize peer-reviewed journals "
            "and cross-reference citations."
        ),
        sub_skills=[
            SkillSubSkill(
                id="citation_analysis",
                trigger="when analyzing citation networks",
            ),
            SkillSubSkill(
                id="methodology_review",
                trigger="when evaluating research methodology",
            ),
        ],
        tools=[
            SkillTool(id="scholar_search", schema={"query": "string", "filters": "object"}),
            SkillTool(id="citation_graph", schema={"paper_ids": "list[string]"}),
        ],
        resources=[
            SkillResource(
                type="template",
                path="/skills/academic_research/templates/literature_review.md",
                load_trigger="when generating a literature review",
            )
        ],
        context_cost={"base_prompt": "800_tokens", "sub_skills": "400_tokens_each"},
        compatible_agent_types=["Worker", "Specialist", "Judge"],
    )
    loader = SkillLoader(SkillRegistry([skill]))

    context = loader.render_context([SkillAssignment(id="academic_research")])

    assert "Academic Research Specialist" in context
    assert "citation_analysis: when analyzing citation networks" in context
    assert "methodology_review: when evaluating research methodology" in context
    assert "scholar_search" in context
    assert "citation_graph" in context
    assert "/skills/academic_research/templates/literature_review.md" in context
    assert "base_prompt=800" in context
    assert "sub_skills=400" in context
    assert "Worker, Specialist, Judge" in context
