from __future__ import annotations

from deep_agents.models import (
    AgentAssignment,
    AgentKind,
    ArtifactRef,
    ContextBudget,
    ExecutionPlan,
    Objective,
    PlanState,
    SkillAssignment,
    SkillDefinition,
    TaskCard,
    Wave,
)
from deep_agents.models.context import ArtifactKind
from deep_agents.runtime import ArtifactStore, ContextAssembler, TaskRunResult
from deep_agents.skills import SkillLoader, SkillRegistry


def build_execution_plan() -> ExecutionPlan:
    assignment = AgentAssignment(
        type=AgentKind.WORKER,
        name="Worker",
        skills=[SkillAssignment(id="technical_writing")],
    )
    return ExecutionPlan(
        id="EP-context",
        objective="Draft and review a project summary",
        waves=[
            Wave(index=0, task_ids=["T1"]),
            Wave(index=1, task_ids=["T2"]),
            Wave(index=2, task_ids=["T3"]),
        ],
        task_cards=[
            TaskCard(id="T1", name="Research", wave=0, assigned_to=assignment),
            TaskCard(id="T2", name="Draft", wave=1, assigned_to=assignment),
            TaskCard(
                id="T3",
                name="Review",
                wave=2,
                blocked_by=["T2"],
                assigned_to=assignment,
            ),
        ],
    )


def test_context_assembler_includes_direct_dependencies_and_summarizes_prior_results() -> None:
    plan = build_execution_plan()
    task = plan.task_cards[2]
    assembler = ContextAssembler(summary_max_chars=32)
    context = assembler.assemble(
        task=task,
        execution_plan=plan,
        plan_state=PlanState(objective=Objective(raw=plan.objective)),
        results={
            "T1": TaskRunResult(task_id="T1", output={"research": "background notes"}),
            "T2": TaskRunResult(task_id="T2", output={"draft": "short summary"}),
        },
    )

    assert list(context.dependency_results) == ["T2"]
    assert context.dependency_results["T2"].output == {"draft": "short summary"}
    assert [summary.task_id for summary in context.prior_result_summaries] == ["T1"]
    assert context.prior_result_summaries[0].output == {}
    assert context.prior_result_summaries[0].summary
    assert context.plan_context["blocked_by"] == ["T2"]
    assert context.budget_report.over_budget is False


def test_context_assembler_includes_dependency_artifacts_and_deduplicates_refs() -> None:
    plan = build_execution_plan()
    artifact = ArtifactRef(
        id="A1",
        kind=ArtifactKind.STRUCTURED,
        uri="memory://summary",
        summary="Draft summary",
    )
    plan.task_cards[2].input_artifacts.append(artifact)
    assembler = ContextAssembler(artifact_store=ArtifactStore())
    context = assembler.assemble(
        task=plan.task_cards[2],
        execution_plan=plan,
        plan_state=PlanState(objective=Objective(raw=plan.objective)),
        results={
            "T2": TaskRunResult(
                task_id="T2",
                output={"draft": "short summary"},
                artifacts=[artifact],
            )
        },
    )

    assert context.artifacts == [artifact]
    assert context.layered_context.artifacts == [artifact]


def test_context_assembler_loads_assigned_skill_context() -> None:
    plan = build_execution_plan()
    skill_loader = SkillLoader(
        SkillRegistry(
            [
                SkillDefinition(
                    id="technical_writing",
                    name="Technical Writing",
                    prompt="Write clearly and concretely.",
                )
            ]
        )
    )

    context = ContextAssembler(skill_loader=skill_loader).assemble(
        task=plan.task_cards[0],
        execution_plan=plan,
        plan_state=PlanState(objective=Objective(raw=plan.objective)),
    )

    assert context.loaded_skill_ids == ["technical_writing"]
    assert "Write clearly and concretely." in context.skill_context
    assert context.layered_context.skill_state["loaded_skill_ids"] == ["technical_writing"]


def test_context_assembler_drops_prior_summaries_before_compacting_dependencies() -> None:
    plan = build_execution_plan()
    task = plan.task_cards[2]
    task.context_budget = ContextBudget(max_tokens=180, reserved_response_tokens=0)
    long_text = "x" * 1200

    context = ContextAssembler(summary_max_chars=180).assemble(
        task=task,
        execution_plan=plan,
        plan_state=PlanState(objective=Objective(raw=plan.objective)),
        results={
            "T1": TaskRunResult(task_id="T1", output={"research": long_text}),
            "T2": TaskRunResult(task_id="T2", output={"draft": long_text}),
        },
    )

    assert context.budget_report.dropped_prior_result_ids == ["T1"]
    assert context.budget_report.compacted_result_ids == ["T2"]
    assert context.prior_result_summaries == []
    assert context.dependency_results["T2"].output == {}
    assert context.dependency_results["T2"].summary
    assert context.budget_report.estimated_input_tokens <= 180
