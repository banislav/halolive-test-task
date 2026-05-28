import pytest
from pydantic import ValidationError

from deep_agents.models import (
    AcceptanceCriterion,
    AgentAssignment,
    AgentKind,
    DiscoveryPlan,
    ExecutionPlan,
    JudgeRecommendation,
    JudgeVerdict,
    Milestone,
    Objective,
    PlanState,
    PromptQueueItem,
    SkillAssignment,
    SkillLoadMode,
    Task,
    TaskCard,
    Wave,
)


def test_discovery_plan_and_plan_state_share_objective() -> None:
    objective = Objective(raw="Build a deep-agent runtime")
    discovery = DiscoveryPlan(
        objective=objective,
        milestones=[
            Milestone(
                id="M1",
                name="Planning",
                tasks=[
                    Task(
                        id="T1",
                        name="Define contracts",
                        acceptance_criteria=[
                            AcceptanceCriterion(description="Core models are typed")
                        ],
                    )
                ],
            )
        ],
    )

    state = PlanState(objective=objective, discovery_plan=discovery)

    assert state.discovery_plan is discovery
    assert state.status == "initializing"


def test_plan_state_rejects_mismatched_discovery_objective() -> None:
    with pytest.raises(ValidationError, match="plan objective must match"):
        PlanState(
            objective=Objective(raw="Original objective"),
            discovery_plan=DiscoveryPlan(objective=Objective(raw="Different objective")),
        )


def test_execution_plan_validates_wave_task_references() -> None:
    task_card = TaskCard(
        id="T1",
        name="Gather sources",
        wave=0,
        assigned_to=AgentAssignment(
            type=AgentKind.WORKER,
            name="ResearchWorker",
            skills=[SkillAssignment(id="academic_research", load_mode=SkillLoadMode.PROGRESSIVE)],
        ),
    )

    plan = ExecutionPlan(
        id="EP1",
        objective="Research a topic",
        waves=[Wave(index=0, task_ids=["T1"])],
        task_cards=[task_card],
    )

    assert plan.task_cards[0].assigned_to.name == "ResearchWorker"


def test_execution_plan_rejects_unknown_wave_task_id() -> None:
    with pytest.raises(ValidationError, match="unknown task ids"):
        ExecutionPlan(
            id="EP1",
            objective="Research a topic",
            waves=[Wave(index=0, task_ids=["missing"])],
            task_cards=[],
        )


def test_judge_verdict_and_prompt_priority() -> None:
    verdict = JudgeVerdict(
        task_id="T1",
        verdict="pass",
        overall_confidence=0.95,
        recommendation=JudgeRecommendation.ADVANCE,
    )
    prompt = PromptQueueItem(id="P1", content="Stop and change direction", priority=1)

    assert verdict.recommendation == "advance"
    assert prompt.is_lifo


def test_judge_verdict_accepts_provider_string_criteria_results() -> None:
    verdict = JudgeVerdict(
        task_id="T1",
        verdict="pass",
        criteria_results=["Output includes a concise project summary: Met"],
        overall_confidence=0.95,
        recommendation=JudgeRecommendation.ADVANCE,
    )

    assert verdict.criteria_results[0].criterion == "Output includes a concise project summary"
    assert verdict.criteria_results[0].met is True
    assert verdict.criteria_results[0].evidence == "Output includes a concise project summary: Met"


def test_judge_verdict_accepts_hold_recommendation() -> None:
    verdict = JudgeVerdict(
        task_id="T1",
        verdict="partial",
        criteria_results=["Output includes a concise project summary: Not met"],
        overall_confidence=0.7,
        recommendation="hold",
    )

    assert verdict.recommendation == "hold"


def test_judge_verdict_accepts_block_recommendation() -> None:
    verdict = JudgeVerdict(
        task_id="T1",
        verdict="partial",
        criteria_results=["Required input is missing: Not met"],
        overall_confidence=0.7,
        recommendation="block",
    )

    assert verdict.recommendation == "block"
