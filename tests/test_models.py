import pytest
from pydantic import ValidationError

from deep_agents.models import (
    AcceptanceCriterion,
    AgentAssignment,
    AgentKind,
    AgentProfile,
    Clarification,
    DiscoveryPlan,
    ExecutionPlan,
    ExecutionPlannerInput,
    GateDecision,
    GateJudgment,
    HandoffStep,
    JudgeRecommendation,
    JudgeVerdict,
    Milestone,
    Objective,
    PlannerInput,
    PlanState,
    PromptQueueItem,
    Risk,
    SkillAssignment,
    SkillLoadMode,
    Task,
    TaskCard,
    TopologyPattern,
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


def test_discovery_plan_matches_architecture_artifacts() -> None:
    plan = DiscoveryPlan(
        objective=Objective(raw="Research a topic"),
        clarifications=[
            Clarification(
                question="Which topic?",
                resolution="Assumed deep-agent architecture",
            )
        ],
        milestones=[
            Milestone(
                id="M1",
                name="Research Phase",
                gates=["G1"],
                tasks=[
                    Task(
                        id="T1",
                        name="Gather sources",
                        acceptance_criteria=[
                            AcceptanceCriterion(
                                description="Minimum 5 credible sources identified"
                            )
                        ],
                        tools_needed=["web_search", "file_write"],
                        skills_needed=["academic_research"],
                        estimated_complexity="medium",
                        risks=[
                            Risk(
                                description="Sources may be paywalled",
                                fallback="Use cached/archive versions",
                            )
                        ],
                    )
                ],
            )
        ],
        capability_map={"T1": ["web_search", "file_write"]},
        skill_assignments={"T1": ["academic_research"]},
        risk_register=[
            Risk(description="Sources may be paywalled", fallback="Use cached/archive versions")
        ],
        dependency_graph={"T1": []},
    )

    assert plan.clarifications[0].resolution == "Assumed deep-agent architecture"
    assert plan.milestones[0].tasks[0].risks[0].fallback == "Use cached/archive versions"


def test_planner_input_models_hold_structured_context() -> None:
    discovery = DiscoveryPlan(objective=Objective(raw="Research a topic"))
    planner_input = PlannerInput(
        objective="Research a topic",
        constraints=["Use credible sources"],
        available_tools=["web_search"],
        available_skills=["academic_research"],
        context={"audience": "engineers"},
    )
    execution_input = ExecutionPlannerInput(
        discovery_plan=discovery,
        available_tools=planner_input.available_tools,
        available_skills=planner_input.available_skills,
        context=planner_input.context,
    )

    assert planner_input.context["audience"] == "engineers"
    assert execution_input.discovery_plan is discovery
    assert execution_input.available_skills == ["academic_research"]


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


def test_task_card_matches_architecture_schema() -> None:
    card = TaskCard(
        id="T3",
        name="Search academic papers",
        wave=1,
        blocked_by=["T1", "T2"],
        blocks=["T6"],
        assigned_to=AgentAssignment(
            type=AgentKind.WORKER,
            name="ResearchWorker",
            skills=[SkillAssignment(id="academic_research", load_mode=SkillLoadMode.PROGRESSIVE)],
        ),
        invocation={
            "method": "async_dispatch",
            "input": {"query": "deep agents", "max_results": 5},
            "input_schema": {"query": "string", "max_results": "int"},
            "expected_output_schema": {
                "results": "list[{title, url, abstract, relevance_score}]",
                "artifacts": "list[filepath]",
            },
            "timeout_seconds": 120,
            "retry_policy": {
                "max_retries": 2,
                "backoff": "exponential",
                "on_exhaust": "escalate_to_replanner",
            },
        },
        acceptance_criteria=[
            AcceptanceCriterion(description="At least 5 results with relevance_score > 0.7")
        ],
        responsiveness={
            "heartbeat_interval_seconds": 15,
            "progress_events": True,
            "early_findings_enabled": True,
        },
        context_budget={"max_tokens": 4000},
        estimated_complexity="medium",
        risks=[Risk(description="Sources may be paywalled", fallback="Use archive versions")],
    )

    assert card.invocation.input["query"] == "deep agents"
    assert card.risks[0].fallback == "Use archive versions"


def test_execution_plan_models_multi_agent_topology_and_handoffs() -> None:
    profile = AgentProfile(
        id="analysis_worker",
        name="AnalysisWorker",
        type=AgentKind.SPECIALIST,
        description="Analyzes extracted data.",
    )
    card = TaskCard(
        id="T1",
        name="Complete browser workflow",
        wave=0,
        assigned_to=AgentAssignment(type=AgentKind.WORKER, name="BrowserWorker"),
        handoff_chain=[
            HandoffStep(
                id="fill_form",
                name="Fill form",
                assigned_to=AgentAssignment(type=AgentKind.SPECIALIST, name="FormFiller"),
                instruction="Fill the required form fields.",
            ),
            HandoffStep(
                id="extract_confirmation",
                name="Extract confirmation",
                assigned_to=AgentAssignment(
                    type=AgentKind.SPECIALIST,
                    name="DataExtractor",
                    agent_id=profile.id,
                ),
                instruction="Extract confirmation data.",
            ),
        ],
    )
    plan = ExecutionPlan(
        id="EP-topology",
        objective="Exercise topology",
        waves=[Wave(index=0, task_ids=["T1"], topology=TopologyPattern.HANDOFFS)],
        task_cards=[card],
    )

    assert plan.waves[0].topology == TopologyPattern.HANDOFFS
    assert plan.task_cards[0].handoff_chain[1].assigned_to.agent_id == "analysis_worker"
    assert profile.type == AgentKind.SPECIALIST


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


def test_gate_judgment_matches_checkpoint_judge_schema() -> None:
    judgment = GateJudgment(
        gate_id="G1",
        milestone_id="M1",
        decision=GateDecision.OPEN,
        criteria_results=["All M1 tasks pass acceptance criteria: Met"],
        overall_confidence=0.91,
        reasoning="All milestone tasks passed their task completion judges.",
    )

    assert judgment.decision == "open"
    assert judgment.criteria_results[0].met is True
