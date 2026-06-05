from __future__ import annotations

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableLambda

from deep_agents.langchain import (
    build_discovery_plan_builder,
    build_execution_planner,
    build_execution_planner_messages,
    build_initial_planner,
    build_initial_planner_messages,
    build_planning_pipeline,
)
from deep_agents.models import (
    AcceptanceCriterion,
    AgentAssignment,
    AgentKind,
    DiscoveryPlan,
    ExecutionPlan,
    ExecutionPlannerInput,
    Milestone,
    Objective,
    PlannerInput,
    SkillAssignment,
    Task,
    TaskCard,
    Wave,
)


class StubStructuredChatModel:
    def __init__(self, response: object) -> None:
        self.response = response
        self.requested_schema: object | None = None

    def with_structured_output(self, schema: object) -> RunnableLambda:
        self.requested_schema = schema
        return RunnableLambda(lambda _: self.response)


def build_discovery_plan() -> DiscoveryPlan:
    return DiscoveryPlan(
        objective=Objective(raw="Research deep agents"),
        milestones=[
            Milestone(
                id="M1",
                name="Research",
                tasks=[
                    Task(
                        id="T1",
                        name="Gather sources",
                        acceptance_criteria=[
                            AcceptanceCriterion(description="Find credible sources")
                        ],
                        skills_needed=["academic_research"],
                    )
                ],
            )
        ],
        capability_map={"T1": ["web_search"]},
        skill_assignments={"T1": ["academic_research"]},
        dependency_graph={"T1": []},
    )


def build_execution_plan() -> ExecutionPlan:
    return ExecutionPlan(
        id="EP1",
        objective="Research deep agents",
        waves=[Wave(index=0, task_ids=["T1"])],
        task_cards=[
            TaskCard(
                id="T1",
                name="Gather sources",
                wave=0,
                assigned_to=AgentAssignment(
                    type=AgentKind.WORKER,
                    name="ResearchWorker",
                    skills=[SkillAssignment(id="academic_research")],
                ),
                invocation={
                    "input_schema": {"query": "string"},
                    "expected_output_schema": {"results": "list"},
                },
            )
        ],
        data_flow={"T1": []},
    )


def test_initial_planner_prompt_includes_inputs_and_json_instruction() -> None:
    messages = build_initial_planner_messages(
        PlannerInput(
            objective="Research deep agents",
            constraints=["Use recent sources"],
            available_tools=["web_search"],
            available_skills=["academic_research"],
            context={"audience": "engineers"},
        )
    )

    content = _joined_message_content(messages)
    assert "JSON" in content
    assert "Research deep agents" in content
    assert "Use recent sources" in content
    assert "web_search" in content
    assert "academic_research" in content


def test_execution_planner_prompt_includes_discovery_plan_and_json_instruction() -> None:
    messages = build_execution_planner_messages(
        ExecutionPlannerInput(
            discovery_plan=build_discovery_plan(),
            available_tools=["web_search"],
            available_skills=["academic_research"],
        )
    )

    content = _joined_message_content(messages)
    assert "JSON" in content
    assert "DiscoveryPlan JSON:" in content
    assert "Gather sources" in content
    assert "web_search" in content


def test_initial_planner_factory_uses_discovery_plan_structured_output() -> None:
    model = StubStructuredChatModel(build_discovery_plan())

    planner = build_initial_planner(model=model)  # type: ignore[arg-type]
    result = planner.invoke(PlannerInput(objective="Research deep agents"))

    assert model.requested_schema is DiscoveryPlan
    assert result.objective.raw == "Research deep agents"


def test_discovery_plan_builder_converts_raw_prompt_with_defaults() -> None:
    captured_inputs: list[PlannerInput] = []
    discovery_plan = build_discovery_plan()

    def run_initial(planner_input: PlannerInput) -> DiscoveryPlan:
        captured_inputs.append(planner_input)
        return discovery_plan

    builder = build_discovery_plan_builder(
        RunnableLambda(run_initial),
        constraints=["Use recent sources"],
        available_tools=["web_search"],
        available_skills=["academic_research"],
        context={"audience": "engineers"},
    )

    result = builder.invoke("Research deep agents")

    assert result is discovery_plan
    assert captured_inputs[0] == PlannerInput(
        objective="Research deep agents",
        constraints=["Use recent sources"],
        available_tools=["web_search"],
        available_skills=["academic_research"],
        context={"audience": "engineers"},
    )


def test_discovery_plan_builder_preserves_planner_input() -> None:
    captured_inputs: list[PlannerInput] = []
    discovery_plan = build_discovery_plan()

    def run_initial(planner_input: PlannerInput) -> DiscoveryPlan:
        captured_inputs.append(planner_input)
        return discovery_plan

    planner_input = PlannerInput(
        objective="Research deep agents",
        constraints=["Use local defaults only if raw string"],
        available_tools=["custom_tool"],
        available_skills=["custom_skill"],
        context={"source": "caller"},
    )
    builder = build_discovery_plan_builder(
        RunnableLambda(run_initial),
        constraints=["ignored"],
        available_tools=["ignored"],
        available_skills=["ignored"],
        context={"ignored": True},
    )

    builder.invoke(planner_input)

    assert captured_inputs[0] is planner_input


def test_discovery_plan_builder_coerces_dict_input() -> None:
    captured_inputs: list[PlannerInput] = []

    def run_initial(planner_input: PlannerInput) -> DiscoveryPlan:
        captured_inputs.append(planner_input)
        return build_discovery_plan()

    builder = build_discovery_plan_builder(RunnableLambda(run_initial))

    builder.invoke(
        {
            "objective": "Research deep agents",
            "constraints": ["Use credible sources"],
            "available_tools": ["web_search"],
            "available_skills": ["academic_research"],
            "context": {"audience": "engineers"},
        }
    )

    assert captured_inputs[0].objective == "Research deep agents"
    assert captured_inputs[0].constraints == ["Use credible sources"]
    assert captured_inputs[0].available_tools == ["web_search"]
    assert captured_inputs[0].available_skills == ["academic_research"]
    assert captured_inputs[0].context == {"audience": "engineers"}


def test_discovery_plan_builder_uses_initial_planner_structured_output() -> None:
    model = StubStructuredChatModel(build_discovery_plan())

    builder = build_discovery_plan_builder(model=model)  # type: ignore[arg-type]
    result = builder.invoke("Research deep agents")

    assert model.requested_schema is DiscoveryPlan
    assert result.objective.raw == "Research deep agents"


def test_execution_planner_factory_uses_execution_plan_structured_output() -> None:
    model = StubStructuredChatModel(build_execution_plan())

    planner = build_execution_planner(model=model)  # type: ignore[arg-type]
    result = planner.invoke(ExecutionPlannerInput(discovery_plan=build_discovery_plan()))

    assert model.requested_schema is ExecutionPlan
    assert result.task_cards[0].assigned_to.name == "ResearchWorker"


def test_planning_pipeline_passes_discovery_plan_to_execution_planner() -> None:
    captured_inputs: list[ExecutionPlannerInput] = []
    discovery_plan = build_discovery_plan()
    execution_plan = build_execution_plan()

    initial = RunnableLambda(lambda _: discovery_plan)

    def run_execution(planner_input: ExecutionPlannerInput) -> ExecutionPlan:
        captured_inputs.append(planner_input)
        return execution_plan

    pipeline = build_planning_pipeline(
        initial_planner=initial,
        execution_planner=RunnableLambda(run_execution),
    )

    result = pipeline.invoke(
        PlannerInput(
            objective="Research deep agents",
            available_tools=["web_search"],
            available_skills=["academic_research"],
        )
    )

    assert result is execution_plan
    assert captured_inputs[0].discovery_plan is discovery_plan
    assert captured_inputs[0].available_tools == ["web_search"]
    assert captured_inputs[0].available_skills == ["academic_research"]


def _joined_message_content(messages: list[BaseMessage]) -> str:
    return "\n".join(str(message.content) for message in messages)
