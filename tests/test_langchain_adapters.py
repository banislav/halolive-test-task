from __future__ import annotations

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableLambda

from deep_agents.config import DeepAgentsSettings
from deep_agents.langchain import (
    build_checkpoint_judge,
    build_checkpoint_judge_messages,
    build_content_reasoning_agent,
    build_content_reasoning_messages,
    build_execution_planner_messages,
    build_judge_messages,
    build_prompt_classifier,
    build_prompt_classifier_messages,
    build_task_completion_judge,
    build_task_worker,
    build_worker_messages,
)
from deep_agents.langchain import judges as judge_module
from deep_agents.langchain import prompt_handlers as prompt_handler_module
from deep_agents.langchain import workers as worker_module
from deep_agents.models import (
    AcceptanceCriterion,
    AgentAssignment,
    AgentKind,
    DiscoveryPlan,
    ExecutionPlan,
    ExecutionPlannerInput,
    Gate,
    GateDecision,
    GateJudgment,
    HandoffStep,
    JudgeRecommendation,
    JudgeVerdict,
    Objective,
    PlanState,
    PromptCategory,
    PromptClassification,
    PromptQueueItem,
    PromptReasoningInput,
    PromptResponse,
    SkillAssignment,
    SkillDefinition,
    TaskCard,
    Wave,
)
from deep_agents.runtime import HandoffStepInput, TaskExecutionContext, TaskRunResult
from deep_agents.skills import SkillLoader, SkillRegistry


class StubStructuredChatModel:
    def __init__(self, response: object) -> None:
        self.response = response
        self.requested_schema: object | None = None

    def with_structured_output(self, schema: object) -> RunnableLambda:
        self.requested_schema = schema
        return RunnableLambda(lambda _: self.response)


def build_task_card() -> TaskCard:
    return TaskCard(
        id="T1",
        name="Draft summary",
        wave=0,
        assigned_to=AgentAssignment(
            type=AgentKind.WORKER,
            name="WriterWorker",
            skills=[SkillAssignment(id="technical_writing")],
        ),
        acceptance_criteria=[
            AcceptanceCriterion(description="Output includes a concise summary")
        ],
    )


def build_execution_plan() -> ExecutionPlan:
    return ExecutionPlan(
        id="EP1",
        objective="Test plan",
        waves=[Wave(index=0, task_ids=["T1"])],
        task_cards=[build_task_card()],
    )


def test_worker_prompt_includes_task_context() -> None:
    messages = build_worker_messages(build_task_card())

    content = _joined_message_content(messages)
    assert "JSON" in content
    assert "Task id: T1" in content
    assert "WriterWorker" in content
    assert "technical_writing" in content


def test_worker_prompt_includes_loaded_skill_context() -> None:
    messages = build_worker_messages(
        build_task_card(),
        skill_context="Loaded skills:\n- Technical Writing\n  instructions:\n    Be concise.",
    )

    content = _joined_message_content(messages)
    assert "Loaded skills:" in content
    assert "Be concise." in content


def test_worker_prompt_includes_assembled_task_context() -> None:
    context = TaskExecutionContext(
        task=build_task_card(),
        objective=Objective(raw="Draft a project summary"),
        plan_context={"blocked_by": ["T0"]},
        dependency_results={
            "T0": {
                "task_id": "T0",
                "output": {"notes": "important background"},
            }
        },
        skill_context="Loaded skills:\n- Technical Writing\n  instructions:\n    Be concise.",
        loaded_skill_ids=["technical_writing"],
    )

    messages = build_worker_messages(context)

    content = _joined_message_content(messages)
    assert "Objective:" in content
    assert "Draft a project summary" in content
    assert "Direct dependency results JSON:" in content
    assert "important background" in content
    assert "Loaded skills:" in content


def test_worker_prompt_includes_handoff_step_context_without_routing_chatter() -> None:
    step_input = HandoffStepInput(
        parent_task=build_task_card(),
        step=HandoffStep(
            id="extract_confirmation",
            name="Extract confirmation",
            assigned_to=AgentAssignment(type=AgentKind.SPECIALIST, name="DataExtractor"),
            instruction="Extract confirmation data.",
        ),
        previous_output={"form": "submitted"},
        shared_state={"fill_form": {"status": "done"}},
    )

    messages = build_worker_messages(step_input)

    content = _joined_message_content(messages)
    assert "intra-task handoff agent" in content
    assert "Handoff step JSON:" in content
    assert "Previous step output JSON:" in content
    assert "Extract confirmation data." in content
    assert "do not paraphrase" in content


def test_judge_prompt_includes_task_result() -> None:
    messages = build_judge_messages(
        build_task_card(),
        TaskRunResult(task_id="T1", output={"summary": "Done"}),
    )

    content = _joined_message_content(messages)
    assert "JSON" in content
    assert "Task result:" in content
    assert '"summary": "Done"' in content


def test_execution_planner_prompt_includes_topology_rules() -> None:
    messages = build_execution_planner_messages(
        ExecutionPlannerInput(
            discovery_plan=DiscoveryPlan(objective=Objective(raw="Test plan")),
        )
    )

    content = _joined_message_content(messages)
    assert "Select a topology per wave" in content
    assert "subagents" in content
    assert "handoff_chain" in content
    assert "Inter-task dependencies are not handoffs" in content


def test_checkpoint_judge_prompt_includes_gate_and_runtime_context() -> None:
    messages = build_checkpoint_judge_messages(
        Gate(
            id="G1",
            condition="All milestone tasks pass acceptance criteria",
            action_on_fail="replan",
        ),
        plan_state=PlanState(objective=Objective(raw="Test plan")),
        execution_plan=build_execution_plan(),
        results={"T1": TaskRunResult(task_id="T1", output={"summary": "Done"})},
    )

    content = _joined_message_content(messages)
    assert "JSON" in content
    assert "Gate JSON:" in content
    assert "G1" in content
    assert "Completed task results JSON:" in content
    assert '"summary": "Done"' in content


def test_prompt_classifier_prompt_includes_prompt_and_json_instruction() -> None:
    messages = build_prompt_classifier_messages(
        PromptQueueItem(id="P1", content="What is the status?")
    )

    content = _joined_message_content(messages)
    assert "JSON" in content
    assert "Queued prompt JSON:" in content
    assert "What is the status?" in content


def test_content_reasoning_prompt_includes_runtime_context() -> None:
    messages = build_content_reasoning_messages(
        PromptReasoningInput(
            prompt=PromptQueueItem(id="P1", content="What is done?"),
            plan_state=PlanState(objective=Objective(raw="Test plan")),
            results={"T1": {"output": {"summary": "Done"}}},
        )
    )

    content = _joined_message_content(messages)
    assert "JSON" in content
    assert "Plan state JSON:" in content
    assert "Completed results JSON:" in content
    assert '"summary": "Done"' in content


def test_task_worker_factory_uses_structured_output() -> None:
    model = StubStructuredChatModel(
        {"task_id": "T1", "output": {"summary": "Done"}, "artifacts": []}
    )

    worker = build_task_worker(model=model)  # type: ignore[arg-type]
    result = worker.invoke(build_task_card())

    assert model.requested_schema is TaskRunResult
    assert result == TaskRunResult(task_id="T1", output={"summary": "Done"})


def test_task_worker_factory_injects_loaded_skills() -> None:
    captured_prompts: list[str] = []

    class CapturingStructuredChatModel(StubStructuredChatModel):
        def with_structured_output(self, schema: object) -> RunnableLambda:
            self.requested_schema = schema

            def capture(messages: list[BaseMessage]) -> dict[str, object]:
                captured_prompts.append(_joined_message_content(messages))
                return {"task_id": "T1", "output": {"summary": "Done"}, "artifacts": []}

            return RunnableLambda(capture)

    skill_loader = SkillLoader(
        SkillRegistry(
            [
                SkillDefinition(
                    id="technical_writing",
                    name="Technical Writing",
                    prompt="Be concise and specific.",
                )
            ]
        )
    )

    worker = build_task_worker(
        model=CapturingStructuredChatModel({}),  # type: ignore[arg-type]
        skill_loader=skill_loader,
    )
    worker.invoke(build_task_card())

    assert "Loaded skills:" in captured_prompts[0]
    assert "Be concise and specific." in captured_prompts[0]


def test_task_completion_judge_factory_uses_structured_output() -> None:
    verdict = JudgeVerdict(
        task_id="T1",
        verdict="pass",
        overall_confidence=0.95,
        recommendation=JudgeRecommendation.ADVANCE,
    )
    model = StubStructuredChatModel(verdict)

    judge = build_task_completion_judge(model=model)  # type: ignore[arg-type]
    result = judge.invoke(
        {
            "task": build_task_card(),
            "result": TaskRunResult(task_id="T1", output={"summary": "Done"}),
        }
    )

    assert model.requested_schema is JudgeVerdict
    assert result == verdict


def test_checkpoint_judge_factory_uses_structured_output() -> None:
    judgment = GateJudgment(
        gate_id="G1",
        decision=GateDecision.OPEN,
        overall_confidence=0.95,
        reasoning="Gate is satisfied.",
    )
    model = StubStructuredChatModel(judgment)

    judge = build_checkpoint_judge(model=model)  # type: ignore[arg-type]
    result = judge.invoke(
        {
            "gate": Gate(id="G1", condition="All tasks pass"),
            "plan_state": {
                "objective": {"raw": "Test plan"},
            },
            "execution_plan": build_execution_plan(),
            "results": {"T1": TaskRunResult(task_id="T1", output={"summary": "Done"})},
        }
    )

    assert model.requested_schema is GateJudgment
    assert result == judgment


def test_prompt_classifier_factory_uses_structured_output() -> None:
    classification = PromptClassification(
        prompt_id="P1",
        category=PromptCategory.CONTENT_REASONING,
        priority=3,
        reasoning="Prompt asks for status.",
    )
    model = StubStructuredChatModel(classification)

    classifier = build_prompt_classifier(model=model)  # type: ignore[arg-type]
    result = classifier.invoke(PromptQueueItem(id="P1", content="What is the status?"))

    assert model.requested_schema is PromptClassification
    assert result == classification


def test_content_reasoning_factory_uses_structured_output() -> None:
    response = PromptResponse(prompt_id="P1", answer="Task T1 is complete.")
    model = StubStructuredChatModel(response)

    reasoner = build_content_reasoning_agent(model=model)  # type: ignore[arg-type]
    result = reasoner.invoke(
        PromptReasoningInput(
            prompt=PromptQueueItem(id="P1", content="What is done?"),
            plan_state=PlanState(objective=Objective(raw="Test plan")),
            results={"T1": {"output": {"summary": "Done"}}},
        )
    )

    assert model.requested_schema is PromptResponse
    assert result == response


def test_runnable_factories_forward_explicit_settings(monkeypatch) -> None:
    settings = DeepAgentsSettings(provider="openrouter", model="qwen/qwen3.6-flash")
    captured: list[DeepAgentsSettings | None] = []

    def build_stub_model(
        received_settings: DeepAgentsSettings | None = None,
    ) -> StubStructuredChatModel:
        captured.append(received_settings)
        return StubStructuredChatModel(
            {"task_id": "T1", "output": {"summary": "Done"}, "artifacts": []}
        )

    monkeypatch.setattr(worker_module, "build_chat_model", build_stub_model)
    build_task_worker(settings=settings)

    assert captured == [settings]


def test_judge_factory_forwards_explicit_settings(monkeypatch) -> None:
    settings = DeepAgentsSettings(provider="openrouter", model="qwen/qwen3.6-flash")
    verdict = JudgeVerdict(
        task_id="T1",
        verdict="pass",
        overall_confidence=0.95,
        recommendation=JudgeRecommendation.ADVANCE,
    )
    captured: list[DeepAgentsSettings | None] = []

    def build_stub_model(
        received_settings: DeepAgentsSettings | None = None,
    ) -> StubStructuredChatModel:
        captured.append(received_settings)
        return StubStructuredChatModel(verdict)

    monkeypatch.setattr(judge_module, "build_chat_model", build_stub_model)
    build_task_completion_judge(settings=settings)

    assert captured == [settings]


def test_checkpoint_judge_factory_forwards_explicit_settings(monkeypatch) -> None:
    settings = DeepAgentsSettings(provider="openrouter", model="qwen/qwen3.6-flash")
    judgment = GateJudgment(
        gate_id="G1",
        decision=GateDecision.OPEN,
        overall_confidence=0.95,
        reasoning="Gate is satisfied.",
    )
    captured: list[DeepAgentsSettings | None] = []

    def build_stub_model(
        received_settings: DeepAgentsSettings | None = None,
    ) -> StubStructuredChatModel:
        captured.append(received_settings)
        return StubStructuredChatModel(judgment)

    monkeypatch.setattr(judge_module, "build_chat_model", build_stub_model)
    build_checkpoint_judge(settings=settings)

    assert captured == [settings]


def test_prompt_classifier_factory_forwards_explicit_settings(monkeypatch) -> None:
    settings = DeepAgentsSettings(provider="openrouter", model="qwen/qwen3.6-flash")
    classification = PromptClassification(
        prompt_id="P1",
        category=PromptCategory.CONTENT_REASONING,
        priority=3,
        reasoning="Prompt asks for status.",
    )
    captured: list[DeepAgentsSettings | None] = []

    def build_stub_model(
        received_settings: DeepAgentsSettings | None = None,
    ) -> StubStructuredChatModel:
        captured.append(received_settings)
        return StubStructuredChatModel(classification)

    monkeypatch.setattr(prompt_handler_module, "build_chat_model", build_stub_model)
    build_prompt_classifier(settings=settings)

    assert captured == [settings]


def _joined_message_content(messages: list[BaseMessage]) -> str:
    return "\n".join(str(message.content) for message in messages)
