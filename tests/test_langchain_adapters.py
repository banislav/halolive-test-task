from __future__ import annotations

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableLambda

from deep_agents.langchain import (
    build_judge_messages,
    build_task_completion_judge,
    build_task_worker,
    build_worker_messages,
)
from deep_agents.models import (
    AcceptanceCriterion,
    AgentAssignment,
    AgentKind,
    JudgeRecommendation,
    JudgeVerdict,
    SkillAssignment,
    TaskCard,
)
from deep_agents.runtime import TaskRunResult


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


def test_worker_prompt_includes_task_context() -> None:
    messages = build_worker_messages(build_task_card())

    content = _joined_message_content(messages)
    assert "Task id: T1" in content
    assert "WriterWorker" in content
    assert "technical_writing" in content


def test_judge_prompt_includes_task_result() -> None:
    messages = build_judge_messages(
        build_task_card(),
        TaskRunResult(task_id="T1", output={"summary": "Done"}),
    )

    content = _joined_message_content(messages)
    assert "Task result:" in content
    assert '"summary": "Done"' in content


def test_task_worker_factory_uses_structured_output() -> None:
    model = StubStructuredChatModel(
        {"task_id": "T1", "output": {"summary": "Done"}, "artifacts": []}
    )

    worker = build_task_worker(model=model)  # type: ignore[arg-type]
    result = worker.invoke(build_task_card())

    assert model.requested_schema is TaskRunResult
    assert result == TaskRunResult(task_id="T1", output={"summary": "Done"})


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


def _joined_message_content(messages: list[BaseMessage]) -> str:
    return "\n".join(str(message.content) for message in messages)
