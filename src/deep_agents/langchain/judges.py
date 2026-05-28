from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda

from deep_agents.langchain.models import build_chat_model
from deep_agents.langchain.prompts import build_judge_messages
from deep_agents.models import JudgeVerdict, TaskCard
from deep_agents.runtime import TaskRunResult


def build_task_completion_judge(
    model: BaseChatModel | None = None,
) -> Runnable[dict[str, Any], JudgeVerdict]:
    """Build a LangChain runnable that judges task results as structured verdicts."""
    chat_model = model or build_chat_model()
    structured_model = chat_model.with_structured_output(JudgeVerdict)
    return RunnableLambda(_build_messages_from_payload) | structured_model | RunnableLambda(
        _coerce_judge_verdict
    )


def _build_messages_from_payload(payload: dict[str, Any]) -> Any:
    task = payload["task"]
    result = payload["result"]
    if not isinstance(task, TaskCard):
        task = TaskCard(**task)
    if not isinstance(result, TaskRunResult):
        result = TaskRunResult(**result)
    return build_judge_messages(task, result)


def _coerce_judge_verdict(value: JudgeVerdict | dict[str, Any]) -> JudgeVerdict:
    if isinstance(value, JudgeVerdict):
        return value
    return JudgeVerdict(**value)
