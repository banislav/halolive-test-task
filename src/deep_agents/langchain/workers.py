from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda

from deep_agents.config import DeepAgentsSettings
from deep_agents.langchain.models import build_chat_model
from deep_agents.langchain.prompts import build_worker_messages
from deep_agents.models import TaskCard
from deep_agents.runtime import TaskRunResult


def build_task_worker(
    model: BaseChatModel | None = None,
    settings: DeepAgentsSettings | None = None,
) -> Runnable[TaskCard, TaskRunResult]:
    """Build a LangChain runnable that executes task cards as structured worker results."""
    chat_model = model or build_chat_model(settings)
    structured_model = chat_model.with_structured_output(TaskRunResult)
    return RunnableLambda(build_worker_messages) | structured_model | RunnableLambda(
        _coerce_task_result
    )


def _coerce_task_result(value: TaskRunResult | dict[str, Any]) -> TaskRunResult:
    if isinstance(value, TaskRunResult):
        return value
    return TaskRunResult(**value)
