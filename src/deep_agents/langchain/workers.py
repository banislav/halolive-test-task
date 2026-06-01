from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda

from deep_agents.config import DeepAgentsSettings
from deep_agents.langchain.models import build_chat_model
from deep_agents.langchain.prompts import build_worker_messages
from deep_agents.models import TaskCard
from deep_agents.runtime import TaskExecutionContext, TaskRunResult
from deep_agents.skills import SkillLoader


def build_task_worker(
    model: BaseChatModel | None = None,
    settings: DeepAgentsSettings | None = None,
    skill_loader: SkillLoader | None = None,
) -> Runnable[TaskCard | TaskExecutionContext, TaskRunResult]:
    """Build a LangChain runnable that executes task cards as structured worker results."""
    chat_model = model or build_chat_model(settings)
    structured_model = chat_model.with_structured_output(TaskRunResult)
    prompt_builder = RunnableLambda(
        lambda task_input: _build_worker_messages(task_input, skill_loader)
    )
    return prompt_builder | structured_model | RunnableLambda(
        _coerce_task_result
    )


def _build_worker_messages(
    task_input: TaskCard | TaskExecutionContext,
    skill_loader: SkillLoader | None,
) -> object:
    if isinstance(task_input, TaskExecutionContext):
        return build_worker_messages(task_input)
    return build_worker_messages(
        task_input,
        skill_loader.render_context(task_input.assigned_to.skills) if skill_loader else None,
    )


def _coerce_task_result(value: TaskRunResult | dict[str, Any]) -> TaskRunResult:
    if isinstance(value, TaskRunResult):
        return value
    return TaskRunResult(**value)
