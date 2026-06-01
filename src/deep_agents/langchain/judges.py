from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda

from deep_agents.config import DeepAgentsSettings
from deep_agents.langchain.models import build_chat_model
from deep_agents.langchain.prompts import build_checkpoint_judge_messages, build_judge_messages
from deep_agents.models import (
    ExecutionPlan,
    Gate,
    GateJudgment,
    JudgeVerdict,
    Milestone,
    PlanState,
    TaskCard,
)
from deep_agents.runtime import TaskRunResult


def build_task_completion_judge(
    model: BaseChatModel | None = None,
    settings: DeepAgentsSettings | None = None,
) -> Runnable[dict[str, Any], JudgeVerdict]:
    """Build a LangChain runnable that judges task results as structured verdicts."""
    chat_model = model or build_chat_model(settings)
    structured_model = chat_model.with_structured_output(JudgeVerdict)
    return RunnableLambda(_build_messages_from_payload) | structured_model | RunnableLambda(
        _coerce_judge_verdict
    )


def build_checkpoint_judge(
    model: BaseChatModel | None = None,
    settings: DeepAgentsSettings | None = None,
) -> Runnable[dict[str, Any], GateJudgment]:
    """Build a LangChain runnable that judges milestone gates as structured output."""
    chat_model = model or build_chat_model(settings)
    structured_model = chat_model.with_structured_output(GateJudgment)
    return (
        RunnableLambda(_build_checkpoint_messages_from_payload)
        | structured_model
        | RunnableLambda(_coerce_gate_judgment)
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


def _build_checkpoint_messages_from_payload(payload: dict[str, Any]) -> Any:
    gate = payload["gate"]
    plan_state = payload["plan_state"]
    execution_plan = payload["execution_plan"]
    milestone = payload.get("milestone")
    results = payload.get("results", {})

    if not isinstance(gate, Gate):
        gate = Gate(**gate)
    if not isinstance(plan_state, PlanState):
        plan_state = PlanState(**plan_state)
    if not isinstance(execution_plan, ExecutionPlan):
        execution_plan = ExecutionPlan(**execution_plan)
    if milestone is not None and not isinstance(milestone, Milestone):
        milestone = Milestone(**milestone)

    coerced_results: dict[str, TaskRunResult] = {}
    for task_id, result in results.items():
        coerced_results[task_id] = (
            result if isinstance(result, TaskRunResult) else TaskRunResult(**result)
        )

    return build_checkpoint_judge_messages(
        gate,
        plan_state,
        execution_plan,
        milestone=milestone,
        results=coerced_results,
    )


def _coerce_gate_judgment(value: GateJudgment | dict[str, Any]) -> GateJudgment:
    if isinstance(value, GateJudgment):
        return value
    return GateJudgment(**value)
