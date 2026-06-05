from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda

from deep_agents.config import DeepAgentsSettings
from deep_agents.langchain.models import build_chat_model
from deep_agents.langchain.prompts import (
    build_execution_planner_messages,
    build_initial_planner_messages,
)
from deep_agents.models import DiscoveryPlan, ExecutionPlan, ExecutionPlannerInput, PlannerInput


def build_initial_planner(
    model: BaseChatModel | None = None,
    settings: DeepAgentsSettings | None = None,
) -> Runnable[PlannerInput, DiscoveryPlan]:
    """Build a LangChain runnable that produces discovery plans."""
    chat_model = model or build_chat_model(settings)
    structured_model = chat_model.with_structured_output(DiscoveryPlan)
    return RunnableLambda(_coerce_planner_input) | RunnableLambda(
        build_initial_planner_messages
    ) | structured_model | RunnableLambda(_coerce_discovery_plan)


def build_discovery_plan_builder(
    initial_planner: Runnable[PlannerInput, DiscoveryPlan] | None = None,
    *,
    model: BaseChatModel | None = None,
    settings: DeepAgentsSettings | None = None,
    constraints: list[str] | None = None,
    available_tools: list[str] | None = None,
    available_skills: list[str] | None = None,
    context: dict[str, Any] | None = None,
) -> Runnable[str | PlannerInput | dict[str, Any], DiscoveryPlan]:
    """Build a convenience runnable from raw prompt text to a discovery plan."""
    planner = initial_planner or build_initial_planner(model=model, settings=settings)

    def run(value: str | PlannerInput | dict[str, Any]) -> DiscoveryPlan:
        planner_input = _coerce_discovery_builder_input(
            value,
            constraints=constraints,
            available_tools=available_tools,
            available_skills=available_skills,
            context=context,
        )
        return planner.invoke(planner_input)

    return RunnableLambda(run)


def build_execution_planner(
    model: BaseChatModel | None = None,
    settings: DeepAgentsSettings | None = None,
) -> Runnable[ExecutionPlannerInput, ExecutionPlan]:
    """Build a LangChain runnable that converts discovery plans to execution plans."""
    chat_model = model or build_chat_model(settings)
    structured_model = chat_model.with_structured_output(ExecutionPlan)
    return RunnableLambda(_coerce_execution_planner_input) | RunnableLambda(
        build_execution_planner_messages
    ) | structured_model | RunnableLambda(_coerce_execution_plan)


def build_planning_pipeline(
    initial_planner: Runnable[PlannerInput, DiscoveryPlan] | None = None,
    execution_planner: Runnable[ExecutionPlannerInput, ExecutionPlan] | None = None,
    model: BaseChatModel | None = None,
    settings: DeepAgentsSettings | None = None,
) -> Runnable[PlannerInput, ExecutionPlan]:
    """Build a pipeline that runs discovery planning, then execution planning."""
    initial = initial_planner or build_initial_planner(model=model, settings=settings)
    execution = execution_planner or build_execution_planner(model=model, settings=settings)

    def run(planner_input: PlannerInput | dict[str, Any]) -> ExecutionPlan:
        resolved_input = _coerce_planner_input(planner_input)
        discovery_plan = initial.invoke(resolved_input)
        return execution.invoke(
            ExecutionPlannerInput(
                discovery_plan=discovery_plan,
                available_tools=resolved_input.available_tools,
                available_skills=resolved_input.available_skills,
                context=resolved_input.context,
            )
        )

    return RunnableLambda(run)


def _coerce_planner_input(value: PlannerInput | dict[str, Any]) -> PlannerInput:
    if isinstance(value, PlannerInput):
        return value
    return PlannerInput(**value)


def _coerce_discovery_builder_input(
    value: str | PlannerInput | dict[str, Any],
    *,
    constraints: list[str] | None,
    available_tools: list[str] | None,
    available_skills: list[str] | None,
    context: dict[str, Any] | None,
) -> PlannerInput:
    if isinstance(value, str):
        return PlannerInput(
            objective=value,
            constraints=list(constraints or []),
            available_tools=list(available_tools or []),
            available_skills=list(available_skills or []),
            context=dict(context or {}),
        )
    return _coerce_planner_input(value)


def _coerce_execution_planner_input(
    value: ExecutionPlannerInput | dict[str, Any],
) -> ExecutionPlannerInput:
    if isinstance(value, ExecutionPlannerInput):
        return value
    return ExecutionPlannerInput(**value)


def _coerce_discovery_plan(value: DiscoveryPlan | dict[str, Any]) -> DiscoveryPlan:
    if isinstance(value, DiscoveryPlan):
        return value
    return DiscoveryPlan(**value)


def _coerce_execution_plan(value: ExecutionPlan | dict[str, Any]) -> ExecutionPlan:
    if isinstance(value, ExecutionPlan):
        return value
    return ExecutionPlan(**value)
