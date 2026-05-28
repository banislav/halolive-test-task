from __future__ import annotations

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from deep_agents.models import ExecutionPlannerInput, PlannerInput, TaskCard
from deep_agents.runtime import TaskRunResult

INITIAL_PLANNER_SYSTEM_PROMPT = """You are InitialPlannerAgent.
Transform a raw user objective into a DiscoveryPlan.
Return only valid JSON matching the DiscoveryPlan schema.
Include objective, clarifications, milestones, gates, capability_map, skill_assignments,
risk_register, and dependency_graph."""

EXECUTION_PLANNER_SYSTEM_PROMPT = """You are ExecutionPlannerAgent.
Convert a DiscoveryPlan into a dispatchable ExecutionPlan.
Return only valid JSON matching the ExecutionPlan schema.
Include waves, task_cards, dependency_graph, and data_flow."""

WORKER_SYSTEM_PROMPT = """You are a deep-agent worker.
Execute exactly the assigned task card.
Return only valid JSON matching the TaskRunResult schema.
Use this JSON shape: {"task_id": "...", "output": {}, "artifacts": []}."""

JUDGE_SYSTEM_PROMPT = """You are a read-only task completion judge.
Evaluate whether the task result satisfies the task acceptance criteria.
Return only valid JSON matching the JudgeVerdict schema.
Use this JSON shape: {"task_id": "...", "verdict": "pass", "criteria_results":
[{"criterion": "...", "met": true, "evidence": "..."}], "overall_confidence": 0.0,
"recommendation": "advance"}.
Use recommendation "hold" when execution should pause for more information.
Use recommendation "block" when the task cannot continue because a dependency or required
input is missing."""


def build_initial_planner_messages(planner_input: PlannerInput) -> list[BaseMessage]:
    """Build LangChain messages for the initial discovery planner."""
    return [
        SystemMessage(content=INITIAL_PLANNER_SYSTEM_PROMPT),
        HumanMessage(
            content="\n".join(
                [
                    f"Objective: {planner_input.objective}",
                    "Constraints:",
                    _bullets(planner_input.constraints),
                    "Available tools:",
                    _bullets(planner_input.available_tools),
                    "Available skills:",
                    _bullets(planner_input.available_skills),
                    "Context JSON:",
                    planner_input.model_dump_json(indent=2),
                    "Expected JSON top-level shape:",
                    (
                        '{"objective": {}, "clarifications": [], "milestones": [], '
                        '"gates": [], "capability_map": {}, "skill_assignments": {}, '
                        '"risk_register": [], "dependency_graph": {}}'
                    ),
                ]
            )
        ),
    ]


def build_execution_planner_messages(planner_input: ExecutionPlannerInput) -> list[BaseMessage]:
    """Build LangChain messages for execution planning."""
    return [
        SystemMessage(content=EXECUTION_PLANNER_SYSTEM_PROMPT),
        HumanMessage(
            content="\n".join(
                [
                    "DiscoveryPlan JSON:",
                    planner_input.discovery_plan.model_dump_json(indent=2),
                    "Available tools:",
                    _bullets(planner_input.available_tools),
                    "Available skills:",
                    _bullets(planner_input.available_skills),
                    "Context JSON:",
                    planner_input.model_dump_json(indent=2),
                    "Expected JSON top-level shape:",
                    (
                        '{"id": "...", "objective": "...", "waves": [], '
                        '"task_cards": [], "dependency_graph": {}, "data_flow": {}}'
                    ),
                ]
            )
        ),
    ]


def build_worker_messages(task: TaskCard, skill_context: str | None = None) -> list[BaseMessage]:
    """Build LangChain messages for executing a task card."""
    human_sections = [_task_card_text(task)]
    if skill_context:
        human_sections.extend(["", skill_context])
    return [
        SystemMessage(content=WORKER_SYSTEM_PROMPT),
        HumanMessage(content="\n".join(human_sections)),
    ]


def build_judge_messages(task: TaskCard, result: TaskRunResult) -> list[BaseMessage]:
    """Build LangChain messages for judging a task result."""
    return [
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(
            content="\n\n".join(
                [
                    _task_card_text(task),
                    "Task result:",
                    result.model_dump_json(indent=2),
                ]
            )
        ),
    ]


def _task_card_text(task: TaskCard) -> str:
    criteria = "\n".join(f"- {item.description}" for item in task.acceptance_criteria)
    skills = "\n".join(f"- {skill.id} ({skill.load_mode})" for skill in task.assigned_to.skills)
    return "\n".join(
        [
            f"Task id: {task.id}",
            f"Task name: {task.name}",
            f"Assigned agent: {task.assigned_to.name} ({task.assigned_to.type})",
            "Acceptance criteria:",
            criteria or "- No explicit criteria provided",
            "Assigned skills:",
            skills or "- No skills assigned",
        ]
    )


def _bullets(items: list[str]) -> str:
    if not items:
        return "- None provided"
    return "\n".join(f"- {item}" for item in items)
