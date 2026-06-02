from __future__ import annotations

import json

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from deep_agents.models import (
    ExecutionPlan,
    ExecutionPlannerInput,
    Gate,
    Milestone,
    PlannerInput,
    PlanState,
    TaskCard,
)
from deep_agents.runtime import TaskExecutionContext, TaskRunResult

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

CHECKPOINT_JUDGE_SYSTEM_PROMPT = """You are a read-only checkpoint judge.
Evaluate whether a milestone gate is ready to open based on plan state, task statuses,
completed results, and the gate condition.
Return only valid JSON matching the GateJudgment schema.
Use this JSON shape: {"gate_id": "...", "milestone_id": "...", "decision": "open",
"criteria_results": [{"criterion": "...", "met": true, "evidence": "..."}],
"overall_confidence": 0.0, "reasoning": "...", "actions": []}.
Use decision "open" when the gate condition is satisfied.
Use decision "hold" when execution should wait for more information or incomplete work.
Use decision "reject" when the checkpoint failed and replanning is needed.
Use decision "escalate" when human input is required."""


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


def build_worker_messages(
    task_input: TaskCard | TaskExecutionContext,
    skill_context: str | None = None,
) -> list[BaseMessage]:
    """Build LangChain messages for executing a task card."""
    if isinstance(task_input, TaskExecutionContext):
        skill_context = skill_context if skill_context is not None else task_input.skill_context
        human_sections = [_task_execution_context_text(task_input)]
    else:
        human_sections = [_task_card_text(task_input)]
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


def build_checkpoint_judge_messages(
    gate: Gate,
    plan_state: PlanState,
    execution_plan: ExecutionPlan,
    *,
    milestone: Milestone | None = None,
    results: dict[str, TaskRunResult] | None = None,
) -> list[BaseMessage]:
    """Build LangChain messages for judging a milestone checkpoint gate."""
    serialized_results = {
        task_id: result.model_dump(mode="json") for task_id, result in (results or {}).items()
    }
    return [
        SystemMessage(content=CHECKPOINT_JUDGE_SYSTEM_PROMPT),
        HumanMessage(
            content="\n\n".join(
                [
                    "Gate JSON:",
                    gate.model_dump_json(indent=2),
                    "Milestone JSON:",
                    milestone.model_dump_json(indent=2) if milestone else "null",
                    "Plan state JSON:",
                    plan_state.model_dump_json(indent=2),
                    "Execution plan JSON:",
                    execution_plan.model_dump_json(indent=2),
                    "Completed task results JSON:",
                    json.dumps(serialized_results, indent=2),
                    "Expected JSON top-level shape:",
                    (
                        '{"gate_id": "...", "milestone_id": "...", "decision": "open", '
                        '"criteria_results": [], "overall_confidence": 0.0, '
                        '"reasoning": "...", "actions": []}'
                    ),
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


def _task_execution_context_text(context: TaskExecutionContext) -> str:
    dependency_results = {
        task_id: result.model_dump(mode="json")
        for task_id, result in context.dependency_results.items()
    }
    prior_summaries = [
        result.model_dump(mode="json") for result in context.prior_result_summaries
    ]
    artifacts = [artifact.model_dump(mode="json") for artifact in context.artifacts]
    return "\n\n".join(
        [
            "Objective:",
            context.objective.raw,
            _task_card_text(context.task),
            "Plan context JSON:",
            json.dumps(context.plan_context, indent=2),
            "Direct dependency results JSON:",
            json.dumps(dependency_results, indent=2),
            "Prior result summaries JSON:",
            json.dumps(prior_summaries, indent=2),
            "Relevant artifacts JSON:",
            json.dumps(artifacts, indent=2),
        ]
    )


def _bullets(items: list[str]) -> str:
    if not items:
        return "- None provided"
    return "\n".join(f"- {item}" for item in items)
