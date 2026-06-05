from __future__ import annotations

import json

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from deep_agents.models import (
    ExecutionPlan,
    ExecutionPlannerInput,
    Gate,
    HandoffStep,
    Milestone,
    PlannerInput,
    PlanState,
    PromptQueueItem,
    PromptReasoningInput,
    TaskCard,
)
from deep_agents.runtime import TaskExecutionContext, TaskRunResult
from deep_agents.runtime.handoffs import HandoffStepInput

INITIAL_PLANNER_SYSTEM_PROMPT = """You are InitialPlannerAgent.
Transform a raw user objective into a DiscoveryPlan.
Return only valid JSON matching the DiscoveryPlan schema.
Include objective, clarifications, milestones, gates, capability_map, skill_assignments,
risk_register, and dependency_graph.
Use the exact schema field names. Do not use title, description, risk, mitigation, or
other alternate field names unless they are part of the schema."""

EXECUTION_PLANNER_SYSTEM_PROMPT = """You are ExecutionPlannerAgent.
Convert a DiscoveryPlan into a dispatchable ExecutionPlan.
Return only valid JSON matching the ExecutionPlan schema.
Include waves, task_cards, dependency_graph, and data_flow.
Use the exact schema field names. Do not use wave id fields, task description,
task tools, task skills, or alternate dependency graph shapes unless they are part
of the schema.
Select a topology per wave using one of: subagents, handoffs, router, skills,
custom_workflow.
Use subagents for independent task invocations with clean context isolation.
Use handoff_chain only for sequential specialist handoffs inside one logical task.
Never model inter-task dependencies as handoffs; the Dispatcher injects artifacts and
context into a new subagent invocation for downstream tasks.
Avoid routing chatter in subagent context."""

WORKER_SYSTEM_PROMPT = """You are a deep-agent worker.
Execute exactly the assigned task card.
Return only valid JSON matching the TaskRunResult schema.
Use this JSON shape: {"task_id": "...", "output": {}, "artifacts": []}."""

HANDOFF_STEP_SYSTEM_PROMPT = """You are a deep-agent intra-task handoff agent.
Execute exactly the assigned handoff step.
Use the parent task context, previous step output, and shared handoff state.
Return only valid JSON matching the TaskRunResult schema.
Forward your final result directly; do not paraphrase or add routing commentary.
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

PROMPT_CLASSIFIER_SYSTEM_PROMPT = """You are PromptClassifierAgent.
Classify a queued user prompt as content_reasoning or plan_update.
Return only valid JSON matching the PromptClassification schema.
Use this JSON shape: {"prompt_id": "...", "category": "content_reasoning",
"priority": 3, "reasoning": "..."}."""

CONTENT_REASONING_SYSTEM_PROMPT = """You are a read-only content reasoning agent.
Answer the queued user prompt using only current plan state and completed results.
Never mutate plan state or request actions.
Return only valid JSON matching the PromptResponse schema.
Use this JSON shape: {"prompt_id": "...", "answer": "...",
"referenced_task_ids": [], "referenced_artifact_ids": []}."""


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
                        '{"objective": {"raw": "...", "normalized": null, '
                        '"constraints": [], "success_criteria": []}, '
                        '"clarifications": [{"question": "...", "resolution": null}], '
                        '"milestones": [{"id": "M1", "name": "...", "gates": [], '
                        '"tasks": [{"id": "T1", "name": "...", "description": null, '
                        '"acceptance_criteria": [{"description": "...", '
                        '"measurable": true}], "tools_needed": [], '
                        '"skills_needed": [], "estimated_complexity": "medium", '
                        '"risks": [], "blocked_by": [], "status": "pending"}]}], '
                        '"gates": [{"id": "G1", "type": "quality_gate", '
                        '"condition": "...", "action_on_fail": "replan"}], '
                        '"capability_map": {"T1": ["tool_id"]}, '
                        '"skill_assignments": {"T1": ["skill_id"]}, '
                        '"risk_register": [{"description": "...", '
                        '"fallback": null, "severity": "medium"}], '
                        '"dependency_graph": {"T1": []}}'
                    ),
                    "Important schema constraints:",
                    (
                        "Milestones use name, not title. Gates use condition, not description. "
                        "Risks use description and fallback, not risk and mitigation. "
                        "capability_map, skill_assignments, and dependency_graph values must "
                        "always be arrays of strings."
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
                        '{"id": "...", "objective": "...", '
                        '"waves": [{"index": 0, "name": "...", "blocked_by": [], '
                        '"task_ids": ["T1"], "topology": "subagents"}], '
                        '"task_cards": [{"id": "T1", "name": "...", "wave": 0, '
                        '"blocked_by": [], "blocks": [], '
                        '"assigned_to": {"type": "worker", "name": "Worker", '
                        '"agent_id": null, "skills": [{"id": "skill_id"}]}, '
                        '"invocation": {"method": "async_dispatch", "input": {}, '
                        '"input_schema": {}, "expected_output_schema": {}, '
                        '"timeout_seconds": 120}, '
                        '"acceptance_criteria": [{"description": "...", '
                        '"measurable": true}], "estimated_complexity": "medium", '
                        '"risks": [], "handoff_chain": []}], '
                        '"dependency_graph": {"blocked_by": {"T1": []}, '
                        '"blocks": {"T1": []}}, "data_flow": {"T1": []}}'
                    ),
                    "Topology rules:",
                    (
                        "Wave.topology defaults to subagents. Add TaskCard.handoff_chain only "
                        "for intra-task handoffs. Inter-task dependencies are not handoffs."
                    ),
                    "Important schema constraints:",
                    (
                        "Waves use numeric index, not id. Task cards use wave and assigned_to. "
                        "Put skills under assigned_to.skills, not task.skills. Put tool inputs "
                        "under invocation.input or invocation.input_schema, not task.tools. "
                        "dependency_graph must contain blocked_by and blocks maps. data_flow "
                        "values must be arrays of strings."
                    ),
                ]
            )
        ),
    ]


def build_worker_messages(
    task_input: TaskCard | TaskExecutionContext | HandoffStepInput,
    skill_context: str | None = None,
) -> list[BaseMessage]:
    """Build LangChain messages for executing a task card."""
    if isinstance(task_input, HandoffStepInput):
        skill_context = skill_context or _assignment_skill_text(task_input.step)
        human_sections = [_handoff_step_input_text(task_input)]
        if skill_context:
            human_sections.extend(["", skill_context])
        return [
            SystemMessage(content=HANDOFF_STEP_SYSTEM_PROMPT),
            HumanMessage(content="\n".join(human_sections)),
        ]
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


def build_prompt_classifier_messages(prompt: PromptQueueItem) -> list[BaseMessage]:
    """Build LangChain messages for classifying a queued user prompt."""
    return [
        SystemMessage(content=PROMPT_CLASSIFIER_SYSTEM_PROMPT),
        HumanMessage(
            content="\n".join(
                [
                    "Queued prompt JSON:",
                    prompt.model_dump_json(indent=2),
                    "Expected JSON top-level shape:",
                    (
                        '{"prompt_id": "...", "category": "content_reasoning", '
                        '"priority": 3, "reasoning": "..."}'
                    ),
                ]
            )
        ),
    ]


def build_content_reasoning_messages(
    reasoning_input: PromptReasoningInput,
) -> list[BaseMessage]:
    """Build LangChain messages for answering a read-only queued prompt."""
    return [
        SystemMessage(content=CONTENT_REASONING_SYSTEM_PROMPT),
        HumanMessage(
            content="\n\n".join(
                [
                    "Queued prompt JSON:",
                    reasoning_input.prompt.model_dump_json(indent=2),
                    "Plan state JSON:",
                    reasoning_input.plan_state.model_dump_json(indent=2),
                    "Completed results JSON:",
                    json.dumps(reasoning_input.results, indent=2),
                    "Context JSON:",
                    json.dumps(reasoning_input.context, indent=2),
                    "Expected JSON top-level shape:",
                    (
                        '{"prompt_id": "...", "answer": "...", '
                        '"referenced_task_ids": [], "referenced_artifact_ids": []}'
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


def _assignment_skill_text(step: HandoffStep) -> str:
    if not step.assigned_to.skills:
        return ""
    skills = "\n".join(
        f"- {skill.id} ({skill.load_mode})" for skill in step.assigned_to.skills
    )
    return f"Assigned handoff step skills:\n{skills}"


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


def _handoff_step_input_text(step_input: HandoffStepInput) -> str:
    parent_context = (
        step_input.parent_context.model_dump(mode="json")
        if step_input.parent_context is not None
        else None
    )
    return "\n\n".join(
        [
            "Parent task JSON:",
            step_input.parent_task.model_dump_json(indent=2),
            "Handoff step JSON:",
            step_input.step.model_dump_json(indent=2),
            "Parent task context JSON:",
            json.dumps(parent_context, indent=2),
            "Previous step output JSON:",
            json.dumps(step_input.previous_output, indent=2),
            "Shared handoff state JSON:",
            json.dumps(step_input.shared_state, indent=2),
        ]
    )


def _bullets(items: list[str]) -> str:
    if not items:
        return "- None provided"
    return "\n".join(f"- {item}" for item in items)
