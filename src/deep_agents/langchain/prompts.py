from __future__ import annotations

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from deep_agents.models import TaskCard
from deep_agents.runtime import TaskRunResult

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


def build_worker_messages(task: TaskCard) -> list[BaseMessage]:
    """Build LangChain messages for executing a task card."""
    return [
        SystemMessage(content=WORKER_SYSTEM_PROMPT),
        HumanMessage(content=_task_card_text(task)),
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
