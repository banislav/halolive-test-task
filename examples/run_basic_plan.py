from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deep_agents.config import DeepAgentsSettings
from deep_agents.instrumentation import configure_logging
from deep_agents.langchain import build_task_completion_judge, build_task_worker
from deep_agents.models import (
    AcceptanceCriterion,
    AgentAssignment,
    AgentKind,
    ExecutionPlan,
    Objective,
    PlanState,
    SkillAssignment,
    SkillDefinition,
    TaskCard,
    Wave,
)
from deep_agents.runtime import RuntimeEngine
from deep_agents.skills import SkillLoader, SkillRegistry


def build_plan() -> ExecutionPlan:
    assignment = AgentAssignment(
        type=AgentKind.WORKER,
        name="WriterWorker",
        skills=[SkillAssignment(id="technical_writing")],
    )
    return ExecutionPlan(
        id="EP-basic",
        objective="Draft and review a short project summary.",
        waves=[
            Wave(index=0, task_ids=["T1"]),
            Wave(index=1, task_ids=["T2"]),
        ],
        task_cards=[
            TaskCard(
                id="T1",
                name="Draft summary",
                wave=0,
                assigned_to=assignment,
                acceptance_criteria=[
                    AcceptanceCriterion(description="Output includes a concise project summary")
                ],
            ),
            TaskCard(
                id="T2",
                name="Refine summary",
                wave=1,
                blocked_by=["T1"],
                assigned_to=assignment,
                acceptance_criteria=[
                    AcceptanceCriterion(description="Output is clearer than the draft")
                ],
            ),
        ],
    )


def main() -> None:
    configure_logging()
    plan = build_plan()
    settings = DeepAgentsSettings(
        provider="openrouter",
        model="qwen/qwen3.6-flash",
    )
    skill_loader = SkillLoader(
        SkillRegistry(
            [
                SkillDefinition(
                    id="technical_writing",
                    name="Technical Writing",
                    prompt=(
                        "Write concise, concrete project prose. Prefer short paragraphs, "
                        "explicit outcomes, and plain engineering language."
                    ),
                )
            ]
        )
    )
    engine = RuntimeEngine(
        worker=build_task_worker(settings=settings, skill_loader=skill_loader),
        judge=build_task_completion_judge(settings=settings),
    )
    final_state = engine.invoke(
        plan,
        PlanState(objective=Objective(raw=plan.objective)),
    )
    print(final_state["plan_state"].model_dump_json(indent=2))


if __name__ == "__main__":
    main()
