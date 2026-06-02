from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.runnables import RunnableLambda

from deep_agents.instrumentation import configure_logging
from deep_agents.models import (
    AcceptanceCriterion,
    AgentAssignment,
    AgentKind,
    CriteriaResult,
    DiscoveryPlan,
    ExecutionPlan,
    ExecutionPlannerInput,
    JudgeRecommendation,
    JudgeVerdict,
    Milestone,
    Objective,
    PlanState,
    RuntimeCommandResult,
    SkillAssignment,
    Task,
    TaskCard,
    Wave,
)
from deep_agents.runtime import (
    ContextAssembler,
    RuntimeEngine,
    RuntimeReplanner,
    TaskExecutionContext,
    TaskRunResult,
)

OBJECTIVE = "Draft and review a short project summary for a runtime replanning loop."


def build_discovery_plan() -> DiscoveryPlan:
    objective = Objective(raw=OBJECTIVE)
    return DiscoveryPlan(
        objective=objective,
        milestones=[
            Milestone(
                id="M1",
                name="Draft and review summary",
                tasks=[
                    Task(
                        id="T1",
                        name="Draft runtime summary",
                        acceptance_criteria=[
                            AcceptanceCriterion(
                                description="Draft names the runtime replanning loop"
                            )
                        ],
                    ),
                    Task(
                        id="T3",
                        name="Review replanned draft",
                        blocked_by=["T1"],
                        acceptance_criteria=[
                            AcceptanceCriterion(description="Review confirms the draft is usable")
                        ],
                    ),
                ],
            )
        ],
        dependency_graph={"T1": [], "T3": ["T1"]},
    )


def build_initial_plan() -> ExecutionPlan:
    assignment = _worker_assignment()
    return ExecutionPlan(
        id="EP-initial",
        objective=OBJECTIVE,
        waves=[
            Wave(index=0, task_ids=["T1"]),
            Wave(index=1, task_ids=["T2"]),
        ],
        task_cards=[
            TaskCard(
                id="T1",
                name="Draft generic summary",
                wave=0,
                assigned_to=assignment,
                acceptance_criteria=[
                    AcceptanceCriterion(description="Output should match the runtime objective")
                ],
            ),
            TaskCard(
                id="T2",
                name="Review generic summary",
                wave=1,
                blocked_by=["T1"],
                assigned_to=assignment,
            ),
        ],
    )


def build_replanned_plan() -> ExecutionPlan:
    assignment = _worker_assignment()
    return ExecutionPlan(
        id="EP-replanned",
        objective=OBJECTIVE,
        waves=[
            Wave(index=0, task_ids=["T1"]),
            Wave(index=1, task_ids=["T3"]),
        ],
        task_cards=[
            TaskCard(
                id="T1",
                name="Draft replanning-loop summary",
                wave=0,
                assigned_to=assignment,
                acceptance_criteria=[
                    AcceptanceCriterion(description="Output names the replanning loop")
                ],
            ),
            TaskCard(
                id="T3",
                name="Review replanned summary",
                wave=1,
                blocked_by=["T1"],
                assigned_to=assignment,
                acceptance_criteria=[
                    AcceptanceCriterion(description="Review confirms the replanned output")
                ],
            ),
        ],
    )


def _worker_assignment() -> AgentAssignment:
    return AgentAssignment(
        type=AgentKind.WORKER,
        name="WriterWorker",
        skills=[SkillAssignment(id="technical_writing")],
    )


def main() -> None:
    configure_logging()
    initial_plan = build_initial_plan()
    plan_state = PlanState(
        objective=Objective(raw=OBJECTIVE),
        discovery_plan=build_discovery_plan(),
    )
    worker_attempts: dict[str, int] = {}
    planner_inputs: list[ExecutionPlannerInput] = []

    def run_task(context: TaskExecutionContext) -> TaskRunResult:
        task = context.task
        worker_attempts[task.id] = worker_attempts.get(task.id, 0) + 1

        if task.id == "T1" and worker_attempts[task.id] == 1:
            return TaskRunResult(
                task_id=task.id,
                output={
                    "summary": "This draft is generic and misses the replanning-loop objective.",
                    "needs_replan": True,
                },
            )

        if task.id == "T1":
            return TaskRunResult(
                task_id=task.id,
                output={
                    "summary": (
                        "Runtime Replanning V1 records a request_replan command, invokes "
                        "an execution replanner, swaps in a replacement plan, and resumes "
                        "dispatch from the reconciled plan state."
                    ),
                    "needs_replan": False,
                },
            )

        draft = context.dependency_results["T1"].output
        return TaskRunResult(
            task_id=task.id,
            output={
                "review": "approved",
                "reviewed_summary": draft["summary"],
                "reason": "The replacement plan produced a summary matching the objective.",
            },
        )

    def judge_task(payload: dict[str, object]) -> JudgeVerdict:
        result = payload["result"]
        assert isinstance(result, TaskRunResult)
        needs_replan = bool(result.output.get("needs_replan"))
        return JudgeVerdict(
            task_id=result.task_id,
            verdict="fail" if needs_replan else "pass",
            criteria_results=[
                CriteriaResult(
                    criterion="Output matches the replanning-loop objective",
                    met=not needs_replan,
                    evidence=str(result.output),
                )
            ],
            overall_confidence=0.95,
            recommendation=(
                JudgeRecommendation.REPLAN
                if needs_replan
                else JudgeRecommendation.ADVANCE
            ),
        )

    def replan(planner_input: ExecutionPlannerInput) -> ExecutionPlan:
        planner_inputs.append(planner_input)
        return build_replanned_plan()

    engine = RuntimeEngine(
        worker=RunnableLambda(run_task),
        judge=RunnableLambda(judge_task),
        context_assembler=ContextAssembler(),
        runtime_replanner=RuntimeReplanner(
            RunnableLambda(replan),
            available_skills=["technical_writing"],
        ),
    )

    final_state = engine.invoke(initial_plan, plan_state)

    print("Initial execution plan:", initial_plan.id)
    print("Final execution plan:", final_state["execution_plan"].id)
    print("\nTask status:")
    print(final_state["plan_state"].model_dump_json(indent=2))

    print("\nRuntime commands:")
    for command in final_state["runtime_commands"]:
        print(command.model_dump_json(indent=2))

    print("\nReplan results:")
    for result in final_state["replan_results"]:
        print(result.model_dump_json(indent=2))

    print("\nReplanner received trigger:")
    if planner_inputs:
        trigger = planner_inputs[0].context["trigger"]
        print(RuntimeCommandResult.model_validate(trigger).model_dump_json(indent=2))

    print("\nTask outputs:")
    for result in final_state["results"].values():
        print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
