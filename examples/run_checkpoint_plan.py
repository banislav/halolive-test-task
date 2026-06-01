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
    Gate,
    GateDecision,
    GateJudgment,
    JudgeRecommendation,
    JudgeVerdict,
    Milestone,
    Objective,
    PlanState,
    SkillAssignment,
    Task,
    TaskCard,
    Wave,
)
from deep_agents.runtime import RuntimeEngine, TaskRunResult

OBJECTIVE = "Draft and review a short project summary for a deep-agent runtime."

OBJECTIVE_OUTPUT = {
    "title": "Deep-agent runtime project summary",
    "summary": (
        "Build a deep-agent runtime that turns an objective into a discovery plan, "
        "execution plan, task execution loop, checkpoint gate review, and advisory "
        "runtime commands."
    ),
    "matches_objective": OBJECTIVE,
}


def build_discovery_plan() -> DiscoveryPlan:
    objective = Objective(raw=OBJECTIVE)
    return DiscoveryPlan(
        objective=objective,
        milestones=[
            Milestone(
                id="M1",
                name="Draft milestone",
                gates=["G1"],
                tasks=[
                    Task(
                        id="T1",
                        name="Draft summary",
                        acceptance_criteria=[
                            AcceptanceCriterion(
                                description="Output includes a concise project summary"
                            )
                        ],
                    )
                ],
            )
        ],
        gates=[
            Gate(
                id="G1",
                condition="Draft milestone task passes its acceptance criteria",
                action_on_fail="hold",
            )
        ],
        dependency_graph={"T1": [], "T2": ["T1"]},
    )


def build_execution_plan() -> ExecutionPlan:
    assignment = AgentAssignment(
        type=AgentKind.WORKER,
        name="WriterWorker",
        skills=[SkillAssignment(id="technical_writing")],
    )
    return ExecutionPlan(
        id="EP-checkpoint",
        objective=OBJECTIVE,
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
                name="Review summary",
                wave=1,
                blocked_by=["T1"],
                assigned_to=assignment,
                acceptance_criteria=[
                    AcceptanceCriterion(description="Review confirms the summary is usable")
                ],
            ),
        ],
    )


def run_task(task: TaskCard) -> TaskRunResult:
    if task.id == "T1":
        output = OBJECTIVE_OUTPUT
    else:
        output = {
            "review": "approved",
            "reason": "The draft directly describes the requested deep-agent runtime.",
            "checked_objective": OBJECTIVE,
        }

    return TaskRunResult(
        task_id=task.id,
        output=output,
    )


def judge_task(payload: dict[str, object]) -> JudgeVerdict:
    result = payload["result"]
    assert isinstance(result, TaskRunResult)
    return JudgeVerdict(
        task_id=result.task_id,
        verdict="pass",
        criteria_results=[
            CriteriaResult(
                criterion="Worker returned structured output",
                met=bool(result.output),
                evidence=str(result.output),
            )
        ],
        overall_confidence=1.0,
        recommendation=JudgeRecommendation.ADVANCE,
    )


def judge_checkpoint(payload: dict[str, object]) -> GateJudgment:
    gate = payload["gate"]
    milestone = payload["milestone"]
    results = payload["results"]
    gate_id = getattr(gate, "id", "G1")
    milestone_id = getattr(milestone, "id", "M1")
    result_count = len(results) if isinstance(results, dict) else 0
    return GateJudgment(
        gate_id=gate_id,
        milestone_id=milestone_id,
        decision=GateDecision.HOLD,
        criteria_results=[
            CriteriaResult(
                criterion="Draft result exists before review begins",
                met=result_count > 0,
                evidence=f"{result_count} completed result(s) available",
            )
        ],
        overall_confidence=0.85,
        reasoning="The gate is advisory in this example, so it records a hold command.",
    )


def main() -> None:
    configure_logging()
    execution_plan = build_execution_plan()
    plan_state = PlanState(
        objective=Objective(raw=OBJECTIVE),
        discovery_plan=build_discovery_plan(),
    )
    engine = RuntimeEngine(
        worker=RunnableLambda(run_task),
        judge=RunnableLambda(judge_task),
        checkpoint_judge=RunnableLambda(judge_checkpoint),
    )

    final_state = engine.invoke(execution_plan, plan_state)

    print("Plan state:")
    print(final_state["plan_state"].model_dump_json(indent=2))
    print("\nTask outputs:")
    for result in final_state["results"].values():
        print(result.model_dump_json(indent=2))
    print("\nGate judgments:")
    for judgment in final_state["gate_judgments"]:
        print(judgment.model_dump_json(indent=2))
    print("\nRuntime commands:")
    for command in final_state["runtime_commands"]:
        print(command.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
