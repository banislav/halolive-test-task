from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.runnables import RunnableLambda

from deep_agents.models import (
    AgentAssignment,
    AgentKind,
    ExecutionPlan,
    JudgeRecommendation,
    JudgeVerdict,
    Objective,
    PlanState,
    TaskCard,
    ToolCallRequest,
    ToolDefinition,
    Wave,
)
from deep_agents.runtime import (
    InMemoryStore,
    MemoryRecorder,
    ProgressSignalBus,
    RuntimeEngine,
    TaskRunResult,
    ToolMiddlewareRunner,
    ToolRegistry,
)


def main() -> None:
    memory_store = InMemoryStore()
    progress_bus = ProgressSignalBus()
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            id="summarize_text",
            name="Summarize Text",
            input_schema={"text": "string"},
            output_schema={"summary": "string"},
        ),
        lambda text: {"summary": text[:80]},
    )
    tool_runner = ToolMiddlewareRunner(
        registry=registry,
        memory_recorder=MemoryRecorder(memory_store),
        progress_bus=progress_bus,
        plan_id="EP-tool-middleware",
    )

    assignment = AgentAssignment(type=AgentKind.WORKER, name="ToolWorker")
    plan = ExecutionPlan(
        id="EP-tool-middleware",
        objective="Summarize text through the tool middleware stack",
        waves=[Wave(index=0, task_ids=["T1"])],
        task_cards=[
            TaskCard(id="T1", name="Summarize source text", wave=0, assigned_to=assignment)
        ],
    )

    def run_task(task: TaskCard) -> TaskRunResult:
        tool_result = tool_runner.invoke(
            ToolCallRequest(
                tool_id="summarize_text",
                task_id=task.id,
                input={
                    "text": (
                        "Execution middleware gives every tool call permission checks, "
                        "validation, progress emission, and output capture."
                    )
                },
                caller_agent=task.assigned_to,
            )
        )
        return TaskRunResult(task_id=task.id, output=tool_result.output)

    def judge_task(payload: dict[str, object]) -> JudgeVerdict:
        result = payload["result"]
        assert isinstance(result, TaskRunResult)
        return JudgeVerdict(
            task_id=result.task_id,
            verdict="pass",
            overall_confidence=1.0,
            recommendation=JudgeRecommendation.ADVANCE,
        )

    final_state = RuntimeEngine(
        worker=RunnableLambda(run_task),
        judge=RunnableLambda(judge_task),
        memory_store=memory_store,
        progress_bus=progress_bus,
    ).invoke(plan, PlanState(objective=Objective(raw=plan.objective)))

    print(final_state["results"]["T1"].output)
    print([record.tags for record in final_state["memory_records"] if "tool_result" in record.tags])


if __name__ == "__main__":
    main()
