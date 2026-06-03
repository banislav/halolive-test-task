from __future__ import annotations

from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from deep_agents.models import (
    AgentAssignment,
    AgentKind,
    ExecutionPlan,
    JudgeRecommendation,
    JudgeVerdict,
    MemoryQuery,
    Objective,
    PlanState,
    TaskCard,
    ToolCallRequest,
    ToolCallStatus,
    ToolDefinition,
    ToolSafetyLevel,
    Wave,
)
from deep_agents.runtime import (
    InMemoryStore,
    MemoryRecorder,
    ProgressSignalBus,
    RuntimeEngine,
    TaskRunResult,
    ToolMiddlewareRunner,
    ToolPolicy,
    ToolRegistry,
)


class SearchArgs(BaseModel):
    query: str
    max_results: int


class LangChainStyleTool:
    args_schema = SearchArgs

    def invoke(self, input_data: dict[str, object]) -> dict[str, object]:
        return {"results": [input_data["query"]], "limit": input_data["max_results"]}


def build_tool_runner(
    *,
    policy: ToolPolicy | None = None,
    memory_store: InMemoryStore | None = None,
    progress_bus: ProgressSignalBus | None = None,
) -> ToolMiddlewareRunner:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            id="echo",
            name="Echo",
            input_schema={"message": "string"},
            output_schema={"message": "string"},
        ),
        lambda message: {"message": message},
    )
    registry.register(
        ToolDefinition(id="runnable", name="Runnable", input_schema={"value": "int"}),
        RunnableLambda(lambda payload: {"value": payload["value"] + 1}),
    )
    registry.register(
        ToolDefinition(id="search", name="Search"),
        LangChainStyleTool(),
    )
    registry.register(
        ToolDefinition(
            id="delete",
            name="Delete",
            input_schema={"path": "string"},
            safety_level=ToolSafetyLevel.DESTRUCTIVE,
        ),
        lambda path: {"deleted": path},
    )
    registry.register(
        ToolDefinition(id="broken", name="Broken", input_schema={"value": "int"}),
        lambda value: (_ for _ in ()).throw(RuntimeError(f"bad value {value}")),
    )
    return ToolMiddlewareRunner(
        registry=registry,
        policy=policy,
        memory_recorder=MemoryRecorder(memory_store) if memory_store is not None else None,
        progress_bus=progress_bus,
        plan_id="EP-tools",
    )


def test_tool_registry_supports_callable_runnable_and_langchain_style_tools() -> None:
    runner = build_tool_runner()

    callable_result = runner.invoke(
        ToolCallRequest(tool_id="echo", task_id="T1", input={"message": "hello"})
    )
    runnable_result = runner.invoke(
        ToolCallRequest(tool_id="runnable", task_id="T1", input={"value": 1})
    )
    langchain_result = runner.invoke(
        ToolCallRequest(
            tool_id="search",
            task_id="T1",
            input={"query": "deep agents", "max_results": 2},
        )
    )

    assert callable_result.status == ToolCallStatus.SUCCEEDED
    assert callable_result.output == {"message": "hello"}
    assert runnable_result.output == {"value": 2}
    assert langchain_result.output == {"results": ["deep agents"], "limit": 2}
    assert all(isinstance(tool, BaseTool) for tool in runner.registry.langchain_tools())


def test_tool_permission_denies_unallowed_tool() -> None:
    runner = build_tool_runner(policy=ToolPolicy(allowed_tool_ids=["search"]))

    result = runner.invoke(ToolCallRequest(tool_id="echo", task_id="T1", input={"message": "x"}))

    assert result.status == ToolCallStatus.DENIED
    assert result.error_type == "permission_denied"


def test_tool_validation_rejects_invalid_simple_and_pydantic_inputs() -> None:
    runner = build_tool_runner()

    simple_result = runner.invoke(
        ToolCallRequest(tool_id="echo", task_id="T1", input={"message": 1})
    )
    pydantic_result = runner.invoke(
        ToolCallRequest(tool_id="search", task_id="T1", input={"query": "x"})
    )

    assert simple_result.status == ToolCallStatus.VALIDATION_FAILED
    assert pydantic_result.status == ToolCallStatus.VALIDATION_FAILED


def test_tool_rate_limit_is_deterministic() -> None:
    runner = build_tool_runner(policy=ToolPolicy(rate_limits={"echo": 1}))

    first = runner.invoke(ToolCallRequest(tool_id="echo", task_id="T1", input={"message": "x"}))
    second = runner.invoke(ToolCallRequest(tool_id="echo", task_id="T1", input={"message": "y"}))

    assert first.status == ToolCallStatus.SUCCEEDED
    assert second.status == ToolCallStatus.RATE_LIMITED


def test_tool_safety_blocks_destructive_tools_without_policy_allowance() -> None:
    blocked = build_tool_runner().invoke(
        ToolCallRequest(tool_id="delete", task_id="T1", input={"path": "/tmp/file"})
    )
    allowed = build_tool_runner(policy=ToolPolicy(allow_destructive=True)).invoke(
        ToolCallRequest(tool_id="delete", task_id="T1", input={"path": "/tmp/file"})
    )

    assert blocked.status == ToolCallStatus.SAFETY_BLOCKED
    assert allowed.status == ToolCallStatus.SUCCEEDED


def test_tool_failure_captures_structured_error_details() -> None:
    result = build_tool_runner().invoke(
        ToolCallRequest(tool_id="broken", task_id="T1", input={"value": 3})
    )

    assert result.status == ToolCallStatus.FAILED
    assert result.error_type == "RuntimeError"
    assert result.error_message == "bad value 3"


def test_tool_runner_records_memory_and_progress_signals() -> None:
    memory_store = InMemoryStore()
    progress_bus = ProgressSignalBus()
    runner = build_tool_runner(memory_store=memory_store, progress_bus=progress_bus)

    result = runner.invoke(
        ToolCallRequest(
            tool_id="echo",
            task_id="T1",
            input={"message": "hello"},
            metadata={"input_tokens": 4, "output_tokens": 2},
        )
    )

    assert result.metadata["accounting"]["input_tokens"] == 4
    assert result.metadata["accounting"]["output_tokens"] == 2
    assert [record.tags[0] for record in memory_store.query(MemoryQuery(task_ids=["T1"]))] == [
        "tool_call",
        "tool_result",
    ]
    assert [signal.payload.status for signal in progress_bus.signals(task_id="T1")] == [
        "tool_started",
        "tool_succeeded",
    ]


def test_runtime_worker_can_call_tool_runner_during_attempt() -> None:
    memory_store = InMemoryStore()
    progress_bus = ProgressSignalBus()
    tool_runner = build_tool_runner(memory_store=memory_store, progress_bus=progress_bus)
    assignment = AgentAssignment(type=AgentKind.WORKER, name="Worker")
    plan = ExecutionPlan(
        id="EP-tools",
        objective="Use a tool",
        waves=[Wave(index=0, task_ids=["T1"])],
        task_cards=[TaskCard(id="T1", name="Echo", wave=0, assigned_to=assignment)],
    )

    def run_task(task: TaskCard) -> TaskRunResult:
        tool_result = tool_runner.invoke(
            ToolCallRequest(
                tool_id="echo",
                task_id=task.id,
                input={"message": "from worker"},
                caller_agent=task.assigned_to,
            )
        )
        return TaskRunResult(task_id=task.id, output={"tool_output": tool_result.output})

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
    ).invoke(plan, PlanState(objective=Objective(raw="Use a tool")))

    assert final_state["results"]["T1"].output == {
        "tool_output": {"message": "from worker"}
    }
    assert final_state["task_attempts"][0].result["tool_calls"] == [
        {
            "tool_id": "echo",
            "status": "succeeded",
            "duration_seconds": final_state["task_attempts"][0].result["tool_calls"][0][
                "duration_seconds"
            ],
            "error_type": None,
        }
    ]
    assert memory_store.query(MemoryQuery(task_ids=["T1"], tags=["tool_result"]))
