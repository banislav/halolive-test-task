import asyncio
import threading
import time

from langchain_core.runnables import RunnableLambda

from deep_agents.models import (
    AgentAssignment,
    AgentKind,
    ExecutionPlan,
    InterruptPriority,
    JudgeRecommendation,
    JudgeVerdict,
    LongRunningTaskConfig,
    Objective,
    PlanState,
    PlanStatus,
    PromptCategory,
    RuntimeCommand,
    RuntimeCommandStatus,
    RuntimeCommandType,
    RuntimeMessageType,
    TaskCard,
    TaskInvocation,
    Wave,
)
from deep_agents.runtime import (
    AsyncRuntimeSession,
    RuntimeEngine,
    TaskRunResult,
    current_long_running_context,
)


def _plan(
    *,
    task_id: str = "T1",
    invocation: TaskInvocation | None = None,
) -> ExecutionPlan:
    assignment = AgentAssignment(type=AgentKind.WORKER, name="Worker")
    return ExecutionPlan(
        id="EP-session",
        objective="Test async session",
        waves=[Wave(index=0, task_ids=[task_id])],
        task_cards=[
            TaskCard(
                id=task_id,
                name="Session task",
                wave=0,
                assigned_to=assignment,
                invocation=invocation or TaskInvocation(),
            )
        ],
    )


def _state() -> PlanState:
    return PlanState(objective=Objective(raw="Test async session"))


def _passing_judge() -> RunnableLambda:
    return RunnableLambda(
        lambda value: JudgeVerdict(
            task_id=value["task"].id,
            verdict="pass",
            overall_confidence=0.9,
            recommendation=JudgeRecommendation.ADVANCE,
        )
    )


async def _drain_events(session: AsyncRuntimeSession) -> list:
    events = []
    async for event in session.events():
        events.append(event)
    return events


def test_async_session_runs_plan_and_streams_messages() -> None:
    async def run() -> None:
        worker = RunnableLambda(
            lambda task: TaskRunResult(task_id=task.id, output={"summary": "done"})
        )
        session = AsyncRuntimeSession(
            RuntimeEngine(worker=worker, judge=_passing_judge()),
            session_id="session-test",
        )

        session.start(_plan(), _state())
        final_state = await session.wait()
        events = await _drain_events(session)

        assert final_state["plan_state"].status == PlanStatus.COMPLETED
        assert session.status == "completed"
        assert {event.type for event in events} >= {
            RuntimeMessageType.PROGRESS,
            RuntimeMessageType.REQUEST,
            RuntimeMessageType.RESULT,
            RuntimeMessageType.VERDICT,
        }
        assert any(event.payload.get("status") == "session_started" for event in events)
        assert any(
            event.payload.get("status") == "session_finished"
            for event in events
        )

    asyncio.run(run())


def test_submit_prompt_queues_fifo_prompt_before_start() -> None:
    async def run() -> None:
        worker = RunnableLambda(
            lambda task: TaskRunResult(task_id=task.id, output={"summary": "done"})
        )
        session = AsyncRuntimeSession(RuntimeEngine(worker=worker, judge=_passing_judge()))

        prompt = session.submit_prompt(
            "what is the current status?",
            category=PromptCategory.CONTENT_REASONING,
        )
        session.start(_plan(), _state())
        final_state = await session.wait()

        assert final_state["prompt_results"][0].prompt.id == prompt.id
        assert final_state["prompt_results"][0].response is not None
        assert final_state["results"]["T1"].output == {"summary": "done"}

    asyncio.run(run())


def test_p1_prompt_cooperatively_cancels_active_long_running_worker() -> None:
    async def run() -> None:
        started = threading.Event()
        cancel_seen = threading.Event()

        def worker(task: TaskCard) -> TaskRunResult:
            context = current_long_running_context()
            assert context is not None
            started.set()
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                context.heartbeat(status="working", percent_complete=25)
                if context.should_cancel():
                    cancel_seen.set()
                    return TaskRunResult(
                        task_id=task.id,
                        output={"cancelled": True},
                    )
                time.sleep(0.02)
            return TaskRunResult(task_id=task.id, output={"cancelled": False})

        invocation = TaskInvocation(
            timeout_seconds=5,
            long_running=LongRunningTaskConfig(
                heartbeat_interval_seconds=1,
                checkpoint_interval_seconds=1,
            ),
        )
        session = AsyncRuntimeSession(
            RuntimeEngine(worker=RunnableLambda(worker), judge=_passing_judge())
        )

        session.start(_plan(invocation=invocation), _state())
        assert await asyncio.to_thread(started.wait, 1)
        prompt = session.submit_prompt("pause this task", priority=InterruptPriority.P1_PAUSE)
        final_state = await session.wait()

        assert prompt.id.startswith("prompt-")
        assert cancel_seen.is_set()
        assert final_state["results"]["T1"].output == {"cancelled": True}
        assert any(
            result.command.type == RuntimeCommandType.PAUSE_TASK
            and result.status == RuntimeCommandStatus.APPLIED
            for result in final_state["command_results"]
        )

    asyncio.run(run())


def test_submit_command_applies_halt_to_live_state() -> None:
    async def run() -> None:
        worker = RunnableLambda(
            lambda task: TaskRunResult(task_id=task.id, output={"summary": "done"})
        )
        session = AsyncRuntimeSession(RuntimeEngine(worker=worker, judge=_passing_judge()))
        session.start(_plan(), _state())

        result = session.submit_command(
            RuntimeCommand(
                type=RuntimeCommandType.HALT,
                reason="Stop before dispatch continues.",
                source="test",
            )
        )
        final_state = await session.wait()

        assert result.status == RuntimeCommandStatus.APPLIED
        assert final_state["plan_state"].status == PlanStatus.PAUSED
        assert "T1" not in final_state.get("results", {})

    asyncio.run(run())


def test_session_snapshot_includes_runtime_state() -> None:
    async def run() -> None:
        worker = RunnableLambda(
            lambda task: TaskRunResult(task_id=task.id, output={"summary": "done"})
        )
        session = AsyncRuntimeSession(RuntimeEngine(worker=worker, judge=_passing_judge()))
        prompt = session.submit_prompt("status?", category=PromptCategory.CONTENT_REASONING)

        pending_snapshot = session.snapshot()
        assert pending_snapshot.pending_prompt_ids == [prompt.id]

        session.start(_plan(), _state())
        await session.wait()
        final_snapshot = session.snapshot()

        assert final_snapshot.status == "completed"
        assert final_snapshot.execution_plan_id == "EP-session"
        assert final_snapshot.results["T1"]["output"] == {"summary": "done"}
        assert final_snapshot.memory_record_count > 0

    asyncio.run(run())
