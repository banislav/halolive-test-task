import logging

from langchain_core.runnables import RunnableLambda

from deep_agents.models import (
    AgentAssignment,
    AgentKind,
    ExecutionPlan,
    JudgeRecommendation,
    JudgeVerdict,
    Objective,
    PlanState,
    PlanStatus,
    PromptQueueItem,
    TaskCard,
    TaskStatus,
    Wave,
)
from deep_agents.runtime import (
    Dispatcher,
    ObserverJudge,
    PlanTracker,
    ProcessJudge,
    ProgressSignalBus,
    PromptQueue,
    RuntimeEngine,
    TaskRunResult,
)


def build_execution_plan() -> ExecutionPlan:
    assignment = AgentAssignment(type=AgentKind.WORKER, name="Worker")
    return ExecutionPlan(
        id="EP1",
        objective="Test plan",
        waves=[
            Wave(index=0, task_ids=["T1"]),
            Wave(index=1, task_ids=["T2"]),
        ],
        task_cards=[
            TaskCard(id="T1", name="First task", wave=0, assigned_to=assignment),
            TaskCard(
                id="T2",
                name="Second task",
                wave=1,
                blocked_by=["T1"],
                assigned_to=assignment,
            ),
        ],
    )


def test_dispatcher_returns_only_unblocked_ready_tasks() -> None:
    dispatcher = Dispatcher(build_execution_plan())

    ready = dispatcher.ready_tasks({"T1": TaskStatus.PENDING, "T2": TaskStatus.PENDING})

    assert [task.id for task in ready] == ["T1"]


def test_plan_tracker_unblocks_dependent_tasks_after_completion() -> None:
    plan = build_execution_plan()
    tracker = PlanTracker(PlanState(objective=Objective(raw="Test plan")), plan)

    assert tracker.state.task_statuses == {
        "T1": "ready",
        "T2": "blocked",
    }

    ready_ids = tracker.apply_task_completion("T1")

    assert ready_ids == ["T2"]
    assert tracker.state.task_statuses["T1"] == "completed"
    assert tracker.state.task_statuses["T2"] == "ready"


def test_plan_tracker_logs_state_changes(caplog) -> None:
    plan = build_execution_plan()
    tracker = PlanTracker(PlanState(objective=Objective(raw="Test plan")), plan)

    with caplog.at_level(logging.INFO, logger="deep_agents"):
        tracker.mark_running("T1")
        tracker.apply_task_completion("T1")

    messages = [record.message for record in caplog.records]
    assert "task status changed" in messages
    assert "plan status changed" in messages


def test_plan_tracker_completes_when_all_tasks_pass_judgment() -> None:
    plan = build_execution_plan()
    tracker = PlanTracker(PlanState(objective=Objective(raw="Test plan")), plan)

    tracker.apply_judge_verdict(
        JudgeVerdict(
            task_id="T1",
            verdict="pass",
            overall_confidence=0.9,
            recommendation=JudgeRecommendation.ADVANCE,
        )
    )
    tracker.apply_judge_verdict(
        JudgeVerdict(
            task_id="T2",
            verdict="pass",
            overall_confidence=0.9,
            recommendation=JudgeRecommendation.ADVANCE,
        )
    )

    assert tracker.state.status == PlanStatus.COMPLETED


def test_plan_tracker_marks_retry_as_ready_refining() -> None:
    plan = build_execution_plan()
    tracker = PlanTracker(PlanState(objective=Objective(raw="Test plan")), plan)

    ready_ids = tracker.apply_judge_verdict(
        JudgeVerdict(
            task_id="T1",
            verdict="fail",
            overall_confidence=0.8,
            recommendation=JudgeRecommendation.RETRY,
        )
    )

    assert ready_ids == ["T1"]
    assert tracker.state.status == "refining"
    assert tracker.state.task_statuses["T1"] == "ready"


def test_plan_tracker_marks_hold_as_paused_refining() -> None:
    plan = build_execution_plan()
    tracker = PlanTracker(PlanState(objective=Objective(raw="Test plan")), plan)

    ready_ids = tracker.apply_judge_verdict(
        JudgeVerdict(
            task_id="T1",
            verdict="partial",
            overall_confidence=0.7,
            recommendation=JudgeRecommendation.HOLD,
        )
    )

    assert ready_ids == []
    assert tracker.state.status == "refining"
    assert tracker.state.task_statuses["T1"] == "paused"


def test_plan_tracker_marks_block_as_paused_refining() -> None:
    plan = build_execution_plan()
    tracker = PlanTracker(PlanState(objective=Objective(raw="Test plan")), plan)

    ready_ids = tracker.apply_judge_verdict(
        JudgeVerdict(
            task_id="T1",
            verdict="partial",
            overall_confidence=0.7,
            recommendation=JudgeRecommendation.BLOCK,
        )
    )

    assert ready_ids == []
    assert tracker.state.status == "refining"
    assert tracker.state.task_statuses["T1"] == "paused"


def test_prompt_queue_places_lifo_interrupts_first() -> None:
    queue = PromptQueue()
    queue.push(PromptQueueItem(id="P1", content="Normal feedback", priority=3))
    queue.push(PromptQueueItem(id="P2", content="Stop now", priority=1))

    assert len(queue) == 2
    assert queue.pop().id == "P2"
    assert queue.pop().id == "P1"
    assert queue.pop() is None


def test_task_run_result_accepts_provider_result_alias_and_status() -> None:
    result = TaskRunResult.model_validate(
        {
            "task_id": "T1",
            "status": "success",
            "result": {"summary": "Done"},
            "artifacts": [],
        }
    )

    assert result.output == {"summary": "Done"}
    assert result.status == "success"


def test_runtime_engine_runs_dependent_tasks_to_completion() -> None:
    def run_task(task: TaskCard) -> TaskRunResult:
        return TaskRunResult(task_id=task.id, output={"message": f"ran {task.id}"})

    def judge_task(payload: dict[str, object]) -> JudgeVerdict:
        result = payload["result"]
        assert isinstance(result, TaskRunResult)
        return JudgeVerdict(
            task_id=result.task_id,
            verdict="pass",
            overall_confidence=1.0,
            recommendation=JudgeRecommendation.ADVANCE,
        )

    engine = RuntimeEngine(
        worker=RunnableLambda(run_task),
        judge=RunnableLambda(judge_task),
    )

    final_state = engine.invoke(
        build_execution_plan(),
        PlanState(objective=Objective(raw="Test plan")),
    )

    assert final_state["plan_state"].status == PlanStatus.COMPLETED
    assert final_state["plan_state"].task_statuses == {
        "T1": "completed",
        "T2": "completed",
    }
    assert list(final_state["results"]) == ["T1", "T2"]


def test_runtime_engine_logs_worker_errors(caplog) -> None:
    def fail_task(_: TaskCard) -> TaskRunResult:
        raise RuntimeError("worker exploded")

    def judge_task(_: dict[str, object]) -> JudgeVerdict:
        raise AssertionError("judge should not run")

    engine = RuntimeEngine(
        worker=RunnableLambda(fail_task),
        judge=RunnableLambda(judge_task),
    )

    with caplog.at_level(logging.ERROR, logger="deep_agents"):
        try:
            engine.invoke(
                build_execution_plan(),
                PlanState(objective=Objective(raw="Test plan")),
            )
        except RuntimeError as exc:
            assert str(exc) == "worker exploded"
        else:
            raise AssertionError("expected worker failure")

    messages = [record.message for record in caplog.records]
    assert "worker failed" in messages
    assert "runtime engine invoke failed" in messages


def test_runtime_engine_emits_progress_signals_and_collects_judgments() -> None:
    def run_task(task: TaskCard) -> TaskRunResult:
        return TaskRunResult(task_id=task.id, output={"message": f"ran {task.id}"})

    def judge_task(payload: dict[str, object]) -> JudgeVerdict:
        result = payload["result"]
        assert isinstance(result, TaskRunResult)
        return JudgeVerdict(
            task_id=result.task_id,
            verdict="pass",
            overall_confidence=1.0,
            recommendation=JudgeRecommendation.ADVANCE,
        )

    bus = ProgressSignalBus()
    bus.subscribe_process(ProcessJudge())
    bus.subscribe_observer(ObserverJudge())
    engine = RuntimeEngine(
        worker=RunnableLambda(run_task),
        judge=RunnableLambda(judge_task),
        progress_bus=bus,
    )

    final_state = engine.invoke(
        build_execution_plan(),
        PlanState(objective=Objective(raw="Test plan")),
    )

    signals = bus.signals()
    signal_statuses = [signal.payload.status for signal in signals]
    assert "dispatched" in signal_statuses
    assert "worker_started" in signal_statuses
    assert "worker_completed" in signal_statuses
    assert "judge_started" in signal_statuses
    assert "judge_completed" in signal_statuses
    assert "verdict_applied" in signal_statuses
    assert any(signal.signal_type == "finding" for signal in signals)
    assert final_state["process_judgments"]
    assert final_state["observer_judgments"]


def test_runtime_engine_emits_error_signal_before_worker_failure_reraises() -> None:
    def fail_task(_: TaskCard) -> TaskRunResult:
        raise RuntimeError("worker exploded")

    def judge_task(_: dict[str, object]) -> JudgeVerdict:
        raise AssertionError("judge should not run")

    bus = ProgressSignalBus()
    engine = RuntimeEngine(
        worker=RunnableLambda(fail_task),
        judge=RunnableLambda(judge_task),
        progress_bus=bus,
    )

    try:
        engine.invoke(
            build_execution_plan(),
            PlanState(objective=Objective(raw="Test plan")),
        )
    except RuntimeError as exc:
        assert str(exc) == "worker exploded"
    else:
        raise AssertionError("expected worker failure")

    signals = bus.signals(task_id="T1")
    assert signals[-1].signal_type == "error"
    assert signals[-1].payload.status == "worker_failed"
