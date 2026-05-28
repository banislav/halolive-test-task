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
from deep_agents.runtime import Dispatcher, PlanTracker, PromptQueue


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


def test_prompt_queue_places_lifo_interrupts_first() -> None:
    queue = PromptQueue()
    queue.push(PromptQueueItem(id="P1", content="Normal feedback", priority=3))
    queue.push(PromptQueueItem(id="P2", content="Stop now", priority=1))

    assert len(queue) == 2
    assert queue.pop().id == "P2"
    assert queue.pop().id == "P1"
    assert queue.pop() is None
