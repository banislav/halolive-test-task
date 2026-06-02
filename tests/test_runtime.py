import logging

from langchain_core.runnables import RunnableLambda

from deep_agents.models import (
    AgentAssignment,
    AgentKind,
    DiscoveryPlan,
    ExecutionPlan,
    Gate,
    GateDecision,
    GateJudgment,
    InterruptPriority,
    JudgeRecommendation,
    JudgeVerdict,
    Milestone,
    Objective,
    ObserverHealth,
    ObserverJudgment,
    PlanState,
    PlanStatus,
    ProcessAction,
    ProcessAssessment,
    ProcessJudgment,
    PromptCategory,
    PromptQueueItem,
    RuntimeCommand,
    RuntimeCommandStatus,
    RuntimeCommandType,
    Task,
    TaskCard,
    TaskStatus,
    Wave,
)
from deep_agents.runtime import (
    ContextAssembler,
    Dispatcher,
    ObserverJudge,
    PlanTracker,
    ProcessJudge,
    ProgressSignalBus,
    PromptHandler,
    PromptQueue,
    RuntimeCommandExecutor,
    RuntimeEngine,
    TaskExecutionContext,
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


def build_plan_state_with_gate(
    *,
    checkpoint_ids: list[str] | None = None,
) -> PlanState:
    objective = Objective(raw="Test plan")
    return PlanState(
        objective=objective,
        discovery_plan=DiscoveryPlan(
            objective=objective,
            milestones=[
                Milestone(
                    id="M1",
                    name="First milestone",
                    gates=["G1"],
                    tasks=[Task(id="T1", name="First task")],
                )
            ],
            gates=[Gate(id="G1", condition="First milestone task is complete")],
        ),
        checkpoint_ids=checkpoint_ids or [],
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


def test_plan_tracker_converts_needs_more_time_judgment_to_timeout_command() -> None:
    plan = build_execution_plan()
    tracker = PlanTracker(PlanState(objective=Objective(raw="Test plan")), plan)

    commands = tracker.apply_process_judgment(
        ProcessJudgment(
            task_id="T1",
            assessment=ProcessAssessment.NEEDS_MORE_TIME,
            reasoning="Task is close to timeout.",
            actions=[ProcessAction(type="adjust_timeout", value=180)],
        )
    )

    assert [command.type for command in commands] == [RuntimeCommandType.ADJUST_TIMEOUT]
    assert commands[0].payload == {"value": 180}
    assert tracker.state.task_statuses["T1"] == TaskStatus.READY


def test_plan_tracker_applies_early_complete_judgment_and_unblocks_dependents() -> None:
    plan = build_execution_plan()
    tracker = PlanTracker(PlanState(objective=Objective(raw="Test plan")), plan)

    commands = tracker.apply_process_judgment(
        ProcessJudgment(
            task_id="T1",
            assessment=ProcessAssessment.HEALTHY,
            reasoning="Finding satisfies acceptance criteria.",
            actions=[ProcessAction(type="mark_early_complete", value=True)],
        )
    )

    assert [command.type for command in commands] == [RuntimeCommandType.MARK_EARLY_COMPLETE]
    assert commands[0].payload == {"ready_task_ids": ["T2"]}
    assert tracker.state.task_statuses["T1"] == TaskStatus.COMPLETED
    assert tracker.state.task_statuses["T2"] == TaskStatus.READY


def test_plan_tracker_terminates_low_relevance_judgment_and_requests_replan() -> None:
    plan = build_execution_plan()
    tracker = PlanTracker(PlanState(objective=Objective(raw="Test plan")), plan)

    commands = tracker.apply_process_judgment(
        ProcessJudgment(
            task_id="T1",
            assessment=ProcessAssessment.EARLY_TERMINATE,
            reasoning="Finding is off track.",
            actions=[ProcessAction(type="terminate_task", value="low_relevance")],
        )
    )

    assert [command.type for command in commands] == [
        RuntimeCommandType.TERMINATE_TASK,
        RuntimeCommandType.REQUEST_REPLAN,
    ]
    assert tracker.state.task_statuses["T1"] == TaskStatus.READY
    assert tracker.state.status == PlanStatus.INITIALIZING


def test_plan_tracker_pauses_on_hitl_process_judgment() -> None:
    plan = build_execution_plan()
    tracker = PlanTracker(PlanState(objective=Objective(raw="Test plan")), plan)

    commands = tracker.apply_process_judgment(
        ProcessJudgment(
            task_id="T1",
            assessment=ProcessAssessment.ESCALATE_HITL,
            reasoning="Task needs user input.",
            actions=[ProcessAction(type="escalate_hitl", value="missing credential")],
        )
    )

    assert [command.type for command in commands] == [RuntimeCommandType.ESCALATE_HITL]
    assert tracker.state.task_statuses["T1"] == TaskStatus.READY
    assert tracker.state.status == PlanStatus.INITIALIZING


def test_plan_tracker_converts_observer_judgments_to_runtime_commands() -> None:
    plan = build_execution_plan()
    tracker = PlanTracker(PlanState(objective=Objective(raw="Test plan")), plan)

    degraded = tracker.apply_observer_judgment(
        ObserverJudgment(
            health=ObserverHealth.DEGRADED,
            reasoning="Repeated errors observed.",
        )
    )
    stuck = tracker.apply_observer_judgment(
        ObserverJudgment(
            health=ObserverHealth.STUCK,
            reasoning="No heartbeat.",
        ),
        task_id="T1",
    )
    diverging = tracker.apply_observer_judgment(
        ObserverJudgment(
            health=ObserverHealth.DIVERGING,
            reasoning="Runtime is off objective.",
        )
    )

    assert [command.type for command in degraded] == [RuntimeCommandType.ESCALATE_HITL]
    assert [command.type for command in stuck] == [RuntimeCommandType.PAUSE_TASK]
    assert [command.type for command in diverging] == [RuntimeCommandType.REQUEST_REPLAN]
    assert tracker.state.task_statuses["T1"] == TaskStatus.READY
    assert tracker.state.status == PlanStatus.INITIALIZING


def test_plan_tracker_converts_gate_judgments_to_runtime_commands() -> None:
    plan = build_execution_plan()
    tracker = PlanTracker(PlanState(objective=Objective(raw="Test plan")), plan)

    opened = tracker.apply_gate_judgment(
        GateJudgment(
            gate_id="G1",
            decision=GateDecision.OPEN,
            overall_confidence=0.95,
            reasoning="Gate is satisfied.",
        )
    )
    held = tracker.apply_gate_judgment(
        GateJudgment(
            gate_id="G1",
            decision=GateDecision.HOLD,
            overall_confidence=0.6,
            reasoning="Milestone still has incomplete work.",
        )
    )
    rejected = tracker.apply_gate_judgment(
        GateJudgment(
            gate_id="G1",
            decision=GateDecision.REJECT,
            overall_confidence=0.8,
            reasoning="Milestone failed its quality gate.",
        )
    )
    escalated = tracker.apply_gate_judgment(
        GateJudgment(
            gate_id="G1",
            decision=GateDecision.ESCALATE,
            overall_confidence=0.7,
            reasoning="Gate needs human approval.",
        )
    )

    assert opened == []
    assert [command.type for command in held] == [RuntimeCommandType.HOLD_GATE]
    assert [command.type for command in rejected] == [RuntimeCommandType.REQUEST_REPLAN]
    assert [command.type for command in escalated] == [RuntimeCommandType.ESCALATE_HITL]
    assert held[0].payload["gate_id"] == "G1"


def test_runtime_command_executor_applies_halt() -> None:
    plan_state = PlanState(objective=Objective(raw="Test plan"))
    executor = RuntimeCommandExecutor()

    result = executor.execute(
        RuntimeCommand(
            type=RuntimeCommandType.HALT,
            reason="Stop now.",
            source="test",
        ),
        plan_state=plan_state,
        execution_plan=build_execution_plan(),
    )

    assert result.status == RuntimeCommandStatus.APPLIED
    assert plan_state.status == PlanStatus.PAUSED


def test_runtime_command_executor_pauses_targeted_running_task() -> None:
    plan = build_execution_plan()
    plan_state = PlanState(
        objective=Objective(raw="Test plan"),
        task_statuses={"T1": TaskStatus.RUNNING, "T2": TaskStatus.BLOCKED},
    )

    result = RuntimeCommandExecutor().execute(
        RuntimeCommand(
            type=RuntimeCommandType.PAUSE_TASK,
            task_id="T1",
            reason="Pause task.",
            source="test",
        ),
        plan_state=plan_state,
        execution_plan=plan,
    )

    assert result.status == RuntimeCommandStatus.APPLIED
    assert result.affected_task_ids == ["T1"]
    assert plan_state.task_statuses["T1"] == TaskStatus.PAUSED


def test_runtime_command_executor_pauses_all_running_tasks() -> None:
    plan_state = PlanState(
        objective=Objective(raw="Test plan"),
        task_statuses={"T1": TaskStatus.RUNNING, "T2": TaskStatus.RUNNING},
    )

    result = RuntimeCommandExecutor().execute(
        RuntimeCommand(
            type=RuntimeCommandType.PAUSE_TASK,
            reason="Pause all.",
            source="test",
        ),
        plan_state=plan_state,
        execution_plan=build_execution_plan(),
    )

    assert result.status == RuntimeCommandStatus.APPLIED
    assert result.affected_task_ids == ["T1", "T2"]
    assert plan_state.status == PlanStatus.PAUSED
    assert set(plan_state.task_statuses.values()) == {TaskStatus.PAUSED}


def test_runtime_command_executor_resumes_paused_unblocked_tasks() -> None:
    plan = build_execution_plan()
    plan_state = PlanState(
        objective=Objective(raw="Test plan"),
        status=PlanStatus.PAUSED,
        task_statuses={"T1": TaskStatus.PAUSED, "T2": TaskStatus.PAUSED},
    )

    result = RuntimeCommandExecutor().execute(
        RuntimeCommand(
            type=RuntimeCommandType.RESUME_TASK,
            task_id="T1",
            reason="Resume task.",
            source="test",
        ),
        plan_state=plan_state,
        execution_plan=plan,
    )

    assert result.status == RuntimeCommandStatus.APPLIED
    assert result.affected_task_ids == ["T1"]
    assert plan_state.task_statuses["T1"] == TaskStatus.READY
    assert plan_state.task_statuses["T2"] == TaskStatus.PAUSED
    assert plan_state.status == PlanStatus.EXECUTING


def test_runtime_command_executor_terminates_task() -> None:
    plan_state = PlanState(
        objective=Objective(raw="Test plan"),
        task_statuses={"T1": TaskStatus.RUNNING, "T2": TaskStatus.BLOCKED},
    )

    result = RuntimeCommandExecutor().execute(
        RuntimeCommand(
            type=RuntimeCommandType.TERMINATE_TASK,
            task_id="T1",
            reason="Terminate task.",
            source="test",
        ),
        plan_state=plan_state,
        execution_plan=build_execution_plan(),
    )

    assert result.status == RuntimeCommandStatus.APPLIED
    assert result.affected_task_ids == ["T1"]
    assert plan_state.task_statuses["T1"] == TaskStatus.TERMINATED
    assert plan_state.status == PlanStatus.REFINING


def test_runtime_command_executor_ignores_advisory_commands() -> None:
    plan_state = PlanState(objective=Objective(raw="Test plan"))

    result = RuntimeCommandExecutor().execute(
        RuntimeCommand(
            type=RuntimeCommandType.REQUEST_REPLAN,
            reason="Needs replan.",
            source="test",
        ),
        plan_state=plan_state,
        execution_plan=build_execution_plan(),
    )

    assert result.status == RuntimeCommandStatus.IGNORED
    assert plan_state.status == PlanStatus.INITIALIZING


def test_prompt_queue_places_lifo_interrupts_first() -> None:
    queue = PromptQueue()
    queue.push(PromptQueueItem(id="P1", content="Normal feedback", priority=3))
    queue.push(PromptQueueItem(id="P2", content="Stop now", priority=1))

    assert len(queue) == 2
    assert queue.pop().id == "P2"
    assert queue.pop().id == "P1"
    assert queue.pop() is None


def test_prompt_queue_drains_in_handling_order() -> None:
    queue = PromptQueue()
    queue.push(PromptQueueItem(id="P1", content="Normal feedback", priority=3))
    queue.push(PromptQueueItem(id="P2", content="Stop now", priority=1))
    queue.push(PromptQueueItem(id="P3", content="What is done?", priority=3))

    assert [item.id for item in queue.items()] == ["P2", "P1", "P3"]
    assert [item.id for item in queue.drain()] == ["P2", "P1", "P3"]
    assert len(queue) == 0


def test_prompt_handler_classifies_content_and_plan_update_prompts() -> None:
    handler = PromptHandler()
    plan = build_execution_plan()
    plan_state = PlanState(objective=Objective(raw="Test plan"))

    content = handler.handle_prompt(
        PromptQueueItem(id="P1", content="What is the current status?"),
        execution_plan=plan,
        plan_state=plan_state,
        results={},
    )
    update = handler.handle_prompt(
        PromptQueueItem(id="P2", content="Change the scope to include security review"),
        execution_plan=plan,
        plan_state=plan_state,
        results={},
    )

    assert content.classification.category == PromptCategory.CONTENT_REASONING
    assert content.response is not None
    assert update.classification.category == PromptCategory.PLAN_UPDATE
    assert [command.type for command in update.commands] == [RuntimeCommandType.REQUEST_REPLAN]


def test_prompt_handler_respects_explicit_prompt_category() -> None:
    result = PromptHandler().handle_prompt(
        PromptQueueItem(
            id="P1",
            content="What sounds like a question but should change the plan?",
            category=PromptCategory.PLAN_UPDATE,
        ),
        execution_plan=build_execution_plan(),
        plan_state=PlanState(objective=Objective(raw="Test plan")),
        results={},
    )

    assert result.classification.category == PromptCategory.PLAN_UPDATE
    assert result.classification.reasoning == "Prompt item already included an explicit category."


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


def test_runtime_engine_answers_content_prompt_at_dispatch_boundary() -> None:
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

    queue = PromptQueue()
    queue.push(PromptQueueItem(id="P1", content="What is the current status?"))
    engine = RuntimeEngine(
        worker=RunnableLambda(run_task),
        judge=RunnableLambda(judge_task),
        prompt_queue=queue,
    )

    final_state = engine.invoke(
        build_execution_plan(),
        PlanState(objective=Objective(raw="Test plan")),
    )

    assert final_state["plan_state"].status == PlanStatus.COMPLETED
    assert (
        final_state["prompt_results"][0].classification.category
        == PromptCategory.CONTENT_REASONING
    )
    assert final_state["prompt_results"][0].response is not None
    assert final_state["runtime_commands"] == []


def test_runtime_engine_records_plan_update_prompt_command() -> None:
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

    queue = PromptQueue()
    queue.push(PromptQueueItem(id="P1", content="Change the scope to include security review"))
    engine = RuntimeEngine(
        worker=RunnableLambda(run_task),
        judge=RunnableLambda(judge_task),
        prompt_queue=queue,
    )

    final_state = engine.invoke(
        build_execution_plan(),
        PlanState(objective=Objective(raw="Test plan")),
    )

    assert final_state["prompt_results"][0].classification.category == PromptCategory.PLAN_UPDATE
    assert any(
        command.type == RuntimeCommandType.REQUEST_REPLAN
        for command in final_state["runtime_commands"]
    )
    assert any(
        result.command.type == RuntimeCommandType.REQUEST_REPLAN
        and result.status == RuntimeCommandStatus.IGNORED
        for result in final_state["command_results"]
    )
    assert final_state["plan_state"].status == PlanStatus.COMPLETED


def test_runtime_engine_records_p1_pause_prompt_command() -> None:
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

    queue = PromptQueue()
    queue.push(
        PromptQueueItem(
            id="P1",
            content="What is the current status?",
            priority=InterruptPriority.P1_PAUSE,
        )
    )
    engine = RuntimeEngine(
        worker=RunnableLambda(run_task),
        judge=RunnableLambda(judge_task),
        prompt_queue=queue,
    )

    final_state = engine.invoke(
        build_execution_plan(),
        PlanState(objective=Objective(raw="Test plan")),
    )

    assert (
        final_state["prompt_results"][0].classification.category
        == PromptCategory.CONTENT_REASONING
    )
    assert any(
        command.type == RuntimeCommandType.PAUSE_TASK
        for command in final_state["runtime_commands"]
    )
    assert any(
        result.command.type == RuntimeCommandType.PAUSE_TASK
        and result.status == RuntimeCommandStatus.IGNORED
        for result in final_state["command_results"]
    )
    assert final_state["plan_state"].status == PlanStatus.COMPLETED


def test_runtime_engine_halts_dispatch_for_p0_prompt() -> None:
    def run_task(_: TaskCard) -> TaskRunResult:
        raise AssertionError("worker should not run after P0 halt")

    def judge_task(_: dict[str, object]) -> JudgeVerdict:
        raise AssertionError("judge should not run after P0 halt")

    queue = PromptQueue()
    queue.push(
        PromptQueueItem(
            id="P1",
            content="Halt execution now",
            priority=InterruptPriority.P0_HALT,
        )
    )
    engine = RuntimeEngine(
        worker=RunnableLambda(run_task),
        judge=RunnableLambda(judge_task),
        prompt_queue=queue,
    )

    final_state = engine.invoke(
        build_execution_plan(),
        PlanState(objective=Objective(raw="Test plan")),
    )

    assert final_state["results"] == {}
    assert any(
        command.type == RuntimeCommandType.HALT
        for command in final_state["runtime_commands"]
    )
    assert any(
        result.command.type == RuntimeCommandType.HALT
        and result.status == RuntimeCommandStatus.APPLIED
        for result in final_state["command_results"]
    )
    assert final_state["plan_state"].status == PlanStatus.PAUSED
    assert final_state["plan_state"].task_statuses == {"T1": "ready", "T2": "blocked"}


def test_runtime_engine_can_send_assembled_context_to_workers() -> None:
    captured_contexts: list[TaskExecutionContext] = []

    def run_task(context: TaskExecutionContext) -> TaskRunResult:
        captured_contexts.append(context)
        if context.task.id == "T2":
            assert context.dependency_results["T1"].output == {"message": "ran T1"}
        return TaskRunResult(
            task_id=context.task.id,
            output={"message": f"ran {context.task.id}"},
        )

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
        context_assembler=ContextAssembler(),
    )

    final_state = engine.invoke(
        build_execution_plan(),
        PlanState(objective=Objective(raw="Test plan")),
    )

    assert final_state["plan_state"].status == PlanStatus.COMPLETED
    assert [context.task.id for context in captured_contexts] == ["T1", "T2"]
    assert captured_contexts[0].dependency_results == {}
    assert list(captured_contexts[1].dependency_results) == ["T1"]


def test_runtime_engine_evaluates_completed_milestone_checkpoint_gate() -> None:
    captured_payloads: list[dict[str, object]] = []

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

    def judge_gate(payload: dict[str, object]) -> GateJudgment:
        captured_payloads.append(payload)
        return GateJudgment(
            gate_id="G1",
            milestone_id="M1",
            decision=GateDecision.HOLD,
            overall_confidence=0.8,
            reasoning="Gate should hold for review.",
        )

    engine = RuntimeEngine(
        worker=RunnableLambda(run_task),
        judge=RunnableLambda(judge_task),
        checkpoint_judge=RunnableLambda(judge_gate),
    )

    final_state = engine.invoke(build_execution_plan(), build_plan_state_with_gate())

    assert len(captured_payloads) == 1
    assert captured_payloads[0]["results"]["T1"].output == {"message": "ran T1"}
    assert final_state["gate_judgments"][0].decision == GateDecision.HOLD
    assert final_state["plan_state"].checkpoint_ids == ["G1"]
    assert any(
        command.type == RuntimeCommandType.HOLD_GATE
        for command in final_state["runtime_commands"]
    )
    assert any(
        result.command.type == RuntimeCommandType.HOLD_GATE
        and result.status == RuntimeCommandStatus.IGNORED
        for result in final_state["command_results"]
    )
    assert final_state["plan_state"].status == PlanStatus.COMPLETED


def test_runtime_engine_records_checkpoint_gate_command_decisions() -> None:
    expected_commands = {
        GateDecision.HOLD: RuntimeCommandType.HOLD_GATE,
        GateDecision.REJECT: RuntimeCommandType.REQUEST_REPLAN,
        GateDecision.ESCALATE: RuntimeCommandType.ESCALATE_HITL,
    }

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

    for decision, command_type in expected_commands.items():
        engine = RuntimeEngine(
            worker=RunnableLambda(run_task),
            judge=RunnableLambda(judge_task),
            checkpoint_judge=RunnableLambda(
                lambda _, decision=decision: GateJudgment(
                    gate_id="G1",
                    milestone_id="M1",
                    decision=decision,
                    overall_confidence=0.8,
                    reasoning=f"Gate decision: {decision}",
                )
            ),
        )

        final_state = engine.invoke(build_execution_plan(), build_plan_state_with_gate())

        assert any(command.type == command_type for command in final_state["runtime_commands"])


def test_runtime_engine_skips_already_checkpointed_gates() -> None:
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

    def judge_gate(_: dict[str, object]) -> GateJudgment:
        raise AssertionError("checkpoint judge should not run")

    engine = RuntimeEngine(
        worker=RunnableLambda(run_task),
        judge=RunnableLambda(judge_task),
        checkpoint_judge=RunnableLambda(judge_gate),
    )

    final_state = engine.invoke(
        build_execution_plan(),
        build_plan_state_with_gate(checkpoint_ids=["G1"]),
    )

    assert final_state["gate_judgments"] == []
    assert final_state["plan_state"].checkpoint_ids == ["G1"]


def test_runtime_engine_skips_checkpoint_gates_without_discovery_plan_or_judge() -> None:
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

    def judge_gate(_: dict[str, object]) -> GateJudgment:
        raise AssertionError("checkpoint judge should not run")

    no_discovery_engine = RuntimeEngine(
        worker=RunnableLambda(run_task),
        judge=RunnableLambda(judge_task),
        checkpoint_judge=RunnableLambda(judge_gate),
    )
    no_checkpoint_judge_engine = RuntimeEngine(
        worker=RunnableLambda(run_task),
        judge=RunnableLambda(judge_task),
    )

    no_discovery_state = no_discovery_engine.invoke(
        build_execution_plan(),
        PlanState(objective=Objective(raw="Test plan")),
    )
    no_judge_state = no_checkpoint_judge_engine.invoke(
        build_execution_plan(),
        build_plan_state_with_gate(),
    )

    assert no_discovery_state["gate_judgments"] == []
    assert no_judge_state["gate_judgments"] == []
    assert no_judge_state["plan_state"].checkpoint_ids == []


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
    assert any(
        command.type == RuntimeCommandType.MARK_EARLY_COMPLETE
        for command in final_state["runtime_commands"]
    )
    assert any(
        result.command.type == RuntimeCommandType.MARK_EARLY_COMPLETE
        and result.status == RuntimeCommandStatus.APPLIED
        for result in final_state["command_results"]
    )


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
