from __future__ import annotations

from langchain_core.runnables import RunnableLambda

from deep_agents.models import (
    AcceptanceCriterion,
    AgentAssignment,
    AgentKind,
    ArtifactRef,
    DiscoveryPlan,
    ExecutionPlan,
    ExecutionPlannerInput,
    Gate,
    GateDecision,
    GateJudgment,
    JudgeRecommendation,
    JudgeVerdict,
    MemoryKind,
    MemoryQuery,
    MemoryRecord,
    Milestone,
    Objective,
    PlanState,
    PromptQueueItem,
    RuntimeReplanStatus,
    Task,
    TaskCard,
    Wave,
)
from deep_agents.models.context import ArtifactKind
from deep_agents.runtime import (
    ContextAssembler,
    InMemoryStore,
    MemoryRecorder,
    PromptQueue,
    RuntimeEngine,
    RuntimeReplanner,
    TaskExecutionContext,
    TaskRunResult,
)


def build_memory_plan(plan_id: str = "EP-memory") -> ExecutionPlan:
    assignment = AgentAssignment(type=AgentKind.WORKER, name="Worker")
    return ExecutionPlan(
        id=plan_id,
        objective="Exercise unified memory",
        waves=[
            Wave(index=0, task_ids=["T1"]),
            Wave(index=1, task_ids=["T2"]),
        ],
        task_cards=[
            TaskCard(
                id="T1",
                name="Draft",
                wave=0,
                assigned_to=assignment,
                acceptance_criteria=[AcceptanceCriterion(description="Draft exists")],
            ),
            TaskCard(
                id="T2",
                name="Review",
                wave=1,
                blocked_by=["T1"],
                assigned_to=assignment,
            ),
        ],
    )


def build_memory_discovery_plan() -> DiscoveryPlan:
    objective = Objective(raw="Exercise unified memory")
    return DiscoveryPlan(
        objective=objective,
        milestones=[
            Milestone(
                id="M1",
                name="Draft milestone",
                gates=["G1"],
                tasks=[Task(id="T1", name="Draft")],
            )
        ],
        gates=[Gate(id="G1", condition="Draft completed")],
    )


def test_in_memory_store_filters_and_preserves_insertion_order() -> None:
    store = InMemoryStore()
    first = MemoryRecord(
        id="M1",
        kind=MemoryKind.WORKING,
        task_id="T1",
        plan_id="EP1",
        tags=["task_result"],
        source="test",
        payload={"summary": "first memory"},
    )
    second = MemoryRecord(
        id="M2",
        kind=MemoryKind.SEMANTIC,
        task_id="T2",
        plan_id="EP1",
        tags=["lesson"],
        source="test",
        payload={"lesson": "second memory"},
    )

    store.put_many([first, second])

    assert store.get("M1") == first
    assert store.by_task("T2") == [second]
    assert [record.id for record in store.query(MemoryQuery(plan_ids=["EP1"]))] == [
        "M1",
        "M2",
    ]
    assert [
        record.id
        for record in store.query(
            MemoryQuery(kinds=[MemoryKind.SEMANTIC], tags=["lesson"], text_query="second")
        )
    ] == ["M2"]
    assert [record.id for record in store.query(MemoryQuery(limit=1))] == ["M1"]


def test_memory_recorder_writes_typed_memory_records() -> None:
    store = InMemoryStore()
    recorder = MemoryRecorder(store)
    artifact = ArtifactRef(
        id="A1",
        kind=ArtifactKind.STRUCTURED,
        uri="memory://artifact",
        summary="Structured artifact",
    )

    recorder.record_task_result(
        TaskRunResult(task_id="T1", output={"draft": "done"}, artifacts=[artifact]),
        plan_id="EP1",
    )
    recorder.record_semantic_fact(
        fact={"lesson": "Prefer deterministic memory tests."},
        source="test",
    )
    recorder.record_episodic_memory(
        payload={"preference": "User prefers bullet-point executive summaries."},
        source="preference_store",
        tags=["user_preference"],
    )
    recorder.record_procedural_memory(
        payload={"pattern": "Review drafts before finalizing."},
        source="pattern_store",
        tags=["execution_pattern"],
    )
    recorder.record_skill_memory(
        payload={"skill_id": "technical_writing", "pairs_well_with": ["review"]},
        source="skill_registry",
        tags=["skill_relationship"],
    )

    records = store.records()
    assert [record.kind for record in records] == [
        MemoryKind.WORKING,
        MemoryKind.SESSION,
        MemoryKind.SEMANTIC,
        MemoryKind.EPISODIC,
        MemoryKind.PROCEDURAL,
        MemoryKind.SKILL,
    ]
    assert records[0].payload["result"]["output"] == {"draft": "done"}
    assert records[1].payload["artifact"]["id"] == "A1"
    assert records[2].payload["fact"]["lesson"] == "Prefer deterministic memory tests."
    assert records[3].payload["preference"] == "User prefers bullet-point executive summaries."
    assert records[4].payload["pattern"] == "Review drafts before finalizing."
    assert records[5].payload["skill_id"] == "technical_writing"


def test_context_assembler_reads_dependency_results_and_artifacts_from_memory() -> None:
    plan = build_memory_plan()
    artifact = ArtifactRef(
        id="A1",
        kind=ArtifactKind.STRUCTURED,
        uri="memory://draft",
        summary="Draft artifact",
    )
    store = InMemoryStore()
    MemoryRecorder(store).record_task_result(
        TaskRunResult(task_id="T1", output={"draft": "from memory"}, artifacts=[artifact]),
        plan_id=plan.id,
    )

    context = ContextAssembler(memory_store=store).assemble(
        task=plan.task_cards[1],
        execution_plan=plan,
        plan_state=PlanState(objective=Objective(raw=plan.objective)),
    )

    assert context.dependency_results["T1"].output == {"draft": "from memory"}
    assert context.artifacts == [artifact]
    assert context.memory_context["working"][0]["payload"]["result"]["output"] == {
        "draft": "from memory"
    }
    assert context.memory_context["session"][0]["payload"]["artifact"]["id"] == "A1"
    assert context.plan_context["memory_context"] == context.memory_context


def test_runtime_engine_records_unified_memory_for_run_events() -> None:
    artifact = ArtifactRef(
        id="A1",
        kind=ArtifactKind.STRUCTURED,
        uri="memory://draft",
        summary="Draft artifact",
    )
    replacement_plan = build_memory_plan("EP-memory-replanned")

    def run_task(context: TaskExecutionContext) -> TaskRunResult:
        output = {"message": f"ran {context.task.id}"}
        artifacts = [artifact] if context.task.id == "T1" else []
        return TaskRunResult(task_id=context.task.id, output=output, artifacts=artifacts)

    def judge_task(payload: dict[str, object]) -> JudgeVerdict:
        result = payload["result"]
        assert isinstance(result, TaskRunResult)
        return JudgeVerdict(
            task_id=result.task_id,
            verdict="pass",
            overall_confidence=1.0,
            recommendation=JudgeRecommendation.ADVANCE,
        )

    def hold_gate(_: dict[str, object]) -> GateJudgment:
        return GateJudgment(
            gate_id="G1",
            milestone_id="M1",
            decision=GateDecision.REJECT,
            overall_confidence=0.8,
            reasoning="Force a replanning memory record.",
        )

    def replan(planner_input: ExecutionPlannerInput) -> ExecutionPlan:
        assert planner_input.context["trigger"]["command"]["source"] == "checkpoint_judge"
        return replacement_plan

    queue = PromptQueue()
    queue.push(PromptQueueItem(id="P1", content="What is the current status?"))
    memory_store = InMemoryStore()
    engine = RuntimeEngine(
        worker=RunnableLambda(run_task),
        judge=RunnableLambda(judge_task),
        checkpoint_judge=RunnableLambda(hold_gate),
        context_assembler=ContextAssembler(),
        prompt_queue=queue,
        runtime_replanner=RuntimeReplanner(RunnableLambda(replan)),
        memory_store=memory_store,
    )

    final_state = engine.invoke(
        build_memory_plan(),
        PlanState(
            objective=Objective(raw="Exercise unified memory"),
            discovery_plan=build_memory_discovery_plan(),
        ),
    )

    records = final_state["memory_records"]
    tags = {tag for record in records for tag in record.tags}
    kinds = {record.kind for record in records}

    assert records == memory_store.records()
    assert final_state["replan_results"][0].status == RuntimeReplanStatus.APPLIED
    assert {
        MemoryKind.WORKING,
        MemoryKind.SESSION,
    }.issubset(kinds)
    assert all(
        record.kind == MemoryKind.WORKING
        for record in records
        if "task_result" in record.tags
    )
    assert all(record.kind == MemoryKind.SESSION for record in records if "artifact" in record.tags)
    assert all(
        record.kind == MemoryKind.SESSION
        for record in records
        if set(record.tags)
        & {
            "plan_snapshot",
            "progress_signal",
            "judge_verdict",
            "prompt_handling",
            "runtime_command",
            "command_result",
            "gate_judgment",
            "replan_result",
            "plan_transition",
            "agent_lifecycle",
            "task_attempt",
        }
    )
    assert {
        "plan_snapshot",
        "task_dispatch",
        "task_context",
        "task_result",
        "artifact",
        "progress_signal",
        "judge_verdict",
        "prompt_handling",
        "runtime_command",
        "command_result",
        "gate_judgment",
        "replan_result",
        "plan_transition",
        "agent_lifecycle",
        "task_attempt",
    }.issubset(tags)
