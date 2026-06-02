from __future__ import annotations

import json
from typing import Protocol
from uuid import uuid4

from deep_agents.models import (
    ArtifactRef,
    ExecutionPlan,
    GateJudgment,
    JudgeVerdict,
    MemoryKind,
    MemoryQuery,
    MemoryRecord,
    ObserverJudgment,
    PlanState,
    ProcessJudgment,
    ProgressSignal,
    PromptHandlingResult,
    RuntimeCommand,
    RuntimeCommandResult,
    RuntimeReplanResult,
)
from deep_agents.models.base import JsonObject
from deep_agents.runtime.context import TaskExecutionContext
from deep_agents.runtime.results import TaskRunResult


class MemoryStore(Protocol):
    """Storage interface for architecture-native memory records."""

    def put(self, record: MemoryRecord) -> MemoryRecord:
        """Store one memory record and return it."""
        ...

    def put_many(self, records: list[MemoryRecord]) -> list[MemoryRecord]:
        """Store multiple records in order and return them."""
        ...

    def get(self, record_id: str) -> MemoryRecord | None:
        """Return one record by id."""
        ...

    def query(self, query: MemoryQuery) -> list[MemoryRecord]:
        """Return records matching the query in deterministic insertion order."""
        ...

    def by_task(self, task_id: str) -> list[MemoryRecord]:
        """Return records associated with one task id."""
        ...

    def records(self) -> list[MemoryRecord]:
        """Return all records in insertion order."""
        ...


class InMemoryStore:
    """Deterministic in-memory implementation of MemoryStore."""

    def __init__(self, records: list[MemoryRecord] | None = None) -> None:
        self._records: dict[str, MemoryRecord] = {}
        self._order: list[str] = []
        self.put_many(records or [])

    def put(self, record: MemoryRecord) -> MemoryRecord:
        """Store one memory record and return it."""
        if record.id not in self._records:
            self._order.append(record.id)
        self._records[record.id] = record
        return record

    def put_many(self, records: list[MemoryRecord]) -> list[MemoryRecord]:
        """Store multiple records in order and return them."""
        for record in records:
            self.put(record)
        return records

    def get(self, record_id: str) -> MemoryRecord | None:
        """Return one record by id."""
        return self._records.get(record_id)

    def query(self, query: MemoryQuery) -> list[MemoryRecord]:
        """Return records matching the query in deterministic insertion order."""
        matches: list[MemoryRecord] = []
        for record in self.records():
            if not self._matches(record, query):
                continue
            matches.append(record)
            if query.limit is not None and len(matches) >= query.limit:
                break
        return matches

    def by_task(self, task_id: str) -> list[MemoryRecord]:
        """Return records associated with one task id."""
        return self.query(MemoryQuery(task_ids=[task_id]))

    def records(self) -> list[MemoryRecord]:
        """Return all records in insertion order."""
        return [self._records[record_id] for record_id in self._order]

    def _matches(self, record: MemoryRecord, query: MemoryQuery) -> bool:
        if query.kinds and record.kind not in query.kinds:
            return False
        if query.task_ids and record.task_id not in query.task_ids:
            return False
        if query.plan_ids and record.plan_id not in query.plan_ids:
            return False
        if query.tags and not set(query.tags).issubset(record.tags):
            return False
        if query.text_query:
            haystack = json.dumps(record.model_dump(mode="json"), sort_keys=True).lower()
            if query.text_query.lower() not in haystack:
                return False
        return True


class MemoryRecorder:
    """Typed helper for writing runtime objects as memory records."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def put(
        self,
        *,
        kind: MemoryKind,
        source: str,
        payload: JsonObject,
        scope: JsonObject | None = None,
        tags: list[str] | None = None,
        task_id: str | None = None,
        plan_id: str | None = None,
    ) -> MemoryRecord:
        """Write one memory record with a generated id."""
        return self.store.put(
            MemoryRecord(
                id=f"mem-{uuid4().hex}",
                kind=kind,
                scope=scope or {},
                payload=payload,
                tags=tags or [],
                task_id=task_id,
                plan_id=plan_id,
                source=source,
            )
        )

    def record_plan_snapshot(
        self,
        *,
        execution_plan: ExecutionPlan,
        plan_state: PlanState,
        results: dict[str, TaskRunResult],
        source: str,
        reason: str,
    ) -> MemoryRecord:
        """Record a current-session plan state snapshot."""
        return self.put(
            kind=MemoryKind.SESSION,
            source=source,
            plan_id=execution_plan.id,
            tags=["plan_snapshot"],
            payload={
                "reason": reason,
                "execution_plan": execution_plan.model_dump(mode="json"),
                "plan_state": plan_state.model_dump(mode="json"),
                "retained_result_task_ids": list(results),
            },
        )

    def record_plan_transition(
        self,
        *,
        previous_plan_id: str,
        new_plan_id: str | None,
        trigger: RuntimeCommandResult,
        preserved_task_ids: list[str],
        dropped_task_ids: list[str],
    ) -> MemoryRecord:
        """Record a current-session replan transition between execution plans."""
        return self.put(
            kind=MemoryKind.SESSION,
            source="runtime_replanner",
            plan_id=new_plan_id or previous_plan_id,
            tags=["plan_transition", "replan"],
            payload={
                "previous_execution_plan_id": previous_plan_id,
                "new_execution_plan_id": new_plan_id,
                "trigger": trigger.model_dump(mode="json"),
                "preserved_task_ids": preserved_task_ids,
                "dropped_task_ids": dropped_task_ids,
            },
        )

    def record_working_context(
        self,
        context: TaskExecutionContext,
        *,
        plan_id: str | None,
    ) -> MemoryRecord:
        """Record the context slice assembled for a worker."""
        return self.put(
            kind=MemoryKind.WORKING,
            source="context_assembler",
            task_id=context.task.id,
            plan_id=plan_id,
            tags=["task_context"],
            payload=context.model_dump(mode="json"),
        )

    def record_task_result(
        self,
        result: TaskRunResult,
        *,
        plan_id: str | None,
    ) -> list[MemoryRecord]:
        """Record a worker result as working memory and any produced artifacts."""
        records = [
            self.put(
                kind=MemoryKind.WORKING,
                source="worker",
                task_id=result.task_id,
                plan_id=plan_id,
                tags=["task_result"],
                payload={"result": result.model_dump(mode="json")},
            )
        ]
        for artifact in result.artifacts:
            records.append(self.record_artifact(artifact, task_id=result.task_id, plan_id=plan_id))
        return records

    def record_artifact(
        self,
        artifact: ArtifactRef,
        *,
        task_id: str | None,
        plan_id: str | None,
    ) -> MemoryRecord:
        """Record an artifact reference with task provenance."""
        return self.put(
            kind=MemoryKind.SESSION,
            source="artifact_store",
            task_id=task_id,
            plan_id=plan_id,
            tags=["artifact"],
            payload={"artifact": artifact.model_dump(mode="json")},
        )

    def record_progress_signal(
        self,
        signal: ProgressSignal,
        *,
        plan_id: str | None,
    ) -> MemoryRecord:
        """Record a current-session progress signal."""
        return self.put(
            kind=MemoryKind.SESSION,
            source="progress_signal_bus",
            task_id=signal.task_id,
            plan_id=plan_id,
            tags=["progress_signal", signal.signal_type],
            payload={"signal": signal.model_dump(mode="json")},
        )

    def record_judge_verdict(
        self,
        verdict: JudgeVerdict,
        *,
        plan_id: str | None,
    ) -> MemoryRecord:
        """Record a task judge verdict."""
        return self.put(
            kind=MemoryKind.SESSION,
            source="task_judge",
            task_id=verdict.task_id,
            plan_id=plan_id,
            tags=["judge_verdict"],
            payload={"verdict": verdict.model_dump(mode="json")},
        )

    def record_process_judgment(
        self,
        judgment: ProcessJudgment,
        *,
        plan_id: str | None,
    ) -> MemoryRecord:
        """Record a process judge judgment."""
        return self.put(
            kind=MemoryKind.SESSION,
            source="process_judge",
            task_id=judgment.task_id,
            plan_id=plan_id,
            tags=["process_judgment"],
            payload={"judgment": judgment.model_dump(mode="json")},
        )

    def record_observer_judgment(
        self,
        judgment: ObserverJudgment,
        *,
        plan_id: str | None,
    ) -> MemoryRecord:
        """Record an observer judge judgment."""
        return self.put(
            kind=MemoryKind.SESSION,
            source="observer_judge",
            plan_id=plan_id,
            tags=["observer_judgment"],
            payload={"judgment": judgment.model_dump(mode="json")},
        )

    def record_prompt_result(
        self,
        prompt_result: PromptHandlingResult,
        *,
        plan_id: str | None,
    ) -> MemoryRecord:
        """Record a prompt handling result."""
        return self.put(
            kind=MemoryKind.SESSION,
            source="prompt_queue",
            plan_id=plan_id,
            tags=["prompt_handling"],
            payload={"prompt_result": prompt_result.model_dump(mode="json")},
        )

    def record_runtime_command(
        self,
        command: RuntimeCommand,
        *,
        plan_id: str | None,
    ) -> MemoryRecord:
        """Record an emitted runtime command."""
        return self.put(
            kind=MemoryKind.SESSION,
            source=command.source,
            task_id=command.task_id,
            plan_id=plan_id,
            tags=["runtime_command", command.type],
            payload={"command": command.model_dump(mode="json")},
        )

    def record_command_result(
        self,
        result: RuntimeCommandResult,
        *,
        plan_id: str | None,
    ) -> MemoryRecord:
        """Record a runtime command execution result."""
        return self.put(
            kind=MemoryKind.SESSION,
            source="runtime_command_executor",
            task_id=result.command.task_id,
            plan_id=plan_id,
            tags=["command_result", result.command.type],
            payload={"command_result": result.model_dump(mode="json")},
        )

    def record_gate_judgment(
        self,
        judgment: GateJudgment,
        *,
        plan_id: str | None,
    ) -> MemoryRecord:
        """Record a checkpoint gate judgment."""
        return self.put(
            kind=MemoryKind.SESSION,
            source="checkpoint_judge",
            plan_id=plan_id,
            tags=["gate_judgment"],
            payload={"judgment": judgment.model_dump(mode="json")},
        )

    def record_replan_result(self, result: RuntimeReplanResult) -> MemoryRecord:
        """Record a runtime replanning result."""
        return self.put(
            kind=MemoryKind.SESSION,
            source="runtime_replanner",
            plan_id=result.new_execution_plan_id or result.previous_execution_plan_id,
            tags=["replan_result"],
            payload={"replan_result": result.model_dump(mode="json")},
        )

    def record_semantic_fact(
        self,
        *,
        fact: JsonObject,
        source: str,
        tags: list[str] | None = None,
        task_id: str | None = None,
        plan_id: str | None = None,
    ) -> MemoryRecord:
        """Record an explicit durable fact or knowledge item."""
        return self.put(
            kind=MemoryKind.SEMANTIC,
            source=source,
            task_id=task_id,
            plan_id=plan_id,
            tags=tags or ["semantic_fact"],
            payload={"fact": fact},
        )

    def record_procedural_memory(
        self,
        *,
        payload: JsonObject,
        source: str,
        tags: list[str] | None = None,
        plan_id: str | None = None,
    ) -> MemoryRecord:
        """Record a proven execution pattern or reusable procedure."""
        return self.put(
            kind=MemoryKind.PROCEDURAL,
            source=source,
            plan_id=plan_id,
            tags=tags or ["procedural"],
            payload=payload,
        )

    def record_episodic_memory(
        self,
        *,
        payload: JsonObject,
        source: str,
        tags: list[str] | None = None,
        plan_id: str | None = None,
    ) -> MemoryRecord:
        """Record persistent cross-session experience, such as user preferences."""
        return self.put(
            kind=MemoryKind.EPISODIC,
            source=source,
            plan_id=plan_id,
            tags=tags or ["episodic"],
            payload=payload,
        )

    def record_skill_memory(
        self,
        *,
        payload: JsonObject,
        source: str,
        tags: list[str] | None = None,
        plan_id: str | None = None,
    ) -> MemoryRecord:
        """Record persistent knowledge about skill usage or skill relationships."""
        return self.put(
            kind=MemoryKind.SKILL,
            source=source,
            plan_id=plan_id,
            tags=tags or ["skill"],
            payload=payload,
        )
