from __future__ import annotations

import json
from typing import Any

from pydantic import Field

from deep_agents.models import (
    ArtifactRecord,
    ArtifactRef,
    DeepAgentsModel,
    ExecutionPlan,
    LayeredContext,
    MemoryKind,
    MemoryQuery,
    Objective,
    PlanState,
    TaskCard,
)
from deep_agents.models.base import JsonObject
from deep_agents.runtime.memory_context import build_memory_context
from deep_agents.runtime.results import TaskRunResult
from deep_agents.skills import SkillLoader


class TaskResultContext(DeepAgentsModel):
    """Compact task result projection for worker context injection."""

    task_id: str
    output: JsonObject = Field(default_factory=dict)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    status: str | None = None
    summary: str | None = None


class ContextBudgetReport(DeepAgentsModel):
    """Estimated context budget usage after deterministic compaction."""

    max_tokens: int = 0
    reserved_response_tokens: int = 0
    available_input_tokens: int = 0
    estimated_input_tokens: int = 0
    over_budget: bool = False
    compacted_result_ids: list[str] = Field(default_factory=list)
    dropped_prior_result_ids: list[str] = Field(default_factory=list)


class TaskExecutionContext(DeepAgentsModel):
    """Need-to-know context assembled for a single worker invocation."""

    task: TaskCard
    objective: Objective
    plan_context: JsonObject = Field(default_factory=dict)
    dependency_results: dict[str, TaskResultContext] = Field(default_factory=dict)
    prior_result_summaries: list[TaskResultContext] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    skill_context: str = ""
    loaded_skill_ids: list[str] = Field(default_factory=list)
    memory_context: JsonObject = Field(default_factory=dict)
    budget_report: ContextBudgetReport = Field(default_factory=ContextBudgetReport)
    layered_context: LayeredContext = Field(default_factory=LayeredContext)
    long_running: Any | None = Field(default=None, exclude=True)


class ArtifactStore:
    """In-memory artifact index for completed task outputs."""

    def __init__(self, records: list[ArtifactRecord] | None = None) -> None:
        self._records: dict[str, ArtifactRecord] = {}
        for record in records or []:
            self.add(record)

    def add(self, record: ArtifactRecord) -> None:
        """Store or replace an artifact record by artifact id."""
        self._records[record.ref.id] = record

    def add_result(self, result: TaskRunResult, *, tags: list[str] | None = None) -> None:
        """Index all artifacts produced by a task result."""
        for artifact in result.artifacts:
            self.add(
                ArtifactRecord(
                    ref=artifact,
                    producer_task_id=result.task_id,
                    tags=tags or [],
                )
            )

    def list(self) -> list[ArtifactRecord]:
        """Return all artifact records in insertion order."""
        return list(self._records.values())

    def for_task_ids(self, task_ids: list[str]) -> list[ArtifactRef]:
        """Return artifact refs produced by any of the provided task ids."""
        wanted = set(task_ids)
        return [
            record.ref
            for record in self._records.values()
            if record.producer_task_id in wanted
        ]

    def get(self, artifact_id: str) -> ArtifactRecord | None:
        """Return an artifact record by id, if present."""
        return self._records.get(artifact_id)


class ContextAssembler:
    """Build per-task context slices from full runtime state."""

    def __init__(
        self,
        *,
        artifact_store: ArtifactStore | None = None,
        memory_store: object | None = None,
        skill_loader: SkillLoader | None = None,
        summary_max_chars: int = 280,
        chars_per_token: int = 4,
    ) -> None:
        self.artifact_store = artifact_store or ArtifactStore()
        self.memory_store = memory_store
        self.skill_loader = skill_loader
        self.summary_max_chars = summary_max_chars
        self.chars_per_token = chars_per_token

    def assemble(
        self,
        *,
        task: TaskCard,
        execution_plan: ExecutionPlan,
        plan_state: PlanState,
        results: dict[str, TaskRunResult] | None = None,
    ) -> TaskExecutionContext:
        """Return the relevant context slice for one task."""
        resolved_results = self._results_from_memory(execution_plan.id)
        resolved_results.update(results or {})
        results = resolved_results
        for result in results.values():
            self.artifact_store.add_result(result)

        dependency_ids = self._dependency_ids(task, execution_plan)
        dependency_results = {
            task_id: self._full_result_context(result)
            for task_id, result in results.items()
            if task_id in dependency_ids
        }
        prior_result_summaries = [
            self._summary_result_context(result)
            for task_id, result in results.items()
            if task_id not in dependency_ids
        ]
        skill_context = ""
        loaded_skill_ids: list[str] = []
        if self.skill_loader is not None:
            loaded = self.skill_loader.load(task.assigned_to.skills)
            loaded_skill_ids = [skill.definition.id for skill in loaded]
            skill_context = self.skill_loader.render_context(task.assigned_to.skills)

        artifacts = self._artifact_refs(task, dependency_ids, execution_plan.id)
        memory_context = build_memory_context(
            self.memory_store,
            plan_id=execution_plan.id,
            task_id=task.id,
            dependency_task_ids=dependency_ids,
        )
        plan_context = self._plan_context(task, execution_plan, plan_state, dependency_ids)
        budget_report = self._apply_budget(
            task=task,
            plan_context=plan_context,
            dependency_results=dependency_results,
            prior_result_summaries=prior_result_summaries,
            artifacts=artifacts,
            skill_context=skill_context,
        )
        if memory_context:
            plan_context["memory_context"] = memory_context
        layered_context = LayeredContext(
            global_objective=plan_state.objective.model_dump(mode="json"),
            plan_state={
                "status": plan_state.status,
                "execution_plan_id": plan_state.execution_plan_id,
                "task_statuses": {
                    task_id: status
                    for task_id, status in plan_state.task_statuses.items()
                    if task_id == task.id or task_id in dependency_ids
                },
            },
            execution_state={
                "current_task_id": task.id,
                "dependency_ids": dependency_ids,
            },
            artifacts=artifacts,
            skill_state={
                "loaded_skill_ids": loaded_skill_ids,
                "assigned_skill_ids": [skill.id for skill in task.assigned_to.skills],
                "skill_memory": memory_context.get("skill", []),
            },
            agent_state={
                "agent_name": task.assigned_to.name,
                "agent_type": task.assigned_to.type,
            },
        )

        return TaskExecutionContext(
            task=task,
            objective=plan_state.objective,
            plan_context=plan_context,
            dependency_results=dependency_results,
            prior_result_summaries=prior_result_summaries,
            artifacts=artifacts,
            skill_context=skill_context,
            loaded_skill_ids=loaded_skill_ids,
            memory_context=memory_context,
            budget_report=budget_report,
            layered_context=layered_context,
        )

    def _dependency_ids(self, task: TaskCard, execution_plan: ExecutionPlan) -> list[str]:
        dependency_ids = list(task.blocked_by)
        graph_dependencies = execution_plan.dependency_graph.blocked_by.get(task.id, [])
        for task_id in graph_dependencies:
            if task_id not in dependency_ids:
                dependency_ids.append(task_id)
        return dependency_ids

    def _artifact_refs(
        self,
        task: TaskCard,
        dependency_ids: list[str],
        execution_plan_id: str,
    ) -> list[ArtifactRef]:
        refs: list[ArtifactRef] = []
        seen: set[str] = set()
        for artifact in [
            *task.input_artifacts,
            *self.artifact_store.for_task_ids(dependency_ids),
            *self._artifact_refs_from_memory(dependency_ids, execution_plan_id),
        ]:
            if artifact.id in seen:
                continue
            refs.append(artifact)
            seen.add(artifact.id)
        return refs

    def _results_from_memory(self, execution_plan_id: str) -> dict[str, TaskRunResult]:
        if self.memory_store is None or not hasattr(self.memory_store, "query"):
            return {}

        records = self.memory_store.query(
            MemoryQuery(
                kinds=[MemoryKind.WORKING],
                plan_ids=[execution_plan_id],
                tags=["task_result"],
            )
        )
        if not records:
            records = self.memory_store.query(
                MemoryQuery(kinds=[MemoryKind.WORKING], tags=["task_result"])
            )

        results: dict[str, TaskRunResult] = {}
        for record in records:
            payload = record.payload.get("result")
            if not isinstance(payload, dict):
                continue
            result = TaskRunResult.model_validate(payload)
            results[result.task_id] = result
        return results

    def _artifact_refs_from_memory(
        self,
        dependency_ids: list[str],
        execution_plan_id: str,
    ) -> list[ArtifactRef]:
        if self.memory_store is None or not hasattr(self.memory_store, "query"):
            return []

        refs: list[ArtifactRef] = []
        for task_id in dependency_ids:
            records = self.memory_store.query(
                MemoryQuery(
                    kinds=[MemoryKind.SESSION],
                    task_ids=[task_id],
                    plan_ids=[execution_plan_id],
                    tags=["artifact"],
                )
            )
            if not records:
                records = self.memory_store.query(
                    MemoryQuery(
                        kinds=[MemoryKind.SESSION],
                        task_ids=[task_id],
                        tags=["artifact"],
                    )
                )
            for record in records:
                payload = record.payload.get("artifact")
                if isinstance(payload, dict):
                    refs.append(ArtifactRef.model_validate(payload))
        return refs

    def _plan_context(
        self,
        task: TaskCard,
        execution_plan: ExecutionPlan,
        plan_state: PlanState,
        dependency_ids: list[str],
    ) -> JsonObject:
        wave = next(
            (wave for wave in execution_plan.waves if wave.index == task.wave),
            None,
        )
        return {
            "execution_plan_id": execution_plan.id,
            "plan_status": plan_state.status,
            "task_id": task.id,
            "wave": task.wave,
            "same_wave_task_ids": wave.task_ids if wave else [],
            "blocked_by": dependency_ids,
            "blocks": task.blocks,
            "context_budget": task.context_budget.model_dump(mode="json"),
        }

    def _full_result_context(self, result: TaskRunResult) -> TaskResultContext:
        return TaskResultContext(
            task_id=result.task_id,
            output=result.output,
            artifacts=result.artifacts,
            status=result.status,
            summary=self._summarize_output(result.output),
        )

    def _summary_result_context(self, result: TaskRunResult) -> TaskResultContext:
        return TaskResultContext(
            task_id=result.task_id,
            status=result.status,
            artifacts=result.artifacts,
            summary=self._summarize_output(result.output),
        )

    def _apply_budget(
        self,
        *,
        task: TaskCard,
        plan_context: JsonObject,
        dependency_results: dict[str, TaskResultContext],
        prior_result_summaries: list[TaskResultContext],
        artifacts: list[ArtifactRef],
        skill_context: str,
    ) -> ContextBudgetReport:
        available = max(
            task.context_budget.max_tokens - task.context_budget.reserved_response_tokens,
            0,
        )
        compacted_result_ids: list[str] = []
        dropped_prior_result_ids: list[str] = []
        estimated = self._estimate_context_tokens(
            plan_context=plan_context,
            dependency_results=dependency_results,
            prior_result_summaries=prior_result_summaries,
            artifacts=artifacts,
            skill_context=skill_context,
        )

        while prior_result_summaries and estimated > available:
            dropped = prior_result_summaries.pop(0)
            dropped_prior_result_ids.append(dropped.task_id)
            estimated = self._estimate_context_tokens(
                plan_context=plan_context,
                dependency_results=dependency_results,
                prior_result_summaries=prior_result_summaries,
                artifacts=artifacts,
                skill_context=skill_context,
            )

        for result in self._largest_results_first(dependency_results):
            if estimated <= available:
                break
            if not result.output:
                continue
            result.output = {}
            compacted_result_ids.append(result.task_id)
            estimated = self._estimate_context_tokens(
                plan_context=plan_context,
                dependency_results=dependency_results,
                prior_result_summaries=prior_result_summaries,
                artifacts=artifacts,
                skill_context=skill_context,
            )

        return ContextBudgetReport(
            max_tokens=task.context_budget.max_tokens,
            reserved_response_tokens=task.context_budget.reserved_response_tokens,
            available_input_tokens=available,
            estimated_input_tokens=estimated,
            over_budget=estimated > available,
            compacted_result_ids=compacted_result_ids,
            dropped_prior_result_ids=dropped_prior_result_ids,
        )

    def _estimate_context_tokens(
        self,
        *,
        plan_context: JsonObject,
        dependency_results: dict[str, TaskResultContext],
        prior_result_summaries: list[TaskResultContext],
        artifacts: list[ArtifactRef],
        skill_context: str,
    ) -> int:
        payload = {
            "plan_context": plan_context,
            "dependency_results": {
                task_id: result.model_dump(mode="json")
                for task_id, result in dependency_results.items()
            },
            "prior_result_summaries": [
                result.model_dump(mode="json") for result in prior_result_summaries
            ],
            "artifacts": [artifact.model_dump(mode="json") for artifact in artifacts],
            "skill_context": skill_context,
        }
        char_count = len(json.dumps(payload, sort_keys=True))
        return max(1, (char_count + self.chars_per_token - 1) // self.chars_per_token)

    def _largest_results_first(
        self,
        dependency_results: dict[str, TaskResultContext],
    ) -> list[TaskResultContext]:
        return sorted(
            dependency_results.values(),
            key=lambda result: len(json.dumps(result.output, sort_keys=True)),
            reverse=True,
        )

    def _summarize_output(self, output: JsonObject) -> str:
        if not output:
            return ""
        text = json.dumps(output, sort_keys=True)
        if len(text) <= self.summary_max_chars:
            return text
        return f"{text[: self.summary_max_chars - 3]}..."
