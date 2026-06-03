from __future__ import annotations

from typing import Any

from langchain_core.runnables import Runnable
from pydantic import Field

from deep_agents.models import HandoffStep, MemoryKind, TaskCard
from deep_agents.models.base import DeepAgentsModel, JsonObject
from deep_agents.runtime.agent_registry import AgentRegistry
from deep_agents.runtime.context import TaskExecutionContext
from deep_agents.runtime.memory import MemoryRecorder
from deep_agents.runtime.results import TaskRunResult


class HandoffStepInput(DeepAgentsModel):
    """Input passed to an intra-task handoff step runnable."""

    parent_task: TaskCard
    step: HandoffStep
    parent_context: TaskExecutionContext | None = None
    previous_output: JsonObject = Field(default_factory=dict)
    shared_state: JsonObject = Field(default_factory=dict)


class HandoffRunner:
    """Execute ordered intra-task handoff chains."""

    def __init__(
        self,
        *,
        agent_registry: AgentRegistry | None = None,
        default_worker: Runnable[Any, Any],
        memory_recorder: MemoryRecorder,
    ) -> None:
        self.agent_registry = agent_registry
        self.default_worker = default_worker
        self.memory_recorder = memory_recorder

    def invoke(
        self,
        *,
        task: TaskCard,
        parent_input: TaskCard | TaskExecutionContext,
        plan_id: str | None,
    ) -> TaskRunResult:
        """Run a task handoff chain and return the final step output as the task result."""
        parent_context = parent_input if isinstance(parent_input, TaskExecutionContext) else None
        previous_output: JsonObject = {}
        shared_state: JsonObject = {}
        artifacts = []
        last_status: str | None = None

        for step in task.handoff_chain:
            step_input = HandoffStepInput(
                parent_task=task,
                step=step,
                parent_context=parent_context,
                previous_output=previous_output,
                shared_state=shared_state,
            )
            runnable = self._resolve_step_runnable(step)
            result = self._coerce_step_result(task.id, step.id, runnable.invoke(step_input))
            previous_output = result.output
            shared_state[step.id] = result.output
            artifacts.extend(result.artifacts)
            last_status = result.status
            self.memory_recorder.put(
                kind=MemoryKind.WORKING,
                source="handoff_runner",
                task_id=task.id,
                plan_id=plan_id,
                tags=["handoff_step_result", step.id],
                payload={
                    "step": step.model_dump(mode="json"),
                    "result": result.model_dump(mode="json"),
                    "shared_state": shared_state,
                },
            )

        return TaskRunResult(
            task_id=task.id,
            output=previous_output,
            artifacts=artifacts,
            status=last_status,
        )

    def _resolve_step_runnable(self, step: HandoffStep) -> Runnable[Any, Any]:
        if self.agent_registry is None:
            return self.default_worker
        return self.agent_registry.resolve(step.assigned_to) or self.default_worker

    def _coerce_step_result(
        self,
        parent_task_id: str,
        step_id: str,
        value: TaskRunResult | dict[str, Any],
    ) -> TaskRunResult:
        if isinstance(value, TaskRunResult):
            return value
        data = dict(value)
        data.setdefault("task_id", f"{parent_task_id}:{step_id}")
        return TaskRunResult(**data)
