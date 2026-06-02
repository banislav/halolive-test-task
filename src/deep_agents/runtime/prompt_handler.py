from __future__ import annotations

from typing import Any

from langchain_core.runnables import Runnable

from deep_agents.models import (
    ExecutionPlan,
    InterruptPriority,
    PlanState,
    PromptCategory,
    PromptClassification,
    PromptHandlingResult,
    PromptQueueItem,
    PromptReasoningInput,
    PromptResponse,
    RuntimeCommand,
    RuntimeCommandType,
)
from deep_agents.runtime.prompt_queue import PromptQueue
from deep_agents.runtime.results import TaskRunResult


class PromptHandler:
    """Classify and route queued user prompts at runtime boundaries."""

    def __init__(
        self,
        *,
        prompt_classifier: Runnable[PromptQueueItem, PromptClassification | dict[str, Any]]
        | None = None,
        content_reasoner: Runnable[PromptReasoningInput, PromptResponse | dict[str, Any]]
        | None = None,
    ) -> None:
        self.prompt_classifier = prompt_classifier
        self.content_reasoner = content_reasoner

    def handle_queue(
        self,
        queue: PromptQueue,
        *,
        execution_plan: ExecutionPlan,
        plan_state: PlanState,
        results: dict[str, TaskRunResult],
        memory_context: dict[str, Any] | None = None,
        current_task_id: str | None = None,
    ) -> list[PromptHandlingResult]:
        """Drain and handle queued prompts in deterministic order."""
        handled: list[PromptHandlingResult] = []
        for prompt in queue.drain():
            handled.append(
                self.handle_prompt(
                    prompt,
                    execution_plan=execution_plan,
                    plan_state=plan_state,
                    results=results,
                    memory_context=memory_context,
                    current_task_id=current_task_id,
                )
            )
        return handled

    def handle_prompt(
        self,
        prompt: PromptQueueItem,
        *,
        execution_plan: ExecutionPlan,
        plan_state: PlanState,
        results: dict[str, TaskRunResult],
        memory_context: dict[str, Any] | None = None,
        current_task_id: str | None = None,
    ) -> PromptHandlingResult:
        """Classify one prompt and produce a read-only response or advisory commands."""
        classification = self._classify(prompt)
        commands = self._interrupt_commands(prompt, current_task_id=current_task_id)
        response: PromptResponse | None = None

        if classification.category == PromptCategory.CONTENT_REASONING:
            response = self._answer_content_prompt(
                prompt,
                plan_state=plan_state,
                results=results,
                execution_plan=execution_plan,
                memory_context=memory_context,
            )
        else:
            commands.append(
                RuntimeCommand(
                    type=RuntimeCommandType.REQUEST_REPLAN,
                    reason=classification.reasoning,
                    payload={
                        "prompt_id": prompt.id,
                        "content": prompt.content,
                        "category": classification.category,
                    },
                    source="prompt_queue",
                )
            )

        return PromptHandlingResult(
            prompt=prompt,
            classification=classification,
            response=response,
            commands=commands,
        )

    def _classify(self, prompt: PromptQueueItem) -> PromptClassification:
        if prompt.category is not None:
            return PromptClassification(
                prompt_id=prompt.id,
                category=prompt.category,
                priority=prompt.priority,
                reasoning="Prompt item already included an explicit category.",
            )
        if self.prompt_classifier is not None:
            result = self.prompt_classifier.invoke(prompt)
            return self._coerce_classification(prompt, result)
        return self._classify_deterministically(prompt)

    def _answer_content_prompt(
        self,
        prompt: PromptQueueItem,
        *,
        plan_state: PlanState,
        results: dict[str, TaskRunResult],
        execution_plan: ExecutionPlan,
        memory_context: dict[str, Any] | None,
    ) -> PromptResponse:
        reasoning_input = PromptReasoningInput(
            prompt=prompt,
            plan_state=plan_state,
            results={
                task_id: result.model_dump(mode="json")
                for task_id, result in results.items()
            },
            context={
                "execution_plan_id": execution_plan.id,
                "memory_context": memory_context or {},
            },
        )
        if self.content_reasoner is not None:
            result = self.content_reasoner.invoke(reasoning_input)
            return self._coerce_response(prompt, result)
        memory_task_ids = self._task_ids_from_memory(memory_context or {})
        referenced_task_ids = list(dict.fromkeys([*results, *memory_task_ids]))
        artifact_ids = self._artifact_ids_from_memory(memory_context or {})
        return PromptResponse(
            prompt_id=prompt.id,
            answer=(
                f"Plan status is {plan_state.status}. "
                f"Known task results are available for: "
                f"{', '.join(referenced_task_ids) or 'none'}."
            ),
            referenced_task_ids=referenced_task_ids,
            referenced_artifact_ids=artifact_ids,
        )

    def _task_ids_from_memory(self, memory_context: dict[str, Any]) -> list[str]:
        task_ids: list[str] = []
        for records in memory_context.values():
            if not isinstance(records, list):
                continue
            for record in records:
                if isinstance(record, dict) and isinstance(record.get("task_id"), str):
                    task_ids.append(record["task_id"])
        return task_ids

    def _artifact_ids_from_memory(self, memory_context: dict[str, Any]) -> list[str]:
        artifact_ids: list[str] = []
        for record in memory_context.get("session", []):
            if not isinstance(record, dict) or "artifact" not in record.get("tags", []):
                continue
            payload = record.get("payload", {})
            artifact = payload.get("artifact") if isinstance(payload, dict) else None
            if isinstance(artifact, dict) and isinstance(artifact.get("id"), str):
                artifact_ids.append(artifact["id"])
        return artifact_ids

    def _interrupt_commands(
        self,
        prompt: PromptQueueItem,
        *,
        current_task_id: str | None,
    ) -> list[RuntimeCommand]:
        if prompt.priority == InterruptPriority.P0_HALT:
            return [
                RuntimeCommand(
                    type=RuntimeCommandType.HALT,
                    task_id=current_task_id,
                    reason="P0 prompt requested runtime halt.",
                    payload={"prompt_id": prompt.id, "content": prompt.content},
                    source="prompt_queue",
                )
            ]
        if prompt.priority == InterruptPriority.P1_PAUSE:
            return [
                RuntimeCommand(
                    type=RuntimeCommandType.PAUSE_TASK,
                    task_id=current_task_id,
                    reason="P1 prompt requested runtime pause.",
                    payload={"prompt_id": prompt.id, "content": prompt.content},
                    source="prompt_queue",
                )
            ]
        return []

    def _classify_deterministically(self, prompt: PromptQueueItem) -> PromptClassification:
        text = prompt.content.lower()
        plan_update_terms = {
            "change",
            "stop",
            "halt",
            "pause",
            "redo",
            "replan",
            "redirect",
            "replace",
            "modify",
            "scope",
        }
        content_terms = {
            "?",
            "status",
            "progress",
            "what",
            "show",
            "result",
            "summary",
            "explain",
        }
        if any(term in text for term in plan_update_terms):
            category = PromptCategory.PLAN_UPDATE
            reasoning = "Prompt appears to change, pause, redirect, or replan work."
        elif any(term in text for term in content_terms):
            category = PromptCategory.CONTENT_REASONING
            reasoning = "Prompt asks for status, progress, results, or explanation."
        else:
            category = PromptCategory.CONTENT_REASONING
            reasoning = "Prompt does not request a plan change, so it is treated as read-only."
        return PromptClassification(
            prompt_id=prompt.id,
            category=category,
            priority=prompt.priority,
            reasoning=reasoning,
        )

    def _coerce_classification(
        self,
        prompt: PromptQueueItem,
        value: PromptClassification | dict[str, Any],
    ) -> PromptClassification:
        if isinstance(value, PromptClassification):
            return value
        data: dict[str, Any] = {"prompt_id": prompt.id, "priority": prompt.priority}
        data.update(value)
        return PromptClassification(**data)

    def _coerce_response(
        self,
        prompt: PromptQueueItem,
        value: PromptResponse | dict[str, Any],
    ) -> PromptResponse:
        if isinstance(value, PromptResponse):
            return value
        data: dict[str, Any] = {"prompt_id": prompt.id}
        data.update(value)
        return PromptResponse(**data)
