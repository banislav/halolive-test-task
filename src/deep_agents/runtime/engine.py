from __future__ import annotations

from typing import Any, NotRequired, TypedDict

from langchain_core.runnables import Runnable
from langgraph.graph import END, StateGraph

from deep_agents.instrumentation import get_logger
from deep_agents.models import (
    ExecutionPlan,
    Gate,
    GateJudgment,
    JudgeRecommendation,
    JudgeVerdict,
    MemoryKind,
    MemoryRecord,
    Milestone,
    ObserverJudgment,
    PlanState,
    PlanStatus,
    ProcessJudgment,
    ProgressSignal,
    ProgressSignalPayload,
    ProgressSignalType,
    PromptClassification,
    PromptHandlingResult,
    PromptQueueItem,
    PromptReasoningInput,
    PromptResponse,
    RuntimeCommand,
    RuntimeCommandResult,
    RuntimeCommandStatus,
    RuntimeCommandType,
    RuntimeReplanResult,
    TaskAttemptRecord,
    TaskAttemptStatus,
    TaskCard,
)
from deep_agents.runtime.agent_registry import AgentRegistry
from deep_agents.runtime.command_executor import RuntimeCommandExecutor
from deep_agents.runtime.context import ContextAssembler, TaskExecutionContext
from deep_agents.runtime.handoffs import HandoffRunner
from deep_agents.runtime.long_running import LongRunningRunRegistry
from deep_agents.runtime.memory import InMemoryStore, MemoryRecorder, MemoryStore
from deep_agents.runtime.memory_context import build_memory_context
from deep_agents.runtime.observability import ProgressSignalBus
from deep_agents.runtime.plan_tracker import PlanTracker
from deep_agents.runtime.prompt_handler import PromptHandler
from deep_agents.runtime.prompt_queue import PromptQueue
from deep_agents.runtime.replanner import RuntimeReplanner
from deep_agents.runtime.results import TaskRunResult
from deep_agents.runtime.task_attempts import TaskAttemptRunError, TaskAttemptRunner

logger = get_logger(__name__)


class RuntimeGraphState(TypedDict):
    """State carried through the LangGraph runtime engine."""
    execution_plan: ExecutionPlan
    plan_state: PlanState
    current_task_id: NotRequired[str | None]
    current_context: NotRequired[TaskExecutionContext | None]
    latest_result: NotRequired[TaskRunResult | None]
    latest_verdict: NotRequired[JudgeVerdict | None]
    results: NotRequired[dict[str, TaskRunResult]]
    process_judgments: NotRequired[list[ProcessJudgment]]
    observer_judgments: NotRequired[list[ObserverJudgment]]
    gate_judgments: NotRequired[list[GateJudgment]]
    prompt_results: NotRequired[list[PromptHandlingResult]]
    runtime_commands: NotRequired[list[RuntimeCommand]]
    command_results: NotRequired[list[RuntimeCommandResult]]
    replan_results: NotRequired[list[RuntimeReplanResult]]
    task_attempts: NotRequired[list[TaskAttemptRecord]]
    memory_records: NotRequired[list[MemoryRecord]]


class RuntimeEngine:
    """LangGraph-backed engine that dispatches, runs, judges, and advances tasks."""
    def __init__(
        self,
        *,
        worker: Runnable[TaskCard | TaskExecutionContext, TaskRunResult | dict[str, Any]],
        judge: Runnable[dict[str, Any], JudgeVerdict | dict[str, Any]],
        checkpoint_judge: Runnable[dict[str, Any], GateJudgment | dict[str, Any]] | None = None,
        context_assembler: ContextAssembler | None = None,
        prompt_queue: PromptQueue | None = None,
        prompt_classifier: Runnable[PromptQueueItem, PromptClassification | dict[str, Any]]
        | None = None,
        content_reasoner: Runnable[PromptReasoningInput, PromptResponse | dict[str, Any]]
        | None = None,
        command_executor: RuntimeCommandExecutor | None = None,
        runtime_replanner: RuntimeReplanner | None = None,
        agent_registry: AgentRegistry | None = None,
        memory_store: MemoryStore | None = None,
        progress_bus: ProgressSignalBus | None = None,
        recursion_limit: int = 100,
    ) -> None:
        """Create an engine from LangChain runnables for task work and judgment."""
        self.worker = worker
        self.judge = judge
        self.checkpoint_judge = checkpoint_judge
        self.memory_store = memory_store or InMemoryStore()
        self.memory_recorder = MemoryRecorder(self.memory_store)
        self.context_assembler = context_assembler
        if self.context_assembler is not None and self.context_assembler.memory_store is None:
            self.context_assembler.memory_store = self.memory_store
        self.prompt_queue = prompt_queue
        self.prompt_handler = PromptHandler(
            prompt_classifier=prompt_classifier,
            content_reasoner=content_reasoner,
        )
        self.long_running_registry = LongRunningRunRegistry()
        self.command_executor = command_executor or RuntimeCommandExecutor()
        if self.command_executor.long_running_registry is None:
            self.command_executor.long_running_registry = self.long_running_registry
        self.runtime_replanner = runtime_replanner
        self.agent_registry = agent_registry
        self.progress_bus = progress_bus or ProgressSignalBus()
        self.recursion_limit = recursion_limit
        self.graph = self._build_graph()

    def invoke(self, execution_plan: ExecutionPlan, plan_state: PlanState) -> RuntimeGraphState:
        """Run the graph synchronously until the plan reaches a terminal state."""
        logger.info(
            "runtime engine invoke started",
            extra={"execution_plan_id": execution_plan.id, "plan_status": plan_state.status},
        )
        initial_state: RuntimeGraphState = {
            "execution_plan": execution_plan,
            "plan_state": plan_state,
            "current_task_id": None,
            "current_context": None,
            "latest_result": None,
            "latest_verdict": None,
            "results": {},
            "process_judgments": [],
            "observer_judgments": [],
            "gate_judgments": [],
            "prompt_results": [],
            "runtime_commands": [],
            "command_results": [],
            "replan_results": [],
            "task_attempts": [],
            "memory_records": [],
        }
        self.memory_recorder.record_plan_snapshot(
            execution_plan=execution_plan,
            plan_state=plan_state,
            results={},
            source="runtime_engine",
            reason="initial_plan",
        )
        self._sync_memory_state(initial_state)
        try:
            final_state = self.graph.invoke(
                initial_state,
                config={"recursion_limit": self.recursion_limit},
            )
        except Exception:
            logger.exception(
                "runtime engine invoke failed",
                extra={"execution_plan_id": execution_plan.id},
            )
            raise
        self._sync_memory_state(final_state)
        logger.info(
            "runtime engine invoke completed",
            extra={
                "execution_plan_id": execution_plan.id,
                "plan_status": final_state["plan_state"].status,
            },
        )
        return final_state

    async def ainvoke(
        self,
        execution_plan: ExecutionPlan,
        plan_state: PlanState,
    ) -> RuntimeGraphState:
        """Run the graph asynchronously until the plan reaches a terminal state."""
        logger.info(
            "runtime engine async invoke started",
            extra={"execution_plan_id": execution_plan.id, "plan_status": plan_state.status},
        )
        initial_state: RuntimeGraphState = {
            "execution_plan": execution_plan,
            "plan_state": plan_state,
            "current_task_id": None,
            "current_context": None,
            "latest_result": None,
            "latest_verdict": None,
            "results": {},
            "process_judgments": [],
            "observer_judgments": [],
            "gate_judgments": [],
            "prompt_results": [],
            "runtime_commands": [],
            "command_results": [],
            "replan_results": [],
            "task_attempts": [],
            "memory_records": [],
        }
        self.memory_recorder.record_plan_snapshot(
            execution_plan=execution_plan,
            plan_state=plan_state,
            results={},
            source="runtime_engine",
            reason="initial_plan",
        )
        self._sync_memory_state(initial_state)
        try:
            final_state = await self.graph.ainvoke(
                initial_state,
                config={"recursion_limit": self.recursion_limit},
            )
        except Exception:
            logger.exception(
                "runtime engine async invoke failed",
                extra={"execution_plan_id": execution_plan.id},
            )
            raise
        self._sync_memory_state(final_state)
        logger.info(
            "runtime engine async invoke completed",
            extra={
                "execution_plan_id": execution_plan.id,
                "plan_status": final_state["plan_state"].status,
            },
        )
        return final_state

    def _build_graph(self) -> Any:
        graph = StateGraph(RuntimeGraphState)
        graph.add_node("dispatch", self._dispatch_node)
        graph.add_node("worker", self._worker_node)
        graph.add_node("judge", self._judge_node)
        graph.add_node("apply_verdict", self._apply_verdict_node)

        graph.set_entry_point("dispatch")
        graph.add_conditional_edges(
            "dispatch",
            self._route_after_dispatch,
            {
                "worker": "worker",
                "end": END,
            },
        )
        graph.add_edge("worker", "judge")
        graph.add_edge("judge", "apply_verdict")
        graph.add_edge("apply_verdict", "dispatch")
        return graph.compile()

    def _dispatch_node(self, state: RuntimeGraphState) -> RuntimeGraphState:
        try:
            tracker = PlanTracker(state["plan_state"], state["execution_plan"])

            if self._is_terminal(tracker.state.status):
                state["current_task_id"] = None
                return state

            if self._handle_prompt_queue(state):
                state["current_task_id"] = None
                return state

            tracker = PlanTracker(state["plan_state"], state["execution_plan"])
            ready_ids = tracker.refresh_readiness()
            if not ready_ids:
                logger.info(
                    "no ready tasks available",
                    extra={"execution_plan_id": state["execution_plan"].id},
                )
                state["current_task_id"] = None
                return state

            task_id = ready_ids[0]
            tracker.mark_running(task_id)
            state["current_task_id"] = task_id
            task = self._current_task(state)
            self.memory_recorder.put(
                kind=MemoryKind.WORKING,
                source="dispatcher",
                task_id=task_id,
                plan_id=state["execution_plan"].id,
                tags=["task_dispatch"],
                payload={
                    "task": task.model_dump(mode="json"),
                    "plan_status": tracker.state.status,
                },
            )
            self._publish_signal(
                state,
                task_id=task_id,
                signal_type=ProgressSignalType.PROGRESS,
                payload=ProgressSignalPayload(
                    status="dispatched",
                    data={
                        "execution_plan_id": state["execution_plan"].id,
                        "wave": task.wave,
                    },
                ),
            )
            logger.info(
                "task dispatched",
                extra={"execution_plan_id": state["execution_plan"].id, "task_id": task_id},
            )
            return state
        except Exception:
            logger.exception("dispatch node failed")
            raise

    def _worker_node(self, state: RuntimeGraphState) -> RuntimeGraphState:
        task = self._current_task(state)
        try:
            logger.info("worker started", extra={"task_id": task.id})
            self._publish_signal(
                state,
                task_id=task.id,
                signal_type=ProgressSignalType.HEARTBEAT,
                payload=ProgressSignalPayload(status="worker_started"),
            )
            worker_input = self._worker_input(task, state)
            result, attempts = TaskAttemptRunner(
                invoker=lambda current_task, current_input: self._invoke_task_agent(
                    current_task,
                    current_input,
                    state,
                ),
                memory_recorder=self.memory_recorder,
                plan_id=state["execution_plan"].id,
                progress_bus=self.progress_bus,
                long_running_registry=self.long_running_registry,
            ).invoke(task, worker_input)
            state.setdefault("task_attempts", []).extend(attempts)
            state["latest_result"] = result
            self.memory_recorder.record_task_result(
                state["latest_result"],
                plan_id=state["execution_plan"].id,
            )
            self._publish_signal(
                state,
                task_id=task.id,
                signal_type=ProgressSignalType.PROGRESS,
                payload=ProgressSignalPayload(status="worker_completed", percent_complete=100),
            )
            if state["latest_result"].output:
                self._publish_signal(
                    state,
                    task_id=task.id,
                    signal_type=ProgressSignalType.FINDING,
                    payload=ProgressSignalPayload(
                        status="worker_output",
                        actionable=True,
                        relevance_score=1.0,
                        data={"output": state["latest_result"].output},
                    ),
            )
            logger.info("worker completed", extra={"task_id": task.id})
            return state
        except TaskAttemptRunError as exc:
            state.setdefault("task_attempts", []).extend(exc.attempts)
            PlanTracker(state["plan_state"], state["execution_plan"]).apply_task_failure(
                task.id,
                recoverable=True,
            )
            self._record_retry_exhaustion_command(state, exc)
            self._publish_signal(
                state,
                task_id=task.id,
                signal_type=ProgressSignalType.ERROR,
                payload=ProgressSignalPayload(
                    status="worker_failed",
                    error_type=exc.last_exception.__class__.__name__,
                    detail=str(exc.last_exception),
                    data={
                        "attempt_count": len(exc.attempts),
                        "last_attempt_status": exc.attempts[-1].status
                        if exc.attempts
                        else None,
                    },
                ),
            )
            logger.exception(
                "worker failed",
                extra={"task_id": task.id, "attempt_count": len(exc.attempts)},
            )
            raise exc.last_exception from exc
        except Exception:
            self._publish_signal(
                state,
                task_id=task.id,
                signal_type=ProgressSignalType.ERROR,
                payload=ProgressSignalPayload(
                    status="worker_failed",
                    error_type="worker_error",
                    detail=f"Worker failed for task {task.id}",
                ),
            )
            logger.exception("worker failed", extra={"task_id": task.id})
            raise

    async def _worker_node_async(self, state: RuntimeGraphState) -> RuntimeGraphState:
        task = self._current_task(state)
        try:
            logger.info("worker async started", extra={"task_id": task.id})
            self._publish_signal(
                state,
                task_id=task.id,
                signal_type=ProgressSignalType.HEARTBEAT,
                payload=ProgressSignalPayload(status="worker_started"),
            )
            worker_input = self._worker_input(task, state)
            result = await self._ainvoke_task_agent(task, worker_input, state)
            state["latest_result"] = self._coerce_result(task.id, result)
            self.memory_recorder.record_task_result(
                state["latest_result"],
                plan_id=state["execution_plan"].id,
            )
            self._publish_signal(
                state,
                task_id=task.id,
                signal_type=ProgressSignalType.PROGRESS,
                payload=ProgressSignalPayload(status="worker_completed", percent_complete=100),
            )
            if state["latest_result"].output:
                self._publish_signal(
                    state,
                    task_id=task.id,
                    signal_type=ProgressSignalType.FINDING,
                    payload=ProgressSignalPayload(
                        status="worker_output",
                        actionable=True,
                        relevance_score=1.0,
                        data={"output": state["latest_result"].output},
                    ),
                )
            logger.info("worker async completed", extra={"task_id": task.id})
            return state
        except Exception:
            self._publish_signal(
                state,
                task_id=task.id,
                signal_type=ProgressSignalType.ERROR,
                payload=ProgressSignalPayload(
                    status="worker_failed",
                    error_type="worker_error",
                    detail=f"Worker failed for task {task.id}",
                ),
            )
            logger.exception("worker async failed", extra={"task_id": task.id})
            raise

    def _judge_node(self, state: RuntimeGraphState) -> RuntimeGraphState:
        task = self._current_task(state)
        result = state.get("latest_result")
        try:
            logger.info("judge started", extra={"task_id": task.id})
            self._publish_signal(
                state,
                task_id=task.id,
                signal_type=ProgressSignalType.PROGRESS,
                payload=ProgressSignalPayload(status="judge_started"),
            )
            verdict = self.judge.invoke({"task": task, "result": result})
            state["latest_verdict"] = self._coerce_verdict(verdict)
            self.memory_recorder.record_judge_verdict(
                state["latest_verdict"],
                plan_id=state["execution_plan"].id,
            )
            self._publish_signal(
                state,
                task_id=task.id,
                signal_type=ProgressSignalType.PROGRESS,
                payload=ProgressSignalPayload(
                    status="judge_completed",
                    data={
                        "verdict": state["latest_verdict"].verdict,
                        "recommendation": state["latest_verdict"].recommendation,
                        "confidence": state["latest_verdict"].overall_confidence,
                    },
                ),
            )
            logger.info("judge completed", extra={"task_id": task.id})
            return state
        except Exception:
            self._publish_signal(
                state,
                task_id=task.id,
                signal_type=ProgressSignalType.ERROR,
                payload=ProgressSignalPayload(
                    status="judge_failed",
                    error_type="judge_error",
                    detail=f"Judge failed for task {task.id}",
                ),
            )
            logger.exception("judge failed", extra={"task_id": task.id})
            raise

    async def _judge_node_async(self, state: RuntimeGraphState) -> RuntimeGraphState:
        task = self._current_task(state)
        result = state.get("latest_result")
        try:
            logger.info("judge async started", extra={"task_id": task.id})
            self._publish_signal(
                state,
                task_id=task.id,
                signal_type=ProgressSignalType.PROGRESS,
                payload=ProgressSignalPayload(status="judge_started"),
            )
            verdict = await self.judge.ainvoke({"task": task, "result": result})
            state["latest_verdict"] = self._coerce_verdict(verdict)
            self.memory_recorder.record_judge_verdict(
                state["latest_verdict"],
                plan_id=state["execution_plan"].id,
            )
            self._publish_signal(
                state,
                task_id=task.id,
                signal_type=ProgressSignalType.PROGRESS,
                payload=ProgressSignalPayload(
                    status="judge_completed",
                    data={
                        "verdict": state["latest_verdict"].verdict,
                        "recommendation": state["latest_verdict"].recommendation,
                        "confidence": state["latest_verdict"].overall_confidence,
                    },
                ),
            )
            logger.info("judge async completed", extra={"task_id": task.id})
            return state
        except Exception:
            self._publish_signal(
                state,
                task_id=task.id,
                signal_type=ProgressSignalType.ERROR,
                payload=ProgressSignalPayload(
                    status="judge_failed",
                    error_type="judge_error",
                    detail=f"Judge failed for task {task.id}",
                ),
            )
            logger.exception("judge async failed", extra={"task_id": task.id})
            raise

    def _apply_verdict_node(self, state: RuntimeGraphState) -> RuntimeGraphState:
        verdict = state.get("latest_verdict")
        result = state.get("latest_result")
        if verdict is None:
            msg = "cannot apply verdict before judge node runs"
            logger.error(msg)
            raise RuntimeError(msg)

        try:
            tracker = PlanTracker(state["plan_state"], state["execution_plan"])
            tracker.apply_judge_verdict(verdict)

            if result is not None:
                results = state.setdefault("results", {})
                results[result.task_id] = result
                logger.info("task result recorded", extra={"task_id": result.task_id})

            if verdict.recommendation == JudgeRecommendation.REPLAN:
                self._record_runtime_commands(
                    state,
                    [
                        RuntimeCommand(
                            type=RuntimeCommandType.REQUEST_REPLAN,
                            task_id=verdict.task_id,
                            reason="Task judge requested replanning.",
                            payload={
                                "verdict": verdict.verdict,
                                "recommendation": verdict.recommendation,
                                "confidence": verdict.overall_confidence,
                            },
                            source="task_judge",
                        )
                    ],
                )
                tracker = PlanTracker(state["plan_state"], state["execution_plan"])

            self._evaluate_checkpoint_gates(state, tracker)

            self._publish_signal(
                state,
                task_id=verdict.task_id,
                signal_type=ProgressSignalType.PROGRESS,
                payload=ProgressSignalPayload(
                    status="verdict_applied",
                    data={
                        "verdict": verdict.verdict,
                        "recommendation": verdict.recommendation,
                        "plan_status": tracker.state.status,
                    },
                ),
            )
            state["current_task_id"] = None
            state["current_context"] = None
            state["latest_result"] = None
            state["latest_verdict"] = None
            return state
        except Exception:
            logger.exception(
                "apply verdict failed",
                extra={"task_id": verdict.task_id, "recommendation": verdict.recommendation},
            )
            self._publish_signal(
                state,
                task_id=verdict.task_id,
                signal_type=ProgressSignalType.ERROR,
                payload=ProgressSignalPayload(
                    status="apply_verdict_failed",
                    error_type="verdict_application_error",
                    detail=f"Could not apply verdict for task {verdict.task_id}",
                ),
            )
            raise

    def _route_after_dispatch(self, state: RuntimeGraphState) -> str:
        if self._is_terminal(state["plan_state"].status):
            return "end"
        if state.get("current_task_id") is None:
            return "end"
        return "worker"

    def _handle_prompt_queue(self, state: RuntimeGraphState) -> bool:
        if self.prompt_queue is None or len(self.prompt_queue) == 0:
            return False

        prompt_results = self.prompt_handler.handle_queue(
            self.prompt_queue,
            execution_plan=state["execution_plan"],
            plan_state=state["plan_state"],
            results=state.get("results", {}),
            memory_context=self._memory_context(state),
            current_task_id=state.get("current_task_id"),
        )
        state.setdefault("prompt_results", []).extend(prompt_results)
        for prompt_result in prompt_results:
            self.memory_recorder.record_prompt_result(
                prompt_result,
                plan_id=state["execution_plan"].id,
            )

        commands = [
            command
            for result in prompt_results
            for command in result.commands
        ]
        if commands:
            self._record_runtime_commands(state, commands)
        logger.info(
            "prompt queue handled",
            extra={
                "prompt_count": len(prompt_results),
                "command_count": len(commands),
                "command_types": [command.type for command in commands],
            },
        )
        return self._has_applied_halt(state)

    def _record_runtime_commands(
        self,
        state: RuntimeGraphState,
        commands: list[RuntimeCommand],
    ) -> list[RuntimeCommandResult]:
        if not commands:
            return []
        state.setdefault("runtime_commands", []).extend(commands)
        for command in commands:
            self.memory_recorder.record_runtime_command(
                command,
                plan_id=state["execution_plan"].id,
            )
        results = self.command_executor.execute_all(
            commands,
            plan_state=state["plan_state"],
            execution_plan=state["execution_plan"],
        )
        state.setdefault("command_results", []).extend(results)
        for result in results:
            self.memory_recorder.record_command_result(
                result,
                plan_id=state["execution_plan"].id,
            )
        self._maybe_replan_from_command_results(state, results)
        logger.info(
            "runtime commands executed",
            extra={
                "command_count": len(commands),
                "applied_count": sum(
                    result.status == RuntimeCommandStatus.APPLIED for result in results
                ),
                "ignored_count": sum(
                    result.status == RuntimeCommandStatus.IGNORED for result in results
                ),
                "failed_count": sum(
                    result.status == RuntimeCommandStatus.FAILED for result in results
                ),
            },
        )
        return results

    def _record_retry_exhaustion_command(
        self,
        state: RuntimeGraphState,
        error: TaskAttemptRunError,
    ) -> None:
        if not error.attempts or error.attempts[-1].status != TaskAttemptStatus.TIMED_OUT:
            return

        command = RuntimeCommand(
            type=RuntimeCommandType.REQUEST_REPLAN,
            task_id=error.task_id,
            reason="Task timed out after retry exhaustion.",
            payload={
                "attempt_ids": [attempt.id for attempt in error.attempts],
                "last_attempt_status": error.attempts[-1].status,
            },
            source="task_attempt_runner",
        )
        result = RuntimeCommandResult(
            command=command,
            status=RuntimeCommandStatus.IGNORED,
            reason=(
                "Recorded timeout exhaustion for audit; worker failure aborts this invocation "
                "before boundary-time replanning can continue."
            ),
            affected_task_ids=[error.task_id],
        )
        state.setdefault("runtime_commands", []).append(command)
        state.setdefault("command_results", []).append(result)
        self.memory_recorder.record_runtime_command(
            command,
            plan_id=state["execution_plan"].id,
        )
        self.memory_recorder.record_command_result(
            result,
            plan_id=state["execution_plan"].id,
        )

    def _maybe_replan_from_command_results(
        self,
        state: RuntimeGraphState,
        results: list[RuntimeCommandResult],
    ) -> None:
        if self.runtime_replanner is None:
            return

        for result in results:
            if result.command.type != RuntimeCommandType.REQUEST_REPLAN:
                continue
            previous_plan = state["execution_plan"]
            previous_result_ids = set(state.get("results", {}))
            self.memory_recorder.record_plan_snapshot(
                execution_plan=previous_plan,
                plan_state=state["plan_state"],
                results=state.get("results", {}),
                source="runtime_replanner",
                reason="before_replan",
            )
            replacement_plan, replan_result = self.runtime_replanner.replan(
                trigger=result,
                execution_plan=previous_plan,
                plan_state=state["plan_state"],
                results=state.setdefault("results", {}),
                memory_context=self._memory_context(state),
            )
            state["execution_plan"] = replacement_plan
            state.setdefault("replan_results", []).append(replan_result)
            self.memory_recorder.record_replan_result(replan_result)
            current_result_ids = set(state.get("results", {}))
            if replan_result.new_execution_plan_id is not None:
                self.memory_recorder.record_plan_transition(
                    previous_plan_id=replan_result.previous_execution_plan_id,
                    new_plan_id=replan_result.new_execution_plan_id,
                    trigger=result,
                    preserved_task_ids=sorted(previous_result_ids & current_result_ids),
                    dropped_task_ids=sorted(previous_result_ids - current_result_ids),
                )
            logger.info(
                "runtime replan handled",
                extra={
                    "status": replan_result.status,
                    "previous_execution_plan_id": replan_result.previous_execution_plan_id,
                    "new_execution_plan_id": replan_result.new_execution_plan_id,
                },
            )

    def _has_applied_halt(self, state: RuntimeGraphState) -> bool:
        return any(
            result.command.type == RuntimeCommandType.HALT
            and result.status == RuntimeCommandStatus.APPLIED
            for result in state.get("command_results", [])
        )

    def _current_task(self, state: RuntimeGraphState) -> TaskCard:
        task_id = state.get("current_task_id")
        if task_id is None:
            msg = "runtime graph has no current task"
            logger.error(msg)
            raise RuntimeError(msg)
        tracker = PlanTracker(state["plan_state"], state["execution_plan"])
        if tracker.dispatcher is None:
            msg = "runtime graph has no dispatcher"
            logger.error(msg)
            raise RuntimeError(msg)
        return tracker.dispatcher.get_task(task_id)

    def _worker_input(
        self,
        task: TaskCard,
        state: RuntimeGraphState,
    ) -> TaskCard | TaskExecutionContext:
        if self.context_assembler is None:
            return task
        context = self.context_assembler.assemble(
            task=task,
            execution_plan=state["execution_plan"],
            plan_state=state["plan_state"],
            results=state.get("results", {}),
        )
        state["current_context"] = context
        self.memory_recorder.record_working_context(
            context,
            plan_id=state["execution_plan"].id,
        )
        logger.info(
            "worker context assembled",
            extra={
                "task_id": task.id,
                "dependency_count": len(context.dependency_results),
                "artifact_count": len(context.artifacts),
                "loaded_skill_count": len(context.loaded_skill_ids),
            },
        )
        return context

    def _invoke_task_agent(
        self,
        task: TaskCard,
        worker_input: TaskCard | TaskExecutionContext,
        state: RuntimeGraphState,
    ) -> TaskRunResult | dict[str, Any]:
        if task.handoff_chain:
            return HandoffRunner(
                agent_registry=self.agent_registry,
                default_worker=self.worker,
                memory_recorder=self.memory_recorder,
            ).invoke(
                task=task,
                parent_input=worker_input,
                plan_id=state["execution_plan"].id,
            )
        runnable = self._resolve_task_runnable(task)
        return runnable.invoke(worker_input)

    async def _ainvoke_task_agent(
        self,
        task: TaskCard,
        worker_input: TaskCard | TaskExecutionContext,
        state: RuntimeGraphState,
    ) -> TaskRunResult | dict[str, Any]:
        if task.handoff_chain:
            return self._invoke_task_agent(task, worker_input, state)
        runnable = self._resolve_task_runnable(task)
        return await runnable.ainvoke(worker_input)

    def _resolve_task_runnable(self, task: TaskCard) -> Runnable[Any, Any]:
        if self.agent_registry is None:
            return self.worker
        return self.agent_registry.resolve(task.assigned_to) or self.worker

    def _is_terminal(self, status: PlanStatus | str) -> bool:
        return PlanStatus(status) in {PlanStatus.COMPLETED, PlanStatus.FAILED, PlanStatus.PAUSED}

    def _sync_memory_state(self, state: RuntimeGraphState) -> None:
        state["memory_records"] = self.memory_store.records()

    def _memory_context(
        self,
        state: RuntimeGraphState,
        *,
        task_id: str | None = None,
        dependency_task_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        return build_memory_context(
            self.memory_store,
            plan_id=state["execution_plan"].id,
            task_id=task_id or state.get("current_task_id"),
            dependency_task_ids=dependency_task_ids,
        )

    def _coerce_result(self, task_id: str, value: TaskRunResult | dict[str, Any]) -> TaskRunResult:
        if isinstance(value, TaskRunResult):
            return value
        data = dict(value)
        data.setdefault("task_id", task_id)
        return TaskRunResult(**data)

    def _coerce_verdict(self, value: JudgeVerdict | dict[str, Any]) -> JudgeVerdict:
        if isinstance(value, JudgeVerdict):
            return value
        return JudgeVerdict(**value)

    def _coerce_gate_judgment(self, value: GateJudgment | dict[str, Any]) -> GateJudgment:
        if isinstance(value, GateJudgment):
            return value
        return GateJudgment(**value)

    def _evaluate_checkpoint_gates(
        self,
        state: RuntimeGraphState,
        tracker: PlanTracker,
    ) -> None:
        if self.checkpoint_judge is None:
            return

        discovery_plan = tracker.state.discovery_plan
        if discovery_plan is None:
            return

        gate_index = {gate.id: gate for gate in discovery_plan.gates}
        for milestone in discovery_plan.milestones:
            if not milestone.gates or not self._milestone_completed(tracker, milestone):
                continue

            for gate_id in milestone.gates:
                if gate_id in tracker.state.checkpoint_ids:
                    continue
                gate = gate_index.get(gate_id)
                if gate is None:
                    logger.warning(
                        "milestone references unknown gate",
                        extra={"milestone_id": milestone.id, "gate_id": gate_id},
                    )
                    continue
                self._evaluate_checkpoint_gate(state, tracker, gate, milestone)

    def _evaluate_checkpoint_gate(
        self,
        state: RuntimeGraphState,
        tracker: PlanTracker,
        gate: Gate,
        milestone: Milestone,
    ) -> None:
        if self.checkpoint_judge is None:
            return

        result = self.checkpoint_judge.invoke(
            {
                "gate": gate,
                "milestone": milestone,
                "plan_state": tracker.state,
                "execution_plan": state["execution_plan"],
                "results": state.get("results", {}),
            }
        )
        judgment = self._coerce_gate_judgment(result)
        state.setdefault("gate_judgments", []).append(judgment)
        self.memory_recorder.record_gate_judgment(
            judgment,
            plan_id=state["execution_plan"].id,
        )
        commands = tracker.apply_gate_judgment(judgment)
        if commands:
            self._record_runtime_commands(state, commands)
            logger.info(
                "checkpoint gate commands recorded",
                extra={
                    "gate_id": gate.id,
                    "command_count": len(commands),
                    "command_types": [command.type for command in commands],
                },
            )
        tracker.state.checkpoint_ids.append(gate.id)
        logger.info(
            "checkpoint gate evaluated",
            extra={
                "gate_id": gate.id,
                "milestone_id": milestone.id,
                "decision": judgment.decision,
            },
        )

    def _milestone_completed(self, tracker: PlanTracker, milestone: Milestone) -> bool:
        task_ids = [task.id for task in milestone.tasks]
        if not task_ids:
            return False
        return all(
            tracker.state.task_statuses.get(task_id) == "completed" for task_id in task_ids
        )

    def _publish_signal(
        self,
        state: RuntimeGraphState,
        *,
        task_id: str,
        signal_type: ProgressSignalType,
        payload: ProgressSignalPayload,
    ) -> None:
        signal = ProgressSignal(task_id=task_id, signal_type=signal_type, payload=payload)
        self.memory_recorder.record_progress_signal(
            signal,
            plan_id=state["execution_plan"].id,
        )
        judgments = self.progress_bus.publish(signal)
        logger.info(
            "progress signal published",
            extra={
                "task_id": task_id,
                "signal_type": signal_type,
                "judgment_count": len(judgments),
            },
        )

        for judgment in judgments:
            tracker = PlanTracker(state["plan_state"], state["execution_plan"])
            if isinstance(judgment, ProcessJudgment):
                state.setdefault("process_judgments", []).append(judgment)
                self.memory_recorder.record_process_judgment(
                    judgment,
                    plan_id=state["execution_plan"].id,
                )
                logger.info(
                    "process judgment recorded",
                    extra={"task_id": judgment.task_id, "assessment": judgment.assessment},
                )
                commands = tracker.apply_process_judgment(judgment)
            elif isinstance(judgment, ObserverJudgment):
                state.setdefault("observer_judgments", []).append(judgment)
                self.memory_recorder.record_observer_judgment(
                    judgment,
                    plan_id=state["execution_plan"].id,
                )
                logger.info(
                    "observer judgment recorded",
                    extra={"health": judgment.health},
                )
                commands = tracker.apply_observer_judgment(
                    judgment,
                    task_id=state.get("current_task_id"),
                )
            else:
                commands = []

            if commands:
                self._record_runtime_commands(state, commands)
                logger.info(
                    "runtime commands recorded",
                    extra={
                        "task_id": task_id,
                        "command_count": len(commands),
                        "command_types": [command.type for command in commands],
                    },
                )
