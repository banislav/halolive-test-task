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
    TaskCard,
)
from deep_agents.runtime.command_executor import RuntimeCommandExecutor
from deep_agents.runtime.context import ContextAssembler, TaskExecutionContext
from deep_agents.runtime.observability import ProgressSignalBus
from deep_agents.runtime.plan_tracker import PlanTracker
from deep_agents.runtime.prompt_handler import PromptHandler
from deep_agents.runtime.prompt_queue import PromptQueue
from deep_agents.runtime.replanner import RuntimeReplanner
from deep_agents.runtime.results import TaskRunResult

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
        progress_bus: ProgressSignalBus | None = None,
        recursion_limit: int = 100,
    ) -> None:
        """Create an engine from LangChain runnables for task work and judgment."""
        self.worker = worker
        self.judge = judge
        self.checkpoint_judge = checkpoint_judge
        self.context_assembler = context_assembler
        self.prompt_queue = prompt_queue
        self.prompt_handler = PromptHandler(
            prompt_classifier=prompt_classifier,
            content_reasoner=content_reasoner,
        )
        self.command_executor = command_executor or RuntimeCommandExecutor()
        self.runtime_replanner = runtime_replanner
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
        }
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
        }
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
            self._publish_signal(
                state,
                task_id=task_id,
                signal_type=ProgressSignalType.PROGRESS,
                payload=ProgressSignalPayload(
                    status="dispatched",
                    data={
                        "execution_plan_id": state["execution_plan"].id,
                        "wave": self._current_task(state).wave,
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
            result = self.worker.invoke(self._worker_input(task, state))
            state["latest_result"] = self._coerce_result(task.id, result)
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
            result = await self.worker.ainvoke(self._worker_input(task, state))
            state["latest_result"] = self._coerce_result(task.id, result)
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
            current_task_id=state.get("current_task_id"),
        )
        state.setdefault("prompt_results", []).extend(prompt_results)

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
        results = self.command_executor.execute_all(
            commands,
            plan_state=state["plan_state"],
            execution_plan=state["execution_plan"],
        )
        state.setdefault("command_results", []).extend(results)
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
            replacement_plan, replan_result = self.runtime_replanner.replan(
                trigger=result,
                execution_plan=state["execution_plan"],
                plan_state=state["plan_state"],
                results=state.setdefault("results", {}),
            )
            state["execution_plan"] = replacement_plan
            state.setdefault("replan_results", []).append(replan_result)
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

    def _is_terminal(self, status: PlanStatus | str) -> bool:
        return PlanStatus(status) in {PlanStatus.COMPLETED, PlanStatus.FAILED, PlanStatus.PAUSED}

    def _coerce_result(self, task_id: str, value: TaskRunResult | dict[str, Any]) -> TaskRunResult:
        if isinstance(value, TaskRunResult):
            return value
        return TaskRunResult(task_id=task_id, **value)

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
                logger.info(
                    "process judgment recorded",
                    extra={"task_id": judgment.task_id, "assessment": judgment.assessment},
                )
                commands = tracker.apply_process_judgment(judgment)
            elif isinstance(judgment, ObserverJudgment):
                state.setdefault("observer_judgments", []).append(judgment)
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
