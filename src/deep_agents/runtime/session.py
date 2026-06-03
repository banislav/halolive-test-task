from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from deep_agents.instrumentation import get_logger
from deep_agents.models import (
    ExecutionPlan,
    InterruptPriority,
    PlanState,
    PlanStatus,
    PromptCategory,
    PromptQueueItem,
    RuntimeCommand,
    RuntimeCommandResult,
    RuntimeCommandStatus,
    RuntimeCommandType,
    RuntimeMessage,
    RuntimeMessageType,
    RuntimeSessionSnapshot,
    RuntimeSessionStatus,
)
from deep_agents.models.base import JsonObject
from deep_agents.models.signals import ProgressSignal
from deep_agents.runtime.engine import RuntimeEngine, RuntimeGraphState
from deep_agents.runtime.observability.progress_bus import (
    ObserverSignalJudge,
    ProcessSignalJudge,
    ProgressSignalBus,
)
from deep_agents.runtime.prompt_queue import PromptQueue

logger = get_logger(__name__)


class RuntimeMessageBus:
    """Async message stream with thread-safe publishing for runtime sessions."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[RuntimeMessage | None] = asyncio.Queue()
        self._messages: list[RuntimeMessage] = []
        self._closed = False

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind the bus to the event loop that will consume streamed events."""
        if self._loop is None:
            self._loop = loop

    def publish(self, message: RuntimeMessage) -> None:
        """Publish a message from the event loop or a worker thread."""
        self._messages.append(message)
        if self._closed:
            return
        if self._loop is not None and self._called_from_other_thread():
            self._loop.call_soon_threadsafe(self._queue.put_nowait, message)
            return
        self._queue.put_nowait(message)

    async def publish_async(self, message: RuntimeMessage) -> None:
        """Async-compatible publish wrapper."""
        self.publish(message)

    async def events(self) -> AsyncIterator[RuntimeMessage]:
        """Yield messages until the bus is closed."""
        while True:
            message = await self._queue.get()
            if message is None:
                break
            yield message

    def messages(self) -> list[RuntimeMessage]:
        """Return messages published so far."""
        return list(self._messages)

    def close(self) -> None:
        """Close the stream after all already queued messages are delivered."""
        if self._closed:
            return
        self._closed = True
        if self._loop is not None and self._called_from_other_thread():
            self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
            return
        self._queue.put_nowait(None)

    def _called_from_other_thread(self) -> bool:
        try:
            return asyncio.get_running_loop() is not self._loop
        except RuntimeError:
            return True


class SessionProgressSignalBus:
    """Progress bus wrapper that mirrors signals into a runtime message stream."""

    def __init__(
        self,
        *,
        delegate: ProgressSignalBus,
        message_bus: RuntimeMessageBus,
        session_id: str,
    ) -> None:
        self.delegate = delegate
        self.message_bus = message_bus
        self.session_id = session_id

    def subscribe_observer(self, judge: ObserverSignalJudge) -> None:
        """Register an observer judge on the underlying progress bus."""
        self.delegate.subscribe_observer(judge)

    def subscribe_process(self, judge: ProcessSignalJudge, task_id: str | None = None) -> None:
        """Register a process judge on the underlying progress bus."""
        self.delegate.subscribe_process(judge, task_id=task_id)

    def publish(self, signal: ProgressSignal) -> list[Any]:
        """Publish a progress signal and mirror it to the async message stream."""
        judgments = self.delegate.publish(signal)
        self.message_bus.publish(
            RuntimeMessage(
                from_agent="progress_signal_bus",
                to_agent="runtime_session",
                type=RuntimeMessageType.SIGNAL,
                payload={
                    "signal": signal.model_dump(mode="json"),
                    "judgment_count": len(judgments),
                },
                correlation_id=signal.task_id or self.session_id,
            )
        )
        return judgments

    def signals(self, task_id: str | None = None) -> list[ProgressSignal]:
        """Return stored signals from the underlying bus."""
        return self.delegate.signals(task_id=task_id)


class AsyncRuntimeSession:
    """Interactive async session around the synchronous runtime execution primitives."""

    def __init__(
        self,
        engine: RuntimeEngine,
        *,
        session_id: str | None = None,
        message_bus: RuntimeMessageBus | None = None,
    ) -> None:
        self.engine = engine
        self.session_id = session_id or f"session-{uuid4().hex}"
        self.message_bus = message_bus or RuntimeMessageBus()
        self.status = RuntimeSessionStatus.CREATED
        self.state: RuntimeGraphState | None = None
        self._task: asyncio.Task[RuntimeGraphState] | None = None
        self._emitted_prompt_results = 0
        self._emitted_command_results = 0
        self._emitted_replan_results = 0
        self._original_progress_bus = engine.progress_bus
        self.engine.progress_bus = SessionProgressSignalBus(
            delegate=engine.progress_bus,
            message_bus=self.message_bus,
            session_id=self.session_id,
        )
        if self.engine.prompt_queue is None:
            self.engine.prompt_queue = PromptQueue()

    def start(self, execution_plan: ExecutionPlan, plan_state: PlanState) -> None:
        """Start executing a plan in the background."""
        if self._task is not None:
            msg = "runtime session has already been started"
            raise RuntimeError(msg)
        loop = asyncio.get_running_loop()
        self.message_bus.bind_loop(loop)
        self.state = self.engine.initial_state(execution_plan, plan_state)
        self.status = RuntimeSessionStatus.RUNNING
        self._publish(
            RuntimeMessageType.PROGRESS,
            from_agent="runtime_session",
            to_agent="user",
            payload={"status": "session_started", "execution_plan_id": execution_plan.id},
            correlation_id=self.session_id,
        )
        self._task = loop.create_task(self._run_loop())

    async def wait(self) -> RuntimeGraphState:
        """Wait for the background run to finish and return the final graph state."""
        if self._task is None:
            msg = "runtime session has not been started"
            raise RuntimeError(msg)
        return await self._task

    def stop(self, reason: str | None = None) -> None:
        """Request a cooperative session halt."""
        if self.state is None:
            self.status = RuntimeSessionStatus.STOPPED
            self.engine.progress_bus = self._original_progress_bus
            self.message_bus.close()
            return
        self.submit_command(
            RuntimeCommand(
                type=RuntimeCommandType.HALT,
                reason=reason or "Runtime session stop requested.",
                payload={"session_id": self.session_id},
                source="runtime_session",
            )
        )

    def submit_prompt(
        self,
        content: str,
        *,
        priority: InterruptPriority = InterruptPriority.P3_FEEDBACK,
        category: PromptCategory | None = None,
        metadata: JsonObject | None = None,
    ) -> PromptQueueItem:
        """Queue a user prompt and immediately apply P0/P1 cooperative interrupts."""
        prompt = PromptQueueItem(
            id=f"prompt-{uuid4().hex}",
            content=content,
            priority=priority,
            category=category,
            metadata=metadata or {},
        )
        if self.engine.prompt_queue is None:
            self.engine.prompt_queue = PromptQueue()
        self.engine.prompt_queue.push(prompt)
        self._publish(
            RuntimeMessageType.PROMPT,
            from_agent="user",
            to_agent="prompt_queue",
            payload={"prompt": prompt.model_dump(mode="json")},
            correlation_id=prompt.id,
        )

        if priority == InterruptPriority.P0_HALT:
            self.submit_command(
                RuntimeCommand(
                    type=RuntimeCommandType.HALT,
                    task_id=self._current_task_id(),
                    reason="P0 user prompt requested halt.",
                    payload={"prompt_id": prompt.id, "content": content},
                    source="prompt_queue",
                )
            )
        elif priority == InterruptPriority.P1_PAUSE:
            self.submit_command(
                RuntimeCommand(
                    type=RuntimeCommandType.PAUSE_TASK,
                    task_id=self._current_task_id(),
                    reason="P1 user prompt requested pause.",
                    payload={"prompt_id": prompt.id, "content": content},
                    source="prompt_queue",
                )
            )
        return prompt

    def submit_command(self, command: RuntimeCommand) -> RuntimeCommandResult:
        """Execute one runtime command against the live session state."""
        if self.state is None:
            msg = "cannot submit a runtime command before session state exists"
            raise RuntimeError(msg)
        results = self.engine._record_runtime_commands(self.state, [command])
        result = results[0]
        self._sync_status_from_command_result(result)
        self._emit_new_runtime_artifacts()
        return result

    async def events(self) -> AsyncIterator[RuntimeMessage]:
        """Stream runtime messages until the session finishes or the bus closes."""
        async for message in self.message_bus.events():
            yield message

    def snapshot(self) -> RuntimeSessionSnapshot:
        """Return a point-in-time view of the live session."""
        state = self.state
        prompt_queue = self.engine.prompt_queue
        if state is None:
            return RuntimeSessionSnapshot(
                session_id=self.session_id,
                status=self.status,
                pending_prompt_ids=[item.id for item in prompt_queue.items()]
                if prompt_queue is not None
                else [],
            )
        return RuntimeSessionSnapshot(
            session_id=self.session_id,
            status=self.status,
            execution_plan_id=state["execution_plan"].id,
            current_task_id=state.get("current_task_id"),
            plan_state=state["plan_state"].model_dump(mode="json"),
            results={
                task_id: result.model_dump(mode="json")
                for task_id, result in state.get("results", {}).items()
            },
            runtime_commands=[
                command.model_dump(mode="json")
                for command in state.get("runtime_commands", [])
            ],
            command_results=[
                result.model_dump(mode="json")
                for result in state.get("command_results", [])
            ],
            prompt_results=[
                result.model_dump(mode="json")
                for result in state.get("prompt_results", [])
            ],
            pending_prompt_ids=[item.id for item in prompt_queue.items()]
            if prompt_queue is not None
            else [],
            memory_record_count=len(state.get("memory_records", [])),
        )

    async def _run_loop(self) -> RuntimeGraphState:
        if self.state is None:
            msg = "runtime session has no initialized state"
            raise RuntimeError(msg)

        try:
            while True:
                self.engine._dispatch_node(self.state)
                self.engine._sync_memory_state(self.state)
                self._emit_new_runtime_artifacts()
                if self.engine._route_after_dispatch(self.state) == "end":
                    break

                task_id = self.state["current_task_id"]
                self._publish(
                    RuntimeMessageType.REQUEST,
                    from_agent="dispatcher",
                    to_agent="worker",
                    payload={"task_id": task_id},
                    correlation_id=task_id or self.session_id,
                )

                await asyncio.to_thread(self.engine._worker_node, self.state)
                self._emit_new_runtime_artifacts()
                result = self.state.get("latest_result")
                if result is not None:
                    self._publish(
                        RuntimeMessageType.RESULT,
                        from_agent="worker",
                        to_agent="plan_tracker",
                        payload={"result": result.model_dump(mode="json")},
                        correlation_id=result.task_id,
                    )

                await asyncio.to_thread(self.engine._judge_node, self.state)
                self._emit_new_runtime_artifacts()
                verdict = self.state.get("latest_verdict")
                if verdict is not None:
                    self._publish(
                        RuntimeMessageType.VERDICT,
                        from_agent="judge",
                        to_agent="plan_tracker",
                        payload={"verdict": verdict.model_dump(mode="json")},
                        correlation_id=verdict.task_id,
                    )

                await asyncio.to_thread(self.engine._apply_verdict_node, self.state)
                self.engine._sync_memory_state(self.state)
                self._emit_new_runtime_artifacts()

            self.engine._sync_memory_state(self.state)
            self._sync_terminal_status()
            self._publish(
                RuntimeMessageType.PROGRESS,
                from_agent="runtime_session",
                to_agent="user",
                payload={
                    "status": "session_finished",
                    "session_status": self.status,
                    "plan_status": self.state["plan_state"].status,
                },
                correlation_id=self.session_id,
            )
            return self.state
        except Exception as exc:
            self.status = RuntimeSessionStatus.FAILED
            self._publish(
                RuntimeMessageType.ERROR,
                from_agent="runtime_session",
                to_agent="user",
                payload={"error_type": exc.__class__.__name__, "detail": str(exc)},
                correlation_id=self.session_id,
            )
            logger.exception("async runtime session failed", extra={"session_id": self.session_id})
            raise
        finally:
            self.engine.progress_bus = self._original_progress_bus
            self.message_bus.close()

    def _publish(
        self,
        message_type: RuntimeMessageType,
        *,
        from_agent: str,
        to_agent: str,
        payload: JsonObject,
        correlation_id: str,
    ) -> None:
        self.message_bus.publish(
            RuntimeMessage(
                from_agent=from_agent,
                to_agent=to_agent,
                type=message_type,
                payload=payload,
                correlation_id=correlation_id,
            )
        )

    def _current_task_id(self) -> str | None:
        if self.state is None:
            return None
        return self.state.get("current_task_id")

    def _emit_new_runtime_artifacts(self) -> None:
        if self.state is None:
            return

        prompt_results = self.state.get("prompt_results", [])
        for prompt_result in prompt_results[self._emitted_prompt_results:]:
            destination = (
                "content_reasoning_agent"
                if prompt_result.response is not None
                else "plan_tracker"
            )
            self._publish(
                RuntimeMessageType.PROMPT,
                from_agent="prompt_queue",
                to_agent=destination,
                payload={"prompt_result": prompt_result.model_dump(mode="json")},
                correlation_id=prompt_result.prompt.id,
            )
        self._emitted_prompt_results = len(prompt_results)

        command_results = self.state.get("command_results", [])
        for command_result in command_results[self._emitted_command_results:]:
            self._publish(
                RuntimeMessageType.COMMAND,
                from_agent=command_result.command.source,
                to_agent="plan_tracker",
                payload={"command_result": command_result.model_dump(mode="json")},
                correlation_id=command_result.command.task_id or self.session_id,
            )
            self._sync_status_from_command_result(command_result)
        self._emitted_command_results = len(command_results)

        replan_results = self.state.get("replan_results", [])
        for replan_result in replan_results[self._emitted_replan_results:]:
            self._publish(
                RuntimeMessageType.COMMAND,
                from_agent="runtime_replanner",
                to_agent="plan_tracker",
                payload={"replan_result": replan_result.model_dump(mode="json")},
                correlation_id=replan_result.new_execution_plan_id
                or replan_result.previous_execution_plan_id,
            )
        self._emitted_replan_results = len(replan_results)

    def _sync_status_from_command_result(self, result: RuntimeCommandResult) -> None:
        if result.status != RuntimeCommandStatus.APPLIED:
            return
        if result.command.type == RuntimeCommandType.HALT:
            self.status = RuntimeSessionStatus.STOPPED
        elif result.command.type == RuntimeCommandType.PAUSE_TASK:
            self.status = RuntimeSessionStatus.PAUSED

    def _sync_terminal_status(self) -> None:
        if self.state is None:
            return
        plan_status = PlanStatus(self.state["plan_state"].status)
        if plan_status == PlanStatus.COMPLETED:
            self.status = RuntimeSessionStatus.COMPLETED
        elif plan_status == PlanStatus.FAILED:
            self.status = RuntimeSessionStatus.FAILED
        elif plan_status == PlanStatus.PAUSED and self.status != RuntimeSessionStatus.STOPPED:
            self.status = RuntimeSessionStatus.PAUSED
