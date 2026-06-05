from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.runnables import Runnable, RunnableLambda

from deep_agents.config import DeepAgentsSettings, load_env
from deep_agents.instrumentation import configure_logging
from deep_agents.langchain import (
    build_discovery_plan_builder,
    build_execution_planner,
    build_task_completion_judge,
    build_task_worker,
)
from deep_agents.models import (
    DiscoveryPlan,
    ExecutionPlan,
    ExecutionPlannerInput,
    InterruptPriority,
    Objective,
    PlannerInput,
    PlanState,
    PromptCategory,
    RuntimeMessage,
    SkillDefinition,
    TaskCard,
)
from deep_agents.runtime import (
    AsyncRuntimeSession,
    ContextAssembler,
    RuntimeEngine,
    TaskExecutionContext,
    TaskRunResult,
)
from deep_agents.skills import SkillLoader, SkillRegistry

RAW_PROMPT = (
    "Create a concise engineering summary of how an async deep-agent runtime accepts "
    "a raw objective, plans work, executes tasks in the background, streams progress, "
    "and handles user prompts during execution."
)

DEFAULT_TOOLS = ["progress_signal_bus", "prompt_queue", "runtime_command_executor"]
DEFAULT_SKILLS = ["technical_writing"]
DEFAULT_CONSTRAINTS = [
    "Keep the output concise and implementation-focused.",
    "Mention discovery planning, execution planning, and async runtime execution.",
]
DEFAULT_CONTEXT = {
    "audience": "engineers evaluating the deep-agent runtime",
    "style": "plain engineering prose",
}


def build_prompt_input(prompt: str = RAW_PROMPT) -> PlannerInput:
    return PlannerInput(
        objective=prompt,
        constraints=DEFAULT_CONSTRAINTS,
        available_tools=DEFAULT_TOOLS,
        available_skills=DEFAULT_SKILLS,
        context=DEFAULT_CONTEXT,
    )


def build_skill_loader() -> SkillLoader:
    return SkillLoader(
        SkillRegistry(
            [
                SkillDefinition(
                    id="technical_writing",
                    name="Technical Writing",
                    prompt=(
                        "Write concise, concrete engineering prose. Prefer short paragraphs, "
                        "plain language, and explicit feature behavior."
                    ),
                )
            ]
        )
    )


def delayed_worker(
    worker: Runnable[TaskCard | TaskExecutionContext, TaskRunResult],
    *,
    delay_seconds: float = 1.5,
) -> Runnable[TaskCard | TaskExecutionContext, TaskRunResult]:
    def invoke_with_delay(task_input: TaskCard | TaskExecutionContext) -> TaskRunResult:
        time.sleep(delay_seconds)
        return worker.invoke(task_input)

    return RunnableLambda(invoke_with_delay)


async def stream_events(session: AsyncRuntimeSession) -> None:
    async for message in session.events():
        print(_message_summary(message))


async def submit_progress_question(session: AsyncRuntimeSession) -> None:
    await asyncio.sleep(0.5)
    prompt = session.submit_prompt(
        "What progress is available so far?",
        priority=InterruptPriority.P3_FEEDBACK,
        category=PromptCategory.CONTENT_REASONING,
    )
    print(f"[prompt queued] {prompt.id}: {prompt.content}")


async def run_session() -> None:
    load_env()
    settings = DeepAgentsSettings(
        provider="openrouter",
        model="qwen/qwen3.6-flash",
    )
    if not settings.openrouter_api_key:
        print(
            "Missing OPENROUTER_API_KEY or DEEP_AGENTS_OPENROUTER_API_KEY. "
            "Add it to .env before running this example."
        )
        return

    configure_logging()
    skill_loader = build_skill_loader()

    discovery_builder = build_discovery_plan_builder(
        settings=settings,
        constraints=DEFAULT_CONSTRAINTS,
        available_tools=DEFAULT_TOOLS,
        available_skills=DEFAULT_SKILLS,
        context=DEFAULT_CONTEXT,
    )
    discovery_plan = discovery_builder.invoke(RAW_PROMPT)

    execution_planner = build_execution_planner(settings=settings)
    execution_plan = execution_planner.invoke(
        ExecutionPlannerInput(
            discovery_plan=discovery_plan,
            available_tools=DEFAULT_TOOLS,
            available_skills=DEFAULT_SKILLS,
            context=DEFAULT_CONTEXT,
        )
    )

    final_state = await run_async_execution(
        discovery_plan=discovery_plan,
        execution_plan=execution_plan,
        settings=settings,
        skill_loader=skill_loader,
    )

    print("\nDiscovery plan:")
    print(discovery_plan.model_dump_json(indent=2))
    print("\nExecution plan:")
    print(execution_plan.model_dump_json(indent=2))
    print("\nPrompt responses:")
    for result in final_state["prompt_results"]:
        if result.response is not None:
            print(result.response.model_dump_json(indent=2))
    print("\nTask outputs:")
    for result in final_state["results"].values():
        print(result.model_dump_json(indent=2))


async def run_async_execution(
    *,
    discovery_plan: DiscoveryPlan,
    execution_plan: ExecutionPlan,
    settings: DeepAgentsSettings,
    skill_loader: SkillLoader,
) -> dict[str, Any]:
    worker = build_task_worker(settings=settings, skill_loader=skill_loader)
    engine = RuntimeEngine(
        worker=delayed_worker(worker),
        judge=build_task_completion_judge(settings=settings),
        context_assembler=ContextAssembler(),
    )
    session = AsyncRuntimeSession(engine, session_id="example-prompt-to-runtime")
    session.start(
        execution_plan,
        PlanState(
            objective=Objective(raw=discovery_plan.objective.raw),
            discovery_plan=discovery_plan,
            execution_plan_id=execution_plan.id,
        ),
    )
    event_task = asyncio.create_task(stream_events(session))
    prompt_task = asyncio.create_task(submit_progress_question(session))

    final_state = await session.wait()
    await prompt_task
    await event_task

    print("\nFinal session snapshot:")
    print(session.snapshot().model_dump_json(indent=2))
    return final_state


def _message_summary(message: RuntimeMessage) -> str:
    detail = _payload_detail(message.payload)
    return (
        f"[event] {message.type} {message.from_agent}->{message.to_agent} "
        f"correlation={message.correlation_id}{detail}"
    )


def _payload_detail(payload: dict[str, Any]) -> str:
    if "status" in payload:
        return f" status={payload['status']}"
    if "signal" in payload:
        signal = payload["signal"]
        if isinstance(signal, dict):
            signal_type = signal.get("signal_type")
            signal_payload = signal.get("payload", {})
            status = signal_payload.get("status") if isinstance(signal_payload, dict) else None
            return f" signal={signal_type} status={status}"
    if "prompt_result" in payload:
        prompt = payload["prompt_result"].get("prompt", {})
        return f" prompt={prompt.get('id')}"
    if "command_result" in payload:
        command = payload["command_result"].get("command", {})
        return f" command={command.get('type')}"
    if "result" in payload:
        result = payload["result"]
        return f" task={result.get('task_id')}" if isinstance(result, dict) else ""
    if "verdict" in payload:
        verdict = payload["verdict"]
        return f" task={verdict.get('task_id')}" if isinstance(verdict, dict) else ""
    return ""


def main() -> None:
    asyncio.run(run_session())


if __name__ == "__main__":
    main()
