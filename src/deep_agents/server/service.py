from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any

from langchain_core.runnables import Runnable
from pydantic import Field

from deep_agents.config import DeepAgentsSettings, load_env
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
    PromptQueueItem,
    RuntimeCommand,
    RuntimeCommandResult,
    RuntimeSessionSnapshot,
    SkillDefinition,
)
from deep_agents.models.base import DeepAgentsModel, JsonObject
from deep_agents.runtime import AsyncRuntimeSession, ContextAssembler, RuntimeEngine
from deep_agents.skills import SkillLoader, SkillRegistry

DEFAULT_TOOLS = ["progress_signal_bus", "prompt_queue", "runtime_command_executor"]
DEFAULT_SKILLS = ["technical_writing"]
DEFAULT_CONSTRAINTS = [
    "Keep the output concise and implementation-focused.",
    "Mention discovery planning, execution planning, and async runtime execution.",
]
DEFAULT_CONTEXT: JsonObject = {
    "audience": "engineers evaluating the deep-agent runtime",
    "style": "plain engineering prose",
}


class PlanFromPromptRequest(DeepAgentsModel):
    prompt: str
    constraints: list[str] = Field(default_factory=list)
    available_tools: list[str] = Field(default_factory=list)
    available_skills: list[str] = Field(default_factory=list)
    context: JsonObject = Field(default_factory=dict)


class PlanFromPromptResponse(DeepAgentsModel):
    discovery_plan: DiscoveryPlan
    execution_plan: ExecutionPlan


class SessionCreateRequest(DeepAgentsModel):
    discovery_plan: DiscoveryPlan
    execution_plan: ExecutionPlan
    session_id: str | None = None


class SessionCreateResponse(DeepAgentsModel):
    session_id: str
    snapshot: RuntimeSessionSnapshot


class PromptSubmitRequest(DeepAgentsModel):
    content: str
    priority: InterruptPriority = InterruptPriority.P3_FEEDBACK
    category: PromptCategory | None = None
    metadata: JsonObject = Field(default_factory=dict)


class SessionPromptResponse(DeepAgentsModel):
    prompt: PromptQueueItem
    snapshot: RuntimeSessionSnapshot


EngineFactory = Callable[[], RuntimeEngine]


class RuntimeBridgeService:
    """Application service that exposes planning and async sessions to HTTP."""

    def __init__(
        self,
        *,
        settings: DeepAgentsSettings | None = None,
        discovery_builder: Runnable[PlannerInput, DiscoveryPlan] | None = None,
        execution_planner: Runnable[ExecutionPlannerInput, ExecutionPlan] | None = None,
        engine_factory: EngineFactory | None = None,
    ) -> None:
        self.settings = settings
        self.discovery_builder = discovery_builder
        self.execution_planner = execution_planner
        self.engine_factory = engine_factory
        self.sessions: dict[str, AsyncRuntimeSession] = {}

    @classmethod
    def from_environment(cls) -> RuntimeBridgeService:
        load_env()
        settings = DeepAgentsSettings(provider="openrouter", model="qwen/qwen3.6-flash")
        return cls(settings=settings)

    async def plan_from_prompt(self, request: PlanFromPromptRequest) -> PlanFromPromptResponse:
        self._ensure_provider_ready()
        planner_input = PlannerInput(
            objective=request.prompt,
            constraints=request.constraints or DEFAULT_CONSTRAINTS,
            available_tools=request.available_tools or DEFAULT_TOOLS,
            available_skills=request.available_skills or DEFAULT_SKILLS,
            context=request.context or DEFAULT_CONTEXT,
        )
        discovery_plan = await asyncio.to_thread(
            self._discovery_builder().invoke,
            planner_input,
        )
        execution_input = ExecutionPlannerInput(
            discovery_plan=discovery_plan,
            available_tools=planner_input.available_tools,
            available_skills=planner_input.available_skills,
            context=planner_input.context,
        )
        execution_plan = await asyncio.to_thread(
            self._execution_planner().invoke,
            execution_input,
        )
        return PlanFromPromptResponse(
            discovery_plan=discovery_plan,
            execution_plan=execution_plan,
        )

    async def create_session(self, request: SessionCreateRequest) -> SessionCreateResponse:
        self._ensure_provider_ready()
        session = AsyncRuntimeSession(
            self._engine_factory()(),
            session_id=request.session_id,
        )
        plan_state = PlanState(
            objective=Objective(raw=request.discovery_plan.objective.raw),
            discovery_plan=request.discovery_plan,
            execution_plan_id=request.execution_plan.id,
        )
        session.start(request.execution_plan, plan_state)
        self.sessions[session.session_id] = session
        return SessionCreateResponse(
            session_id=session.session_id,
            snapshot=session.snapshot(),
        )

    def get_session(self, session_id: str) -> AsyncRuntimeSession | None:
        return self.sessions.get(session_id)

    def submit_prompt(
        self,
        session_id: str,
        request: PromptSubmitRequest,
    ) -> SessionPromptResponse:
        session = self.sessions[session_id]
        prompt = session.submit_prompt(
            request.content,
            priority=request.priority,
            category=request.category,
            metadata=request.metadata,
        )
        return SessionPromptResponse(prompt=prompt, snapshot=session.snapshot())

    def submit_command(self, session_id: str, command: RuntimeCommand) -> RuntimeCommandResult:
        return self.sessions[session_id].submit_command(command)

    async def stream_events(self, session_id: str) -> AsyncIterator[dict[str, Any]]:
        session = self.sessions[session_id]
        seen = {id(message) for message in session.message_bus.messages()}
        for message in session.message_bus.messages():
            yield message.model_dump(mode="json", by_alias=True)
        async for message in session.events():
            if id(message) in seen:
                continue
            yield message.model_dump(mode="json", by_alias=True)

    def _discovery_builder(self) -> Runnable[PlannerInput, DiscoveryPlan]:
        if self.discovery_builder is None:
            self.discovery_builder = build_discovery_plan_builder(settings=self.settings)
        return self.discovery_builder

    def _execution_planner(self) -> Runnable[ExecutionPlannerInput, ExecutionPlan]:
        if self.execution_planner is None:
            self.execution_planner = build_execution_planner(settings=self.settings)
        return self.execution_planner

    def _engine_factory(self) -> EngineFactory:
        if self.engine_factory is None:
            self.engine_factory = self._default_engine_factory
        return self.engine_factory

    def _default_engine_factory(self) -> RuntimeEngine:
        worker = build_task_worker(
            settings=self.settings,
            skill_loader=_technical_writing_loader(),
        )
        return RuntimeEngine(
            worker=worker,
            judge=build_task_completion_judge(settings=self.settings),
            context_assembler=ContextAssembler(),
        )

    def _ensure_provider_ready(self) -> None:
        if self.settings is None:
            return
        if self.settings.provider == "openrouter" and not self.settings.openrouter_api_key:
            msg = (
                "Missing OPENROUTER_API_KEY or DEEP_AGENTS_OPENROUTER_API_KEY. "
                "Add it to .env before using the runtime bridge."
            )
            raise RuntimeError(msg)


def _technical_writing_loader() -> SkillLoader:
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
