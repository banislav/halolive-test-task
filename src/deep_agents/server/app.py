from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from deep_agents.models import RuntimeCommand
from deep_agents.server.service import (
    PlanFromPromptRequest,
    PlanFromPromptResponse,
    PromptSubmitRequest,
    RuntimeBridgeService,
    SessionCreateRequest,
    SessionCreateResponse,
    SessionPromptResponse,
)


def create_app(service: RuntimeBridgeService | None = None) -> FastAPI:
    """Create the local API bridge for the async runtime frontend."""
    runtime_service = service or RuntimeBridgeService.from_environment()
    app = FastAPI(title="Deep Agents Runtime Bridge")
    app.state.runtime_service = runtime_service

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/api/plans/from-prompt", response_model=PlanFromPromptResponse)
    async def plan_from_prompt(request: PlanFromPromptRequest) -> PlanFromPromptResponse:
        try:
            return await runtime_service.plan_from_prompt(request)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/sessions", response_model=SessionCreateResponse)
    async def create_session(request: SessionCreateRequest) -> SessionCreateResponse:
        try:
            return await runtime_service.create_session(request)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/sessions/{session_id}/snapshot")
    async def session_snapshot(session_id: str) -> dict:
        session = runtime_service.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="runtime session not found")
        return session.snapshot().model_dump(mode="json")

    @app.post("/api/sessions/{session_id}/prompts", response_model=SessionPromptResponse)
    async def submit_prompt(
        session_id: str,
        request: PromptSubmitRequest,
    ) -> SessionPromptResponse:
        try:
            return runtime_service.submit_prompt(session_id, request)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="runtime session not found") from exc

    @app.post("/api/sessions/{session_id}/commands")
    async def submit_command(session_id: str, command: RuntimeCommand) -> dict:
        try:
            return runtime_service.submit_command(session_id, command).model_dump(mode="json")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="runtime session not found") from exc

    @app.get("/api/sessions/{session_id}/events")
    async def session_events(session_id: str) -> StreamingResponse:
        if runtime_service.get_session(session_id) is None:
            raise HTTPException(status_code=404, detail="runtime session not found")
        return StreamingResponse(
            _sse(runtime_service.stream_events(session_id)),
            media_type="text/event-stream",
        )

    return app


async def _sse(events: AsyncIterator[dict]) -> AsyncIterator[str]:
    async for event in events:
        yield f"data: {json.dumps(event)}\n\n"
