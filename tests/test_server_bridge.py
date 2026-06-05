from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableLambda

from deep_agents.models import (
    AgentAssignment,
    AgentKind,
    DiscoveryPlan,
    ExecutionPlan,
    JudgeRecommendation,
    JudgeVerdict,
    Objective,
    PlannerInput,
    TaskCard,
    Wave,
)
from deep_agents.runtime import RuntimeEngine, TaskRunResult
from deep_agents.server import RuntimeBridgeService, create_app
from deep_agents.server.service import PlanFromPromptRequest, SessionCreateRequest


def _discovery_plan(objective: str = "Build a demo") -> DiscoveryPlan:
    return DiscoveryPlan(objective=Objective(raw=objective))


def _execution_plan(objective: str = "Build a demo") -> ExecutionPlan:
    return ExecutionPlan(
        id="EP-web",
        objective=objective,
        waves=[Wave(index=0, task_ids=["T1"])],
        task_cards=[
            TaskCard(
                id="T1",
                name="Write summary",
                wave=0,
                assigned_to=AgentAssignment(type=AgentKind.WORKER, name="Worker"),
            )
        ],
    )


def _engine() -> RuntimeEngine:
    worker = RunnableLambda(lambda task: TaskRunResult(task_id=task.id, output={"summary": "done"}))
    judge = RunnableLambda(
        lambda value: JudgeVerdict(
            task_id=value["task"].id,
            verdict="pass",
            overall_confidence=0.9,
            recommendation=JudgeRecommendation.ADVANCE,
        )
    )
    return RuntimeEngine(worker=worker, judge=judge)


def _service() -> RuntimeBridgeService:
    captured_inputs: list[PlannerInput] = []

    def build_discovery(value: PlannerInput) -> DiscoveryPlan:
        captured_inputs.append(value)
        return _discovery_plan(value.objective)

    service = RuntimeBridgeService(
        discovery_builder=RunnableLambda(build_discovery),
        execution_planner=RunnableLambda(
            lambda value: _execution_plan(value.discovery_plan.objective.raw)
        ),
        engine_factory=_engine,
    )
    service.captured_inputs = captured_inputs  # type: ignore[attr-defined]
    return service


def test_service_builds_plans_with_stubbed_planners() -> None:
    service = _service()

    response = _run(service.plan_from_prompt(PlanFromPromptRequest(prompt="Explain runtime")))

    assert response.discovery_plan.objective.raw == "Explain runtime"
    assert response.execution_plan.id == "EP-web"
    assert service.captured_inputs[0].available_tools
    assert service.captured_inputs[0].available_skills


def test_plan_endpoint_returns_discovery_and_execution_plans() -> None:
    client = TestClient(create_app(_service()))

    response = client.post("/api/plans/from-prompt", json={"prompt": "Explain runtime"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["discovery_plan"]["objective"]["raw"] == "Explain runtime"
    assert payload["execution_plan"]["id"] == "EP-web"


def test_session_endpoint_starts_runtime_and_snapshot_can_be_read() -> None:
    client = TestClient(create_app(_service()))
    plan_response = client.post("/api/plans/from-prompt", json={"prompt": "Explain runtime"})

    response = client.post(
        "/api/sessions",
        json={
            "discovery_plan": plan_response.json()["discovery_plan"],
            "execution_plan": plan_response.json()["execution_plan"],
            "session_id": "session-web",
        },
    )

    assert response.status_code == 200
    assert response.json()["session_id"] == "session-web"
    snapshot = client.get("/api/sessions/session-web/snapshot").json()
    assert snapshot["session_id"] == "session-web"


def test_prompt_endpoint_queues_prompt_for_active_session() -> None:
    client = TestClient(create_app(_service()))
    plan_response = client.post("/api/plans/from-prompt", json={"prompt": "Explain runtime"})
    client.post(
        "/api/sessions",
        json={
            "discovery_plan": plan_response.json()["discovery_plan"],
            "execution_plan": plan_response.json()["execution_plan"],
            "session_id": "session-prompts",
        },
    )

    response = client.post(
        "/api/sessions/session-prompts/prompts",
        json={"content": "What progress is available so far?"},
    )

    assert response.status_code == 200
    assert response.json()["prompt"]["content"] == "What progress is available so far?"


def test_service_event_stream_replays_completed_runtime_messages() -> None:
    async def run() -> None:
        service = _service()
        plan_response = await service.plan_from_prompt(
            PlanFromPromptRequest(prompt="Explain runtime")
        )
        session_response = await service.create_session(
            SessionCreateRequest(
                discovery_plan=plan_response.discovery_plan,
                execution_plan=plan_response.execution_plan,
                session_id="session-service-events",
            )
        )
        session = service.get_session(session_response.session_id)
        assert session is not None
        await session.wait()

        messages = [
            message async for message in service.stream_events("session-service-events")
        ]
        assert any(message["type"] == "result" for message in messages)
        assert any(message["type"] == "verdict" for message in messages)

    _run(run())


def test_events_endpoint_streams_runtime_messages() -> None:
    client = TestClient(create_app(_service()))
    plan_response = client.post("/api/plans/from-prompt", json={"prompt": "Explain runtime"})
    client.post(
        "/api/sessions",
        json={
            "discovery_plan": plan_response.json()["discovery_plan"],
            "execution_plan": plan_response.json()["execution_plan"],
            "session_id": "session-events",
        },
    )
    with client.stream("GET", "/api/sessions/session-events/events") as response:
        assert response.status_code == 200
        messages = _decode_sse(response.iter_lines())

    assert any(message["type"] == "progress" for message in messages)
    required_keys = {"from", "to", "type", "payload", "correlation_id"}
    assert all(required_keys <= set(message) for message in messages)


def _decode_sse(lines: Any) -> list[dict[str, Any]]:
    messages = []
    for line in lines:
        if not line.startswith("data: "):
            continue
        messages.append(json.loads(line.removeprefix("data: ")))
    return messages


def _run(awaitable: Any) -> Any:
    import asyncio

    return asyncio.run(awaitable)
