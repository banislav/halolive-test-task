from __future__ import annotations

import pytest
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool

from deep_agents.models import (
    AgentAssignment,
    AgentKind,
    ExecutionPlan,
    JudgeRecommendation,
    JudgeVerdict,
    MemoryQuery,
    Objective,
    PlanState,
    TaskCard,
    ToolCallRequest,
    ToolCallStatus,
    ToolSafetyLevel,
    Wave,
)
from deep_agents.runtime import (
    BROWSER_TOOL_IDS,
    BrowserSession,
    InMemoryStore,
    MemoryRecorder,
    ProgressSignalBus,
    RuntimeEngine,
    TaskRunResult,
    ToolMiddlewareRunner,
    ToolPolicy,
    build_browser_tool_registry,
)


class FakeBrowserSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.url = "about:blank"

    def navigate(self, url: str) -> dict[str, object]:
        """Navigate to a URL."""
        self.calls.append(("navigate", {"url": url}))
        self.url = url
        return {"url": self.url}

    def current_page(self) -> dict[str, object]:
        """Return the current page."""
        self.calls.append(("current_page", {}))
        return {"url": self.url}

    def extract_text(self) -> dict[str, object]:
        """Extract page text."""
        self.calls.append(("extract_text", {}))
        return {"text": "Example page text"}

    def extract_links(self, limit: int = 50) -> dict[str, object]:
        """Extract page links."""
        self.calls.append(("extract_links", {"limit": limit}))
        return {"links": [{"text": "Docs", "href": "https://example.com/docs"}]}

    def get_elements(self, selector: str, limit: int = 20) -> dict[str, object]:
        """Get matching elements."""
        self.calls.append(("get_elements", {"selector": selector, "limit": limit}))
        return {"selector": selector, "elements": [{"text": "Submit", "tag": "button"}]}

    def click(self, selector: str) -> dict[str, object]:
        """Click an element."""
        self.calls.append(("click", {"selector": selector}))
        return {"selector": selector, "url": self.url}

    def back(self) -> dict[str, object]:
        """Navigate back."""
        self.calls.append(("back", {}))
        self.url = "https://example.com"
        return {"url": self.url, "navigated": True}


def build_fake_browser_runner(
    session: FakeBrowserSession,
    *,
    memory_store: InMemoryStore | None = None,
    progress_bus: ProgressSignalBus | None = None,
) -> ToolMiddlewareRunner:
    return ToolMiddlewareRunner(
        registry=build_browser_tool_registry(session),
        policy=ToolPolicy(allow_sensitive=True),
        memory_recorder=MemoryRecorder(memory_store) if memory_store is not None else None,
        progress_bus=progress_bus,
        plan_id="EP-browser",
    )


def test_browser_factory_registers_expected_langchain_tools_and_safety_levels() -> None:
    registry = build_browser_tool_registry(FakeBrowserSession())

    assert [definition.id for definition in registry.definitions()] == BROWSER_TOOL_IDS
    assert all(isinstance(tool, BaseTool) for tool in registry.langchain_tools())
    assert registry.definition("browser_navigate").safety_level == ToolSafetyLevel.SENSITIVE
    assert registry.definition("browser_click").safety_level == ToolSafetyLevel.SENSITIVE
    assert registry.definition("browser_extract_text").safety_level == ToolSafetyLevel.SAFE


def test_browser_navigation_rejects_non_http_urls_without_starting_browser() -> None:
    runner = ToolMiddlewareRunner(
        registry=build_browser_tool_registry(BrowserSession()),
        policy=ToolPolicy(allow_sensitive=True),
    )

    result = runner.invoke(
        ToolCallRequest(
            tool_id="browser_navigate",
            task_id="T1",
            input={"url": "file:///private/etc/hosts"},
        )
    )

    assert result.status == ToolCallStatus.FAILED
    assert result.error_type == "ValueError"
    assert "http and https" in result.error_message


def test_browser_tools_call_session_methods_with_expected_inputs() -> None:
    session = FakeBrowserSession()
    runner = build_fake_browser_runner(session)

    assert runner.invoke(
        ToolCallRequest(
            tool_id="browser_navigate",
            task_id="T1",
            input={"url": "https://example.com"},
        )
    ).output == {"url": "https://example.com"}
    assert runner.invoke(
        ToolCallRequest(tool_id="browser_current_page", task_id="T1")
    ).output == {"url": "https://example.com"}
    assert runner.invoke(
        ToolCallRequest(tool_id="browser_extract_text", task_id="T1")
    ).output == {"text": "Example page text"}
    assert runner.invoke(
        ToolCallRequest(tool_id="browser_extract_links", task_id="T1", input={"limit": 5})
    ).output["links"][0]["href"] == "https://example.com/docs"
    assert runner.invoke(
        ToolCallRequest(
            tool_id="browser_get_elements",
            task_id="T1",
            input={"selector": "button", "limit": 3},
        )
    ).output["elements"][0]["tag"] == "button"
    assert runner.invoke(
        ToolCallRequest(tool_id="browser_click", task_id="T1", input={"selector": "button"})
    ).output == {"selector": "button", "url": "https://example.com"}
    assert runner.invoke(ToolCallRequest(tool_id="browser_back", task_id="T1")).output == {
        "url": "https://example.com",
        "navigated": True,
    }

    assert [name for name, _ in session.calls] == [
        "navigate",
        "current_page",
        "extract_text",
        "extract_links",
        "get_elements",
        "click",
        "back",
    ]


def test_browser_tools_record_memory_and_progress_through_middleware() -> None:
    memory_store = InMemoryStore()
    progress_bus = ProgressSignalBus()
    runner = build_fake_browser_runner(
        FakeBrowserSession(),
        memory_store=memory_store,
        progress_bus=progress_bus,
    )

    result = runner.invoke(
        ToolCallRequest(
            tool_id="browser_navigate",
            task_id="T1",
            input={"url": "https://example.com"},
        )
    )

    assert result.status == ToolCallStatus.SUCCEEDED
    assert [record.tags[0] for record in memory_store.query(MemoryQuery(task_ids=["T1"]))] == [
        "tool_call",
        "tool_result",
    ]
    assert [signal.payload.status for signal in progress_bus.signals(task_id="T1")] == [
        "tool_started",
        "tool_succeeded",
    ]


def test_runtime_worker_records_browser_tool_summaries_in_attempt() -> None:
    memory_store = InMemoryStore()
    progress_bus = ProgressSignalBus()
    browser_runner = build_fake_browser_runner(
        FakeBrowserSession(),
        memory_store=memory_store,
        progress_bus=progress_bus,
    )
    assignment = AgentAssignment(type=AgentKind.WORKER, name="BrowserWorker")
    plan = ExecutionPlan(
        id="EP-browser",
        objective="Inspect a page",
        waves=[Wave(index=0, task_ids=["T1"])],
        task_cards=[TaskCard(id="T1", name="Inspect", wave=0, assigned_to=assignment)],
    )

    def run_task(task: TaskCard) -> TaskRunResult:
        navigate = browser_runner.invoke(
            ToolCallRequest(
                tool_id="browser_navigate",
                task_id=task.id,
                input={"url": "https://example.com"},
                caller_agent=task.assigned_to,
            )
        )
        text = browser_runner.invoke(
            ToolCallRequest(
                tool_id="browser_extract_text",
                task_id=task.id,
                caller_agent=task.assigned_to,
            )
        )
        return TaskRunResult(
            task_id=task.id,
            output={"url": navigate.output["url"], "text": text.output["text"]},
        )

    def judge_task(payload: dict[str, object]) -> JudgeVerdict:
        result = payload["result"]
        assert isinstance(result, TaskRunResult)
        return JudgeVerdict(
            task_id=result.task_id,
            verdict="pass",
            overall_confidence=1.0,
            recommendation=JudgeRecommendation.ADVANCE,
        )

    final_state = RuntimeEngine(
        worker=RunnableLambda(run_task),
        judge=RunnableLambda(judge_task),
        memory_store=memory_store,
        progress_bus=progress_bus,
    ).invoke(plan, PlanState(objective=Objective(raw="Inspect a page")))

    assert final_state["results"]["T1"].output == {
        "url": "https://example.com",
        "text": "Example page text",
    }
    assert [
        summary["tool_id"] for summary in final_state["task_attempts"][0].result["tool_calls"]
    ] == ["browser_navigate", "browser_extract_text"]


def test_real_playwright_browser_smoke_is_optional() -> None:
    pytest.importorskip("playwright.sync_api")
    pytest.skip("Real browser smoke test requires installed Playwright browser binaries.")
