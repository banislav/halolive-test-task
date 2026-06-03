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
    BrowserWorker,
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
        self.tabs = ["about:blank"]
        self.current_tab = 0

    def navigate(self, url: str) -> dict[str, object]:
        """Navigate to a URL."""
        self.calls.append(("navigate", {"url": url}))
        self.url = url
        self.tabs[self.current_tab] = url
        return self.current_page()

    def current_page(self) -> dict[str, object]:
        """Return the current page."""
        self.calls.append(("current_page", {}))
        return {
            "url": self.url,
            "title": "Example",
            "tab_index": self.current_tab,
            "tab_count": len(self.tabs),
        }

    def open_tab(self, url: str | None = None) -> dict[str, object]:
        """Open a new tab."""
        self.calls.append(("open_tab", {"url": url}))
        self.url = url or "about:blank"
        self.tabs.append(self.url)
        self.current_tab = len(self.tabs) - 1
        return self.current_page()

    def switch_tab(self, index: int) -> dict[str, object]:
        """Switch tabs."""
        self.calls.append(("switch_tab", {"index": index}))
        self.current_tab = index
        self.url = self.tabs[index]
        return self.current_page()

    def close_tab(self, index: int | None = None) -> dict[str, object]:
        """Close a tab."""
        self.calls.append(("close_tab", {"index": index}))
        target = self.current_tab if index is None else index
        self.tabs.pop(target)
        self.current_tab = max(min(target, len(self.tabs) - 1), 0)
        self.url = self.tabs[self.current_tab] if self.tabs else "about:blank"
        if not self.tabs:
            self.tabs.append(self.url)
        return {"closed_index": target, **self.current_page()}

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

    def screenshot(self, full_page: bool = True) -> dict[str, object]:
        """Capture screenshot."""
        self.calls.append(("screenshot", {"full_page": full_page}))
        return {
            "url": self.url,
            "mime_type": "image/png",
            "full_page": full_page,
            "screenshot_base64": "ZmFrZQ==",
        }

    def snapshot(self, limit: int = 50) -> dict[str, object]:
        """Snapshot page elements."""
        self.calls.append(("snapshot", {"limit": limit}))
        return {"url": self.url, "elements": [{"tag": "button", "aria_label": "Submit"}]}

    def type_text(self, selector: str, text: str, clear_first: bool = True) -> dict[str, object]:
        """Type text."""
        self.calls.append(
            ("type_text", {"selector": selector, "text": text, "clear_first": clear_first})
        )
        return {"selector": selector, "text_length": len(text), "url": self.url}

    def scroll(self, x: int = 0, y: int = 800) -> dict[str, object]:
        """Scroll page."""
        self.calls.append(("scroll", {"x": x, "y": y}))
        return {"x": x, "y": y, "url": self.url}

    def wait(self, selector: str | None = None, timeout_ms: int | None = None) -> dict[str, object]:
        """Wait for page state."""
        self.calls.append(("wait", {"selector": selector, "timeout_ms": timeout_ms}))
        return {"status": "selector_ready", "selector": selector, "timeout_ms": timeout_ms or 1000}

    def extract_tables(self, limit: int = 20) -> dict[str, object]:
        """Extract tables."""
        self.calls.append(("extract_tables", {"limit": limit}))
        return {"tables": [{"headers": ["Name"], "rows": [["Ada"]]}]}

    def extract_images(self, limit: int = 50) -> dict[str, object]:
        """Extract images."""
        self.calls.append(("extract_images", {"limit": limit}))
        return {"images": [{"src": "https://example.com/a.png", "alt": "A"}]}

    def extract_structured_data(self) -> dict[str, object]:
        """Extract structured data."""
        self.calls.append(("extract_structured_data", {}))
        return {"url": self.url, "structured_data": {"title": "Example"}}

    def detect_forms(self) -> dict[str, object]:
        """Detect forms."""
        self.calls.append(("detect_forms", {}))
        return {"forms": [{"selector": "form#contact", "fields": [{"selector": "#email"}]}]}

    def fill_form(
        self,
        fields: dict[str, str],
        submit_selector: str | None = None,
    ) -> dict[str, object]:
        """Fill a form."""
        self.calls.append(("fill_form", {"fields": fields, "submit_selector": submit_selector}))
        return {"filled": list(fields), "submitted": submit_selector is not None, "url": self.url}

    def click(self, selector: str) -> dict[str, object]:
        """Click an element."""
        self.calls.append(("click", {"selector": selector}))
        return {"selector": selector, "url": self.url}

    def back(self) -> dict[str, object]:
        """Navigate back."""
        self.calls.append(("back", {}))
        self.url = "https://example.com"
        return {"url": self.url, "navigated": True}

    def close(self) -> None:
        """Close the fake browser session."""
        self.calls.append(("close", {}))


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
    assert registry.definition("browser_type").safety_level == ToolSafetyLevel.SENSITIVE
    assert registry.definition("browser_fill_form").safety_level == ToolSafetyLevel.SENSITIVE
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
    ).output == {
        "url": "https://example.com",
        "title": "Example",
        "tab_index": 0,
        "tab_count": 1,
    }
    assert runner.invoke(
        ToolCallRequest(tool_id="browser_current_page", task_id="T1")
    ).output == {
        "url": "https://example.com",
        "title": "Example",
        "tab_index": 0,
        "tab_count": 1,
    }
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

    assert [name for name, _ in session.calls if name != "current_page"] == [
        "navigate",
        "extract_text",
        "extract_links",
        "get_elements",
        "click",
        "back",
    ]


def test_browser_v1_tools_call_session_methods_with_expected_inputs() -> None:
    session = FakeBrowserSession()
    runner = build_fake_browser_runner(session)

    assert runner.invoke(
        ToolCallRequest(
            tool_id="browser_open_tab",
            task_id="T1",
            input={"url": "https://example.com/docs"},
        )
    ).output["tab_count"] == 2
    assert runner.invoke(
        ToolCallRequest(tool_id="browser_switch_tab", task_id="T1", input={"index": 0})
    ).output["tab_index"] == 0
    assert runner.invoke(
        ToolCallRequest(tool_id="browser_screenshot", task_id="T1", input={"full_page": False})
    ).output == {
        "url": "about:blank",
        "mime_type": "image/png",
        "full_page": False,
        "screenshot_base64": "ZmFrZQ==",
    }
    assert runner.invoke(
        ToolCallRequest(tool_id="browser_snapshot", task_id="T1", input={"limit": 3})
    ).output["elements"][0]["aria_label"] == "Submit"
    assert runner.invoke(
        ToolCallRequest(
            tool_id="browser_type",
            task_id="T1",
            input={"selector": "#email", "text": "ada@example.com"},
        )
    ).output["text_length"] == 15
    assert runner.invoke(
        ToolCallRequest(tool_id="browser_scroll", task_id="T1", input={"x": 1, "y": 2})
    ).output == {"x": 1, "y": 2, "url": "about:blank"}
    assert runner.invoke(
        ToolCallRequest(
            tool_id="browser_wait",
            task_id="T1",
            input={"selector": "#ready", "timeout_ms": 250},
        )
    ).output["status"] == "selector_ready"
    assert runner.invoke(
        ToolCallRequest(tool_id="browser_extract_tables", task_id="T1", input={"limit": 1})
    ).output["tables"][0]["headers"] == ["Name"]
    assert runner.invoke(
        ToolCallRequest(tool_id="browser_extract_images", task_id="T1", input={"limit": 1})
    ).output["images"][0]["alt"] == "A"
    assert runner.invoke(
        ToolCallRequest(tool_id="browser_extract_structured_data", task_id="T1")
    ).output["structured_data"]["title"] == "Example"
    assert runner.invoke(
        ToolCallRequest(tool_id="browser_detect_forms", task_id="T1")
    ).output["forms"][0]["selector"] == "form#contact"
    assert runner.invoke(
        ToolCallRequest(
            tool_id="browser_fill_form",
            task_id="T1",
            input={"fields": {"#email": "ada@example.com"}, "submit_selector": "button"},
        )
    ).output == {"filled": ["#email"], "submitted": True, "url": "about:blank"}
    assert runner.invoke(
        ToolCallRequest(tool_id="browser_close_tab", task_id="T1", input={"index": 1})
    ).output["closed_index"] == 1

    assert [name for name, _ in session.calls if name != "current_page"] == [
        "open_tab",
        "switch_tab",
        "screenshot",
        "snapshot",
        "type_text",
        "scroll",
        "wait",
        "extract_tables",
        "extract_images",
        "extract_structured_data",
        "detect_forms",
        "fill_form",
        "close_tab",
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


def test_browser_worker_facade_invokes_tools_through_middleware() -> None:
    memory_store = InMemoryStore()
    progress_bus = ProgressSignalBus()
    worker = BrowserWorker(
        session=FakeBrowserSession(),
        policy=ToolPolicy(allow_sensitive=True),
        memory_recorder=MemoryRecorder(memory_store),
        progress_bus=progress_bus,
        plan_id="EP-browser",
    )

    result = worker.invoke(
        ToolCallRequest(
            tool_id="browser_type",
            task_id="T1",
            input={"selector": "#email", "text": "ada@example.com"},
        )
    )

    assert result.status == ToolCallStatus.SUCCEEDED
    assert result.output["text_length"] == 15
    assert [record.tags[0] for record in memory_store.query(MemoryQuery(task_ids=["T1"]))] == [
        "tool_call",
        "tool_result",
    ]
    assert [signal.payload.status for signal in progress_bus.signals(task_id="T1")] == [
        "tool_started",
        "tool_succeeded",
    ]
    worker.close()


def test_runtime_worker_records_browser_tool_summaries_in_attempt() -> None:
    memory_store = InMemoryStore()
    progress_bus = ProgressSignalBus()
    browser_worker = BrowserWorker(
        session=FakeBrowserSession(),
        policy=ToolPolicy(allow_sensitive=True),
        memory_recorder=MemoryRecorder(memory_store),
        progress_bus=progress_bus,
        plan_id="EP-browser",
    )
    assignment = AgentAssignment(type=AgentKind.WORKER, name="BrowserWorker")
    plan = ExecutionPlan(
        id="EP-browser",
        objective="Inspect a page",
        waves=[Wave(index=0, task_ids=["T1"])],
        task_cards=[TaskCard(id="T1", name="Inspect", wave=0, assigned_to=assignment)],
    )

    def run_task(task: TaskCard) -> TaskRunResult:
        navigate = browser_worker.invoke(
            ToolCallRequest(
                tool_id="browser_navigate",
                task_id=task.id,
                input={"url": "https://example.com"},
                caller_agent=task.assigned_to,
            )
        )
        form = browser_worker.invoke(
            ToolCallRequest(
                tool_id="browser_fill_form",
                task_id=task.id,
                input={"fields": {"#email": "ada@example.com"}},
                caller_agent=task.assigned_to,
            )
        )
        text = browser_worker.invoke(
            ToolCallRequest(
                tool_id="browser_extract_text",
                task_id=task.id,
                caller_agent=task.assigned_to,
            )
        )
        return TaskRunResult(
            task_id=task.id,
            output={
                "url": navigate.output["url"],
                "filled": form.output["filled"],
                "text": text.output["text"],
            },
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
        "filled": ["#email"],
        "text": "Example page text",
    }
    assert [
        summary["tool_id"] for summary in final_state["task_attempts"][0].result["tool_calls"]
    ] == ["browser_navigate", "browser_fill_form", "browser_extract_text"]
    browser_worker.close()


def test_real_playwright_browser_smoke_is_optional() -> None:
    pytest.importorskip("playwright.sync_api")
    pytest.skip("Real browser smoke test requires installed Playwright browser binaries.")
