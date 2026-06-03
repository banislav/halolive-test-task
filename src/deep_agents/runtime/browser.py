from __future__ import annotations

import base64
import time
from typing import Any
from urllib.parse import urlparse

from deep_agents.models import ToolCallRequest, ToolCallResult, ToolDefinition, ToolSafetyLevel
from deep_agents.models.base import JsonObject
from deep_agents.runtime.memory import MemoryRecorder
from deep_agents.runtime.observability import ProgressSignalBus
from deep_agents.runtime.tools import ToolMiddlewareRunner, ToolPolicy, ToolRegistry

BROWSER_TOOL_IDS = [
    "browser_navigate",
    "browser_current_page",
    "browser_open_tab",
    "browser_switch_tab",
    "browser_close_tab",
    "browser_extract_text",
    "browser_extract_links",
    "browser_get_elements",
    "browser_screenshot",
    "browser_snapshot",
    "browser_type",
    "browser_scroll",
    "browser_wait",
    "browser_extract_tables",
    "browser_extract_images",
    "browser_extract_structured_data",
    "browser_detect_forms",
    "browser_fill_form",
    "browser_click",
    "browser_back",
]


class BrowserRuntimeError(RuntimeError):
    """Raised when the Playwright browser runtime cannot be started."""


class BrowserSession:
    """Lazy synchronous Playwright browser session used by browser tools."""

    def __init__(
        self,
        *,
        browser_type: str = "chromium",
        headless: bool = True,
        viewport: JsonObject | None = None,
        user_agent: str | None = None,
        default_timeout_ms: int = 30_000,
        action_delay_seconds: float = 0,
    ) -> None:
        self.browser_type = browser_type
        self.headless = headless
        self.viewport = viewport
        self.user_agent = user_agent
        self.default_timeout_ms = default_timeout_ms
        self.action_delay_seconds = action_delay_seconds
        self._playwright: Any | None = None
        self._browser: Any | None = None
        self._context: Any | None = None
        self._page: Any | None = None

    def navigate(self, url: str) -> JsonObject:
        """Navigate the current page to an HTTP(S) URL."""
        _validate_http_url(url)
        page = self._ensure_page()
        page.goto(url)
        self._delay()
        return self.current_page()

    def current_page(self) -> JsonObject:
        """Return metadata for the current page and tab set."""
        page = self._ensure_page()
        return self._page_metadata(page)

    def open_tab(self, url: str | None = None) -> JsonObject:
        """Open a new tab and optionally navigate it to an HTTP(S) URL."""
        if url is not None:
            _validate_http_url(url)
        context = self._ensure_context()
        page = context.new_page()
        self._page = page
        if url is not None:
            page.goto(url)
        self._delay()
        return self.current_page()

    def switch_tab(self, index: int) -> JsonObject:
        """Switch the active page to an existing tab by zero-based index."""
        pages = self._pages()
        if index < 0 or index >= len(pages):
            raise ValueError(f"Tab index {index} is out of range.")
        self._page = pages[index]
        return self.current_page()

    def close_tab(self, index: int | None = None) -> JsonObject:
        """Close a tab and select another available tab."""
        pages = self._pages()
        target = self._current_tab_index() if index is None else index
        if target < 0 or target >= len(pages):
            raise ValueError(f"Tab index {target} is out of range.")
        pages[target].close()
        remaining = self._pages(create_if_missing=False)
        self._page = remaining[min(target, len(remaining) - 1)] if remaining else None
        self._delay()
        return {"closed_index": target, **self.current_page()}

    def extract_text(self) -> JsonObject:
        """Extract visible body text from the current page."""
        page = self._ensure_page()
        return {"text": page.locator("body").inner_text()}

    def extract_links(self, limit: int = 50) -> JsonObject:
        """Extract hyperlinks from the current page."""
        page = self._ensure_page()
        links = page.eval_on_selector_all(
            "a[href]",
            """(anchors, limit) => anchors.slice(0, limit).map((anchor) => ({
                text: anchor.innerText || anchor.textContent || "",
                href: anchor.href
            }))""",
            limit,
        )
        return {"links": links}

    def get_elements(self, selector: str, limit: int = 20) -> JsonObject:
        """Return basic text and attributes for elements matching a selector."""
        page = self._ensure_page()
        elements = page.eval_on_selector_all(
            selector,
            """(elements, limit) => elements.slice(0, limit).map((element) => ({
                text: element.innerText || element.textContent || "",
                tag: element.tagName ? element.tagName.toLowerCase() : "",
                id: element.id || "",
                name: element.getAttribute("name") || "",
                aria_label: element.getAttribute("aria-label") || ""
            }))""",
            limit,
        )
        return {"selector": selector, "elements": elements}

    def screenshot(self, full_page: bool = True) -> JsonObject:
        """Capture a PNG screenshot as base64 JSON."""
        page = self._ensure_page()
        data = page.screenshot(full_page=full_page)
        return {
            "url": page.url,
            "mime_type": "image/png",
            "full_page": full_page,
            "screenshot_base64": base64.b64encode(data).decode("ascii"),
        }

    def snapshot(self, limit: int = 50) -> JsonObject:
        """Return a deterministic DOM/ARIA-style element summary."""
        page = self._ensure_page()
        elements = page.eval_on_selector_all(
            "a, button, input, textarea, select, [role], [aria-label]",
            """(elements, limit) => elements.slice(0, limit).map((element) => ({
                tag: element.tagName ? element.tagName.toLowerCase() : "",
                text: element.innerText || element.textContent || "",
                id: element.id || "",
                name: element.getAttribute("name") || "",
                role: element.getAttribute("role") || "",
                aria_label: element.getAttribute("aria-label") || "",
                type: element.getAttribute("type") || ""
            }))""",
            limit,
        )
        return {"url": page.url, "elements": elements}

    def type_text(self, selector: str, text: str, clear_first: bool = True) -> JsonObject:
        """Type text into an element by CSS selector."""
        page = self._ensure_page()
        if clear_first:
            page.fill(selector, "")
        page.type(selector, text)
        self._delay()
        return {"selector": selector, "text_length": len(text), "url": page.url}

    def scroll(self, x: int = 0, y: int = 800) -> JsonObject:
        """Scroll the current page by a pixel offset."""
        page = self._ensure_page()
        page.evaluate("(offset) => window.scrollBy(offset.x, offset.y)", {"x": x, "y": y})
        self._delay()
        return {"x": x, "y": y, "url": page.url}

    def wait(self, selector: str | None = None, timeout_ms: int | None = None) -> JsonObject:
        """Wait for a selector or page load state."""
        page = self._ensure_page()
        resolved_timeout = timeout_ms or self.default_timeout_ms
        if selector:
            page.wait_for_selector(selector, timeout=resolved_timeout)
            status = "selector_ready"
        else:
            page.wait_for_load_state("domcontentloaded", timeout=resolved_timeout)
            status = "page_ready"
        return {"status": status, "selector": selector, "timeout_ms": resolved_timeout}

    def extract_tables(self, limit: int = 20) -> JsonObject:
        """Extract table rows from HTML tables on the current page."""
        page = self._ensure_page()
        tables = page.eval_on_selector_all(
            "table",
            """(tables, limit) => tables.slice(0, limit).map((table) => ({
                headers: Array.from(table.querySelectorAll("th")).map((cell) => cell.innerText),
                rows: Array.from(table.querySelectorAll("tr")).map((row) =>
                    Array.from(row.querySelectorAll("th,td")).map((cell) => cell.innerText)
                )
            }))""",
            limit,
        )
        return {"tables": tables}

    def extract_images(self, limit: int = 50) -> JsonObject:
        """Extract image metadata from the current page."""
        page = self._ensure_page()
        images = page.eval_on_selector_all(
            "img",
            """(images, limit) => images.slice(0, limit).map((image) => ({
                src: image.currentSrc || image.src || "",
                alt: image.alt || "",
                width: image.naturalWidth || image.width || 0,
                height: image.naturalHeight || image.height || 0
            }))""",
            limit,
        )
        return {"images": images}

    def extract_structured_data(self) -> JsonObject:
        """Extract common structured page metadata."""
        page = self._ensure_page()
        data = page.evaluate(
            """() => ({
                title: document.title || "",
                description: document.querySelector('meta[name="description"]')?.content || "",
                canonical_url: document.querySelector('link[rel="canonical"]')?.href || "",
                headings: Array.from(document.querySelectorAll("h1,h2,h3")).map((heading) => ({
                    level: heading.tagName.toLowerCase(),
                    text: heading.innerText || heading.textContent || ""
                })),
                json_ld: Array.from(document.querySelectorAll('script[type="application/ld+json"]'))
                    .map((script) => script.textContent || "")
            })"""
        )
        return {"url": page.url, "structured_data": data}

    def detect_forms(self) -> JsonObject:
        """Detect forms and form fields with explicit selectors."""
        page = self._ensure_page()
        forms = page.eval_on_selector_all(
            "form",
            """(forms) => forms.map((form, formIndex) => ({
                selector: form.id ? `form#${form.id}` : `form:nth-of-type(${formIndex + 1})`,
                fields: Array.from(form.querySelectorAll("input, textarea, select")).map(
                    (field, fieldIndex) => ({
                        selector: field.id ? `#${field.id}` :
                            field.name ? `[name="${field.name}"]` :
                            `form:nth-of-type(${formIndex + 1}) ` +
                                `input:nth-of-type(${fieldIndex + 1})`,
                        name: field.getAttribute("name") || "",
                        type: field.getAttribute("type") || field.tagName.toLowerCase(),
                        label: field.getAttribute("aria-label") || ""
                    })
                )
            }))"""
        )
        return {"forms": forms}

    def fill_form(self, fields: dict[str, str], submit_selector: str | None = None) -> JsonObject:
        """Fill form fields by explicit CSS selectors and optionally submit."""
        page = self._ensure_page()
        filled: list[str] = []
        for selector, value in fields.items():
            page.fill(selector, value)
            filled.append(selector)
        submitted = False
        if submit_selector:
            page.click(submit_selector)
            submitted = True
        self._delay()
        return {"filled": filled, "submitted": submitted, "url": page.url}

    def click(self, selector: str) -> JsonObject:
        """Click an element on the current page by CSS selector."""
        page = self._ensure_page()
        page.click(selector)
        self._delay()
        return {"selector": selector, "url": page.url}

    def back(self) -> JsonObject:
        """Navigate back in browser history."""
        page = self._ensure_page()
        response = page.go_back()
        self._delay()
        return {"url": page.url, "navigated": response is not None}

    def close(self) -> None:
        """Close the browser session and Playwright runtime."""
        if self._context is not None:
            self._context.close()
            self._context = None
        if self._browser is not None:
            self._browser.close()
            self._browser = None
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None
        self._page = None

    def _ensure_page(self) -> Any:
        if self._page is not None:
            return self._page

        context = self._ensure_context()
        self._page = context.new_page()
        return self._page

    def _ensure_context(self) -> Any:
        if self._context is not None:
            return self._context

        try:
            from playwright.sync_api import Error as PlaywrightError
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise BrowserRuntimeError(
                "Playwright is required for browser tools. Install dependencies with "
                "`pip install playwright` and browser binaries with "
                "`python -m playwright install chromium`."
            ) from exc

        try:
            self._playwright = sync_playwright().start()
            browser_factory = getattr(self._playwright, self.browser_type)
            self._browser = browser_factory.launch(headless=self.headless)
            context_kwargs: JsonObject = {}
            if self.viewport is not None:
                context_kwargs["viewport"] = self.viewport
            if self.user_agent is not None:
                context_kwargs["user_agent"] = self.user_agent
            self._context = self._browser.new_context(**context_kwargs)
            self._context.set_default_timeout(self.default_timeout_ms)
        except PlaywrightError as exc:
            self.close()
            raise BrowserRuntimeError(
                "Could not start Playwright browser. Install browser binaries with "
                "`python -m playwright install chromium`."
            ) from exc
        return self._context

    def _pages(self, *, create_if_missing: bool = True) -> list[Any]:
        context = self._ensure_context()
        pages = list(context.pages)
        if not pages and create_if_missing:
            pages = [self._ensure_page()]
        return pages

    def _current_tab_index(self) -> int:
        pages = self._pages()
        page = self._ensure_page()
        return pages.index(page) if page in pages else 0

    def _page_metadata(self, page: Any) -> JsonObject:
        pages = self._pages()
        return {
            "url": page.url,
            "title": page.title(),
            "tab_index": pages.index(page) if page in pages else 0,
            "tab_count": len(pages),
        }

    def _delay(self) -> None:
        if self.action_delay_seconds > 0:
            time.sleep(self.action_delay_seconds)


class BrowserWorker:
    """Tool-backed BrowserWorker facade for browser task execution."""

    def __init__(
        self,
        *,
        session: BrowserSession | None = None,
        policy: ToolPolicy | None = None,
        memory_recorder: MemoryRecorder | None = None,
        progress_bus: ProgressSignalBus | None = None,
        plan_id: str | None = None,
    ) -> None:
        self.session = session or BrowserSession()
        self.registry = build_browser_tool_registry(self.session)
        self.runner = ToolMiddlewareRunner(
            registry=self.registry,
            policy=policy,
            memory_recorder=memory_recorder,
            progress_bus=progress_bus,
            plan_id=plan_id,
        )

    def invoke(self, request: ToolCallRequest | dict[str, Any]) -> ToolCallResult:
        """Invoke one BrowserWorker tool through the runtime middleware stack."""
        return self.runner.invoke(request)

    def close(self) -> None:
        """Close the underlying browser session."""
        self.session.close()


def build_browser_tool_registry(
    session: BrowserSession | None = None,
    *,
    registry: ToolRegistry | None = None,
) -> ToolRegistry:
    """Register application-owned browser tools into a ToolRegistry."""
    resolved_session = session or BrowserSession()
    resolved_registry = registry or ToolRegistry()
    resolved_registry.register(
        ToolDefinition(
            id="browser_navigate",
            name="Browser Navigate",
            input_schema={"url": "string"},
            output_schema={"url": "string"},
            safety_level=ToolSafetyLevel.SENSITIVE,
        ),
        resolved_session.navigate,
    )
    resolved_registry.register(
        ToolDefinition(
            id="browser_current_page",
            name="Browser Current Page",
            output_schema={
                "url": "string",
                "title": "string",
                "tab_index": "int",
                "tab_count": "int",
            },
        ),
        resolved_session.current_page,
    )
    resolved_registry.register(
        ToolDefinition(
            id="browser_open_tab",
            name="Browser Open Tab",
            input_schema={"url": "string"},
            output_schema={"url": "string", "tab_index": "int", "tab_count": "int"},
            safety_level=ToolSafetyLevel.SENSITIVE,
        ),
        resolved_session.open_tab,
    )
    resolved_registry.register(
        ToolDefinition(
            id="browser_switch_tab",
            name="Browser Switch Tab",
            input_schema={"index": "int"},
            output_schema={"url": "string", "tab_index": "int", "tab_count": "int"},
        ),
        resolved_session.switch_tab,
    )
    resolved_registry.register(
        ToolDefinition(
            id="browser_close_tab",
            name="Browser Close Tab",
            input_schema={"index": "int"},
            output_schema={"closed_index": "int", "url": "string"},
            safety_level=ToolSafetyLevel.SENSITIVE,
        ),
        resolved_session.close_tab,
    )
    resolved_registry.register(
        ToolDefinition(
            id="browser_extract_text",
            name="Browser Extract Text",
            output_schema={"text": "string"},
        ),
        resolved_session.extract_text,
    )
    resolved_registry.register(
        ToolDefinition(
            id="browser_extract_links",
            name="Browser Extract Links",
            output_schema={"links": "list"},
        ),
        resolved_session.extract_links,
    )
    resolved_registry.register(
        ToolDefinition(
            id="browser_get_elements",
            name="Browser Get Elements",
            input_schema={"selector": "string"},
            output_schema={"elements": "list"},
        ),
        resolved_session.get_elements,
    )
    resolved_registry.register(
        ToolDefinition(
            id="browser_screenshot",
            name="Browser Screenshot",
            input_schema={"full_page": "bool"},
            output_schema={"mime_type": "string", "screenshot_base64": "string"},
        ),
        resolved_session.screenshot,
    )
    resolved_registry.register(
        ToolDefinition(
            id="browser_snapshot",
            name="Browser Snapshot",
            input_schema={"limit": "int"},
            output_schema={"elements": "list"},
        ),
        resolved_session.snapshot,
    )
    resolved_registry.register(
        ToolDefinition(
            id="browser_type",
            name="Browser Type",
            input_schema={"selector": "string", "text": "string"},
            output_schema={"selector": "string", "text_length": "int"},
            safety_level=ToolSafetyLevel.SENSITIVE,
        ),
        resolved_session.type_text,
    )
    resolved_registry.register(
        ToolDefinition(
            id="browser_scroll",
            name="Browser Scroll",
            input_schema={"x": "int", "y": "int"},
            output_schema={"x": "int", "y": "int"},
        ),
        resolved_session.scroll,
    )
    resolved_registry.register(
        ToolDefinition(
            id="browser_wait",
            name="Browser Wait",
            input_schema={"selector": "string", "timeout_ms": "int"},
            output_schema={"status": "string", "timeout_ms": "int"},
        ),
        resolved_session.wait,
    )
    resolved_registry.register(
        ToolDefinition(
            id="browser_extract_tables",
            name="Browser Extract Tables",
            input_schema={"limit": "int"},
            output_schema={"tables": "list"},
        ),
        resolved_session.extract_tables,
    )
    resolved_registry.register(
        ToolDefinition(
            id="browser_extract_images",
            name="Browser Extract Images",
            input_schema={"limit": "int"},
            output_schema={"images": "list"},
        ),
        resolved_session.extract_images,
    )
    resolved_registry.register(
        ToolDefinition(
            id="browser_extract_structured_data",
            name="Browser Extract Structured Data",
            output_schema={"structured_data": "object"},
        ),
        resolved_session.extract_structured_data,
    )
    resolved_registry.register(
        ToolDefinition(
            id="browser_detect_forms",
            name="Browser Detect Forms",
            output_schema={"forms": "list"},
        ),
        resolved_session.detect_forms,
    )
    resolved_registry.register(
        ToolDefinition(
            id="browser_fill_form",
            name="Browser Fill Form",
            input_schema={"fields": "object"},
            output_schema={"filled": "list", "submitted": "bool"},
            safety_level=ToolSafetyLevel.SENSITIVE,
        ),
        resolved_session.fill_form,
    )
    resolved_registry.register(
        ToolDefinition(
            id="browser_click",
            name="Browser Click",
            input_schema={"selector": "string"},
            output_schema={"url": "string"},
            safety_level=ToolSafetyLevel.SENSITIVE,
        ),
        resolved_session.click,
    )
    resolved_registry.register(
        ToolDefinition(
            id="browser_back",
            name="Browser Back",
            output_schema={"url": "string", "navigated": "bool"},
        ),
        resolved_session.back,
    )
    return resolved_registry


def _validate_http_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Browser navigation only allows http and https URLs.")
    if not parsed.netloc:
        raise ValueError("Browser navigation requires an absolute URL.")
