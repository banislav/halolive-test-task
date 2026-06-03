from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from deep_agents.models import ToolDefinition, ToolSafetyLevel
from deep_agents.models.base import JsonObject
from deep_agents.runtime.tools import ToolRegistry

BROWSER_TOOL_IDS = [
    "browser_navigate",
    "browser_current_page",
    "browser_extract_text",
    "browser_extract_links",
    "browser_get_elements",
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
    ) -> None:
        self.browser_type = browser_type
        self.headless = headless
        self._playwright: Any | None = None
        self._browser: Any | None = None
        self._context: Any | None = None
        self._page: Any | None = None

    def navigate(self, url: str) -> JsonObject:
        """Navigate the current page to an HTTP(S) URL."""
        _validate_http_url(url)
        page = self._ensure_page()
        page.goto(url)
        return {"url": page.url}

    def current_page(self) -> JsonObject:
        """Return the current page URL."""
        page = self._ensure_page()
        return {"url": page.url}

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

    def click(self, selector: str) -> JsonObject:
        """Click an element on the current page by CSS selector."""
        page = self._ensure_page()
        page.click(selector)
        return {"selector": selector, "url": page.url}

    def back(self) -> JsonObject:
        """Navigate back in browser history."""
        page = self._ensure_page()
        response = page.go_back()
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
            self._context = self._browser.new_context()
            self._page = self._context.new_page()
        except PlaywrightError as exc:
            self.close()
            raise BrowserRuntimeError(
                "Could not start Playwright browser. Install browser binaries with "
                "`python -m playwright install chromium`."
            ) from exc
        return self._page


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
            output_schema={"url": "string"},
        ),
        resolved_session.current_page,
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
