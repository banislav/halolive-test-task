from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deep_agents.models import ToolCallRequest
from deep_agents.runtime import (
    BrowserRuntimeError,
    BrowserSession,
    InMemoryStore,
    MemoryRecorder,
    ProgressSignalBus,
    ToolMiddlewareRunner,
    ToolPolicy,
    build_browser_tool_registry,
)


def main() -> None:
    memory_store = InMemoryStore()
    progress_bus = ProgressSignalBus()
    session = BrowserSession(headless=True)
    runner = ToolMiddlewareRunner(
        registry=build_browser_tool_registry(session),
        policy=ToolPolicy(allow_sensitive=True),
        memory_recorder=MemoryRecorder(memory_store),
        progress_bus=progress_bus,
        plan_id="EP-browser-example",
    )

    try:
        navigate = runner.invoke(
            ToolCallRequest(
                tool_id="browser_navigate",
                task_id="T1",
                input={"url": "https://example.com"},
            )
        )
        if navigate.error_message:
            raise BrowserRuntimeError(navigate.error_message)
        text = runner.invoke(ToolCallRequest(tool_id="browser_extract_text", task_id="T1"))
        if text.error_message:
            raise BrowserRuntimeError(text.error_message)
    except BrowserRuntimeError as exc:
        print(exc)
        print("Install browser binaries with: python -m playwright install chromium")
    finally:
        session.close()

    print([record.tags for record in memory_store.records() if "tool_result" in record.tags])


if __name__ == "__main__":
    main()
