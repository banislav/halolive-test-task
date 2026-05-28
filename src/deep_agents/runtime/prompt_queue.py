from __future__ import annotations

from collections import deque

from deep_agents.models import InterruptPriority, PromptQueueItem


class PromptQueue:
    """FIFO prompt queue with priority interrupts at the front."""

    def __init__(self) -> None:
        self._queue: deque[PromptQueueItem] = deque()

    def push(self, item: PromptQueueItem) -> None:
        if item.priority in {InterruptPriority.P0_HALT, InterruptPriority.P1_PAUSE}:
            self._queue.appendleft(item)
            return
        self._queue.append(item)

    def pop(self) -> PromptQueueItem | None:
        if not self._queue:
            return None
        return self._queue.popleft()

    def peek(self) -> PromptQueueItem | None:
        if not self._queue:
            return None
        return self._queue[0]

    def __len__(self) -> int:
        return len(self._queue)
