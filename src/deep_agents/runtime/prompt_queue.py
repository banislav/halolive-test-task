from __future__ import annotations

from collections import deque
from collections.abc import Iterator

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

    def items(self) -> list[PromptQueueItem]:
        """Return queued prompts in handling order without mutating the queue."""
        return list(self._queue)

    def drain(self) -> list[PromptQueueItem]:
        """Remove and return all queued prompts in deterministic handling order."""
        items = self.items()
        self._queue.clear()
        return items

    def __iter__(self) -> Iterator[PromptQueueItem]:
        return iter(self.items())

    def __len__(self) -> int:
        return len(self._queue)
