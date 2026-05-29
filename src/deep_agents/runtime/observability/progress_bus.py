from __future__ import annotations

from collections import defaultdict
from typing import Protocol

from deep_agents.models import ObserverJudgment, ProcessJudgment, ProgressSignal


class ProcessSignalJudge(Protocol):
    """Evaluates task-scoped progress signal history."""

    def evaluate(
        self,
        signal_history: list[ProgressSignal],
        latest_signal: ProgressSignal,
    ) -> ProcessJudgment | None:
        ...


class ObserverSignalJudge(Protocol):
    """Evaluates runtime-wide progress signal history."""

    def evaluate(
        self,
        all_signals: list[ProgressSignal],
        latest_signal: ProgressSignal,
    ) -> ObserverJudgment | None:
        ...


class ProgressSignalBus:
    """Synchronous in-memory signal store and deterministic judge fanout."""

    def __init__(self) -> None:
        self._signals: list[ProgressSignal] = []
        self._observer_judges: list[ObserverSignalJudge] = []
        self._global_process_judges: list[ProcessSignalJudge] = []
        self._task_process_judges: dict[str, list[ProcessSignalJudge]] = defaultdict(list)

    def subscribe_observer(self, judge: ObserverSignalJudge) -> None:
        """Register an observer judge that receives every published signal."""
        self._observer_judges.append(judge)

    def subscribe_process(self, judge: ProcessSignalJudge, task_id: str | None = None) -> None:
        """Register a process judge for all tasks or a specific task id."""
        if task_id is None:
            self._global_process_judges.append(judge)
            return
        self._task_process_judges[task_id].append(judge)

    def publish(self, signal: ProgressSignal) -> list[ProcessJudgment | ObserverJudgment]:
        """Store a signal and synchronously deliver it to matching judges."""
        self._signals.append(signal)
        judgments: list[ProcessJudgment | ObserverJudgment] = []

        task_history = self.signals(task_id=signal.task_id)
        process_judges = [
            *self._global_process_judges,
            *self._task_process_judges.get(signal.task_id, []),
        ]
        for judge in process_judges:
            judgment = judge.evaluate(task_history, signal)
            if judgment is not None:
                judgments.append(judgment)

        all_signals = self.signals()
        for judge in self._observer_judges:
            judgment = judge.evaluate(all_signals, signal)
            if judgment is not None:
                judgments.append(judgment)

        return judgments

    def signals(self, task_id: str | None = None) -> list[ProgressSignal]:
        """Return all stored signals or only the history for one task."""
        if task_id is None:
            return list(self._signals)
        return [signal for signal in self._signals if signal.task_id == task_id]
