from __future__ import annotations

from datetime import datetime

from deep_agents.models import (
    ObserverHealth,
    ObserverJudgment,
    ProcessAction,
    ProgressSignal,
    ProgressSignalType,
)


class ObserverJudge:
    """Rule-based runtime-wide judge for progress signal histories."""

    def __init__(
        self,
        *,
        stale_progress_seconds: int = 120,
        repeated_error_threshold: int = 2,
        low_relevance_threshold: float = 0.25,
        divergence_threshold: int = 2,
    ) -> None:
        self.stale_progress_seconds = stale_progress_seconds
        self.repeated_error_threshold = repeated_error_threshold
        self.low_relevance_threshold = low_relevance_threshold
        self.divergence_threshold = divergence_threshold

    def evaluate(
        self,
        all_signals: list[ProgressSignal],
        latest_signal: ProgressSignal,
    ) -> ObserverJudgment | None:
        """Return a runtime health judgment for the latest signal history."""
        if not all_signals:
            return None

        if self._has_explicit_divergence(latest_signal) or self._low_relevance_count(
            all_signals
        ) >= self.divergence_threshold:
            return ObserverJudgment(
                health=ObserverHealth.DIVERGING,
                reasoning="Runtime is producing repeated low-relevance or divergence signals.",
                actions=[ProcessAction(type="escalate_hitl", value="diverging")],
            )

        if self._error_count(all_signals) >= self.repeated_error_threshold:
            return ObserverJudgment(
                health=ObserverHealth.DEGRADED,
                reasoning="Runtime has emitted repeated error signals.",
                actions=[ProcessAction(type="escalate_hitl", value="repeated_errors")],
            )

        if self._is_stuck(all_signals, latest_signal):
            return ObserverJudgment(
                health=ObserverHealth.STUCK,
                reasoning="Runtime has not emitted recent progress or heartbeat signals.",
                actions=[ProcessAction(type="escalate_hitl", value="stuck")],
            )

        if latest_signal.signal_type in {
            ProgressSignalType.HEARTBEAT,
            ProgressSignalType.PROGRESS,
            ProgressSignalType.FINDING,
        }:
            return ObserverJudgment(
                health=ObserverHealth.HEALTHY,
                reasoning="Runtime emitted a recent progress, heartbeat, or finding signal.",
            )

        return None

    def _error_count(self, signals: list[ProgressSignal]) -> int:
        return sum(signal.signal_type == ProgressSignalType.ERROR for signal in signals[-5:])

    def _low_relevance_count(self, signals: list[ProgressSignal]) -> int:
        return sum(
            signal.signal_type == ProgressSignalType.FINDING
            and signal.payload.relevance_score is not None
            and signal.payload.relevance_score <= self.low_relevance_threshold
            for signal in signals[-5:]
        )

    def _has_explicit_divergence(self, signal: ProgressSignal) -> bool:
        return bool(
            signal.payload.data.get("diverging")
            or signal.payload.status == "diverging"
            or signal.payload.reason == "diverging"
        )

    def _is_stuck(
        self,
        all_signals: list[ProgressSignal],
        latest_signal: ProgressSignal,
    ) -> bool:
        if latest_signal.payload.status == "stuck" or latest_signal.payload.data.get(
            "missed_heartbeat"
        ):
            return True

        latest_progress = next(
            (
                signal
                for signal in reversed(all_signals)
                if signal.signal_type
                in {ProgressSignalType.HEARTBEAT, ProgressSignalType.PROGRESS}
            ),
            None,
        )
        if latest_progress is None:
            return False

        try:
            progress_time = datetime.fromisoformat(latest_progress.timestamp)
            latest_time = datetime.fromisoformat(latest_signal.timestamp)
        except ValueError:
            return False

        return (latest_time - progress_time).total_seconds() > self.stale_progress_seconds
