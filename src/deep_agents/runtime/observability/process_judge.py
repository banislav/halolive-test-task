from __future__ import annotations

from deep_agents.models import (
    ProcessAction,
    ProcessAssessment,
    ProcessJudgment,
    ProgressSignal,
    ProgressSignalType,
)


class ProcessJudge:
    """Rule-based task-local judge for progress signal histories."""

    def __init__(
        self,
        *,
        actionable_relevance_threshold: float = 0.75,
        low_relevance_threshold: float = 0.25,
        timeout_pressure_seconds: int = 30,
        slow_progress_percent: float = 25,
    ) -> None:
        self.actionable_relevance_threshold = actionable_relevance_threshold
        self.low_relevance_threshold = low_relevance_threshold
        self.timeout_pressure_seconds = timeout_pressure_seconds
        self.slow_progress_percent = slow_progress_percent

    def evaluate(
        self,
        signal_history: list[ProgressSignal],
        latest_signal: ProgressSignal,
    ) -> ProcessJudgment | None:
        """Return a process judgment for the latest task signal when a rule fires."""
        payload = latest_signal.payload

        if latest_signal.signal_type in {
            ProgressSignalType.ERROR,
            ProgressSignalType.ESCALATION,
        }:
            return ProcessJudgment(
                task_id=latest_signal.task_id,
                assessment=ProcessAssessment.ESCALATE_HITL,
                reasoning="Task emitted an error or escalation signal.",
                actions=[
                    ProcessAction(
                        type="escalate_hitl",
                        value=payload.reason or payload.error_type or payload.detail,
                    )
                ],
            )

        if latest_signal.signal_type == ProgressSignalType.FINDING:
            relevance = payload.relevance_score
            if relevance is not None and relevance <= self.low_relevance_threshold:
                return ProcessJudgment(
                    task_id=latest_signal.task_id,
                    assessment=ProcessAssessment.EARLY_TERMINATE,
                    reasoning="Latest finding has low relevance to the assigned task.",
                    actions=[ProcessAction(type="terminate_task", value="low_relevance")],
                )

            if payload.actionable and (
                relevance is None or relevance >= self.actionable_relevance_threshold
            ):
                return ProcessJudgment(
                    task_id=latest_signal.task_id,
                    assessment=ProcessAssessment.HEALTHY,
                    reasoning="Latest finding is actionable and sufficiently relevant.",
                    actions=[ProcessAction(type="mark_early_complete", value=True)],
                )

        if latest_signal.signal_type == ProgressSignalType.PROGRESS:
            remaining = payload.estimated_remaining_seconds
            percent = payload.percent_complete
            if (
                remaining is not None
                and remaining <= self.timeout_pressure_seconds
                and (percent is None or percent <= self.slow_progress_percent)
            ):
                return ProcessJudgment(
                    task_id=latest_signal.task_id,
                    assessment=ProcessAssessment.NEEDS_MORE_TIME,
                    reasoning="Task reports low completion with little estimated time remaining.",
                    actions=[
                        ProcessAction(
                            type="adjust_timeout",
                            value=max(self.timeout_pressure_seconds, remaining * 2),
                        )
                    ],
                )

        if latest_signal.signal_type == ProgressSignalType.HEARTBEAT and signal_history:
            return ProcessJudgment(
                task_id=latest_signal.task_id,
                assessment=ProcessAssessment.HEALTHY,
                reasoning="Task emitted a heartbeat signal.",
            )

        return None
