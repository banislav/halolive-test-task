from __future__ import annotations

from deep_agents.models import (
    ObserverHealth,
    ObserverJudgment,
    ProcessAssessment,
    ProcessJudgment,
    ProgressSignal,
    ProgressSignalPayload,
    ProgressSignalType,
)
from deep_agents.runtime import ObserverJudge, ProcessJudge, ProgressSignalBus


def build_signal(
    task_id: str = "T1",
    signal_type: ProgressSignalType = ProgressSignalType.PROGRESS,
    payload: ProgressSignalPayload | None = None,
    timestamp: str = "2026-01-01T00:00:00+00:00",
) -> ProgressSignal:
    return ProgressSignal(
        task_id=task_id,
        signal_type=signal_type,
        payload=payload or ProgressSignalPayload(status="running"),
        timestamp=timestamp,
    )


class RecordingProcessJudge:
    def __init__(self) -> None:
        self.calls: list[tuple[list[ProgressSignal], ProgressSignal]] = []

    def evaluate(
        self,
        signal_history: list[ProgressSignal],
        latest_signal: ProgressSignal,
    ) -> ProcessJudgment | None:
        self.calls.append((signal_history, latest_signal))
        return ProcessJudgment(
            task_id=latest_signal.task_id,
            assessment=ProcessAssessment.HEALTHY,
            reasoning="recorded",
        )


class RecordingObserverJudge:
    def __init__(self) -> None:
        self.calls: list[tuple[list[ProgressSignal], ProgressSignal]] = []

    def evaluate(
        self,
        all_signals: list[ProgressSignal],
        latest_signal: ProgressSignal,
    ) -> ObserverJudgment | None:
        self.calls.append((all_signals, latest_signal))
        return ObserverJudgment(health=ObserverHealth.HEALTHY, reasoning="recorded")


def test_progress_signal_bus_stores_signals_in_order_and_filters_by_task() -> None:
    bus = ProgressSignalBus()
    first = build_signal(task_id="T1")
    second = build_signal(task_id="T2")

    bus.publish(first)
    bus.publish(second)

    assert bus.signals() == [first, second]
    assert bus.signals(task_id="T1") == [first]
    assert bus.signals(task_id="missing") == []


def test_progress_signal_bus_calls_observer_judges_for_all_signals() -> None:
    bus = ProgressSignalBus()
    observer = RecordingObserverJudge()
    bus.subscribe_observer(observer)

    signal = build_signal()
    judgments = bus.publish(signal)

    assert len(observer.calls) == 1
    assert observer.calls[0][0] == [signal]
    assert judgments[0].reasoning == "recorded"


def test_progress_signal_bus_calls_only_matching_task_process_judges() -> None:
    bus = ProgressSignalBus()
    matching = RecordingProcessJudge()
    other = RecordingProcessJudge()
    bus.subscribe_process(matching, task_id="T1")
    bus.subscribe_process(other, task_id="T2")

    signal = build_signal(task_id="T1")
    judgments = bus.publish(signal)

    assert len(matching.calls) == 1
    assert matching.calls[0][0] == [signal]
    assert other.calls == []
    assert len(judgments) == 1


def test_progress_signal_bus_calls_global_process_judges_for_all_task_signals() -> None:
    bus = ProgressSignalBus()
    process = RecordingProcessJudge()
    bus.subscribe_process(process)

    bus.publish(build_signal(task_id="T1"))
    bus.publish(build_signal(task_id="T2"))

    assert len(process.calls) == 2


def test_process_judge_marks_actionable_findings_as_healthy_early_complete() -> None:
    judge = ProcessJudge()
    signal = build_signal(
        signal_type=ProgressSignalType.FINDING,
        payload=ProgressSignalPayload(actionable=True, relevance_score=0.9),
    )

    judgment = judge.evaluate([signal], signal)

    assert judgment is not None
    assert judgment.assessment == ProcessAssessment.HEALTHY
    assert judgment.actions[0].type == "mark_early_complete"


def test_process_judge_recommends_early_termination_for_low_relevance_findings() -> None:
    judge = ProcessJudge()
    signal = build_signal(
        signal_type=ProgressSignalType.FINDING,
        payload=ProgressSignalPayload(relevance_score=0.1),
    )

    judgment = judge.evaluate([signal], signal)

    assert judgment is not None
    assert judgment.assessment == ProcessAssessment.EARLY_TERMINATE
    assert judgment.actions[0].type == "terminate_task"


def test_process_judge_recommends_more_time_for_slow_progress_near_timeout() -> None:
    judge = ProcessJudge(timeout_pressure_seconds=30, slow_progress_percent=20)
    signal = build_signal(
        payload=ProgressSignalPayload(
            percent_complete=10,
            estimated_remaining_seconds=20,
        ),
    )

    judgment = judge.evaluate([signal], signal)

    assert judgment is not None
    assert judgment.assessment == ProcessAssessment.NEEDS_MORE_TIME
    assert judgment.actions[0].type == "adjust_timeout"


def test_process_judge_escalates_error_signals() -> None:
    judge = ProcessJudge()
    signal = build_signal(
        signal_type=ProgressSignalType.ERROR,
        payload=ProgressSignalPayload(error_type="worker_error"),
    )

    judgment = judge.evaluate([signal], signal)

    assert judgment is not None
    assert judgment.assessment == ProcessAssessment.ESCALATE_HITL
    assert judgment.actions[0].type == "escalate_hitl"


def test_observer_judge_reports_healthy_for_recent_progress() -> None:
    judge = ObserverJudge()
    signal = build_signal(signal_type=ProgressSignalType.HEARTBEAT)

    judgment = judge.evaluate([signal], signal)

    assert judgment is not None
    assert judgment.health == ObserverHealth.HEALTHY


def test_observer_judge_reports_degraded_for_repeated_errors() -> None:
    judge = ObserverJudge(repeated_error_threshold=2)
    signals = [
        build_signal(task_id="T1", signal_type=ProgressSignalType.ERROR),
        build_signal(task_id="T2", signal_type=ProgressSignalType.ERROR),
    ]

    judgment = judge.evaluate(signals, signals[-1])

    assert judgment is not None
    assert judgment.health == ObserverHealth.DEGRADED


def test_observer_judge_reports_stuck_when_progress_is_stale() -> None:
    judge = ObserverJudge(stale_progress_seconds=60)
    signals = [
        build_signal(timestamp="2026-01-01T00:00:00+00:00"),
        build_signal(
            signal_type=ProgressSignalType.ERROR,
            timestamp="2026-01-01T00:02:00+00:00",
        ),
    ]

    judgment = judge.evaluate(signals, signals[-1])

    assert judgment is not None
    assert judgment.health == ObserverHealth.STUCK


def test_observer_judge_reports_diverging_for_repeated_low_relevance_findings() -> None:
    judge = ObserverJudge(divergence_threshold=2)
    signals = [
        build_signal(
            signal_type=ProgressSignalType.FINDING,
            payload=ProgressSignalPayload(relevance_score=0.1),
        ),
        build_signal(
            signal_type=ProgressSignalType.FINDING,
            payload=ProgressSignalPayload(relevance_score=0.2),
        ),
    ]

    judgment = judge.evaluate(signals, signals[-1])

    assert judgment is not None
    assert judgment.health == ObserverHealth.DIVERGING
