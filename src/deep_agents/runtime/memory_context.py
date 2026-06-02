from __future__ import annotations

from deep_agents.models import MemoryKind, MemoryQuery, MemoryRecord
from deep_agents.models.base import JsonObject


def build_memory_context(
    memory_store: object | None,
    *,
    plan_id: str | None = None,
    task_id: str | None = None,
    dependency_task_ids: list[str] | None = None,
    limit_per_kind: int = 8,
) -> JsonObject:
    """Return direct memory matches grouped by architecture memory kind."""
    if memory_store is None or not hasattr(memory_store, "query"):
        return {}

    dependency_ids = dependency_task_ids or []
    context: JsonObject = {}
    task_ids = [task for task in [task_id, *dependency_ids] if task is not None]

    working_records: list[MemoryRecord] = []
    for candidate_task_id in task_ids:
        working_records.extend(
            memory_store.query(
                MemoryQuery(
                    kinds=[MemoryKind.WORKING],
                    task_ids=[candidate_task_id],
                )
            )
        )
    _add_records(context, MemoryKind.WORKING, working_records, limit_per_kind)

    if plan_id is not None:
        session_records = memory_store.query(
            MemoryQuery(
                kinds=[MemoryKind.SESSION],
                plan_ids=[plan_id],
            )
        )
        _add_records(context, MemoryKind.SESSION, session_records, limit_per_kind)

    for kind in (
        MemoryKind.SEMANTIC,
        MemoryKind.EPISODIC,
        MemoryKind.PROCEDURAL,
        MemoryKind.SKILL,
    ):
        _add_records(
            context,
            kind,
            memory_store.query(MemoryQuery(kinds=[kind])),
            limit_per_kind,
        )

    return context


def _add_records(
    context: JsonObject,
    kind: MemoryKind,
    records: list[MemoryRecord],
    limit: int,
) -> None:
    seen: set[str] = set()
    projected: list[JsonObject] = []
    for record in records[-limit:]:
        if record.id in seen:
            continue
        projected.append(_project_record(record))
        seen.add(record.id)
        if len(projected) >= limit:
            break
    if projected:
        context[kind.value] = projected


def _project_record(record: MemoryRecord) -> JsonObject:
    return {
        "id": record.id,
        "kind": record.kind,
        "source": record.source,
        "tags": record.tags,
        "task_id": record.task_id,
        "plan_id": record.plan_id,
        "timestamp": record.timestamp,
        "payload": record.payload,
    }
