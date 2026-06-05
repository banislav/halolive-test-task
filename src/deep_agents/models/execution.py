from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from deep_agents.models.agents import AgentAssignment, AgentKind
from deep_agents.models.base import DeepAgentsModel, JsonObject, TimestampedModel
from deep_agents.models.context import ArtifactRef, ContextBudget
from deep_agents.models.planning import AcceptanceCriterion, Risk
from deep_agents.models.skills import SkillAssignment


class RetryPolicy(DeepAgentsModel):
    max_retries: int = Field(default=2, ge=0)
    backoff: str = "exponential"
    on_exhaust: str = "escalate_to_replanner"


class LongRunningTaskConfig(DeepAgentsModel):
    heartbeat_interval_seconds: int = Field(default=30, gt=0)
    checkpoint_interval_seconds: int = Field(default=60, gt=0)
    progress_reporting: bool = True
    early_findings_enabled: bool = True
    timeout_seconds: int | None = Field(default=None, gt=0)
    resumable: bool = True
    max_memory_mb: float | None = Field(default=None, gt=0)
    max_cpu_time_seconds: int | None = Field(default=None, gt=0)
    max_elapsed_seconds: int | None = Field(default=None, gt=0)


class TaskInvocation(DeepAgentsModel):
    method: str = "async_dispatch"
    input: JsonObject = Field(default_factory=dict)
    input_schema: JsonObject = Field(default_factory=dict)
    expected_output_schema: JsonObject = Field(default_factory=dict)
    timeout_seconds: int = Field(default=120, gt=0)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    long_running: LongRunningTaskConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "timeout_seconds" not in data and "timeout" in data:
            data["timeout_seconds"] = data.pop("timeout")
        return data


class TaskResponsiveness(DeepAgentsModel):
    heartbeat_interval_seconds: int = Field(default=15, gt=0)
    progress_events: bool = True
    early_findings_enabled: bool = True


class TopologyPattern(StrEnum):
    SUBAGENTS = "subagents"
    HANDOFFS = "handoffs"
    ROUTER = "router"
    SKILLS = "skills"
    CUSTOM_WORKFLOW = "custom_workflow"


class HandoffStep(DeepAgentsModel):
    id: str
    name: str
    assigned_to: AgentAssignment
    instruction: str
    input_schema: JsonObject = Field(default_factory=dict)
    expected_output_schema: JsonObject = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "instruction" not in data:
            data["instruction"] = data.pop("description", data.get("name", ""))
        if "assigned_to" not in data:
            data["assigned_to"] = _assignment_from_value(
                data.get("agent") or data.get("assigned_agent") or data.get("name"),
                skills=data.pop("skills", []),
            )
        return data


class TaskCard(DeepAgentsModel):
    id: str
    name: str
    wave: int = Field(ge=0)
    blocked_by: list[str] = Field(default_factory=list)
    blocks: list[str] = Field(default_factory=list)
    assigned_to: AgentAssignment
    invocation: TaskInvocation = Field(default_factory=TaskInvocation)
    acceptance_criteria: list[AcceptanceCriterion] = Field(default_factory=list)
    responsiveness: TaskResponsiveness = Field(default_factory=TaskResponsiveness)
    context_budget: ContextBudget = Field(default_factory=ContextBudget)
    input_artifacts: list[ArtifactRef] = Field(default_factory=list)
    estimated_complexity: str = "medium"
    risks: list[Risk] = Field(default_factory=list)
    handoff_chain: list[HandoffStep] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        description = data.pop("description", None)
        tools = data.pop("tools", data.pop("tools_needed", []))
        skills = data.pop("skills", data.pop("skills_needed", []))
        if "wave" not in data:
            data["wave"] = 0
        if data.get("handoff_chain") is None:
            data["handoff_chain"] = []
        if "assigned_to" not in data:
            data["assigned_to"] = _assignment_from_value(
                data.pop("agent", data.pop("assigned_agent", None)),
                skills=skills,
            )
        if "invocation" not in data:
            invocation_input: JsonObject = {}
            if description is not None:
                invocation_input["description"] = description
            if tools:
                invocation_input["tools"] = list(tools)
            data["invocation"] = {"input": invocation_input}
        criteria = data.get("acceptance_criteria")
        if isinstance(criteria, list):
            data["acceptance_criteria"] = [
                {"description": criterion} if isinstance(criterion, str) else criterion
                for criterion in criteria
            ]
        return data


class Wave(DeepAgentsModel):
    index: int = Field(ge=0)
    name: str | None = None
    blocked_by: list[str] = Field(default_factory=list)
    task_ids: list[str] = Field(default_factory=list)
    topology: TopologyPattern = TopologyPattern.SUBAGENTS

    @model_validator(mode="before")
    @classmethod
    def normalize_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        wave_id = data.pop("id", None)
        if "index" not in data:
            data["index"] = _index_from_wave_id(wave_id)
        data.pop("description", None)
        return data


class DependencyGraph(DeepAgentsModel):
    blocked_by: dict[str, list[str]] = Field(default_factory=dict)
    blocks: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        if "blocked_by" in value or "blocks" in value:
            data = dict(value)
            if "blocked_by" in data:
                data["blocked_by"] = _string_list_map(data["blocked_by"])
            if "blocks" in data:
                data["blocks"] = _string_list_map(data["blocks"])
            return data
        return {"blocked_by": _string_list_map(value)}


class ExecutionPlan(TimestampedModel):
    id: str
    objective: str
    waves: list[Wave] = Field(default_factory=list)
    task_cards: list[TaskCard] = Field(default_factory=list)
    dependency_graph: DependencyGraph = Field(default_factory=DependencyGraph)
    data_flow: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "task_cards" not in data:
            data["task_cards"] = data.pop("tasks", data.pop("taskCards", []))
        data["waves"] = _normalize_execution_waves(data.get("waves", []))
        dependency_graph = data.get("dependency_graph", {})
        blocked_by = (
            dependency_graph.get("blocked_by", {})
            if isinstance(dependency_graph, dict) and "blocked_by" in dependency_graph
            else dependency_graph
        )
        task_to_wave = _task_to_wave_index(data["waves"])
        data["task_cards"] = _normalize_execution_tasks(
            data.get("task_cards", []),
            task_to_wave=task_to_wave,
            blocked_by=_string_list_map(blocked_by) if isinstance(blocked_by, dict) else {},
        )
        if isinstance(data.get("dependency_graph"), dict):
            data["dependency_graph"] = DependencyGraph.model_validate(data["dependency_graph"])
        if isinstance(data.get("data_flow"), dict):
            data["data_flow"] = _normalize_data_flow(data["data_flow"])
        return data

    @model_validator(mode="after")
    def task_ids_must_be_unique_and_referenced(self) -> ExecutionPlan:
        task_ids = [task.id for task in self.task_cards]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("task card ids must be unique")

        known = set(task_ids)
        referenced = {task_id for wave in self.waves for task_id in wave.task_ids}
        missing = referenced - known
        if missing:
            raise ValueError(f"wave references unknown task ids: {sorted(missing)}")
        return self


def _assignment_from_value(value: Any, *, skills: Any) -> dict[str, Any]:
    if isinstance(value, AgentAssignment):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        data = dict(value)
        data.setdefault("type", AgentKind.WORKER)
        data.setdefault("name", "Worker")
        if "skills" not in data:
            data["skills"] = _skill_assignments(skills)
        return data
    return {
        "type": AgentKind.WORKER,
        "name": str(value or "Worker"),
        "skills": _skill_assignments(skills),
    }


def _skill_assignments(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        value = [value] if value else []
    assignments: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, SkillAssignment):
            assignments.append(item.model_dump(mode="json"))
        elif isinstance(item, dict):
            assignments.append(dict(item))
        elif isinstance(item, str):
            assignments.append({"id": item})
    return assignments


def _index_from_wave_id(value: Any) -> int:
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, str):
        digits = "".join(character for character in value if character.isdigit())
        if digits:
            return max(int(digits) - 1, 0)
    return 0


def _normalize_execution_waves(value: Any) -> list[Any]:
    if not isinstance(value, list):
        return []
    return [dict(wave) if isinstance(wave, dict) else wave for wave in value]


def _task_to_wave_index(waves: list[Any]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for fallback_index, wave in enumerate(waves):
        if not isinstance(wave, dict):
            continue
        index = wave.get("index", _index_from_wave_id(wave.get("id", fallback_index)))
        for task_id in wave.get("task_ids", []):
            mapping[str(task_id)] = int(index)
    return mapping


def _normalize_execution_tasks(
    value: Any,
    *,
    task_to_wave: dict[str, int],
    blocked_by: dict[str, list[str]],
) -> list[Any]:
    if not isinstance(value, list):
        return []
    tasks: list[Any] = []
    for task in value:
        if not isinstance(task, dict):
            tasks.append(task)
            continue
        data = dict(task)
        task_id = str(data.get("id", ""))
        if task_id in task_to_wave:
            data.setdefault("wave", task_to_wave[task_id])
        if task_id in blocked_by:
            data.setdefault("blocked_by", blocked_by[task_id])
        tasks.append(data)
    return tasks


def _string_list_map(value: dict[Any, Any]) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {}
    for key, item in value.items():
        if isinstance(item, list):
            normalized[str(key)] = [str(entry) for entry in item]
        elif item is None:
            normalized[str(key)] = []
        else:
            normalized[str(key)] = [str(item)]
    return normalized


def _normalize_data_flow(value: dict[Any, Any]) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {}
    for key, item in value.items():
        if isinstance(item, list):
            normalized[str(key)] = [str(entry) for entry in item]
        elif isinstance(item, dict):
            entries: list[str] = []
            for flow_key in ("consumes", "produces"):
                flow_value = item.get(flow_key, [])
                if isinstance(flow_value, list):
                    entries.extend(str(entry) for entry in flow_value)
                elif flow_value is not None:
                    entries.append(str(flow_value))
            normalized[str(key)] = entries
        elif item is None:
            normalized[str(key)] = []
        else:
            normalized[str(key)] = [str(item)]
    return normalized
