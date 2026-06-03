from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, StructuredTool, convert_runnable_to_tool
from pydantic import BaseModel, ValidationError

from deep_agents.models import (
    ProgressSignal,
    ProgressSignalPayload,
    ProgressSignalType,
    ToolCallRequest,
    ToolCallResult,
    ToolCallStatus,
    ToolDefinition,
    ToolSafetyLevel,
)
from deep_agents.models.base import JsonObject, utc_now
from deep_agents.runtime.memory import MemoryRecorder
from deep_agents.runtime.observability import ProgressSignalBus

ToolRuntime = BaseTool | Runnable[Any, Any] | Callable[..., Any]

_current_tool_attempt_id: ContextVar[str | None] = ContextVar(
    "current_tool_attempt_id",
    default=None,
)


@contextmanager
def tool_attempt_context(attempt_id: str | None) -> Iterator[None]:
    """Set the current task attempt id for explicit tool calls made inside a worker."""
    token = _current_tool_attempt_id.set(attempt_id)
    try:
        yield
    finally:
        _current_tool_attempt_id.reset(token)


class ToolRegistry:
    """Registry of tool definitions and LangChain tool implementations."""

    def __init__(self) -> None:
        self._definitions: dict[str, ToolDefinition] = {}
        self._tools: dict[str, BaseTool] = {}

    def register(self, definition: ToolDefinition, tool: ToolRuntime) -> None:
        """Register a tool implementation behind a stable definition."""
        langchain_tool = self._to_langchain_tool(definition, tool)
        self._definitions[definition.id] = self._definition_with_schema(definition, langchain_tool)
        self._tools[definition.id] = langchain_tool

    def definition(self, tool_id: str) -> ToolDefinition | None:
        """Return a registered tool definition by id."""
        return self._definitions.get(tool_id)

    def resolve(self, tool_id: str) -> BaseTool | None:
        """Return a registered LangChain tool implementation by id."""
        return self._tools.get(tool_id)

    def langchain_tools(self) -> list[BaseTool]:
        """Return registered tools for LangChain agent/tool binding."""
        return list(self._tools.values())

    def definitions(self) -> list[ToolDefinition]:
        """Return all definitions in registration order."""
        return list(self._definitions.values())

    def _to_langchain_tool(self, definition: ToolDefinition, tool: ToolRuntime) -> BaseTool:
        if isinstance(tool, BaseTool):
            return tool
        if isinstance(tool, Runnable):
            return convert_runnable_to_tool(
                tool,
                name=definition.id,
                description=definition.description or definition.name,
                arg_types=_arg_types_from_schema(definition.input_schema),
            )
        invoke = getattr(tool, "invoke", None)
        args_schema = getattr(tool, "args_schema", None)
        if callable(invoke):
            return StructuredTool.from_function(
                func=lambda **kwargs: invoke(kwargs),
                name=definition.id,
                description=definition.description or definition.name,
                args_schema=args_schema if isinstance(args_schema, type) else None,
                infer_schema=args_schema is None,
            )
        return StructuredTool.from_function(
            func=tool,
            name=definition.id,
            description=definition.description or definition.name,
            infer_schema=True,
        )

    def _definition_with_schema(
        self,
        definition: ToolDefinition,
        tool: BaseTool,
    ) -> ToolDefinition:
        args_schema = getattr(tool, "args_schema", None)
        if definition.input_schema or args_schema is None:
            return definition
        if not isinstance(args_schema, type) or not issubclass(args_schema, BaseModel):
            return definition
        schema = {
            name: _annotation_to_schema(field.annotation)
            for name, field in args_schema.model_fields.items()
        }
        return definition.model_copy(update={"input_schema": schema})


class ToolPolicy:
    """Deterministic permission, safety, and rate-limit policy for tool calls."""

    def __init__(
        self,
        *,
        allowed_tool_ids: list[str] | None = None,
        task_allowed_tools: dict[str, list[str]] | None = None,
        agent_allowed_tools: dict[str, list[str]] | None = None,
        allow_sensitive: bool = False,
        allow_destructive: bool = False,
        allow_hitl_required: bool = False,
        rate_limits: dict[str, int] | None = None,
    ) -> None:
        self.allowed_tool_ids = set(allowed_tool_ids or [])
        self.task_allowed_tools = {
            task_id: set(tool_ids) for task_id, tool_ids in (task_allowed_tools or {}).items()
        }
        self.agent_allowed_tools = {
            agent_id: set(tool_ids) for agent_id, tool_ids in (agent_allowed_tools or {}).items()
        }
        self.allow_sensitive = allow_sensitive
        self.allow_destructive = allow_destructive
        self.allow_hitl_required = allow_hitl_required
        self.rate_limits = rate_limits or {}
        self._call_counts: dict[str, int] = defaultdict(int)

    def check_permission(
        self,
        request: ToolCallRequest,
        definition: ToolDefinition,
    ) -> str | None:
        """Return a denial reason when the caller is not permitted to use a tool."""
        if self.allowed_tool_ids and request.tool_id not in self.allowed_tool_ids:
            return f"Tool {request.tool_id} is not in the allowed tool set."
        if request.task_id in self.task_allowed_tools:
            if request.tool_id not in self.task_allowed_tools[request.task_id]:
                return f"Tool {request.tool_id} is not allowed for task {request.task_id}."
        caller = request.caller_agent
        if caller is not None:
            agent_key = caller.agent_id or caller.name
            if agent_key in self.agent_allowed_tools:
                if request.tool_id not in self.agent_allowed_tools[agent_key]:
                    return f"Tool {request.tool_id} is not allowed for agent {agent_key}."
            if definition.allowed_agent_ids and caller.agent_id not in definition.allowed_agent_ids:
                return f"Agent {agent_key} is not allowed by tool definition."
            if definition.allowed_agent_names and caller.name not in definition.allowed_agent_names:
                return f"Agent {caller.name} is not allowed by tool definition."
            if definition.allowed_agent_types and caller.type not in definition.allowed_agent_types:
                return f"Agent type {caller.type} is not allowed by tool definition."
        if definition.allowed_task_ids and request.task_id not in definition.allowed_task_ids:
            return f"Task {request.task_id} is not allowed by tool definition."
        return None

    def check_safety(self, definition: ToolDefinition) -> str | None:
        """Return a block reason when the tool safety level is not currently allowed."""
        if definition.safety_level == ToolSafetyLevel.SAFE:
            return None
        if definition.safety_level == ToolSafetyLevel.SENSITIVE and self.allow_sensitive:
            return None
        if definition.safety_level == ToolSafetyLevel.DESTRUCTIVE and self.allow_destructive:
            return None
        if definition.safety_level == ToolSafetyLevel.HITL_REQUIRED and self.allow_hitl_required:
            return None
        return f"Tool safety level {definition.safety_level} is blocked by policy."

    def check_rate_limit(self, tool_id: str) -> str | None:
        """Return a rate-limit reason when a call would exceed the configured budget."""
        limit = self.rate_limits.get(tool_id)
        if limit is None:
            return None
        if self._call_counts[tool_id] >= limit:
            return f"Tool {tool_id} exceeded rate limit of {limit} call(s)."
        return None

    def record_call(self, tool_id: str) -> None:
        """Count one executable tool call for rate-limit tracking."""
        self._call_counts[tool_id] += 1


class ToolMiddlewareRunner:
    """Execute registered tools through the architecture middleware stack."""

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        policy: ToolPolicy | None = None,
        memory_recorder: MemoryRecorder | None = None,
        progress_bus: ProgressSignalBus | None = None,
        plan_id: str | None = None,
    ) -> None:
        self.registry = registry
        self.policy = policy or ToolPolicy()
        self.memory_recorder = memory_recorder
        self.progress_bus = progress_bus
        self.plan_id = plan_id

    def invoke(self, request: ToolCallRequest | dict[str, Any]) -> ToolCallResult:
        """Run one tool call through permission, validation, safety, execution, and capture."""
        resolved_request = self._resolve_request(request)
        definition = self.registry.definition(resolved_request.tool_id)
        tool = self.registry.resolve(resolved_request.tool_id)
        started_at = utc_now()
        if definition is None or tool is None:
            result = self._result(
                request=resolved_request,
                status=ToolCallStatus.DENIED,
                started_at=started_at,
                error_type="unknown_tool",
                error_message=f"Unknown tool id: {resolved_request.tool_id}",
            )
            self._capture(resolved_request, result)
            return result

        self._record_tool_call(resolved_request, definition, started_at)
        self._publish(
            resolved_request,
            "tool_started",
            {"tool_id": definition.id, "safety_level": definition.safety_level},
        )

        permission_reason = self.policy.check_permission(resolved_request, definition)
        if permission_reason is not None:
            result = self._blocked_result(
                resolved_request,
                ToolCallStatus.DENIED,
                started_at,
                "permission_denied",
                permission_reason,
            )
            self._capture(resolved_request, result)
            return result

        validation_error, validated_input = self._validate_input(
            resolved_request.input,
            definition,
            tool,
        )
        if validation_error is not None:
            result = self._blocked_result(
                resolved_request,
                ToolCallStatus.VALIDATION_FAILED,
                started_at,
                "validation_failed",
                validation_error,
            )
            self._capture(resolved_request, result)
            return result

        rate_limit_reason = self.policy.check_rate_limit(definition.id)
        if rate_limit_reason is not None:
            result = self._blocked_result(
                resolved_request,
                ToolCallStatus.RATE_LIMITED,
                started_at,
                "rate_limited",
                rate_limit_reason,
            )
            self._capture(resolved_request, result)
            return result

        safety_reason = self.policy.check_safety(definition)
        if safety_reason is not None:
            result = self._blocked_result(
                resolved_request,
                ToolCallStatus.SAFETY_BLOCKED,
                started_at,
                "safety_blocked",
                safety_reason,
            )
            self._capture(resolved_request, result)
            return result

        self.policy.record_call(definition.id)
        try:
            output = self._execute_tool(tool, validated_input)
        except Exception as exc:
            result = self._result(
                request=resolved_request,
                status=ToolCallStatus.FAILED,
                started_at=started_at,
                error_type=exc.__class__.__name__,
                error_message=str(exc),
            )
            self._capture(resolved_request, result)
            return result

        result = self._result(
            request=resolved_request,
            status=ToolCallStatus.SUCCEEDED,
            started_at=started_at,
            output=self._normalize_output(output),
        )
        self._capture(resolved_request, result)
        return result

    def _resolve_request(self, request: ToolCallRequest | dict[str, Any]) -> ToolCallRequest:
        if isinstance(request, ToolCallRequest):
            resolved = request
        else:
            resolved = ToolCallRequest(**request)
        if resolved.attempt_id is not None:
            return resolved
        attempt_id = _current_tool_attempt_id.get()
        if attempt_id is None:
            return resolved
        return resolved.model_copy(update={"attempt_id": attempt_id})

    def _validate_input(
        self,
        input_data: JsonObject,
        definition: ToolDefinition,
        tool: BaseTool,
    ) -> tuple[str | None, JsonObject]:
        if definition.input_schema:
            errors = _validate_simple_schema(input_data, definition.input_schema)
            if errors:
                return "; ".join(errors), input_data

        args_schema = tool.args_schema
        if isinstance(args_schema, type) and issubclass(args_schema, BaseModel):
            try:
                model = args_schema.model_validate(input_data)
            except ValidationError as exc:
                return str(exc), input_data
            return None, model.model_dump()

        if isinstance(args_schema, dict):
            errors = _validate_simple_schema(input_data, _schema_properties(args_schema))
            if errors:
                return "; ".join(errors), input_data
            return None, input_data

        return None, input_data

    def _execute_tool(self, tool: BaseTool, input_data: JsonObject) -> Any:
        return tool.invoke(input_data)

    def _blocked_result(
        self,
        request: ToolCallRequest,
        status: ToolCallStatus,
        started_at: Any,
        error_type: str,
        error_message: str,
    ) -> ToolCallResult:
        return self._result(
            request=request,
            status=status,
            started_at=started_at,
            error_type=error_type,
            error_message=error_message,
        )

    def _result(
        self,
        *,
        request: ToolCallRequest,
        status: ToolCallStatus,
        started_at: Any,
        output: JsonObject | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> ToolCallResult:
        completed_at = utc_now()
        output_data = output or {}
        return ToolCallResult(
            tool_id=request.tool_id,
            task_id=request.task_id,
            attempt_id=request.attempt_id,
            status=status,
            output=output_data,
            error_type=error_type,
            error_message=error_message,
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            duration_seconds=max((completed_at - started_at).total_seconds(), 0),
            metadata={
                "accounting": {
                    "input_bytes": _json_size(request.input),
                    "output_bytes": _json_size(output_data),
                    "input_tokens": request.metadata.get("input_tokens"),
                    "output_tokens": request.metadata.get("output_tokens"),
                },
                "request_metadata": request.metadata,
            },
        )

    def _record_tool_call(
        self,
        request: ToolCallRequest,
        definition: ToolDefinition,
        started_at: Any,
    ) -> None:
        if self.memory_recorder is None:
            return
        self.memory_recorder.record_tool_call(
            request=request,
            definition=definition,
            plan_id=self.plan_id,
            started_at=started_at.isoformat(),
        )

    def _capture(self, request: ToolCallRequest, result: ToolCallResult) -> None:
        if self.memory_recorder is not None:
            self.memory_recorder.record_tool_result(result, plan_id=self.plan_id)
        self._publish(
            request,
            f"tool_{result.status}",
            {
                "tool_id": result.tool_id,
                "status": result.status,
                "error_type": result.error_type,
                "error_message": result.error_message,
            },
            signal_type=ProgressSignalType.ERROR
            if result.status == ToolCallStatus.FAILED
            else ProgressSignalType.PROGRESS,
        )

    def _publish(
        self,
        request: ToolCallRequest,
        status: str,
        data: JsonObject,
        *,
        signal_type: ProgressSignalType = ProgressSignalType.PROGRESS,
    ) -> None:
        if self.progress_bus is None:
            return
        self.progress_bus.publish(
            ProgressSignal(
                task_id=request.task_id,
                signal_type=signal_type,
                payload=ProgressSignalPayload(status=status, data=data),
            )
        )

    def _normalize_output(self, output: Any) -> JsonObject:
        if isinstance(output, BaseModel):
            return output.model_dump()
        if isinstance(output, dict):
            return output
        return {"value": output}


def _validate_simple_schema(input_data: JsonObject, schema: JsonObject) -> list[str]:
    errors: list[str] = []
    for key, expected in schema.items():
        if key not in input_data:
            errors.append(f"Missing required input: {key}")
            continue
        if not _matches_schema_type(input_data[key], expected):
            errors.append(f"Input {key} does not match expected type {expected}")
    return errors


def _schema_properties(schema: JsonObject) -> JsonObject:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return schema
    return {
        key: value.get("type") if isinstance(value, dict) else value
        for key, value in properties.items()
    }


def _matches_schema_type(value: Any, expected: Any) -> bool:
    if isinstance(expected, type):
        return isinstance(value, expected)
    if not isinstance(expected, str):
        return True
    normalized = expected.lower().replace(" ", "")
    if normalized in {"string", "str"}:
        return isinstance(value, str)
    if normalized in {"integer", "int"}:
        return isinstance(value, int) and not isinstance(value, bool)
    if normalized in {"number", "float"}:
        return (isinstance(value, int | float)) and not isinstance(value, bool)
    if normalized in {"boolean", "bool"}:
        return isinstance(value, bool)
    if normalized in {"object", "dict"}:
        return isinstance(value, dict)
    if normalized.startswith("list") or normalized == "array":
        return isinstance(value, list)
    return True


def _arg_types_from_schema(schema: JsonObject) -> dict[str, type] | None:
    if not schema:
        return None
    arg_types: dict[str, type] = {}
    for key, expected in schema.items():
        arg_types[key] = _python_type_from_schema(expected)
    return arg_types


def _python_type_from_schema(expected: Any) -> type:
    if isinstance(expected, type):
        return expected
    if not isinstance(expected, str):
        return object
    normalized = expected.lower().replace(" ", "")
    if normalized in {"string", "str"}:
        return str
    if normalized in {"integer", "int"}:
        return int
    if normalized in {"number", "float"}:
        return float
    if normalized in {"boolean", "bool"}:
        return bool
    if normalized in {"object", "dict"}:
        return dict
    if normalized.startswith("list") or normalized == "array":
        return list
    return object


def _normalize_annotation(annotation: Any) -> str:
    if annotation is str:
        return "string"
    if annotation is int:
        return "int"
    if annotation is float:
        return "float"
    if annotation is bool:
        return "bool"
    if annotation is dict:
        return "object"
    if annotation is list:
        return "list"
    return "object"


def _annotation_to_schema(annotation: Any) -> str:
    origin = getattr(annotation, "__origin__", None)
    if origin is list:
        return "list"
    if origin is dict:
        return "object"
    return _normalize_annotation(annotation)


def _json_size(payload: Any) -> int:
    return len(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))
