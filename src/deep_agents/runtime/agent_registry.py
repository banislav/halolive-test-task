from __future__ import annotations

from typing import Any

from langchain_core.runnables import Runnable

from deep_agents.models import AgentAssignment, AgentKind, AgentProfile


class AgentRegistry:
    """Resolve architecture agent assignments to LangChain runnables."""

    def __init__(self) -> None:
        self._profiles: dict[str, AgentProfile] = {}
        self._runnables: dict[str, Runnable[Any, Any]] = {}
        self._name_index: dict[str, str] = {}
        self._type_index: dict[AgentKind, str] = {}

    def register(self, profile: AgentProfile, runnable: Runnable[Any, Any]) -> None:
        """Register one runnable agent by id, name, and type fallback."""
        self._profiles[profile.id] = profile
        self._runnables[profile.id] = runnable
        self._name_index[profile.name] = profile.id
        self._type_index.setdefault(profile.type, profile.id)

    def resolve(self, assignment: AgentAssignment) -> Runnable[Any, Any] | None:
        """Resolve an assignment by agent id, name, then agent type."""
        candidate_ids = [
            assignment.agent_id,
            self._name_index.get(assignment.name),
            self._type_index.get(assignment.type),
        ]
        for candidate_id in candidate_ids:
            if candidate_id and candidate_id in self._runnables:
                return self._runnables[candidate_id]
        return None

    def profile_for(self, assignment: AgentAssignment) -> AgentProfile | None:
        """Return registry metadata for an assignment when available."""
        candidate_ids = [
            assignment.agent_id,
            self._name_index.get(assignment.name),
            self._type_index.get(assignment.type),
        ]
        for candidate_id in candidate_ids:
            if candidate_id and candidate_id in self._profiles:
                return self._profiles[candidate_id]
        return None
