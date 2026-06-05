from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def test_async_runtime_session_example_builds_two_task_plan() -> None:
    module = _load_example("run_async_runtime_session")

    plan = module.build_plan()

    assert plan.id == "EP-async-session"
    assert plan.objective == module.OBJECTIVE
    assert [wave.task_ids for wave in plan.waves] == [["T1"], ["T2"]]
    assert [task.id for task in plan.task_cards] == ["T1", "T2"]
    assert plan.task_cards[1].blocked_by == ["T1"]


def test_prompt_to_async_runtime_example_builds_prompt_input_defaults() -> None:
    module = _load_example("run_prompt_to_async_runtime")

    planner_input = module.build_prompt_input()

    assert planner_input.objective == module.RAW_PROMPT
    assert planner_input.constraints == module.DEFAULT_CONSTRAINTS
    assert planner_input.available_tools == module.DEFAULT_TOOLS
    assert planner_input.available_skills == module.DEFAULT_SKILLS
    assert planner_input.context == module.DEFAULT_CONTEXT


def _load_example(name: str) -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
