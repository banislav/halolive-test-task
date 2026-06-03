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


def _load_example(name: str) -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
