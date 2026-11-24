"""Tests for the M2 rules-first agentic layer — no torch/lerobot/libero required.

Drives the full orchestration loop with a fake env + fake policy runner so the state
machine, schema validation, skill registry, and recovery loop are exercised end-to-end.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agentic_vla.agents import (  # noqa: E402
    CriticAgent, PerceptionAgent, PlannerAgent, SkillSelectorAgent,
)
from agentic_vla.orchestrator import Orchestrator  # noqa: E402
from agentic_vla.schemas import validate_message  # noqa: E402
from agentic_vla.skills import SkillRegistry  # noqa: E402
from agentic_vla.state import Status, TaskState  # noqa: E402


# -- fakes -----------------------------------------------------------------------------
class _FakePolicy:
    name_or_path = "fake/smolvla"


class FakeRunner:
    def __init__(self):
        self.policy = _FakePolicy()

    def reset(self):
        pass

    def select_action(self, obs, instruction):
        return np.zeros(7, dtype=np.float32)


class FakeEnv:
    """Succeeds after `success_after` total steps; optionally never succeeds."""

    def __init__(self, instruction, success_after=5):
        self.instruction = instruction
        self.success_after = success_after
        self._steps = 0

    def reset(self, seed=None):
        self._steps = 0
        return {"agentview_image": np.zeros((8, 8, 3), np.uint8)}

    def step(self, action):
        from agentic_vla.envs.libero_env import EnvStep
        self._steps += 1
        succ = self._steps >= self.success_after
        return EnvStep(obs={"agentview_image": np.zeros((8, 8, 3), np.uint8)},
                       reward=1.0 if succ else 0.0, done=succ, success=succ, info={})

    def check_success(self, done=False, info=None):
        return self._steps >= self.success_after

    def oracle_state(self):
        return {"object-state": [0.0, 0.0, 0.0]}

    def images(self):
        return {"agentview_image": np.zeros((8, 8, 3), np.uint8)}

    def close(self):
        pass


# -- unit tests ------------------------------------------------------------------------
def test_planner_pick_place_decomposition():
    msg = PlannerAgent().plan("pick up the red block and place it in the drawer")
    validate_message(msg)
    assert len(msg.subtasks) >= 2
    assert all(s.completion_condition for s in msg.subtasks)


def test_perception_never_fabricates_positions():
    msg = PerceptionAgent().perceive(
        "pick up the red block", {"agentview_image": np.zeros((4, 4, 3))}
    )
    validate_message(msg)
    assert any("block" in o.id for o in msg.objects)
    for o in msg.objects:
        assert o.position is None       # honest unknown, not fabricated
        assert o.confidence is None


def test_registry_unsupported_primitive_falls_back_not_silent():
    reg = SkillRegistry()
    assert reg.is_supported("full_task")
    assert not reg.is_supported("open_drawer")       # declared but not standalone-executable
    assert reg.resolve("open_drawer").name == "full_task"


def test_skill_selector_reports_fallback_explicitly():
    sel = SkillSelectorAgent(SkillRegistry())
    from agentic_vla.state import Subtask
    msg = sel.select(Subtask(id="s1", description="open the drawer", completion_condition="x"))
    validate_message(msg)
    assert msg.selected_skill == "full_task"
    assert msg.status == "SELECTED_WITH_FALLBACK"    # not silently pretending open_drawer ran


def test_state_transition_log():
    st = TaskState(instruction="x")
    st.transition(Status.PERCEIVING, note="go")
    assert st.status == Status.PERCEIVING
    assert st.transition_log[-1]["to"] == "PERCEIVING"


def test_orchestrator_success_path():
    env = FakeEnv("pick up the red block and place it in the drawer", success_after=3)
    orch = Orchestrator(env, FakeRunner(), chunk_steps=5, max_chunks=5, retry_budget=3)
    rep = orch.run(seed=0)
    assert rep["success"] is True
    assert rep["policy_calls"] >= 1


def test_orchestrator_failure_and_recovery_budget():
    env = FakeEnv("pick up the red block", success_after=10_000)  # never succeeds
    orch = Orchestrator(env, FakeRunner(), chunk_steps=2, max_chunks=2, retry_budget=3)
    rep = orch.run(seed=0)
    assert rep["success"] is False
    assert rep["retries"] >= 1                        # recovery was attempted, then gave up
    assert len(rep["failure_history"]) >= 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
