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


# -- M3: LLM-backed planner/critic (fake LLM, no transformers needed) ------------------
class FakeLLM:
    residency = "none"

    def __init__(self, response):
        self.response = response
        self.calls = 0

    def generate(self, prompt, system=None):
        self.calls += 1
        return self.response

    def unload(self):
        pass


def test_llm_planner_parses_strict_json():
    from agentic_vla.agents.llm_backend import plan_with_llm
    llm = FakeLLM('Here you go: {"subtasks":[{"id":"s1","description":"open drawer",'
                  '"completion_condition":"drawer open"},{"id":"s2","description":"place",'
                  '"completion_condition":"in drawer"}]}')
    msg = plan_with_llm(llm, "put the block in the drawer", {"objects": []})
    validate_message(msg)
    assert len(msg.subtasks) == 2
    assert "LLM decomposition" in msg.reason


def test_llm_planner_falls_back_on_garbage():
    from agentic_vla.agents.llm_backend import plan_with_llm
    llm = FakeLLM("sorry I cannot produce JSON")
    msg = plan_with_llm(llm, "pick up the red block and place it in the drawer", None)
    validate_message(msg)
    assert len(msg.subtasks) >= 1                     # rules fallback kicked in


def test_critic_llm_diagnosis_rejects_invalid_category():
    from agentic_vla.agents.llm_backend import diagnose_with_llm
    llm = FakeLLM('{"failure_category":"NOT_A_REAL_CATEGORY",'
                  '"recommended_next_action":"regrasp","reason":"x"}')
    diag = diagnose_with_llm(llm, "no success", False)
    assert diag["failure_category"] == "NOT_A_REAL_CATEGORY"   # raw passthrough
    # the critic itself validates against the taxonomy:
    from agentic_vla.agents.critic import _VALID_CATEGORIES
    assert "NOT_A_REAL_CATEGORY" not in _VALID_CATEGORIES


# -- M4: episodic memory ---------------------------------------------------------------
def test_memory_store_and_retrieve(tmp_path):
    from agentic_vla.memory import EpisodicMemory
    mem = EpisodicMemory(str(tmp_path / "mem"))

    class _S:
        selected_skill = "full_task"

    mem.store(_S(), {
        "task_id": "t1", "instruction": "pick up the red block and place it in the drawer",
        "scene_graph": {"objects": [{"id": "red_block"}, {"id": "drawer"}]},
        "subtasks_completed": 3, "success": True, "failure_history": [],
        "policy_calls": 40, "duration_s": 12.0,
    })
    hint = mem.retrieve("pick up the red block and put it in the drawer",
                        {"objects": [{"id": "red_block"}, {"id": "drawer"}]}, k=3)
    assert hint is not None
    assert hint["advisory"] is True          # advisory only, never authoritative
    assert "full_task" in hint["suggested_skills"]


def test_memory_empty_returns_none(tmp_path):
    from agentic_vla.memory import EpisodicMemory
    mem = EpisodicMemory(str(tmp_path / "mem2"))
    assert mem.retrieve("anything", {"objects": []}) is None


def test_config_c_orchestrator_uses_memory(tmp_path):
    from agentic_vla.memory import EpisodicMemory
    mem = EpisodicMemory(str(tmp_path / "mem3"))
    env = FakeEnv("pick up the red block", success_after=2)
    orch = Orchestrator(env, FakeRunner(), chunk_steps=3, max_chunks=3,
                        retry_budget=2, memory=mem)
    rep = orch.run(seed=0)
    assert rep["success"] is True
    # episode was persisted and is retrievable afterwards
    assert mem.retrieve("pick up the red block", rep["scene_graph"]) is not None


# -- M5: benchmark metrics -------------------------------------------------------------
def _mk_report(seed, success, retries=0, failures=None, subtasks_planned=2, subtasks_completed=2):
    return {
        "seed": seed, "success": success, "duration_s": 10.0, "policy_calls": 20,
        "subtasks_planned": subtasks_planned, "subtasks_completed": subtasks_completed,
        "retries": retries, "replans": 0, "failure_history": failures or [],
    }


def test_aggregate_metrics():
    from agentic_vla.eval import aggregate
    reports = [
        _mk_report(0, True),
        _mk_report(1, False, retries=3,
                   failures=[{"category": "GRASP_FAILED"}, {"category": "GRASP_FAILED"}],
                   subtasks_completed=1),
        _mk_report(2, True, retries=1, failures=[{"category": "OBJECT_SLIPPED"}]),
    ]
    agg = aggregate(reports)
    assert agg["episodes"] == 3
    assert abs(agg["task_success_rate"] - 2 / 3) < 1e-9
    assert agg["total_failed_grasps"] == 2
    assert agg["total_retries"] == 4
    assert agg["failure_distribution"]["GRASP_FAILED"] == 2
    # one of two failing-recovery episodes recovered (seed 2 had a failure but succeeded)
    assert agg["recovery_success_rate"] == 0.5


def test_format_comparison_table():
    from agentic_vla.eval import aggregate, format_comparison
    a = aggregate([_mk_report(0, True), _mk_report(1, False)])
    b = aggregate([_mk_report(0, True), _mk_report(1, True)])
    table = format_comparison({"A": a, "B": b})
    assert "Task success rate" in table
    assert "A baseline" in table and "B agentic" in table
    assert "50.0%" in table and "100.0%" in table


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
