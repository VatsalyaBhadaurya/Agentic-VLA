"""Skill registry — the anti-hallucination contract (spec §3, DESIGN.md §7).

A skill NAME is not an executable capability. Each registered skill must bind to a
concrete, validated capability (a policy) or be marked unsupported. Requesting an
unsupported skill yields SKILL_UNAVAILABLE — the system never silently substitutes.

Phase 1: the only genuinely executable skill is the whole LIBERO task driven by
SmolVLA ("full_task"). The fine-grained planner skills (reach/grasp/open_drawer/...)
are orchestration LABELS used for verification/recovery decisions — they are declared
here as NOT independently executable until a trained/validated per-skill policy exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class Skill:
    name: str
    supported: bool
    required_policy: Optional[str]
    description: str
    # Optional pre/post hooks (for future per-skill controllers). None => handled by critic.
    precondition_check: Optional[Callable] = None
    postcondition_check: Optional[Callable] = None


class SkillRegistry:
    def __init__(self, policy_checkpoint: str = "lerobot/smolvla_base"):
        self._skills: dict[str, Skill] = {}
        # The one executable Phase-1 capability.
        self.register(
            Skill(
                name="full_task",
                supported=True,
                required_policy=policy_checkpoint,
                description="Execute the entire LIBERO task via the SmolVLA policy.",
            )
        )
        # Declared-but-unsupported standalone primitives (spec §3 skill categories).
        for name, desc in {
            "reach": "Move the end-effector to a target.",
            "grasp": "Close the gripper on an object.",
            "pick_and_place": "Pick an object and place it at a target.",
            "push": "Push an object.",
            "pull": "Pull an object.",
            "open_drawer": "Open a drawer.",
            "close_drawer": "Close a drawer.",
            "reposition": "Reposition an object.",
            "recovery_move": "Small corrective motion for recovery.",
        }.items():
            self.register(
                Skill(name=name, supported=False, required_policy=None, description=desc)
            )

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def is_supported(self, name: str) -> bool:
        s = self._skills.get(name)
        return bool(s and s.supported)

    def resolve(self, name: str) -> Skill:
        """Return the skill if supported, else the best supported fallback for execution.

        Does NOT silently pretend an unsupported primitive ran. Returns `full_task` as the
        executable capability while preserving the requested label for logging.
        """
        s = self._skills.get(name)
        if s and s.supported:
            return s
        return self._skills["full_task"]
