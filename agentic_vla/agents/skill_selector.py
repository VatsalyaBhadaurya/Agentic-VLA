"""Skill-selection agent (spec §4 AGENT 3).

Maps the next subtask to a SUPPORTED, executable skill via the registry. If the mapped
primitive is unsupported, it does NOT silently substitute: it falls back to the
executable `full_task` capability and SAYS SO, or returns SKILL_UNAVAILABLE when even
that is impossible. Rules-first; no LLM needed here.
"""
from __future__ import annotations

import re

from ..schemas import SkillSelectionMessage
from ..skills import SkillRegistry
from ..state import Subtask

_KEYWORD_SKILL = [
    (r"\bopen\b.*\bdrawer\b", "open_drawer"),
    (r"\bclose\b.*\bdrawer\b", "close_drawer"),
    (r"\bgrasp\b|\bgrip\b", "grasp"),
    (r"\breach\b|\bapproach\b", "reach"),
    (r"\bpush\b", "push"),
    (r"\bpull\b", "pull"),
    (r"\bpick\b|\bplace\b|\bput\b|\bmove\b", "pick_and_place"),
]


class SkillSelectorAgent:
    name = "skill_selector"

    def __init__(self, registry: SkillRegistry):
        self.registry = registry

    def select(self, subtask: Subtask) -> SkillSelectionMessage:
        requested = self._requested_skill(subtask.description)
        skill = self.registry.get(requested)

        if skill and skill.supported:
            return SkillSelectionMessage(
                status="SELECTED", reason=f"Supported skill {requested!r} selected.",
                selected_skill=requested, required_policy=skill.required_policy,
            )

        # Requested primitive is a label with no standalone controller yet.
        executable = self.registry.resolve(requested)   # -> full_task
        if executable is None:
            return SkillSelectionMessage(
                status="SKILL_UNAVAILABLE",
                reason=f"No supported skill for {requested!r} and no fallback available.",
                selected_skill=None, required_policy=None,
            )
        return SkillSelectionMessage(
            status="SELECTED_WITH_FALLBACK",
            reason=(
                f"Requested {requested!r} has no standalone controller; executing via "
                f"{executable.name!r} (SmolVLA). Label preserved for verification/recovery."
            ),
            selected_skill=executable.name,
            required_policy=executable.required_policy,
            confidence=None,
        )

    def _requested_skill(self, description: str) -> str:
        d = description.lower()
        for pattern, skill in _KEYWORD_SKILL:
            if re.search(pattern, d):
                return skill
        return "full_task"
