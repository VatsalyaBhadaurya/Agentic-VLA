"""Planner agent (spec §4 AGENT 1).

Rules-first (M2). Decomposes the instruction into ordered subtasks with completion
conditions. It does NOT invent object positions/properties (spec §4). In M3 an LLM
backend replaces the heuristic for ambiguous/long-horizon instructions.

Important: the subtasks drive ORCHESTRATION and VERIFICATION, not the string handed to
SmolVLA — the executor uses the original instruction (DESIGN.md P1).
"""
from __future__ import annotations

import re

from ..schemas import PlannerMessage, SubtaskSpec


class PlannerAgent:
    name = "planner"

    def __init__(self, llm=None):
        self.llm = llm   # M3: optional reasoning LLM backend

    def plan(self, instruction: str, scene_graph: dict | None = None) -> PlannerMessage:
        if self.llm is not None:
            return self._plan_with_llm(instruction, scene_graph)
        return self._plan_rules(instruction)

    # -- rules-first decomposition -----------------------------------------------------
    def _plan_rules(self, instruction: str) -> PlannerMessage:
        instr = instruction.strip().lower()
        subtasks: list[SubtaskSpec] = []

        # Heuristic verb/structure detection for common LIBERO phrasings.
        has_open = bool(re.search(r"\bopen\b", instr))
        has_place_in = bool(re.search(r"\b(in|into|inside)\b.*\bdrawer\b", instr))
        is_pick_place = bool(re.search(r"\b(pick|put|place|move)\b", instr))

        n = 0
        def add(desc: str, cond: str):
            nonlocal n
            n += 1
            subtasks.append(SubtaskSpec(id=f"s{n}", description=desc, completion_condition=cond))

        if has_open and has_place_in:
            add("Open the drawer", "drawer is open")
            add("Pick up the target object", "object grasped")
            add("Place the object inside the drawer", "object inside drawer (LIBERO success)")
        elif is_pick_place:
            add("Locate and grasp the target object", "object grasped")
            add("Transport and place the object at the goal", "object at goal (LIBERO success)")
        else:
            # Unknown structure: single subtask = whole task via policy. Honest, not a guess.
            add(instruction.strip(), "LIBERO success predicate true")

        return PlannerMessage(
            status="PLANNED",
            reason=f"Rule-based decomposition into {len(subtasks)} subtask(s).",
            subtasks=subtasks,
            confidence=None,   # heuristic — confidence genuinely unknown
        )

    def _plan_with_llm(self, instruction: str, scene_graph: dict | None) -> PlannerMessage:
        from .llm_backend import plan_with_llm

        return plan_with_llm(self.llm, instruction, scene_graph)
