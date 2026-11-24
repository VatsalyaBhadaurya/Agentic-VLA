"""Perception agent (spec §4 AGENT 2).

Builds a structured scene graph from observations. HARD RULE: positions and confidences
originate from actual perception or explicit simulator data — NEVER fabricated (spec §4).

M2 (rules-first): without a detector wired in, it reports the objects referenced by the
instruction with position=None and confidence=None (honest "unknown"), records which
camera views are available, and flags ambiguity. V1-a plugs SmolVLA's VLM backbone here
in a later pass; a dedicated detector remains optional.
"""
from __future__ import annotations

import re
from typing import Any

from ..schemas import PerceptionMessage, SceneObject, SceneRelation

_COLORS = ("red", "blue", "green", "yellow", "black", "white", "orange", "purple", "pink")
_NOUNS = ("block", "cube", "bowl", "plate", "mug", "cup", "drawer", "box", "can",
          "bottle", "cabinet", "butter", "milk", "cream", "ketchup", "bbq", "sauce")


class PerceptionAgent:
    name = "perception"

    def __init__(self, vlm=None, confidence_threshold: float = 0.4):
        self.vlm = vlm
        self.confidence_threshold = confidence_threshold

    def perceive(self, instruction: str, images: dict[str, Any] | None) -> PerceptionMessage:
        images = images or {}
        objects = self._objects_from_instruction(instruction)
        relationships: list[SceneRelation] = []

        # Spatial relations hinted by the instruction (relation only — no coordinates).
        if re.search(r"\b(in|into|inside)\b", instruction.lower()):
            ids = [o.id for o in objects]
            if len(ids) >= 2:
                relationships.append(
                    SceneRelation(subject=ids[0], relation="target_container", object=ids[-1])
                )

        available_views = list(images.keys())
        uncertain = len(objects) == 0 or not available_views
        return PerceptionMessage(
            status="UNCERTAIN" if uncertain else "SCENE_BUILT",
            reason=(
                f"Parsed {len(objects)} referenced object(s); views={available_views}. "
                "No metric positions estimated (no detector wired) — reported as unknown."
            ),
            objects=objects,
            relationships=relationships,
            evidence=[{"available_camera_views": available_views}],
            confidence=None,
        )

    def _objects_from_instruction(self, instruction: str) -> list[SceneObject]:
        instr = instruction.lower()
        objects: list[SceneObject] = []
        seen: set[str] = set()
        for noun in _NOUNS:
            if re.search(rf"\b{noun}\b", instr):
                color = next((c for c in _COLORS if re.search(rf"\b{c}\b.*\b{noun}\b", instr)), None)
                oid = f"{color}_{noun}" if color else noun
                if oid in seen:
                    continue
                seen.add(oid)
                attrs = {"color": color} if color else {}
                objects.append(
                    SceneObject(
                        id=oid, **{"class": noun}, attributes=attrs,
                        position=None, confidence=None,   # unknown, never fabricated
                    )
                )
        return objects
