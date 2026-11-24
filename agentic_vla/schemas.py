"""Validated agent message schemas (spec §11).

Every agent returns one of these. The orchestrator REJECTS malformed/incomplete
messages (fail loud). `confidence=None` means genuinely unknown — never fabricated.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# -- failure taxonomy (spec §7) --------------------------------------------------------
FailureCategory = Literal[
    "OBJECT_NOT_DETECTED",
    "OBJECT_POSE_UNCERTAIN",
    "GRASP_FAILED",
    "OBJECT_SLIPPED",
    "PLACEMENT_FAILED",
    "COLLISION_OR_OBSTRUCTION",
    "SKILL_UNAVAILABLE",
    "POLICY_EXECUTION_ERROR",
    "TASK_STATE_MISMATCH",
    "VERIFICATION_UNCERTAIN",
]


class AgentMessage(BaseModel):
    """Base envelope every agent returns."""

    agent: Literal["planner", "perception", "skill_selector", "executor", "critic"]
    status: str
    reason: str
    evidence: list[Any] = Field(default_factory=list)
    recommended_next_action: Optional[str] = None
    confidence: Optional[float] = None   # None = unknown; do NOT fabricate


# -- planner ---------------------------------------------------------------------------
class SubtaskSpec(BaseModel):
    id: str
    description: str
    completion_condition: str


class PlannerMessage(AgentMessage):
    agent: Literal["planner"] = "planner"
    subtasks: list[SubtaskSpec]


# -- perception ------------------------------------------------------------------------
class SceneObject(BaseModel):
    id: str
    cls: str = Field(alias="class")
    attributes: dict[str, Any] = Field(default_factory=dict)
    position: Optional[list[float]] = None    # None unless truly perceived (no fabrication)
    confidence: Optional[float] = None

    model_config = {"populate_by_name": True}


class SceneRelation(BaseModel):
    subject: str
    relation: str
    object: str


class PerceptionMessage(AgentMessage):
    agent: Literal["perception"] = "perception"
    objects: list[SceneObject] = Field(default_factory=list)
    relationships: list[SceneRelation] = Field(default_factory=list)


# -- skill selector --------------------------------------------------------------------
class SkillSelectionMessage(AgentMessage):
    agent: Literal["skill_selector"] = "skill_selector"
    selected_skill: Optional[str] = None       # None => SKILL_UNAVAILABLE
    required_policy: Optional[str] = None


# -- executor --------------------------------------------------------------------------
ExecStatus = Literal["CHUNK_DONE", "DONE", "POLICY_EXECUTION_ERROR"]


class ExecutorMessage(AgentMessage):
    agent: Literal["executor"] = "executor"
    status: ExecStatus
    steps_executed: int = 0
    policy_calls: int = 0
    env_success: bool = False                  # official signal observed during the chunk


# -- critic ----------------------------------------------------------------------------
CriticStatus = Literal[
    "SUCCESS", "PARTIAL_SUCCESS", "FAILURE", "UNCERTAIN", "RECOVERY_REQUIRED"
]


class CriticMessage(AgentMessage):
    agent: Literal["critic"] = "critic"
    status: CriticStatus
    failure_category: Optional[FailureCategory] = None
    oracle_assisted: bool = False              # True if any evidence came from oracle state


def validate_message(msg: BaseModel) -> BaseModel:
    """Re-validate an agent message; raise on malformed (fail loud, spec §11)."""
    return msg.model_validate(msg.model_dump(by_alias=True))
