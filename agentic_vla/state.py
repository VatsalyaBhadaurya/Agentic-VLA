"""Shared task state — the single source of truth (spec §5).

Only the orchestrator mutates this. Agents read it and return recommendations. Every
mutation is appended to an immutable transition log so a run is fully reconstructable.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class Status(str, Enum):
    PLANNING = "PLANNING"
    PERCEIVING = "PERCEIVING"
    SELECTING = "SELECTING"
    EXECUTING = "EXECUTING"
    CRITIQUING = "CRITIQUING"
    RECOVERING = "RECOVERING"
    COMPLETE = "COMPLETE"
    ABORTED = "ABORTED"


@dataclass
class Subtask:
    id: str
    description: str
    completion_condition: str
    skill: str | None = None
    status: str = "pending"          # pending | active | done | failed


@dataclass
class TaskState:
    instruction: str
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: Status = Status.PLANNING
    current_subtask: Subtask | None = None
    completed_subtasks: list[Subtask] = field(default_factory=list)
    pending_subtasks: list[Subtask] = field(default_factory=list)
    scene_graph: dict[str, Any] = field(default_factory=dict)
    selected_skill: str | None = None
    execution_attempt: int = 0
    retry_count: int = 0
    failure_history: list[dict[str, Any]] = field(default_factory=list)
    last_observation_id: str | None = None
    verification_status: str = "UNCERTAIN"
    final_result: str | None = None
    transition_log: list[dict[str, Any]] = field(default_factory=list)

    # -- controlled mutation -----------------------------------------------------------
    def transition(self, new_status: Status, note: str = "", **fields: Any) -> None:
        prev = self.status
        self.status = new_status
        for k, v in fields.items():
            setattr(self, k, v)
        self.transition_log.append(
            {
                "t": round(time.time(), 3),
                "from": prev.value if isinstance(prev, Status) else prev,
                "to": new_status.value,
                "note": note,
                "retry_count": self.retry_count,
            }
        )

    def record_failure(self, category: str, detail: str) -> None:
        self.failure_history.append(
            {"t": round(time.time(), 3), "category": category, "detail": detail,
             "subtask": self.current_subtask.id if self.current_subtask else None}
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d
