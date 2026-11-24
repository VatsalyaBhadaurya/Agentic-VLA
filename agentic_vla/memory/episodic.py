"""Episodic memory (spec §8, Config C).

Local JSONL store, one record per episode. Used to retrieve similar prior tasks and
bias planning/recovery. Two hard rules are enforced here:

  1. Retrieval NEVER overrides current visual evidence — `retrieve()` returns a *hint*,
     and the orchestrator treats it as advisory only (spec §8).
  2. Storing episodes does NOT train or adapt SmolVLA. This is experience logging, not
     model improvement (spec §8). No weights are touched.

Similarity is a lightweight lexical score (token Jaccard over the instruction + object
overlap). It is intentionally simple and offline — no embedding model is loaded, keeping
VRAM free for SmolVLA.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class EpisodicMemory:
    def __init__(self, store_dir: str = "memory_store"):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.store_dir / "episodes.jsonl"

    # -- write -------------------------------------------------------------------------
    def store(self, state, report: dict[str, Any]) -> None:
        """Append one episode record (spec §8 fields)."""
        objs = [o.get("id") for o in (report.get("scene_graph", {}).get("objects") or [])]
        successful_recoveries = [
            f for f in report.get("failure_history", [])
        ] if report.get("success") else []
        record = {
            "t": round(time.time(), 3),
            "task_id": report.get("task_id"),
            "instruction": report.get("instruction"),
            "suite_task": report.get("suite_task"),
            "scene_objects": objs,
            "subtasks_completed": report.get("subtasks_completed"),
            "selected_skill": getattr(state, "selected_skill", None),
            "success": bool(report.get("success")),
            "failure_history": report.get("failure_history", []),
            "successful_recoveries": successful_recoveries,
            "policy_calls": report.get("policy_calls"),
            "duration_s": report.get("duration_s"),
        }
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def _all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        return out

    # -- read (advisory only) ----------------------------------------------------------
    def retrieve(self, instruction: str, scene_graph: dict | None, k: int = 3) -> dict[str, Any] | None:
        """Return an ADVISORY hint from similar prior episodes, or None.

        The hint suggests strategies that worked and failure categories to avoid. It must
        NOT override current perception — callers use it only to bias planning/recovery.
        """
        episodes = self._all()
        if not episodes:
            return None

        q_tokens = _tokens(instruction)
        q_objs = {o.get("id") for o in (scene_graph or {}).get("objects", [])}

        scored = []
        for ep in episodes:
            s = _jaccard(q_tokens, _tokens(ep.get("instruction", "")))
            if q_objs:
                s += 0.5 * _jaccard(q_objs, set(ep.get("scene_objects") or []))
            scored.append((s, ep))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [ep for s, ep in scored[:k] if s > 0.0]
        if not top:
            return None

        suggested = []
        avoid = set()
        for ep in top:
            if ep.get("success") and ep.get("selected_skill"):
                suggested.append(ep["selected_skill"])
            for f in ep.get("failure_history", []):
                if not ep.get("success"):
                    avoid.add(f.get("category"))

        return {
            "advisory": True,            # NEVER authoritative over current perception
            "num_similar": len(top),
            "suggested_skills": list(dict.fromkeys(suggested)),
            "avoid_failure_categories": sorted(c for c in avoid if c),
            "best_match_instruction": top[0].get("instruction"),
            "best_match_success": top[0].get("success"),
        }
