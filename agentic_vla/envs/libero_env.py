"""Thin, honest wrapper around a LIBERO task environment.

Responsibilities (DESIGN.md §3):
  - reset / step with a stable interface
  - expose the ORIGINAL language instruction (needed by SmolVLA, DESIGN.md P1)
  - expose the OFFICIAL success predicate (the only thing that scores a task)
  - expose ground-truth object state SEPARATELY and clearly labeled oracle-assisted,
    so it can never leak into a vision-only performance claim (spec §9)

No interface detail is assumed silently: camera/image/action specifics come from the
installed env at runtime, and should match interface_report.json (see scripts/00).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class EnvStep:
    obs: dict[str, Any]
    reward: float
    done: bool
    success: bool          # official LIBERO predicate (vision-agnostic scoring signal)
    info: dict[str, Any]


class LiberoEnv:
    """Wraps a single LIBERO task from a benchmark suite."""

    def __init__(
        self,
        suite: str = "libero_object",
        task_id: int = 0,
        image_size: int = 256,
        camera_names: list[str] | None = None,
    ):
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        benchmark_dict = benchmark.get_benchmark_dict()
        if suite not in benchmark_dict:
            raise ValueError(f"Unknown suite {suite!r}; available: {list(benchmark_dict)}")

        self.suite = suite
        self.task_id = task_id
        self._task_suite = benchmark_dict[suite]()
        self.num_tasks = self._task_suite.n_tasks
        self._task = self._task_suite.get_task(task_id)

        self.instruction: str = getattr(self._task, "language", "") or ""
        bddl = os.path.join(
            get_libero_path("bddl_files"),
            self._task.problem_folder,
            self._task.bddl_file,
        )

        env_kwargs: dict[str, Any] = dict(
            bddl_file_name=bddl,
            camera_heights=image_size,
            camera_widths=image_size,
        )
        if camera_names:
            env_kwargs["camera_names"] = camera_names
        self._env = OffScreenRenderEnv(**env_kwargs)
        self._last_obs: dict[str, Any] | None = None
        self.action_dim = self._infer_action_dim()

    # -- lifecycle ---------------------------------------------------------------------
    def _infer_action_dim(self) -> int:
        try:
            return len(self._env.env.action_spec[0])
        except Exception:
            return int(getattr(self._env, "action_dim", 7))

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self._env.seed(seed)
        obs = self._env.reset()
        self._last_obs = obs
        return obs

    def step(self, action: np.ndarray) -> EnvStep:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        ret = self._env.step(action)
        if len(ret) == 4:
            obs, reward, done, info = ret
        else:  # gymnasium 5-tuple
            obs, reward, terminated, truncated, info = ret
            done = bool(terminated or truncated)
        self._last_obs = obs
        return EnvStep(
            obs=obs,
            reward=float(reward),
            done=bool(done),
            success=self.check_success(done=done, info=info),
            info=info if isinstance(info, dict) else {},
        )

    def close(self) -> None:
        try:
            self._env.close()
        except Exception:
            pass

    # -- the OFFICIAL success signal (scoring) -----------------------------------------
    def check_success(self, done: bool = False, info: dict | None = None) -> bool:
        """Official LIBERO success predicate. The ONLY signal used for benchmark scoring."""
        for cand in ("check_success", "_check_success"):
            for target in (self._env, getattr(self._env, "env", None)):
                fn = getattr(target, cand, None)
                if callable(fn):
                    try:
                        return bool(fn())
                    except Exception:
                        continue
        if isinstance(info, dict) and "success" in info:
            return bool(info["success"])
        return bool(done)

    # -- ORACLE diagnostics (never mixed into vision-only claims) -----------------------
    def oracle_state(self) -> dict[str, Any]:
        """Ground-truth low-dim object/robot state. LABELLED oracle-assisted (spec §9).

        For internal diagnostics / critic cross-checks ONLY. Callers must tag any metric
        derived from this as oracle-assisted.
        """
        obs = self._last_obs or {}
        return {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in obs.items()
            if k.endswith("-state") or k.endswith("_state") or k in ("object-state",)
        }

    # -- convenience accessors ----------------------------------------------------------
    def images(self) -> dict[str, np.ndarray]:
        obs = self._last_obs or {}
        return {k: v for k, v in obs.items() if k.endswith("_image")}

    def proprio(self) -> dict[str, np.ndarray]:
        obs = self._last_obs or {}
        return {
            k: v
            for k, v in obs.items()
            if any(t in k for t in ("eef", "gripper", "joint")) and not k.endswith("_image")
        }
