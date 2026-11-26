"""LeRobot-backed execution backend for the agentic loop (real SmolVLA on MuJoCo).

This is the bridge that makes Config B/C actually run the policy. It reuses LeRobot's
*validated* pipeline end-to-end — env construction, the 4 processor stages
(env-pre -> policy-pre -> policy -> policy-post -> env-post), observation preprocessing,
and task injection — rather than re-implementing normalization by hand (which the system
spec forbids: "never silently reshape / preserve the policy's expected preprocessing").

It exposes two thin adapters, `BackendEnv` and `BackendRunner`, that present exactly the
interfaces the Orchestrator/ExecutorAgent/CriticAgent already expect:
    env:    reset(seed) / step(action)->EnvStep / check_success() / images() /
            oracle_state() / instruction
    runner: reset() / select_action(obs, instruction) / policy

Both adapters share ONE backend instance, so `runner.select_action()` reads the backend's
current observation and `env.step()` advances it — preserving OBSERVE->INFER->ACT->VERIFY.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..envs.libero_env import EnvStep


class LeRobotMuJoCoBackend:
    """Owns the real LeRobot env + policy + processors for one (suite, task)."""

    def __init__(
        self,
        policy_path: str,
        suite: str = "libero_object",
        task_id: int = 0,
        device: str = "cuda",
    ):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.factory import make_env, make_env_pre_post_processors
        from lerobot.envs.utils import add_envs_task, preprocess_observation
        from lerobot.policies.factory import make_policy, make_pre_post_processors

        self._torch = torch
        self._preprocess_observation = preprocess_observation
        self._add_envs_task = add_envs_task

        # --- environment (single, non-vectorized rollout: n_envs=1) ---
        from lerobot.envs.factory import make_env_config

        env_cfg = make_env_config("libero", task=suite, task_ids=[task_id])
        envs = make_env(env_cfg, n_envs=1)
        self.env = envs[suite][task_id]           # gym vector env, batch dim 1
        self.env_cfg = env_cfg

        # --- policy + processors (loaded from the finetuned checkpoint) ---
        policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
        policy_cfg.pretrained_path = policy_path
        policy_cfg.device = device
        self.policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
        self.policy.eval()

        preprocessor_overrides = {
            "device_processor": {"device": str(self.policy.config.device)},
        }
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=policy_path,
            preprocessor_overrides=preprocessor_overrides,
        )
        self.env_pre, self.env_post = make_env_pre_post_processors(
            env_cfg=env_cfg, policy_cfg=policy_cfg
        )

        self.device = device
        self._raw_obs: dict[str, Any] | None = None
        self._last_success = False
        self.record = False
        self.frames: list[np.ndarray] = []
        # instruction comes straight from the LIBERO task (what SmolVLA was trained on)
        try:
            self.instruction = self.env.call("task_description")[0]
        except Exception:
            self.instruction = ""

    # -- lifecycle ---------------------------------------------------------------------
    def reset(self, seed: int = 0) -> dict[str, Any]:
        self.policy.reset()
        obs, _info = self.env.reset(seed=[seed])
        self._raw_obs = obs
        self._last_success = False
        return obs

    def select_action(self) -> np.ndarray:
        """Full LeRobot inference pipeline on the current observation -> env action."""
        from lerobot.utils.constants import ACTION

        obs = self._preprocess_observation(self._raw_obs)
        obs = self._add_envs_task(self.env, obs)
        obs = self.env_pre(obs)
        obs = self.preprocessor(obs)
        with self._torch.inference_mode():
            action = self.policy.select_action(obs)
        action = self.postprocessor(action)
        action = self.env_post({ACTION: action})[ACTION]
        a = action.to("cpu").numpy()
        if a.ndim == 1:
            a = a[None, :]                        # ensure (batch=1, action_dim)
        if not np.all(np.isfinite(a)):
            raise ValueError("Policy produced non-finite action (spec safety stop).")
        return a

    def step(self, action: np.ndarray) -> EnvStep:
        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 1:
            action = action[None, :]
        if self.record:
            self._grab_frame()
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._raw_obs = obs
        success = self._extract_success(info)
        self._last_success = self._last_success or success
        done = bool(np.any(terminated) or np.any(truncated))
        return EnvStep(
            obs=obs, reward=float(np.asarray(reward).reshape(-1)[0]),
            done=done, success=success, info=info if isinstance(info, dict) else {},
        )

    # -- helpers -----------------------------------------------------------------------
    @staticmethod
    def _extract_success(info: dict) -> bool:
        # LeRobot vector env stores per-env success under final_info on termination.
        if not isinstance(info, dict):
            return False
        fi = info.get("final_info")
        if fi is not None:
            for item in (fi if isinstance(fi, (list, tuple)) else [fi]):
                if isinstance(item, dict) and item.get("is_success"):
                    return True
        v = info.get("is_success")
        return bool(np.any(v)) if v is not None else False

    def _grab_frame(self) -> None:
        """Capture the agentview RGB frame for optional video recording."""
        try:
            frames = self.env.call("render")
            f = frames[0] if isinstance(frames, (list, tuple)) else frames
            if f is not None:
                self.frames.append(np.asarray(f, dtype=np.uint8))
        except Exception:
            pass

    def check_success(self) -> bool:
        return self._last_success

    def images(self) -> dict[str, np.ndarray]:
        """Best-effort camera views for the perception agent (advisory only)."""
        raw = self._raw_obs or {}
        out: dict[str, np.ndarray] = {}
        pixels = raw.get("pixels") if isinstance(raw, dict) else None
        if isinstance(pixels, dict):
            for k, v in pixels.items():
                out[k] = np.asarray(v)
        return out

    def oracle_state(self) -> dict[str, Any]:
        # diagnostics only; the agentic loop scores via check_success(), not this
        return {}


# ---- thin adapters presenting the Orchestrator's expected interfaces -----------------
class BackendEnv:
    """Presents the LiberoEnv-like interface backed by LeRobotMuJoCoBackend."""

    def __init__(self, backend: LeRobotMuJoCoBackend):
        self._b = backend
        self.instruction = backend.instruction

    def reset(self, seed: int = 0):
        return self._b.reset(seed=seed)

    def step(self, action) -> EnvStep:
        return self._b.step(action)

    def check_success(self, done: bool = False, info: dict | None = None) -> bool:
        return self._b.check_success()

    def images(self):
        return self._b.images()

    def oracle_state(self):
        return self._b.oracle_state()

    def close(self):
        try:
            self._b.env.close()
        except Exception:
            pass


class BackendRunner:
    """Presents the SmolVLARunner-like interface backed by the same backend."""

    def __init__(self, backend: LeRobotMuJoCoBackend):
        self._b = backend
        self.policy = backend.policy

    def reset(self):
        # policy.reset() is handled by backend.reset(); nothing extra needed here
        pass

    def select_action(self, obs=None, instruction: str | None = None) -> np.ndarray:
        # obs/instruction are informational; the backend holds the authoritative obs
        return self._b.select_action()


def build_agentic_backend(policy_path: str, suite: str, task_id: int, device: str = "cuda"):
    """Convenience: returns (env_adapter, runner_adapter) sharing one backend."""
    backend = LeRobotMuJoCoBackend(policy_path, suite=suite, task_id=task_id, device=device)
    return BackendEnv(backend), BackendRunner(backend), backend
