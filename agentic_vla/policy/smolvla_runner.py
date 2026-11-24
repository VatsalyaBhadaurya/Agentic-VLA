"""SmolVLA policy runner — the Executor's hands.

Wraps the LeRobot SmolVLA policy and owns the EXPLICIT, tested adapters:
    LIBERO obs  ->  LeRobot batch   (images + state + task string)
    policy action chunk  ->  env action vector

Mapping mismatches are BLOCKING errors, never silent reshapes (DESIGN.md §4).
The policy manages its own action-chunk queue internally: `select_action` returns one
action per call and is refilled every `n_action_steps` calls. Call `reset()` per episode.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

_REPORT = Path(__file__).resolve().parent.parent.parent / "interface_report.json"


def _load_report() -> dict[str, Any]:
    if _REPORT.exists():
        try:
            return json.loads(_REPORT.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


class SmolVLARunner:
    def __init__(
        self,
        checkpoint: str = "lerobot/smolvla_base",
        device: str = "cuda",
        dtype: str = "bfloat16",
        camera_map: dict[str, str] | None = None,
        flip_images: bool = True,   # robosuite offscreen frames are vertically flipped [TO-VERIFY]
    ):
        import torch
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.device = device
        self._torch = torch
        self._dtype = getattr(torch, dtype)
        self.flip_images = flip_images

        self.policy = SmolVLAPolicy.from_pretrained(checkpoint)
        self.policy.to(device)
        self.policy.eval()

        cfg = self.policy.config
        self.input_features = dict(getattr(cfg, "input_features", {}) or {})
        # camera keys the checkpoint expects, e.g. "observation.images.top"
        self.expected_image_keys = [k for k in self.input_features if "image" in k.lower()]
        self.expected_state_key = next(
            (k for k in self.input_features if "state" in k.lower()), "observation.state"
        )
        self.state_dim = self._expected_state_dim()

        # LIBERO cam -> policy image key. If not provided, derive from the report/heuristic.
        self.camera_map = camera_map or self._default_camera_map()

    # -- introspection helpers ---------------------------------------------------------
    def _expected_state_dim(self) -> int | None:
        feat = self.input_features.get(self.expected_state_key)
        shape = getattr(feat, "shape", None)
        if shape:
            return int(np.prod(shape))
        rep = _load_report().get("policy", {}).get("input_features", {})
        shp = rep.get("observation.state", {}).get("shape")
        return int(np.prod(shp)) if shp else None

    def _default_camera_map(self) -> dict[str, str]:
        """Best-effort LIBERO->policy camera mapping; verify against interface_report."""
        libero_cams = ["agentview_image", "robot0_eye_in_hand_image"]
        keys = self.expected_image_keys or ["observation.images.image"]
        mapping: dict[str, str] = {}
        for i, pk in enumerate(keys):
            if i < len(libero_cams):
                mapping[libero_cams[i]] = pk
        return mapping

    # -- episode lifecycle -------------------------------------------------------------
    def reset(self) -> None:
        """Clear the internal action-chunk queue. Call at the start of every episode."""
        self.policy.reset()

    # -- the adapters ------------------------------------------------------------------
    def _img_to_tensor(self, img: np.ndarray):
        arr = img
        if self.flip_images:
            arr = arr[::-1]  # undo robosuite vertical flip
        arr = np.ascontiguousarray(arr).astype(np.float32) / 255.0  # HWC [0,1]
        t = self._torch.from_numpy(arr).permute(2, 0, 1)            # CHW
        return t.unsqueeze(0).to(self.device)

    def _build_state(self, obs: dict[str, np.ndarray]):
        """Assemble proprio vector in a fixed order; fail loud on dim mismatch."""
        parts = []
        for k in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"):
            if k in obs:
                parts.append(np.asarray(obs[k], dtype=np.float32).reshape(-1))
        if not parts:
            raise KeyError(
                "No proprio keys found in observation; cannot build observation.state. "
                f"Got keys: {list(obs)}"
            )
        vec = np.concatenate(parts)
        if self.state_dim is not None and vec.shape[0] != self.state_dim:
            raise ValueError(
                f"Proprio dim {vec.shape[0]} != policy-expected {self.state_dim}. "
                "Fix the assembly order/content — do NOT pad silently. See DESIGN.md §4."
            )
        return self._torch.from_numpy(vec).unsqueeze(0).to(self.device)

    def build_batch(self, obs: dict[str, Any], instruction: str) -> dict[str, Any]:
        batch: dict[str, Any] = {}
        for libero_key, policy_key in self.camera_map.items():
            if libero_key not in obs:
                raise KeyError(
                    f"Expected camera {libero_key!r} missing from obs ({list(obs)}). "
                    "Camera map is wrong — validate against interface_report.json."
                )
            batch[policy_key] = self._img_to_tensor(obs[libero_key])
        batch[self.expected_state_key] = self._build_state(obs)
        batch["task"] = [instruction]
        return batch

    # -- inference ---------------------------------------------------------------------
    def select_action(self, obs: dict[str, Any], instruction: str) -> np.ndarray:
        """Return ONE action for the current step (policy buffers the chunk internally)."""
        batch = self.build_batch(obs, instruction)
        with self._torch.no_grad():
            action = self.policy.select_action(batch)
        a = action.squeeze(0).float().cpu().numpy()
        if not np.all(np.isfinite(a)):
            raise ValueError("Policy produced non-finite action — aborting (spec §9).")
        return a
