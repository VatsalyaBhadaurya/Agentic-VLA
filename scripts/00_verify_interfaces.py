#!/usr/bin/env python
"""Interface verification — resolves every [TO-VERIFY] in DESIGN.md.

This script does NOT assume any action format, observation layout, control frequency,
or policy input contract. It introspects the *actually installed* LIBERO environment
and the SmolVLA checkpoint, then writes `interface_report.json`.

Run this before relying on any interface in the rest of the codebase.

    python scripts/00_verify_interfaces.py --suite libero_object --task 0

Exit code 0 only if BOTH the env and the policy were introspected successfully.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

REPORT_PATH = Path(__file__).resolve().parent.parent / "interface_report.json"


def _shape_of(x: Any):
    try:
        import numpy as np

        if isinstance(x, np.ndarray):
            return list(x.shape), str(x.dtype)
    except Exception:
        pass
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return list(x.shape), str(x.dtype)
    except Exception:
        pass
    return None, type(x).__name__


# --------------------------------------------------------------------------------------
# LIBERO environment introspection
# --------------------------------------------------------------------------------------
def verify_env(suite: str, task_id: int, image_size: int) -> dict[str, Any]:
    out: dict[str, Any] = {"ok": False, "suite": suite, "task_id": task_id}
    try:
        import os

        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        benchmark_dict = benchmark.get_benchmark_dict()
        if suite not in benchmark_dict:
            out["error"] = f"suite '{suite}' not in {list(benchmark_dict)}"
            return out
        task_suite = benchmark_dict[suite]()
        out["num_tasks"] = task_suite.n_tasks
        task = task_suite.get_task(task_id)
        out["task_name"] = getattr(task, "name", None)
        out["language_instruction"] = getattr(task, "language", None)

        bddl = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        out["bddl_file"] = bddl

        env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=image_size,
            camera_widths=image_size,
        )
        env.seed(0)
        obs = env.reset()

        # --- observation contract (real keys + shapes) ---
        obs_contract = {}
        for k, v in obs.items():
            shape, dtype = _shape_of(v)
            obs_contract[k] = {"shape": shape, "dtype": dtype}
        out["observation_keys"] = obs_contract
        out["image_keys"] = [k for k in obs if k.endswith("_image")]
        out["state_keys"] = [
            k for k in obs if ("eef" in k or "gripper" in k or "joint" in k)
        ]

        # --- action contract ---
        try:
            import numpy as np

            low = getattr(env.env.action_spec[0], "tolist", lambda: None)()
            high = getattr(env.env.action_spec[1], "tolist", lambda: None)()
            action_dim = len(env.env.action_spec[0])
        except Exception:
            action_dim = getattr(env, "action_dim", None)
            low = high = None
        out["action_dim"] = action_dim
        out["action_low"] = low
        out["action_high"] = high

        # --- one real step to confirm step signature + success predicate ---
        import numpy as np

        zero = np.zeros(action_dim if action_dim else 7, dtype=np.float32)
        step_ret = env.step(zero)
        out["step_return_len"] = len(step_ret)
        if len(step_ret) == 4:
            _, reward, done, info = step_ret
        else:
            _, reward, done, _trunc, info = step_ret
        out["step_return"] = {
            "reward_type": type(reward).__name__,
            "done_type": type(done).__name__,
            "info_keys": list(info.keys()) if isinstance(info, dict) else None,
        }
        # success predicate — try the known candidates without assuming one exists
        success_signal = None
        for cand in ("check_success", "_check_success"):
            fn = getattr(env, cand, None) or getattr(getattr(env, "env", None), cand, None)
            if callable(fn):
                try:
                    success_signal = {"source": cand, "value": bool(fn())}
                    break
                except Exception:
                    continue
        if success_signal is None:
            success_signal = {"source": "done_flag", "value": bool(done)}
        out["success_signal"] = success_signal

        env.close()
        out["ok"] = True
    except ImportError as e:
        out["error"] = f"LIBERO not importable: {e}. Install it (see README M0)."
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()
    return out


# --------------------------------------------------------------------------------------
# SmolVLA policy introspection
# --------------------------------------------------------------------------------------
def verify_policy(checkpoint: str) -> dict[str, Any]:
    out: dict[str, Any] = {"ok": False, "checkpoint": checkpoint}
    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        policy = SmolVLAPolicy.from_pretrained(checkpoint)
        cfg = policy.config

        def _features(feats):
            d = {}
            for name, feat in (feats or {}).items():
                d[name] = {
                    "type": getattr(getattr(feat, "type", None), "name", str(getattr(feat, "type", None))),
                    "shape": list(getattr(feat, "shape", []) or []),
                }
            return d

        out["input_features"] = _features(getattr(cfg, "input_features", {}))
        out["output_features"] = _features(getattr(cfg, "output_features", {}))
        out["n_action_steps"] = getattr(cfg, "n_action_steps", None)
        out["chunk_size"] = getattr(cfg, "chunk_size", None)
        out["n_obs_steps"] = getattr(cfg, "n_obs_steps", None)
        out["normalization_mapping"] = {
            k: getattr(v, "name", str(v))
            for k, v in (getattr(cfg, "normalization_mapping", {}) or {}).items()
        }
        # camera names the checkpoint expects (image input features)
        out["expected_image_inputs"] = [
            k for k in out["input_features"] if "image" in k.lower()
        ]
        out["expected_state_input"] = [
            k for k in out["input_features"] if "state" in k.lower()
        ]
        # param count / dtype (VRAM sanity)
        try:
            import torch

            n_params = sum(p.numel() for p in policy.parameters())
            out["num_params_millions"] = round(n_params / 1e6, 1)
            out["param_dtype"] = str(next(policy.parameters()).dtype)
        except Exception:
            pass
        out["ok"] = True
    except ImportError as e:
        out["error"] = f"LeRobot/SmolVLA not importable: {e}. pip install lerobot[smolvla]."
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()
    return out


def verify_runtime() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        import torch

        out["torch"] = torch.__version__
        out["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            out["vram_gb"] = round(props.total_memory / 1e9, 2)
    except Exception as e:
        out["torch_error"] = str(e)
    try:
        import mujoco

        out["mujoco"] = mujoco.__version__
    except Exception as e:
        out["mujoco_error"] = str(e)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--checkpoint", default="lerobot/smolvla_base")
    args = ap.parse_args()

    report = {
        "runtime": verify_runtime(),
        "environment": verify_env(args.suite, args.task, args.image_size),
        "policy": verify_policy(args.checkpoint),
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nInterface report written to: {REPORT_PATH}")

    env_ok = report["environment"].get("ok")
    pol_ok = report["policy"].get("ok")
    if env_ok and pol_ok:
        print("\n[OK] Both environment and policy interfaces verified.")
        print("     Transcribe these real values into config/env.yaml + config/models.yaml,")
        print("     replacing every [TO-VERIFY] tag.")
        return 0
    print("\n[INCOMPLETE] One or more interfaces could not be verified:")
    if not env_ok:
        print("   env:", report["environment"].get("error"))
    if not pol_ok:
        print("   policy:", report["policy"].get("error"))
    return 1


if __name__ == "__main__":
    sys.exit(main())
