#!/usr/bin/env python
"""Config A — BASELINE VLA.   Instruction -> SmolVLA -> LIBERO.

No agents, no planning, no recovery. This establishes the measured baseline success
rate that Configs B and C must beat. Scoring uses LIBERO's official success predicate
only (a task is NOT 'done' just because the loop finished issuing actions).

    python scripts/01_run_baseline.py --suite libero_object --task 0 --episodes 5

Results (JSON) are written under runs/baseline_<suite>_<ts>/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agentic_vla.config import load_config          # noqa: E402
from agentic_vla.envs import LiberoEnv               # noqa: E402
from agentic_vla.policy import SmolVLARunner         # noqa: E402


def run_episode(env: LiberoEnv, runner: SmolVLARunner, seed: int, max_steps: int) -> dict:
    obs = env.reset(seed=seed)
    runner.reset()
    t0 = time.time()
    policy_calls = 0
    success = False
    steps = 0
    for steps in range(1, max_steps + 1):
        action = runner.select_action(obs, env.instruction)
        policy_calls += 1
        step = env.step(action)
        obs = step.obs
        if step.success:
            success = True
            break
        if step.done:
            break
    return {
        "seed": seed,
        "success": success,            # official LIBERO signal
        "steps": steps,
        "policy_calls": policy_calls,
        "duration_s": round(time.time() - t0, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default=None)
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    suite = args.suite or cfg.suite
    max_steps = args.max_steps or cfg.env.get("max_steps", 600)
    seeds = (cfg.env.get("seeds") or [0])
    seeds = (seeds * args.episodes)[: args.episodes]

    print(f"[Config A] suite={suite} task={args.task} episodes={args.episodes}")
    env = LiberoEnv(
        suite=suite,
        task_id=args.task,
        image_size=cfg.env.get("image_size", 256),
    )
    runner = SmolVLARunner(
        checkpoint=cfg.checkpoint,
        device=cfg.device,
        dtype=cfg.models["policy"].get("dtype", "bfloat16"),
    )
    print(f"  instruction: {env.instruction!r}")

    results = []
    for i, seed in enumerate(seeds):
        r = run_episode(env, runner, seed=seed, max_steps=max_steps)
        results.append(r)
        print(f"  ep {i+1}/{len(seeds)} seed={seed} "
              f"success={r['success']} steps={r['steps']} {r['duration_s']}s")
    env.close()

    sr = sum(r["success"] for r in results) / len(results)
    summary = {
        "config": "A",
        "suite": suite,
        "task": args.task,
        "episodes": len(results),
        "success_rate": sr,
        "avg_steps": sum(r["steps"] for r in results) / len(results),
        "avg_duration_s": sum(r["duration_s"] for r in results) / len(results),
        "episodes_detail": results,
    }

    out_dir = ROOT / "runs" / f"baseline_{suite}_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[Config A] success_rate = {sr:.2%}  ({out_dir/'results.json'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
