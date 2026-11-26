#!/usr/bin/env python
"""Config B/C — the AGENTIC layer driving REAL SmolVLA on MuJoCo.

Unlike scripts/01 (baseline) and lerobot-eval, this routes execution through the full
multi-agent orchestrator (planner -> perception -> skill-selector -> executor -> critic,
with recovery/retry and optional episodic memory). The executor drives SmolVLA through
LeRobot's validated processor pipeline (agentic_vla/policy/lerobot_backend.py).

    python scripts/03_run_agentic_mujoco.py \
        --policy-path <snapshot_or_repo> --suite libero_object --task 0 \
        --episodes 1 --config B --chunk-steps 50 --max-chunks 8

Scoring = official LIBERO success predicate (surfaced by the critic).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agentic_vla  # noqa: F401  (env bootstrap: USE_TF=0, MUJOCO_GL fix)
from agentic_vla.orchestrator import Orchestrator  # noqa: E402
from agentic_vla.policy.lerobot_backend import build_agentic_backend  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy-path", required=True, help="finetuned SmolVLA snapshot or HF repo id")
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--config", choices=["B", "C"], default="B")
    ap.add_argument("--chunk-steps", type=int, default=50)
    ap.add_argument("--max-chunks", type=int, default=8)
    ap.add_argument("--retry-budget", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--record", default=None, help="path to write an mp4 of episode 0")
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    print(f"[Config {args.config}] AGENTIC -> SmolVLA -> MuJoCo | suite={args.suite} "
          f"task={args.task} episodes={args.episodes}")
    print(f"  building LeRobot backend from: {args.policy_path}")
    env, runner, backend = build_agentic_backend(
        args.policy_path, suite=args.suite, task_id=args.task, device=args.device
    )
    print(f"  instruction: {env.instruction!r}")

    memory = None
    if args.config == "C":
        from agentic_vla.memory import EpisodicMemory
        memory = EpisodicMemory("memory_store")

    from agentic_vla.agents.planner import SingleTaskPlanner

    orch = Orchestrator(
        env, runner,
        retry_budget=args.retry_budget,
        chunk_steps=args.chunk_steps,
        max_chunks=args.max_chunks,
        use_oracle_diagnostics=False,     # vision-only scoring; no oracle
        memory=memory,
        planner=SingleTaskPlanner(),      # one end-to-end policy executes the whole task
    )

    if args.record:
        backend.record = True

    reports = []
    for i in range(args.episodes):
        t0 = time.time()
        if args.record and i == 0:
            backend.frames = []            # capture only episode 0
        rep = orch.run(seed=i)
        reports.append(rep)
        print(f"  ep {i+1}/{args.episodes}: success={rep['success']} "
              f"subtasks={rep['subtasks_completed']} calls={rep['policy_calls']} "
              f"retries={rep['retries']} replans={rep['replans']} "
              f"{time.time()-t0:.1f}s")
        # show the agentic trace for the first episode
        if i == 0:
            print("  --- agentic plan ---")
            print("  scene objects:",
                  [o.get("id") for o in rep["scene_graph"].get("objects", [])])
            print("  transitions:",
                  " -> ".join(t["to"] for t in rep["transition_log"][:12]),
                  "..." if len(rep["transition_log"]) > 12 else "")
    if args.record and backend.frames:
        import numpy as np
        from lerobot.utils.io_utils import write_video
        out_path = Path(args.record)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_video(str(out_path), np.stack(backend.frames), fps=args.fps)
        print(f"  video: {out_path}  ({len(backend.frames)} frames @ {args.fps}fps)")

    env.close()

    sr = sum(r["success"] for r in reports) / len(reports)
    summary = {
        "config": args.config, "policy": args.policy_path, "suite": args.suite,
        "task": args.task, "episodes": len(reports), "success_rate": sr,
        "scoring": "official LIBERO success predicate (vision-only)",
        "reports": reports,
    }
    out_dir = ROOT / "runs" / f"agentic{args.config}_mujoco_{args.suite}_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[Config {args.config}] success_rate = {sr:.1%}  ({out_dir/'results.json'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
