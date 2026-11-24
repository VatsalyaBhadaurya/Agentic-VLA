#!/usr/bin/env python
"""Config B / C — AGENTIC VLA.   Instruction -> multi-agent orchestration -> SmolVLA.

  B: multi-agent orchestration (planner/perception/skill-selector/executor/critic)
  C: B + episodic memory (retrieval biases planning/recovery; never overrides perception)

    python scripts/02_run_agentic.py --suite libero_object --episodes 5 --config B

Scoring uses LIBERO's official success predicate only. Results -> runs/agentic_*/.
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
from agentic_vla.orchestrator import Orchestrator    # noqa: E402
from agentic_vla.policy import SmolVLARunner         # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default=None)
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--config", choices=["B", "C"], default="B")
    ap.add_argument("--no-llm", action="store_true", help="force rules-first agents")
    args = ap.parse_args()

    cfg = load_config()
    suite = args.suite or cfg.suite
    seeds = (cfg.env.get("seeds") or [0])
    seeds = (seeds * args.episodes)[: args.episodes]

    env = LiberoEnv(suite=suite, task_id=args.task, image_size=cfg.env.get("image_size", 256))
    runner = SmolVLARunner(
        checkpoint=cfg.checkpoint, device=cfg.device,
        dtype=cfg.models["policy"].get("dtype", "bfloat16"),
    )

    # Optional reasoning LLM (M3). Rules-first if unavailable or --no-llm.
    llm = None
    if not args.no_llm:
        try:
            from agentic_vla.agents.llm_backend import load_llm
            llm = load_llm(cfg.models.get("reasoning_llm", {}))
        except Exception as e:
            print(f"[warn] reasoning LLM unavailable ({e}); using rules-first agents.")

    # Episodic memory (Config C, M4).
    memory = None
    if args.config == "C":
        try:
            from agentic_vla.memory import EpisodicMemory
            memory = EpisodicMemory(cfg.run.get("memory", {}).get("store_dir", "memory_store"))
        except Exception as e:
            print(f"[warn] memory unavailable ({e}); running as config B.")

    rc = cfg.run.get("recovery", {})
    lc = cfg.run.get("loop", {})
    orch = Orchestrator(
        env, runner, llm=llm,
        retry_budget=rc.get("retry_budget", 3),
        chunk_steps=cfg.models["policy"].get("n_action_steps", 30),
        max_chunks=lc.get("max_chunks_per_subtask", 20),
        use_oracle_diagnostics=cfg.run.get("logging", {}).get("oracle_diagnostics", True),
        memory=memory,
    )

    print(f"[Config {args.config}] suite={suite} task={args.task} "
          f"episodes={args.episodes} llm={'on' if llm else 'off'} "
          f"memory={'on' if memory else 'off'}")
    print(f"  instruction: {env.instruction!r}")

    reports = []
    for i, seed in enumerate(seeds):
        rep = orch.run(seed=seed)
        reports.append(rep)
        print(f"  ep {i+1}/{len(seeds)} seed={seed} success={rep['success']} "
              f"subtasks={rep['subtasks_completed']} calls={rep['policy_calls']} "
              f"retries={rep['retries']} {rep['duration_s']}s")
    env.close()

    sr = sum(r["success"] for r in reports) / len(reports)
    summary = {
        "config": args.config,
        "suite": suite,
        "task": args.task,
        "episodes": len(reports),
        "success_rate": sr,
        "recovery_events": sum(len(r["failure_history"]) for r in reports),
        "avg_policy_calls": sum(r["policy_calls"] for r in reports) / len(reports),
        "reports": reports,
    }
    out_dir = ROOT / "runs" / f"agentic{args.config}_{suite}_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[Config {args.config}] success_rate = {sr:.2%}  ({out_dir/'results.json'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
