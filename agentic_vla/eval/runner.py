"""A/B/C benchmark driver (spec §10).

Runs the three configurations on IDENTICAL tasks/seeds and emits aggregated metrics plus
a comparison table:

  A  baseline : Instruction -> SmolVLA -> env
  B  agentic  : Instruction -> multi-agent orchestration -> SmolVLA -> env
  C  +memory  : B + episodic memory

    python -m agentic_vla.eval.runner --suite libero_object --episodes 10 --seeds 0,1,2

Heavy deps (torch/lerobot/libero) are imported lazily so metrics stay unit-testable.
Scoring = official LIBERO success. Residency mode is recorded in the report header.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

from .metrics import aggregate, format_comparison  # noqa: E402


def _baseline_episode(env, runner, seed: int, max_steps: int) -> dict:
    obs = env.reset(seed=seed)
    runner.reset()
    t0 = time.time()
    calls = 0
    success = False
    steps = 0
    for steps in range(1, max_steps + 1):
        action = runner.select_action(obs, env.instruction)
        calls += 1
        step = env.step(action)
        obs = step.obs
        if step.success:
            success = True
            break
        if step.done:
            break
    # normalized report shape (matches orchestrator reports for aggregate()):
    return {
        "seed": seed, "success": success, "duration_s": round(time.time() - t0, 2),
        "policy_calls": calls, "subtasks_planned": 1,
        "subtasks_completed": 1 if success else 0, "retries": 0, "replans": 0,
        "failure_history": [],
    }


def run_config(config: str, suite: str, task: int, seeds: list[int], cfg) -> list[dict]:
    from ..envs import LiberoEnv
    from ..policy import SmolVLARunner

    env = LiberoEnv(suite=suite, task_id=task, image_size=cfg.env.get("image_size", 256))
    runner = SmolVLARunner(
        checkpoint=cfg.checkpoint, device=cfg.device,
        dtype=cfg.models["policy"].get("dtype", "bfloat16"),
    )
    max_steps = cfg.env.get("max_steps", 600)
    reports: list[dict] = []

    if config == "A":
        for seed in seeds:
            reports.append(_baseline_episode(env, runner, seed, max_steps))
        env.close()
        return reports

    # B / C
    from ..orchestrator import Orchestrator

    llm = None
    try:
        from ..agents.llm_backend import load_llm
        llm = load_llm(cfg.models.get("reasoning_llm", {}))
    except Exception as e:
        print(f"[warn] LLM unavailable for config {config}: {e}; rules-first.")

    memory = None
    if config == "C":
        from ..memory import EpisodicMemory
        memory = EpisodicMemory(cfg.run.get("memory", {}).get("store_dir", "memory_store"))

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
    for seed in seeds:
        reports.append(orch.run(seed=seed))
    env.close()
    return reports


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default=None)
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seeds", default=None, help="comma-separated, e.g. 0,1,2")
    ap.add_argument("--configs", default="A,B,C")
    args = ap.parse_args()

    from ..config import load_config

    cfg = load_config()
    suite = args.suite or cfg.suite
    if args.seeds:
        base_seeds = [int(s) for s in args.seeds.split(",")]
    else:
        base_seeds = cfg.env.get("seeds", [0])
    # identical seed list across configs (spec §10 consistency)
    seeds = (base_seeds * args.episodes)[: max(args.episodes, len(base_seeds))]

    residency = cfg.models.get("reasoning_llm", {}).get("residency", "time_sliced")
    configs = [c.strip().upper() for c in args.configs.split(",")]

    aggregated: dict[str, dict] = {}
    raw: dict[str, list] = {}
    for c in configs:
        print(f"\n=== Running Config {c} | suite={suite} seeds={seeds} ===")
        reports = run_config(c, suite, args.task, seeds, cfg)
        raw[c] = reports
        aggregated[c] = aggregate(reports)
        print(f"  Config {c}: task_success_rate = {aggregated[c]['task_success_rate']:.1%}")

    table = format_comparison(aggregated)
    out_dir = ROOT / "runs" / f"benchmark_{suite}_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "aggregated.json").write_text(json.dumps(aggregated, indent=2), encoding="utf-8")
    (out_dir / "raw_reports.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")

    report_md = (
        f"# Agentic VLA — A/B/C Benchmark\n\n"
        f"- Suite: `{suite}`  Task: {args.task}  Seeds: {seeds}\n"
        f"- Checkpoint: `{cfg.checkpoint}`  Residency: `{residency}` (V1-a)\n"
        f"- Scoring: official LIBERO success predicate (vision-only; oracle figures excluded)\n\n"
        f"{table}\n\n"
        f"_Measured results only. Report reflects the actual run above._\n"
    )
    (out_dir / "REPORT.md").write_text(report_md, encoding="utf-8")

    print("\n" + table)
    print(f"\nWritten: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
