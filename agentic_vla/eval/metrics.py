"""Benchmark metrics (spec §10). Pure functions over lists of episode reports.

Reports come from scripts/01 (Config A) and the Orchestrator (Config B/C). Scoring is
always the official LIBERO success flag. Any oracle-assisted figure stays labeled and is
NOT folded into these vision-only success numbers.
"""
from __future__ import annotations

from collections import Counter
from statistics import pstdev
from typing import Any


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def aggregate(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the spec §10 metric set from episode reports."""
    n = len(reports)
    if n == 0:
        return {"episodes": 0}

    successes = [bool(r.get("success")) for r in reports]
    failure_cats: Counter = Counter()
    failed_grasps = 0
    total_retries = 0
    total_replans = 0
    eps_with_failures = 0
    recovered_eps = 0

    for r in reports:
        fh = r.get("failure_history", []) or []
        if fh:
            eps_with_failures += 1
            if r.get("success"):
                recovered_eps += 1
        for f in fh:
            cat = f.get("category")
            if cat:
                failure_cats[cat] += 1
            if cat == "GRASP_FAILED":
                failed_grasps += 1
        total_retries += int(r.get("retries", 0) or 0)
        total_replans += int(r.get("replans", 0) or 0)

    # per-seed spread of success
    by_seed: dict[Any, list[bool]] = {}
    for r in reports:
        by_seed.setdefault(r.get("seed"), []).append(bool(r.get("success")))
    seed_rates = {s: _mean([1.0 if x else 0.0 for x in v]) for s, v in by_seed.items()}

    return {
        "episodes": n,
        "task_success_rate": _mean([1.0 if s else 0.0 for s in successes]),
        "subtask_success_rate": _mean(
            [
                (r.get("subtasks_completed", 0) / r["subtasks_planned"])
                for r in reports
                if r.get("subtasks_planned")
            ]
        ),
        "recovery_success_rate": (recovered_eps / eps_with_failures) if eps_with_failures else None,
        "avg_completion_time_s": _mean([float(r.get("duration_s", 0) or 0) for r in reports]),
        "avg_policy_calls": _mean([float(r.get("policy_calls", 0) or 0) for r in reports]),
        "total_failed_grasps": failed_grasps,
        "total_retries": total_retries,
        "total_replans": total_replans,
        "episodes_with_failures": eps_with_failures,
        "failure_distribution": dict(failure_cats),
        "per_seed_success_rate": seed_rates,
        "seed_success_std": pstdev(list(seed_rates.values())) if len(seed_rates) > 1 else 0.0,
    }


def format_comparison(results_by_config: dict[str, dict[str, Any]]) -> str:
    """Render an A/B/C comparison markdown table from aggregated metric dicts."""
    configs = [c for c in ("A", "B", "C") if c in results_by_config]
    rows = [
        ("Task success rate", "task_success_rate", "{:.1%}"),
        ("Subtask success rate", "subtask_success_rate", "{:.1%}"),
        ("Recovery success rate", "recovery_success_rate", "{:.1%}"),
        ("Avg completion time (s)", "avg_completion_time_s", "{:.1f}"),
        ("Avg policy calls", "avg_policy_calls", "{:.1f}"),
        ("Failed grasps", "total_failed_grasps", "{:d}"),
        ("Retries", "total_retries", "{:d}"),
        ("Replans", "total_replans", "{:d}"),
        ("Seed success std", "seed_success_std", "{:.3f}"),
    ]
    header = "| Metric | " + " | ".join(
        {"A": "A baseline", "B": "B agentic", "C": "C +memory"}[c] for c in configs
    ) + " |"
    sep = "|" + "---|" * (len(configs) + 1)
    lines = [header, sep]
    for label, key, fmt in rows:
        cells = []
        for c in configs:
            v = results_by_config[c].get(key)
            cells.append("n/a" if v is None else fmt.format(v))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
