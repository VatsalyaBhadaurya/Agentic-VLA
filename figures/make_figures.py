#!/usr/bin/env python
"""Generate research-grade figures for the Agentic VLA README.

Implements the scientific-figure-making skill conventions (PALETTE, publication
rcParams, vector+raster export). Produces:
  1. architecture.(png|pdf)      - 5-agent pipeline + closed-loop state machine
  2. stage_timeline.(png|pdf)    - Gantt of the REAL transition_log (measured timings)
  3. execution_chunks.(png|pdf)  - SmolVLA policy calls per bounded chunk
  4. agentic_flow.gif            - animated stage flow following the transition_log

Data source: runs/agenticB_mujoco_libero_object_1795731001/results.json (Config B).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as manim
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
RESULTS = ROOT / "runs" / "agenticB_mujoco_libero_object_1795731001" / "results.json"

PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
    "neutral": "#CFCECE", "highlight": "#FFD700",
    "teal": "#42949E", "violet": "#9A4D8E", "ink": "#1A1A1A",
}

STAGE_COLOR = {
    "PERCEIVING": PALETTE["teal"],
    "PLANNING": PALETTE["blue_main"],
    "SELECTING": PALETTE["violet"],
    "EXECUTING": PALETTE["blue_secondary"],
    "CRITIQUING": PALETTE["green_3"],
    "RECOVERING": PALETTE["red_strong"],
    "COMPLETE": PALETTE["green_2"],
}


def apply_style(font_size: int = 15, axes_lw: float = 2.0) -> None:
    plt.rcParams.update({
        "font.family": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
        "font.size": font_size,
        "axes.linewidth": axes_lw,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": font_size + 3, "axes.titleweight": "bold",
        "legend.frameon": False,
        "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def save(fig, basename: str, formats=("png", "pdf"), dpi: int = 300) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(FIG / f"{basename}.{fmt}", dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"  wrote figures/{basename}.{'/'.join(formats)}")


def load_report() -> dict:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    return data["reports"][0]


# ----------------------------------------------------------------------------------
# 1. Architecture: 5-agent pipeline + closed loop
# ----------------------------------------------------------------------------------
def fig_architecture() -> None:
    apply_style(14, 1.6)
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.2); ax.axis("off")

    def box(x, y, w, h, label, color, text_color="white"):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                           linewidth=1.8, edgecolor=PALETTE["ink"], facecolor=color, zorder=3)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=12.5, fontweight="bold", color=text_color, zorder=4)
        return (x + w / 2, y + h / 2)

    def arrow(p1, p2, color=PALETTE["ink"], style="-|>", rad=0.0, lw=2.0):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=16,
                                     lw=lw, color=color, zorder=2,
                                     connectionstyle=f"arc3,rad={rad}"))

    # orchestrator banner
    box(0.4, 5.3, 11.2, 0.72, "ORCHESTRATOR  —  shared TaskState  ·  sole authorizer of execution",
        PALETTE["ink"])

    y = 3.4
    perc = box(0.4, y, 1.9, 1.1, "Perception\nscene graph", STAGE_COLOR["PERCEIVING"])
    plan = box(2.7, y, 1.9, 1.1, "Planner\nsubtasks", STAGE_COLOR["PLANNING"])
    sel = box(5.0, y, 1.9, 1.1, "Skill\nSelector", STAGE_COLOR["SELECTING"])
    exe = box(7.3, y, 1.9, 1.1, "Executor\nSmolVLA", STAGE_COLOR["EXECUTING"])
    crit = box(9.6, y, 2.0, 1.1, "Critic\nverify success", STAGE_COLOR["CRITIQUING"])

    for a, b in [(perc, plan), (plan, sel), (sel, exe), (exe, crit)]:
        arrow((a[0] + 0.95, a[1]), (b[0] - 0.95, b[1]))

    # closed-loop feedback: critic -> executor (retry) and critic -> planner (replan)
    arrow((crit[0], crit[1] - 0.58), (exe[0], exe[1] - 0.58),
          color=PALETTE["blue_secondary"], rad=0.45, lw=1.8)
    ax.text((crit[0] + exe[0]) / 2, 2.35, "re-observe / retry chunk",
            ha="center", fontsize=10.5, color=PALETTE["blue_secondary"], style="italic")
    arrow((crit[0], crit[1] - 0.58), (plan[0], plan[1] - 0.58),
          color=PALETTE["red_strong"], rad=0.32, lw=1.8)
    ax.text((crit[0] + plan[0]) / 2, 1.55, "replan on failure (retry budget = 3)",
            ha="center", fontsize=10.5, color=PALETTE["red_strong"], style="italic")

    # env + policy backend
    env = box(7.3, 0.35, 4.3, 0.78, "MuJoCo · LIBERO (Franka Panda) — official success predicate",
              PALETTE["neutral"], text_color=PALETTE["ink"])
    arrow((exe[0] - 0.3, exe[1] - 0.55), (env[0] - 1.6, env[1] + 0.39),
          color=PALETTE["ink"], rad=-0.2, lw=1.6)
    # instruction input
    ax.text(1.35, 2.75, "instruction", ha="center", fontsize=10.5, style="italic",
            color=PALETTE["ink"])
    arrow((1.35, 2.95), (perc[0], perc[1] - 0.58), color=PALETTE["ink"], rad=0.0, lw=1.6)

    ax.set_title("Agentic VLA — multi-agent closed loop over SmolVLA",
                 loc="left", pad=10)
    save(fig, "architecture")


# ----------------------------------------------------------------------------------
# 2. Stage timeline (Gantt) from the real transition_log
# ----------------------------------------------------------------------------------
def fig_stage_timeline(rep: dict) -> None:
    apply_style(14, 1.8)
    log = rep["transition_log"]
    t0 = log[0]["t"]
    # build (stage, start, end) intervals from consecutive transitions
    intervals = []
    for i in range(len(log) - 1):
        stage = log[i]["to"]
        start = log[i]["t"] - t0
        end = log[i + 1]["t"] - t0
        intervals.append((stage, start, max(end, start + 0.15)))  # min width for visibility
    # final COMPLETE marker
    complete_t = log[-1]["t"] - t0

    lanes = ["PERCEIVING", "PLANNING", "SELECTING", "EXECUTING", "CRITIQUING", "COMPLETE"]
    lane_y = {s: len(lanes) - 1 - i for i, s in enumerate(lanes)}

    fig, ax = plt.subplots(figsize=(13, 5.0))
    chunk_calls = {1: 50, 2: 50, 3: 50, 4: 9}  # chunk_steps=50; total 159
    ci = 0
    for stage, s, e in intervals:
        y = lane_y[stage]
        ax.barh(y, e - s, left=s, height=0.55, color=STAGE_COLOR[stage],
                edgecolor=PALETTE["ink"], linewidth=1.3, zorder=3)
        if stage == "EXECUTING":
            ci += 1
            ax.text((s + e) / 2, y, f"chunk {ci}\n{chunk_calls.get(ci,'')} calls",
                    ha="center", va="center", fontsize=9.5, color="white", fontweight="bold")
        elif stage == "CRITIQUING":
            ax.scatter((s + e) / 2, y, marker="D", s=70, color=STAGE_COLOR["CRITIQUING"],
                       edgecolor=PALETTE["ink"], zorder=5)

    ax.scatter(complete_t, lane_y["COMPLETE"], marker="*", s=360,
               color=PALETTE["highlight"], edgecolor=PALETTE["ink"], zorder=6)
    ax.text(complete_t, lane_y["COMPLETE"], "  SUCCESS", va="center", fontsize=12,
            fontweight="bold", color=PALETTE["ink"])

    ax.set_yticks(list(lane_y.values()))
    ax.set_yticklabels(["Perception", "Planner", "Skill Select", "Executor (SmolVLA)",
                        "Critic", "Complete"])
    ax.set_xlabel("wall-clock time (s)")
    ax.set_xlim(-2, complete_t + 14)
    ax.set_title("Measured agentic execution — Config B, libero_object task 0 "
                 "(159 policy calls, 4 verify cycles, 0 retries)", loc="left", pad=10)
    ax.grid(axis="x", linestyle=":", alpha=0.4, zorder=0)
    save(fig, "stage_timeline")


# ----------------------------------------------------------------------------------
# 3. Execution chunk breakdown
# ----------------------------------------------------------------------------------
def fig_execution_chunks() -> None:
    apply_style(15, 2.0)
    chunks = ["chunk 1", "chunk 2", "chunk 3", "chunk 4"]
    calls = [50, 50, 50, 9]
    cum = np.cumsum(calls)
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    bars = ax.bar(chunks, calls, color=PALETTE["blue_secondary"],
                  edgecolor=PALETTE["ink"], linewidth=2.0, zorder=3, label="policy calls / chunk")
    for b, c, cu in zip(bars, calls, cum):
        ax.text(b.get_x() + b.get_width() / 2, c + 1.2, str(c), ha="center",
                fontsize=12, fontweight="bold")
    ax.scatter(range(4), cum, color=PALETTE["red_strong"], s=60, zorder=5)
    ax.plot(range(4), cum, color=PALETTE["red_strong"], lw=2.2, marker="o",
            label="cumulative calls")
    ax.axhline(159, ls="--", color=PALETTE["ink"], lw=1.2, alpha=0.6)
    ax.text(3.1, 150, "success @ 159", fontsize=10.5, color=PALETTE["ink"])
    ax.set_ylabel("SmolVLA inference calls")
    ax.set_title("Bounded-chunk execution with between-chunk verification", loc="left", pad=10)
    ax.legend(loc="upper left")
    ax.set_ylim(0, 175)
    save(fig, "execution_chunks")


# ----------------------------------------------------------------------------------
# 4. Animated stage-flow GIF following the transition_log
# ----------------------------------------------------------------------------------
def fig_flow_animation(rep: dict) -> None:
    apply_style(13, 1.6)
    log = rep["transition_log"]
    # node layout on a horizontal pipeline with a feedback loop
    nodes = {
        "PERCEIVING": (1.5, 3.2), "PLANNING": (3.3, 3.2), "SELECTING": (5.1, 3.2),
        "EXECUTING": (7.2, 3.2), "CRITIQUING": (9.3, 3.2), "COMPLETE": (11.0, 3.2),
    }
    label = {"PERCEIVING": "Perceive", "PLANNING": "Plan", "SELECTING": "Select",
             "EXECUTING": "Execute\n(SmolVLA)", "CRITIQUING": "Critic", "COMPLETE": "Complete"}
    seq = [e["to"] for e in log]  # the ordered stages actually visited

    fig, ax = plt.subplots(figsize=(12.5, 4.6))

    def base_draw(active_idx):
        ax.clear()
        ax.set_xlim(0, 12.3); ax.set_ylim(1.3, 4.6); ax.axis("off")
        active = seq[active_idx]
        # static edges
        order = ["PERCEIVING", "PLANNING", "SELECTING", "EXECUTING", "CRITIQUING", "COMPLETE"]
        for a, b in zip(order[:-1], order[1:]):
            ax.add_patch(FancyArrowPatch(nodes[a], nodes[b], arrowstyle="-|>",
                         mutation_scale=14, lw=1.6, color=PALETTE["neutral"], zorder=1,
                         shrinkA=24, shrinkB=24))
        # feedback loop critic->execute
        ax.add_patch(FancyArrowPatch((nodes["CRITIQUING"][0], nodes["CRITIQUING"][1] - 0.5),
                     (nodes["EXECUTING"][0], nodes["EXECUTING"][1] - 0.5),
                     arrowstyle="-|>", mutation_scale=13, lw=1.6,
                     color=PALETTE["blue_secondary"], zorder=1,
                     connectionstyle="arc3,rad=0.5", shrinkA=18, shrinkB=18))
        ax.text(8.25, 2.2, "verify / retry", ha="center", fontsize=10,
                color=PALETTE["blue_secondary"], style="italic")
        # nodes
        for name, (x, y) in nodes.items():
            is_active = (name == active)
            r = 0.62 if is_active else 0.52
            circ = plt.Circle((x, y), r, color=STAGE_COLOR[name],
                              ec=PALETTE["highlight"] if is_active else PALETTE["ink"],
                              lw=3.2 if is_active else 1.6, zorder=3, alpha=1.0 if is_active else 0.55)
            ax.add_patch(circ)
            ax.text(x, y, label[name], ha="center", va="center", fontsize=10.5,
                    fontweight="bold", color="white", zorder=4)
        # progress caption
        cyc = sum(1 for s in seq[:active_idx + 1] if s == "EXECUTING")
        cap = f"step {active_idx + 1}/{len(seq)}:  {active}"
        if active == "EXECUTING":
            cap += f"   (chunk {cyc} — SmolVLA drives 50 actions, then re-verify)"
        if active == "CRITIQUING":
            cap += "   (official LIBERO success predicate)"
        if active == "COMPLETE":
            cap = "TASK COMPLETE — success verified (vision-only)"
        ax.text(0.2, 4.35, "Agentic VLA — stage flow (Config B, libero_object task 0)",
                fontsize=14, fontweight="bold", color=PALETTE["ink"])
        ax.text(0.2, 1.55, cap, fontsize=12, color=PALETTE["ink"])

    def update(i):
        base_draw(i)
        return []

    anim = manim.FuncAnimation(fig, update, frames=len(seq), interval=850, blit=False)
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "agentic_flow.gif"
    anim.save(out, writer=manim.PillowWriter(fps=1.3))
    plt.close(fig)
    print(f"  wrote figures/agentic_flow.gif ({len(seq)} frames)")


def main() -> None:
    rep = load_report()
    print("Generating research-grade figures...")
    fig_architecture()
    fig_stage_timeline(rep)
    fig_execution_chunks()
    fig_flow_animation(rep)
    print("Done.")


if __name__ == "__main__":
    main()
