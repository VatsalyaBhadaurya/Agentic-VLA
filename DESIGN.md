# Agentic VLA — Design Document & Implementation Plan

> Status: **DRAFT v0.1** · Author: project owner · Date: 2026-11-24
> Scope: design + interface contracts only. **No runtime code yet.**
> Golden rule (from system spec): *validate interfaces before execution; never assume
> an action format, camera config, robot-state layout, or checkpoint is compatible.*
> Everything tagged **[TO-VERIFY]** must be confirmed against the actually-installed
> LIBERO / LeRobot / SmolVLA before it is relied on in code.

---

## 0. Environment Baseline (measured, 2026-11-24)

| Component | Finding | Notes |
|---|---|---|
| GPU | NVIDIA RTX 4050 Laptop, **6 GB VRAM** | Hard constraint. Drives every model-sizing decision below. |
| Torch | `2.4.1+cu121`, CUDA available | On the system `python` (3.8.10). |
| MuJoCo | `3.2.3` | Present. |
| LIBERO | source tree at `D:\vatty\work\Adaptive-MoE\LIBERO` (editable) | **Not importable** from system Py3.8 — needs install into the project env. |
| LeRobot / SmolVLA | not in system Py3.8; present in sibling projects (`smolVLA-implementation-TARA`, `lerobot-genesis-train/lerobot_env`) | LeRobot requires **Python ≥ 3.10**. |
| robosuite | not importable from system Py3.8 | LIBERO depends on robosuite; comes via LIBERO install. |
| Checkpoint | `lerobot/smolvla_base` (HF, ~500 MB) | Confirmed by `smolVLA-implementation-TARA/SETUP_SMOLVLA.md`. |
| Known-good eval path | `lerobot-eval --policy.path=lerobot/smolvla_base --env.type=libero --env.task=libero_object --policy.device=cuda` | This is our **Config A baseline** harness. |

**Decision E1 — one project env.** Create a dedicated Python **3.10** venv for this repo
(reuse the pattern of `lerobot_env`). Install: `lerobot[smolvla]`, LIBERO (editable from
the Adaptive-MoE tree *or* a pinned clone), robosuite (LIBERO-pinned version), mujoco.
Pin everything in `requirements.lock`. This env is the single source of truth for all
interface contracts below.

**Decision E2 — Py3.8 is dead for this project.** The system Py3.8 is kept only as
evidence that CUDA/MuJoCo work on this machine. All code targets 3.10.

---

## 1. VRAM Budget (the real design driver)

6 GB cannot hold SmolVLA **and** a separate reasoning LLM **and** a separate VLM
concurrently. Rough resident footprints (bf16):

| Model | Approx VRAM | Role |
|---|---|---|
| SmolVLA base (SmolVLM-2 backbone + action expert, ~0.45 B) | ~1.5–2.5 GB incl. activations | Executor policy |
| MuJoCo render + env | ~0.3–0.8 GB | Simulation |
| A 7B reasoning LLM (even 4-bit) | ~4.5–5 GB | ❌ will not co-reside |
| A 3B LLM 4-bit (e.g. Qwen2.5-3B-Instruct GGUF/AWQ) | ~2.5–3 GB | Planner/critic — only if time-sliced |

**Decision V1 — model residency policy.** Choose ONE of:

- **V1-a (default, recommended): reuse SmolVLA's own VLM for perception** + a *small*
  text LLM (≤3 B, 4-bit) for Planner/Critic, loaded **lazily and time-sliced** with the
  policy. Perception does not load a third large model.
- **V1-b: CPU/quantized reasoning.** Run the Planner/Critic LLM on CPU (llama.cpp) so
  the GPU is reserved for SmolVLA + MuJoCo. Slower reasoning, but zero VRAM contention.
- **V1-c: external/offline LLM** via a local server (Ollama) that manages its own
  load/unload.

All three keep SmolVLA resident during execution and only reason *between* execution
chunks (consistent with the OBSERVE→INFER→ACT→OBSERVE→VERIFY loop, which already
serializes reasoning and acting). **Benchmark must record which residency mode was used.**

---

## 2. Repository Layout

```
agentic-vla/
├── DESIGN.md                      # this file
├── README.md
├── requirements.lock              # pinned, generated from the project venv
├── config/
│   ├── env.yaml                   # LIBERO suite/task, cameras, control freq, seeds
│   ├── models.yaml                # checkpoint ids, residency mode (V1-a/b/c), dtypes
│   └── run.yaml                   # retry budget, chunk size, logging, config A/B/C switch
├── agentic_vla/
│   ├── orchestrator.py            # the ONLY component that authorizes an execution step
│   ├── state.py                   # TaskState dataclass + transition log (Section 5)
│   ├── schemas.py                 # pydantic models for every agent message (Section 11)
│   ├── envs/
│   │   └── libero_env.py          # thin wrapper: reset/obs/step/success-signal
│   ├── policy/
│   │   └── smolvla_runner.py      # load checkpoint, preprocess obs, infer action chunk
│   ├── agents/
│   │   ├── planner.py
│   │   ├── perception.py
│   │   ├── skill_selector.py
│   │   ├── executor.py
│   │   └── critic.py
│   ├── skills/
│   │   └── registry.py            # skill -> concrete capability binding (Section 7)
│   ├── memory/
│   │   └── episodic.py            # local JSONL/SQLite episodic store (Section 8)
│   └── eval/
│       ├── runner.py              # Config A/B/C driver
│       └── metrics.py             # Section 10 metrics
├── scripts/
│   ├── 00_verify_interfaces.py    # prints real action/obs contracts -> fills [TO-VERIFY]
│   ├── 01_run_baseline.py         # Config A
│   └── 02_run_agentic.py          # Config B/C
└── tests/
    └── test_schemas.py
```

---

## 3. Interface Contracts — Environment (LIBERO)

> These follow standard LIBERO/robosuite conventions but are **[TO-VERIFY]** by
> `scripts/00_verify_interfaces.py` against the installed version before any reliance.

### 3.1 Action space  **[TO-VERIFY]**
- Expected: **7-D continuous**, `OSC_POSE` controller:
  `[dx, dy, dz, drx, dry, drz, gripper]`, each normalized to `[-1, 1]`;
  gripper: `-1` open / `+1` close (sign **[TO-VERIFY]**).
- Control frequency: **[TO-VERIFY]** (LIBERO default commonly 20 Hz control over a
  500 Hz sim). Must match what SmolVLA emits.

### 3.2 Observation space  **[TO-VERIFY]**
Expected dict keys from the LIBERO env:
- `agentview_image` — RGB, HxWx3 (resolution config-set, **[TO-VERIFY]**, often 256²).
- `robot0_eye_in_hand_image` — wrist RGB.
- `robot0_eef_pos` (3), `robot0_eef_quat` (4), `robot0_gripper_qpos` (2).
- `object-state` / task-specific low-dim — **used only for oracle diagnostics**, never
  as perception input in vision-only runs (spec §9).

### 3.3 Success signal  **[TO-VERIFY]**
- LIBERO exposes a per-step boolean task-success predicate (`env.check_success()` /
  `done`). This is the **official benchmark signal** for scoring (spec §4 critic, §10).
- Ground-truth object poses available for diagnostics only → must be labeled
  **oracle-assisted** and excluded from vision-only performance claims.

---

## 4. Interface Contracts — Policy (SmolVLA)

> **[TO-VERIFY]** against `lerobot/policies/smolvla` in the installed LeRobot.

- **Inputs (LeRobot batch convention):**
  - `observation.images.<cam>` — one or more camera tensors, preprocessed to the
    checkpoint's expected resolution/normalization. The **camera naming and count must
    match the checkpoint's training config** (do not rename).
  - `observation.state` — proprioceptive vector; dimension/order **[TO-VERIFY]** and
    must be assembled from the LIBERO obs in the exact order SmolVLA expects.
  - `task` — **the natural-language instruction string.** For pretrained SmolVLA the
    executor must pass the **original LIBERO task instruction** (spec §4), not a
    paraphrased planner subtask.
- **Output:** an **action chunk** (sequence of `n_action_steps`, commonly 50 **[TO-VERIFY]**)
  in the action space of §3.1.
- **Mapping risk:** SmolVLA's native action space / proprio layout may differ from raw
  LIBERO. The runner owns an explicit, tested `obs→policy` and `policy→env` adapter.
  Any mismatch is a **blocking** interface error, not a silent reshape.

**Decision P1 — executor language input.** Executor uses the original task instruction
for the policy call. The planner's decomposition drives *orchestration & verification*,
not the string handed to SmolVLA, unless/until we train skill-conditioned policies.

---

## 5. Shared Task State & State Machine

Single source of truth, owned by the orchestrator (spec §5). Dataclass in `state.py`,
every mutation appended to an immutable `transition_log`.

```
status ∈ { PLANNING, PERCEIVING, SELECTING, EXECUTING,
           CRITIQUING, RECOVERING, COMPLETE, ABORTED }
```

State machine (phases map 1:1 to spec §6):

```
INIT ─► PERCEIVE ─► PLAN ─► SELECT ─► EXECUTE ─► CRITIQUE
                     ▲                              │
                     │              ┌───────────────┤
                     │        SUCCESS/PARTIAL   FAILURE/UNCERTAIN
                     │               │               │
                 (next subtask) ◄────┘          RECOVER (reobserve,
                                                 replan, alt skill)
                                                     │
                                       retry_count < budget ─► SELECT
                                       retry_count ≥ budget ─► ABORTED
   all subtasks done + final verify (LIBERO success) ─► COMPLETE
```

Invariants:
- Only the orchestrator calls `env.step`. Agents return *recommendations*, not actions.
- A subtask is never marked done on executor completion alone — only on **critic
  verification** (spec §4: policy-completion ≠ subtask-completion ≠ task-completion).
- `retry_count` resets per subtask; default **budget = 3** (spec §7).
- Recovery must **reobserve before retry** and must not repeat the identical failed action.

---

## 6. Agent Responsibilities (concrete cut)

| Agent | Model (V1-a default) | Input | Output (schema §11) |
|---|---|---|---|
| Planner | small text LLM (≤3B/4-bit) | instruction + scene graph | ordered subtasks + completion conditions |
| Perception | SmolVLA VLM backbone / detector | camera obs | scene graph (objects, relations, confidence) |
| Skill Selector | rules + registry (§7) | next subtask + scene graph | selected skill *or* SKILL_UNAVAILABLE |
| Executor | **SmolVLA** | obs + robot state + task string | action chunk + exec status |
| Critic | LLM + env success predicate | pre/post obs + success signal | SUCCESS/PARTIAL/FAILURE/UNCERTAIN/RECOVERY_REQUIRED |

**Decision A1 — start rules-first where possible.** Skill Selector and Critic begin as
deterministic rule + env-signal logic (cheap, reproducible, no VRAM). LLM reasoning is
added only where rules are insufficient (ambiguous planning, failure diagnosis). This
keeps Config B honest and the VRAM budget sane.

---

## 7. Skill Registry (anti-hallucination contract)

Spec §3: *a skill name is not an executable capability.* The registry maps each skill
to a **concrete, validated binding** or marks it unsupported.

```python
Skill = { name, required_policy, supported: bool,
          precondition_check, postcondition_check, evidence_source }
```

- Phase-1 supported skill: **the whole LIBERO task via SmolVLA** (the policy is itself a
  long-horizon pick/place behavior). The planner's fine-grained skills
  (reach/grasp/open-drawer…) are **orchestration labels** used for verification and
  recovery decisions, not separate executable controllers — until we have trained/validated
  per-skill policies.
- Requesting an unsupported skill ⇒ `SKILL_UNAVAILABLE` ⇒ Planner must replan or abort.
  **Never silently substitute.**

---

## 8. Episodic Memory

Local store (`JSONL` first, optional SQLite later). One record per episode: instruction,
task id, seed, initial scene graph, decomposition, selected skills, policy-call refs,
action/obs trajectory refs, failure categories, recovery attempts, outcome, duration.

- Config C retrieval: nearest prior episode by (task id, instruction embedding) to bias
  planning/recovery.
- Hard rule (spec §8): retrieval **never overrides current visual evidence**, and storing
  episodes **does not** constitute training/adaptation of SmolVLA.

---

## 9. Failure Taxonomy → Recovery Map

The 10 categories (spec §7) each get a recovery policy, e.g.:

| Category | Detector | First recovery |
|---|---|---|
| OBJECT_NOT_DETECTED | perception conf < τ | reobserve (alt camera / settle frames) |
| GRASP_FAILED | gripper closed but no object rise / contact | re-approach with offset, re-grasp |
| OBJECT_SLIPPED | post-grasp pose delta | regrasp |
| PLACEMENT_FAILED | success predicate false post-place | re-plan placement pose |
| SKILL_UNAVAILABLE | registry miss | replan / abort |
| POLICY_EXECUTION_ERROR | malformed/NaN action | stop safely, log |
| VERIFICATION_UNCERTAIN | success signal ambiguous | reobserve, escalate to critic |

Each recovery must change strategy, reobserve first, and respect the retry budget.

---

## 10. Evaluation Plan

Three configs, **identical** tasks/seeds/cameras/control freq:
- **A** Instruction → SmolVLA → env.
- **B** Instruction → multi-agent → SmolVLA → env.
- **C** B + episodic memory.

Metrics (spec §10): task SR, subtask SR, recovery SR, avg completion time, #policy calls,
#failed grasps, #retries, #replans, failure distribution, per-seed spread.

Rules: report **measured** numbers only; LIBERO success is the scoring signal; any
oracle-assisted diagnostic is labeled and excluded from vision-only claims; VRAM
residency mode (V1-a/b/c) recorded per run.

Suite choice (start small on 6 GB): begin with **`libero_object`** (matches the known
working eval command), then extend to `libero_spatial` / `libero_goal` / `libero_10`.

---

## 11. Agent Message Schemas

All agent I/O validated with `pydantic` (spec §11). Base envelope:

```python
class AgentMessage(BaseModel):
    agent: Literal["planner","perception","skill_selector","executor","critic"]
    status: str                 # agent-specific enum
    reason: str
    evidence: list = []
    recommended_next_action: str | None = None
    confidence: float | None = None   # None = genuinely unknown; never fabricated
```

Malformed/incomplete messages are **rejected** by the orchestrator (fail loud).
Scene-graph, subtask, and skill objects get their own strict sub-schemas.

---

## 12. Milestones

- **M0 — Env lockdown.** Create Py3.10 venv, install stack, generate `requirements.lock`.
  Run `scripts/00_verify_interfaces.py` → fill every **[TO-VERIFY]** with real values.
  *Exit:* a printed, committed interface report.
- **M1 — Config A baseline.** `libero_env.py` + `smolvla_runner.py` + `01_run_baseline.py`;
  reproduce `lerobot-eval` numbers on `libero_object`. *Exit:* measured baseline SR.
- **M2 — State + schemas + orchestrator skeleton** with closed OBSERVE→…→VERIFY loop
  using rules-only critic/selector. *Exit:* Config B runs end-to-end, no LLM yet.
- **M3 — LLM planner/critic** under chosen residency mode; recovery loop + retry budget.
- **M4 — Episodic memory** (Config C).
- **M5 — Full A/B/C benchmark** across seeds + report.

**Immediate next action after this doc is approved: M0.**

---

## 13. Resolved Decisions (owner-confirmed 2026-11-24)

1. **Residency mode = V1-a.** SmolVLA resident during execution; perception reuses
   SmolVLA's VLM backbone; reasoning LLM loaded lazily and time-sliced between chunks.
2. **Reasoning LLM = Qwen2.5-3B-Instruct, 4-bit.** Used by Planner/Critic where rules
   are insufficient (ambiguous planning, failure diagnosis). Must fit alongside SmolVLA
   under V1-a — validate resident footprint at M3; fall back to Qwen2.5-1.5B if tight.
3. **LIBERO install = editable** from `D:\vatty\work\Adaptive-MoE\LIBERO`
   (`pip install -e`). Record the tree's commit/state in `requirements.lock` notes for
   reproducibility.
4. **Starting suite = `libero_object`** (matches the known-good eval command).
5. **Perception = reuse SmolVLA VLM backbone** (consistent with V1-a; no dedicated
   detector in Phase 1). Revisit only if perception accuracy blocks progress.
