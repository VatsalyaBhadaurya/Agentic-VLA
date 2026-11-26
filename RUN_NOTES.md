# Run Notes — MuJoCo bring-up (2026-11-27)

Record of actually installing the stack and running on MuJoCo on the target machine
(RTX 4050, 6 GB, Windows 11, Python 3.10). Measured facts only.

## Environment built

Dedicated project venv `.venv` (created with `--system-site-packages` to reuse the
working `torch 2.6.0+cu124`, CUDA available). Installed:

- `mujoco==2.3.7`, `robosuite==1.4.0`, `bddl==1.0.1`, `gym==0.25.2`, hydra/easydict/einops
- `LIBERO` editable from `D:/vatty/work/Adaptive-MoE/LIBERO` (namespace-package layout;
  `libero/__init__.py` is absent, so it resolves as a PEP-420 namespace package)
- `lerobot[smolvla]` → `lerobot 0.4.4` (editable, repointed to the valid D: source after
  the pre-existing system editable pointed at a dead `E:` path), `transformers 4.57.6`
- **Pinned `numpy==1.26.4`** — robosuite 1.4.0 / gym 0.25 need numpy<2, and numpy 2 also
  crashed TensorFlow's bf16 shim during transformers' TF autodetect.

### Windows fixups (automated in `scripts/setup_windows_env.py`)
1. robosuite hardcodes `/tmp/robosuite.log` → create `D:\tmp` (and `C:\tmp`).
2. robosuite loads `mujoco.dll` from its own `utils/` → copy it from the mujoco package.
3. robosuite forces `MUJOCO_GL=egl` (Linux-only) via `MUJOCO_GPU_RENDERING=True` →
   set that macro `False` so it uses the GLFW offscreen context.
4. LIBERO prompts interactively on first import → pre-write `~/.libero/config.yaml`.
5. `agentic_vla/__init__.py` sets `USE_TF=0` (skip TF import) and clears inherited `egl`.

## What ran on MuJoCo (measured)

| Check | Result |
|---|---|
| LIBERO env create/reset/step (`libero_object`, task 0) | ✓ reset ~30s, 256×256×3 offscreen RGB |
| Full 120-step closed-loop episode via our `LiberoEnv` wrapper | ✓ 4.9s; `check_success()` works; clean close |
| Official success predicate `env.check_success()` | ✓ functioning (scoring source of truth) |
| `scripts/00_verify_interfaces.py` env half | ✓ action dim 7, 4-tuple step, image/proprio/oracle keys |
| SmolVLA load (`lerobot/smolvla_base`) | ✓ 450M bf16; VLM = SmolVLM2-500M (16 layers) |
| `scripts/00_verify_interfaces.py` policy half | ✓ chunk 50; input/output features captured |
| lerobot-eval pipeline (SmolVLA→LIBERO) | builds all 10 vec envs + policy, then **correctly stops at the feature validator** |

## ✅ SmolVLA driving the arm on MuJoCo — SUCCESS (measured)

Using the **LIBERO-finetuned** checkpoint `HuggingFaceVLA/smolvla_libero` (2 cameras
`image`/`image2`, 7-D action, 8-D state — matches LIBERO, passes the feature validator):

```
lerobot-eval --policy.path=<snapshot> --env.type=libero --env.task=libero_object \
             --env.task_ids="[0]" --eval.n_episodes=1 --policy.device=cuda
```

| Metric | Value |
|---|---|
| Task | `pick up the alphabet soup and place it in the basket` (libero_object, task 0) |
| **Success (official LIBERO predicate)** | **100% — 1/1 episode**, solved at step ~185/280 |
| avg_sum_reward / avg_max_reward | 1.0 / 1.0 |
| Wall time | 148 s on the RTX 4050 (~1.3 policy steps/s) |
| Video | `outputs/eval/2026-11-27/03-22-03_libero_smolvla/.../eval_episode_0.mp4` |

Result saved at `runs/smolvla_libero_object_task0_1ep.json`. This is the genuine Config A
baseline: the policy closed the loop on MuJoCo and completed the manipulation task.

## The honest SmolVLA-on-LIBERO finding (base vs finetuned)

`lerobot/smolvla_base` is a **generic pretrain, not LIBERO-finetuned**. Its config
requires **3 cameras (`camera1/2/3`), 6-D state, 6-D action** with its own MEAN_STD
normalization stats. LIBERO (via lerobot's `envs/libero.py`) provides **2 cameras
(`image`/`image2`), 7-D action**, and a different state layout.

LeRobot's `validate_visual_features_consistency` **refuses to run** this mismatched
pairing (missing `camera1/2/3`, extra `image`/`image2`). That is the correct,
spec-mandated behavior — *validate interfaces before execution; never silently reshape*
(system spec §4/§9, DESIGN.md §4). SmolVLA's `max_state_dim/max_action_dim=32` would pad
the 6-vs-7 dof gap, but the **2-vs-3 camera** gap and base-vs-LIBERO **normalization
stats** cannot be reconciled without a LIBERO-matched config.

### To get a scored SmolVLA rollout (two valid paths)
1. **Use a LIBERO-finetuned SmolVLA checkpoint** (matching 2-cam/7-dof + LIBERO norm
   stats), then `lerobot-eval --policy.path=<ckpt> --env.type=libero`.
2. **Finetune** `smolvla_base` on a LIBERO dataset to produce such a checkpoint.

A base checkpoint forced onto LIBERO would score ≈0% and is not a meaningful baseline;
we do not report fabricated numbers.

## Reproduce

```bash
py -3.10 -m venv .venv --system-site-packages
.venv\Scripts\python.exe -m pip install mujoco==2.3.7 robosuite==1.4.0 bddl==1.0.1 \
  hydra-core==1.2.0 easydict==1.9 einops gym==0.25.2 opencv-python thop cloudpickle
.venv\Scripts\python.exe -m pip install -e D:/vatty/work/Adaptive-MoE/LIBERO --no-deps --config-settings editable_mode=compat
.venv\Scripts\python.exe -m pip install "lerobot[smolvla]"
.venv\Scripts\python.exe -m pip install numpy==1.26.4 --no-deps
.venv\Scripts\python.exe scripts\setup_windows_env.py
.venv\Scripts\python.exe scripts\00_verify_interfaces.py --suite libero_object
```
