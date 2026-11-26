#!/usr/bin/env python
"""Post-install Windows fixups for the LIBERO/robosuite stack (idempotent).

The pinned LIBERO stack (robosuite 1.4.0 + mujoco + bddl) has three Windows-specific
snags that this script resolves after `pip install`. Run once inside the project venv:

    .venv\\Scripts\\python.exe scripts\\setup_windows_env.py

Fixes applied:
  1. robosuite hardcodes `/tmp/robosuite.log` -> ensure a writable tmp dir exists.
  2. robosuite loads `mujoco.dll` from its own utils/ dir -> copy it from the mujoco pkg.
  3. robosuite forces MUJOCO_GL=egl (Linux-only) when MUJOCO_GPU_RENDERING=True ->
     flip that macro off so it uses the GLFW context on Windows.
  4. LIBERO prompts interactively on first import -> pre-write ~/.libero/config.yaml.

All changes are to the *installed* packages / user config, not the source trees.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def ensure_tmp() -> None:
    for drive in ("C:/tmp", os.path.splitdrive(os.getcwd())[0] + "/tmp"):
        Path(drive).mkdir(parents=True, exist_ok=True)
    print("[1/4] tmp dir(s) ensured")


def copy_mujoco_dll() -> None:
    import mujoco
    import robosuite

    src = Path(mujoco.__file__).parent / "mujoco.dll"
    dst = Path(robosuite.__file__).parent / "utils" / "mujoco.dll"
    if not dst.exists() and src.exists():
        shutil.copy2(src, dst)
        print(f"[2/4] copied mujoco.dll -> {dst}")
    else:
        print("[2/4] mujoco.dll already present (or source missing)")


def disable_gpu_rendering_macro() -> None:
    import robosuite

    macros = Path(robosuite.__file__).parent / "macros.py"
    text = macros.read_text(encoding="utf-8")
    if "MUJOCO_GPU_RENDERING = True" in text:
        text = text.replace(
            "MUJOCO_GPU_RENDERING = True",
            "MUJOCO_GPU_RENDERING = False  # Windows: avoid forced MUJOCO_GL=egl",
        )
        macros.write_text(text, encoding="utf-8")
        print("[3/4] robosuite MUJOCO_GPU_RENDERING -> False")
    else:
        print("[3/4] robosuite GPU rendering already disabled")


def write_libero_config() -> None:
    import yaml

    cfg_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", Path.home() / ".libero"))
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_file = cfg_dir / "config.yaml"
    if cfg_file.exists():
        print(f"[4/4] LIBERO config already present: {cfg_file}")
        return
    # locate the real package root WITHOUT importing libero.libero (which would prompt).
    import importlib.util

    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        print("[4/4] libero not importable; skipping config write")
        return
    root = Path(list(spec.submodule_search_locations)[0]) / "libero"
    cfg = {
        "benchmark_root": str(root),
        "bddl_files": str(root / "bddl_files"),
        "init_states": str(root / "init_files"),
        "datasets": str(root / ".." / "datasets"),
        "assets": str(root / "assets"),
    }
    cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    print(f"[4/4] wrote LIBERO config: {cfg_file}")


def main() -> int:
    ensure_tmp()
    try:
        copy_mujoco_dll()
        disable_gpu_rendering_macro()
    except Exception as e:
        print(f"[robosuite fixups] skipped/failed: {e}")
    try:
        write_libero_config()
    except Exception as e:
        print(f"[libero config] skipped/failed: {e}")
    print("\nDone. Verify with: python scripts/00_verify_interfaces.py --suite libero_object")
    return 0


if __name__ == "__main__":
    sys.exit(main())
