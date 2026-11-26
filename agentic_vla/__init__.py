"""Agentic VLA — multi-agent Vision-Language-Action system over SmolVLA / LIBERO.

See DESIGN.md for architecture. Public surface is intentionally small; import the
submodules you need (orchestrator, agents, envs, policy, memory, eval).
"""

# --- environment bootstrap (must run before torch/transformers/robosuite import) ------
import os as _os

# transformers would otherwise try to import TensorFlow (present in system site-packages)
# which crashes under some numpy builds; we only use the torch backend.
_os.environ.setdefault("USE_TF", "0")
_os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

# robosuite forces MUJOCO_GL=egl on import; egl is Linux-only. On Windows, clear any
# inherited egl so the GLFW offscreen context is used (see scripts/setup_windows_env.py).
if _os.name == "nt" and _os.environ.get("MUJOCO_GL", "").lower() == "egl":
    _os.environ.pop("MUJOCO_GL", None)

__version__ = "0.1.0"
