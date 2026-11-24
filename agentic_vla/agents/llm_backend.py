"""Reasoning-LLM backend for Planner/Critic (DESIGN.md §1, residency V1-a).

Pluggable backends:
  - transformers : Qwen2.5-3B-Instruct, 4-bit (bitsandbytes nf4)
  - ollama       : local Ollama server (manages its own load/unload)
  - llama_cpp    : GGUF via llama-cpp-python (CPU-friendly fallback, V1-b)

V1-a residency = time_sliced: the LLM loads lazily, reasons BETWEEN SmolVLA action
chunks, and `unload()` frees its VRAM so it never co-resides with the policy during
execution. If a backend is unavailable, callers fall back to rules-first agents.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from ..schemas import PlannerMessage, SubtaskSpec

_PLANNER_SYS = (
    "You are the PLANNER of a robotic manipulation system. Decompose the instruction "
    "into an ordered list of concrete subtasks. Do NOT invent object positions or "
    "properties. Respond with STRICT JSON only: "
    '{"subtasks":[{"id":"s1","description":"...","completion_condition":"..."}]}'
)

_CRITIC_SYS = (
    "You are the CRITIC of a robotic manipulation system. Given an execution summary and "
    "whether the official success signal fired, classify the outcome. Respond with STRICT "
    'JSON only: {"failure_category":"<one of the taxonomy or null>",'
    '"recommended_next_action":"...","reason":"..."}'
)


class LLMBackend:
    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-3B-Instruct",
        backend: str = "transformers",
        quantization: str = "4bit",
        device: str = "cuda",
        max_new_tokens: int = 512,
        residency: str = "time_sliced",
    ):
        self.model_name = model
        self.backend = backend
        self.quantization = quantization
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.residency = residency
        self._model = None
        self._tokenizer = None

    # -- lifecycle (V1-a time-slicing) -------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self.backend != "transformers" or self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
        if self.quantization == "4bit":
            try:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                kwargs["device_map"] = {"": 0}
            except Exception:
                pass  # fall back to bf16 full-precision load
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        if "device_map" not in kwargs:
            self._model.to(self.device)
        self._model.eval()

    def unload(self) -> None:
        """Free LLM VRAM so SmolVLA can execute without contention (V1-a)."""
        if self._model is None:
            return
        try:
            import torch

            del self._model
            self._model = None
            self._tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            self._model = None

    # -- generation --------------------------------------------------------------------
    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        if self.backend == "ollama":
            return self._gen_ollama(prompt, system)
        if self.backend == "llama_cpp":
            return self._gen_llama_cpp(prompt, system)
        return self._gen_transformers(prompt, system)

    def _gen_transformers(self, prompt: str, system: Optional[str]) -> str:
        import torch

        self._ensure_loaded()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(gen, skip_special_tokens=True)

    def _gen_ollama(self, prompt: str, system: Optional[str]) -> str:
        import urllib.request

        payload = {
            "model": self.model_name, "prompt": prompt, "system": system or "",
            "stream": False, "options": {"num_predict": self.max_new_tokens},
        }
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode()).get("response", "")

    def _gen_llama_cpp(self, prompt: str, system: Optional[str]) -> str:
        from llama_cpp import Llama  # model path expected in self.model_name

        if self._model is None:
            self._model = Llama(model_path=self.model_name, n_ctx=4096, verbose=False)
        full = (f"<|system|>\n{system}\n" if system else "") + f"<|user|>\n{prompt}\n<|assistant|>\n"
        out = self._model(full, max_tokens=self.max_new_tokens, temperature=0.0)
        return out["choices"][0]["text"]


def load_llm(cfg: dict[str, Any]) -> LLMBackend:
    """Factory from config/models.yaml:reasoning_llm. Raises if the backend can't init."""
    be = LLMBackend(
        model=cfg.get("model", "Qwen/Qwen2.5-3B-Instruct"),
        backend=cfg.get("backend", "transformers"),
        quantization=cfg.get("quantization", "4bit"),
        device=cfg.get("device", "cuda"),
        max_new_tokens=cfg.get("max_new_tokens", 512),
        residency=cfg.get("residency", "time_sliced"),
    )
    return be


# -- JSON extraction + planner/critic helpers ------------------------------------------
def _extract_json(text: str) -> Optional[dict]:
    # tolerate code fences / prose around the JSON object
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def plan_with_llm(llm: LLMBackend, instruction: str, scene_graph: dict | None) -> PlannerMessage:
    """LLM decomposition with strict-JSON parsing; falls back to rules on any failure."""
    from .planner import PlannerAgent  # local import avoids cycle

    prompt = (
        f"Instruction: {instruction}\n"
        f"Known objects (positions may be unknown/null): "
        f"{json.dumps((scene_graph or {}).get('objects', []))}\n"
        "Produce the subtask JSON now."
    )
    try:
        raw = llm.generate(prompt, system=_PLANNER_SYS)
        data = _extract_json(raw)
        subs = (data or {}).get("subtasks") or []
        subtasks = [
            SubtaskSpec(
                id=s.get("id", f"s{i+1}"),
                description=s["description"],
                completion_condition=s.get("completion_condition", "LIBERO success"),
            )
            for i, s in enumerate(subs)
            if s.get("description")
        ]
        if subtasks:
            return PlannerMessage(
                status="PLANNED",
                reason=f"LLM decomposition into {len(subtasks)} subtask(s).",
                subtasks=subtasks, confidence=None,
            )
    except Exception:
        pass
    finally:
        if getattr(llm, "residency", "") == "time_sliced":
            llm.unload()   # free VRAM before execution (V1-a)
    # fallback: rules-first
    return PlannerAgent(llm=None).plan(instruction, scene_graph)


def diagnose_with_llm(llm: LLMBackend, exec_summary: str, env_success: bool) -> dict:
    """Return {failure_category, recommended_next_action, reason} or {} on failure."""
    prompt = f"Execution summary: {exec_summary}\nOfficial success signal fired: {env_success}\nClassify now."
    try:
        raw = llm.generate(prompt, system=_CRITIC_SYS)
        return _extract_json(raw) or {}
    except Exception:
        return {}
    finally:
        if getattr(llm, "residency", "") == "time_sliced":
            llm.unload()
