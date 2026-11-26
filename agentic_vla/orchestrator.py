"""Orchestrator — coordinates the five agents (spec §6, §12).

It is the ONLY component that authorizes an execution step and the ONLY writer of the
shared TaskState. Agents return schema-validated recommendations; the orchestrator
drives the state machine:

    INIT -> PERCEIVE -> PLAN -> SELECT -> EXECUTE -> CRITIQUE
                                   ^                     |
                                   +------ RECOVER <-----+   (reobserve, retry<=budget)

Phase-1 note (DESIGN.md §7): the executable capability is the whole task via SmolVLA.
The planner's subtasks are recorded for structure/verification/recovery; execution
proceeds as bounded chunks of the full task, verified by the critic each chunk.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from .agents import (
    CriticAgent,
    ExecutorAgent,
    PerceptionAgent,
    PlannerAgent,
    SkillSelectorAgent,
)
from .envs import LiberoEnv
from .policy import SmolVLARunner
from .schemas import validate_message
from .skills import SkillRegistry
from .state import Status, Subtask, TaskState


class Orchestrator:
    def __init__(
        self,
        env: LiberoEnv,
        runner: SmolVLARunner,
        *,
        llm=None,
        retry_budget: int = 3,
        chunk_steps: int = 30,
        max_chunks: int = 20,
        use_oracle_diagnostics: bool = True,
        memory=None,
        planner=None,
    ):
        self.env = env
        self.runner = runner
        self.registry = SkillRegistry(policy_checkpoint=runner_checkpoint(runner))
        # A custom planner may be injected. With a single end-to-end policy (design P1),
        # SmolVLA executes the whole task from the original instruction and the LIBERO env
        # auto-resets on success — so a multi-subtask plan cannot be executed as separate
        # rollouts. In that setup, inject a planner that yields one `full_task` subtask;
        # the planner's richer decomposition still informs verification/recovery reasoning.
        self.planner = planner if planner is not None else PlannerAgent(llm=llm)
        self.perception = PerceptionAgent(confidence_threshold=0.4)
        self.selector = SkillSelectorAgent(self.registry)
        self.executor = ExecutorAgent(env, runner)
        self.critic = CriticAgent(env, llm=llm, use_oracle_diagnostics=use_oracle_diagnostics)
        self.retry_budget = retry_budget
        self.chunk_steps = chunk_steps
        self.max_chunks = max_chunks
        self.memory = memory

    # ---------------------------------------------------------------------------------
    def run(self, seed: int = 0) -> dict[str, Any]:
        t0 = time.time()
        state = TaskState(instruction=self.env.instruction)

        # PHASE 1 — INIT
        obs = self.env.reset(seed=seed)
        self.executor.reset(obs)
        state.last_observation_id = f"obs_{seed}_0"

        # PHASE 2 — PERCEIVE
        state.transition(Status.PERCEIVING)
        pmsg = validate_message(self.perception.perceive(state.instruction, self.env.images()))
        state.scene_graph = {
            "objects": [o.model_dump(by_alias=True) for o in pmsg.objects],
            "relationships": [r.model_dump() for r in pmsg.relationships],
        }

        # PHASE 3 — PLAN
        state.transition(Status.PLANNING)
        plan = validate_message(self.planner.plan(state.instruction, state.scene_graph))
        state.pending_subtasks = [
            Subtask(id=s.id, description=s.description, completion_condition=s.completion_condition)
            for s in plan.subtasks
        ]

        # Retrieve prior experience (Config C); never overrides current perception (spec §8).
        memory_hint = None
        if self.memory is not None:
            memory_hint = self.memory.retrieve(state.instruction, state.scene_graph)

        # PHASE 4-7 — per-subtask select/execute/critique/recover
        result_status = "FAILURE"
        total_policy_calls = 0
        total_steps = 0
        replans = 0

        while state.pending_subtasks:
            subtask = state.pending_subtasks.pop(0)
            subtask.status = "active"
            state.current_subtask = subtask
            state.retry_count = 0

            # PHASE 4 — SELECT
            state.transition(Status.SELECTING, note=subtask.id)
            sel = validate_message(self.selector.select(subtask))
            if sel.selected_skill is None:
                state.record_failure("SKILL_UNAVAILABLE", subtask.description)
                replans += 1
                # try a replan once; if planner yields nothing new, abort
                state.transition(Status.PLANNING, note="replan after SKILL_UNAVAILABLE")
                continue
            subtask.skill = sel.selected_skill
            state.selected_skill = sel.selected_skill

            # PHASE 5-7 — EXECUTE / CRITIQUE / RECOVER
            outcome, calls, steps = self._run_subtask(state, subtask)
            total_policy_calls += calls
            total_steps += steps

            if outcome in ("SUCCESS", "PARTIAL_SUCCESS"):
                subtask.status = "done"
                state.completed_subtasks.append(subtask)
                if outcome == "SUCCESS":
                    result_status = "SUCCESS"
                    break
            else:
                subtask.status = "failed"
                result_status = "FAILURE"
                break

        # PHASE 8 — COMPLETION (final verification via official signal)
        final_success = self.env.check_success()
        state.verification_status = "SUCCESS" if final_success else "FAILURE"
        state.final_result = "SUCCESS" if final_success else result_status
        state.transition(Status.COMPLETE if final_success else Status.ABORTED)

        report = {
            "task_id": state.task_id,
            "instruction": state.instruction,
            "seed": seed,
            "success": final_success,           # official signal — the scoring truth
            "final_result": state.final_result,
            "subtasks_planned": len(state.completed_subtasks) + len(state.pending_subtasks) + (1 if state.current_subtask and state.current_subtask.status == "failed" else 0),
            "subtasks_completed": len(state.completed_subtasks),
            "policy_calls": total_policy_calls,
            "steps": total_steps,
            "retries": sum(1 for f in state.failure_history),
            "replans": replans,
            "failure_history": state.failure_history,
            "scene_graph": state.scene_graph,
            "memory_hint_used": memory_hint is not None,
            "duration_s": round(time.time() - t0, 2),
            "transition_log": state.transition_log,
        }
        if self.memory is not None:
            self.memory.store(state, report)
        return report

    # ---------------------------------------------------------------------------------
    def _run_subtask(self, state: TaskState, subtask: Subtask):
        """Execute bounded chunks for one subtask with the recovery/retry loop."""
        calls = 0
        steps = 0
        chunks_used = 0
        is_last = len(state.pending_subtasks) == 0

        while True:
            state.transition(Status.EXECUTING, note=f"{subtask.id} chunk {chunks_used+1}")
            exec_msg = validate_message(
                self.executor.execute_chunk(self.env.instruction, self.chunk_steps)
            )
            calls += exec_msg.policy_calls
            steps += exec_msg.steps_executed
            chunks_used += 1

            state.transition(Status.CRITIQUING, note=subtask.id)
            critic_msg = validate_message(
                self.critic.evaluate(exec_msg, is_last, chunks_used, self.max_chunks)
            )

            if critic_msg.status in ("SUCCESS", "PARTIAL_SUCCESS"):
                return critic_msg.status, calls, steps

            if critic_msg.status == "UNCERTAIN":
                # reobserve (executor keeps latest obs) and continue
                continue

            # RECOVERY_REQUIRED / FAILURE
            if critic_msg.failure_category:
                state.record_failure(critic_msg.failure_category, critic_msg.reason)
            recovered = self._recover(state, subtask, critic_msg)
            if not recovered:
                return "FAILURE", calls, steps
            # reset per-subtask chunk budget for the retry
            chunks_used = 0

    def _recover(self, state: TaskState, subtask: Subtask, critic_msg) -> bool:
        """Reobserve, respect retry budget, change strategy; never blind-repeat (spec §7)."""
        if state.retry_count >= self.retry_budget:
            state.transition(Status.ABORTED, note=f"retry budget exhausted on {subtask.id}")
            return False
        state.retry_count += 1
        state.transition(Status.RECOVERING, note=f"{subtask.id} retry {state.retry_count}")
        # reobserve: refresh perception on current frame (updates scene graph)
        pmsg = self.perception.perceive(state.instruction, self.env.images())
        state.scene_graph = {
            "objects": [o.model_dump(by_alias=True) for o in pmsg.objects],
            "relationships": [r.model_dump() for r in pmsg.relationships],
        }
        # strategy change hook: a per-skill recovery controller would plug in here.
        # Phase-1: re-observe + retry the policy chunk (the policy is closed-loop).
        return True


def runner_checkpoint(runner: SmolVLARunner) -> str:
    try:
        return runner.policy.name_or_path  # type: ignore[attr-defined]
    except Exception:
        return "lerobot/smolvla_base"
