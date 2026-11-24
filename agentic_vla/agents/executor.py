"""Executor agent (spec §4 AGENT 4).

Translates the selected skill into robot actions via the SmolVLA runner, executing a
BOUNDED action chunk (not the whole task blindly) so the critic can verify between
chunks: OBSERVE -> INFER -> ACT -> OBSERVE -> VERIFY (spec §4).

It NEVER claims success; it only reports what the environment did. The official success
flag observed mid-chunk is surfaced for the critic to judge.
"""
from __future__ import annotations

from ..envs import LiberoEnv
from ..policy import SmolVLARunner
from ..schemas import ExecutorMessage


class ExecutorAgent:
    name = "executor"

    def __init__(self, env: LiberoEnv, runner: SmolVLARunner):
        self.env = env
        self.runner = runner
        self._last_obs: dict | None = None

    def reset(self, obs: dict) -> None:
        self.runner.reset()
        self._last_obs = obs

    def execute_chunk(self, instruction: str, max_steps: int) -> ExecutorMessage:
        if self._last_obs is None:
            raise RuntimeError("Executor.reset(obs) must be called before execute_chunk.")

        obs = self._last_obs
        steps = 0
        policy_calls = 0
        env_success = False
        done = False
        try:
            for _ in range(max_steps):
                action = self.runner.select_action(obs, instruction)
                policy_calls += 1
                step = self.env.step(action)
                steps += 1
                obs = step.obs
                self._last_obs = obs
                if step.success:
                    env_success = True
                    break
                if step.done:
                    done = True
                    break
        except Exception as e:  # malformed/non-finite action, policy failure (spec §9)
            return ExecutorMessage(
                status="POLICY_EXECUTION_ERROR",
                reason=f"{type(e).__name__}: {e}",
                steps_executed=steps, policy_calls=policy_calls, env_success=False,
            )

        status = "DONE" if (env_success or done) else "CHUNK_DONE"
        return ExecutorMessage(
            status=status,
            reason=f"Executed {steps} step(s); env_success={env_success}.",
            steps_executed=steps, policy_calls=policy_calls, env_success=env_success,
        )
