"""Critic agent (spec §4 AGENT 5).

Decides SUCCESS / PARTIAL_SUCCESS / FAILURE / UNCERTAIN / RECOVERY_REQUIRED. It cleanly
separates three conditions (spec §4): policy-execution completion != subtask completion
!= overall task completion.

Scoring uses the OFFICIAL LIBERO success predicate. Oracle ground-truth may be consulted
for internal diagnostics ONLY, and any such use is flagged oracle_assisted=True so it is
never mixed into a vision-only performance claim (spec §4, §9).

M2: rules-first. M3 adds an LLM to reason about ambiguous failures.
"""
from __future__ import annotations

from ..envs import LiberoEnv
from ..schemas import CriticMessage, ExecutorMessage


class CriticAgent:
    name = "critic"

    def __init__(self, env: LiberoEnv, llm=None, use_oracle_diagnostics: bool = True):
        self.env = env
        self.llm = llm
        self.use_oracle_diagnostics = use_oracle_diagnostics

    def evaluate(
        self,
        exec_msg: ExecutorMessage,
        is_last_subtask: bool,
        chunks_used: int,
        max_chunks: int,
    ) -> CriticMessage:
        # 1) hard execution error -> recovery
        if exec_msg.status == "POLICY_EXECUTION_ERROR":
            return CriticMessage(
                status="RECOVERY_REQUIRED",
                reason="Executor reported a policy execution error.",
                failure_category="POLICY_EXECUTION_ERROR",
                recommended_next_action="Reobserve and retry with a bounded chunk.",
            )

        # 2) official success signal (the scoring source of truth)
        env_success = exec_msg.env_success or self.env.check_success()
        if env_success:
            status = "SUCCESS" if is_last_subtask else "PARTIAL_SUCCESS"
            return CriticMessage(
                status=status,
                reason="LIBERO official success predicate satisfied.",
                evidence=[{"signal": "libero_success", "value": True}],
                confidence=1.0,
            )

        # 3) not yet successful — decide continue vs recover vs fail
        oracle_note = None
        oracle_assisted = False
        if self.use_oracle_diagnostics:
            oracle = self.env.oracle_state()        # DIAGNOSTICS ONLY, labeled below
            if oracle:
                oracle_note = {"oracle_state_keys": list(oracle.keys())}
                oracle_assisted = True

        if chunks_used < max_chunks:
            return CriticMessage(
                status="UNCERTAIN",
                reason=(
                    "No success yet; execution budget remains. Continue with another "
                    "bounded chunk after reobserving."
                ),
                failure_category="VERIFICATION_UNCERTAIN",
                evidence=[oracle_note] if oracle_note else [],
                oracle_assisted=oracle_assisted,
                recommended_next_action="Execute another chunk.",
                confidence=None,
            )

        # budget for this subtask exhausted
        return CriticMessage(
            status="RECOVERY_REQUIRED",
            reason="Execution budget for this subtask exhausted without success.",
            failure_category="TASK_STATE_MISMATCH",
            evidence=[oracle_note] if oracle_note else [],
            oracle_assisted=oracle_assisted,
            recommended_next_action="Reobserve, replan, or try an alternative strategy.",
            confidence=None,
        )
