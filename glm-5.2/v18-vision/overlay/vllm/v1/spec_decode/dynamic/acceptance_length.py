# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from math import floor
from typing import Sequence


@dataclass(frozen=True)
class AcceptanceLengthUpdate:
    previous_num_spec_tokens: int
    num_spec_tokens: int
    mean_num_accepted_tokens: float
    mean_num_draft_tokens: float
    raw_target_num_spec_tokens: int = 0
    acceptance_ratchet: bool = False
    decision_reason: str = ""
    position_conditional_acceptance: tuple[float, ...] = ()
    observation_window: int = 0
    tail_gain_23: float = 0.0
    position_4_gain: float = 0.0


class AcceptanceLengthController:
    """Adjust speculative depth from observed acceptance, using a depth ladder."""

    def __init__(
        self,
        max_num_spec_tokens: int,
        observation_window: int,
        depth_ladder: Sequence[int] | None = None,
    ) -> None:
        if max_num_spec_tokens <= 0:
            raise ValueError("max_num_spec_tokens must be greater than zero.")
        if observation_window <= 0:
            raise ValueError("observation_window must be greater than zero.")

        self.max_num_spec_tokens = max_num_spec_tokens
        self.observation_window = observation_window
        ladder = sorted(
            {
                int(depth)
                for depth in (depth_ladder or [])
                if 1 <= int(depth) <= max_num_spec_tokens
            }
        )
        if not ladder:
            ladder = list(range(1, max_num_spec_tokens + 1))
        if ladder[-1] != max_num_spec_tokens:
            ladder.append(max_num_spec_tokens)
        self.depth_ladder = tuple(ladder)
        self.num_spec_tokens = self.depth_ladder[0]

        self._num_observation_steps = 0
        self._num_drafts = 0
        self._num_draft_tokens = 0
        self._num_accepted_tokens = 0
        self._position_eligible = [0] * max_num_spec_tokens
        self._position_accepted = [0] * max_num_spec_tokens

    def observe_batch(
        self,
        *,
        num_drafts: int,
        num_draft_tokens: int,
        num_accepted_tokens: int,
        position_eligible: Sequence[int] | None = None,
        position_accepted: Sequence[int] | None = None,
    ) -> AcceptanceLengthUpdate | None:
        """Observe one scheduler step and occasionally update the depth."""
        if num_drafts < 0 or num_draft_tokens < 0 or num_accepted_tokens < 0:
            raise ValueError("Speculative decoding counts must be non-negative.")
        if num_accepted_tokens > num_draft_tokens:
            raise ValueError(
                "num_accepted_tokens must not exceed num_draft_tokens."
            )
        if num_drafts == 0:
            if num_draft_tokens or num_accepted_tokens:
                raise ValueError("Token counts require at least one draft.")
            return None

        self._num_observation_steps += 1
        self._num_drafts += num_drafts
        self._num_draft_tokens += num_draft_tokens
        self._num_accepted_tokens += num_accepted_tokens
        if position_eligible is not None or position_accepted is not None:
            if position_eligible is None or position_accepted is None:
                raise ValueError(
                    "Position eligibility and acceptance must be provided together."
                )
            if len(position_eligible) != len(position_accepted):
                raise ValueError(
                    "Position eligibility and acceptance lengths must match."
                )
            if len(position_eligible) > self.max_num_spec_tokens:
                raise ValueError("Position vectors exceed max_num_spec_tokens.")
            for i, (eligible, accepted) in enumerate(
                zip(position_eligible, position_accepted)
            ):
                if eligible < 0 or accepted < 0 or accepted > eligible:
                    raise ValueError("Invalid per-position acceptance counts.")
                self._position_eligible[i] += eligible
                self._position_accepted[i] += accepted

        # Stay conservative at the k=2 baseline, but resolve exploratory
        # k=4/k=5 probes quickly. The shorter probe window bounds the cost of
        # inheriting a high depth when workload character changes.
        effective_observation_window = (
            self.observation_window
            if self.num_spec_tokens == self.depth_ladder[0]
            else max(1, self.observation_window // 2)
        )
        if self._num_observation_steps < effective_observation_window:
            return None

        mean_num_accepted_tokens = self._num_accepted_tokens / self._num_drafts
        mean_num_draft_tokens = self._num_draft_tokens / self._num_drafts
        position_rates = tuple(
            accepted / eligible if eligible else 0.0
            for eligible, accepted in zip(
                self._position_eligible, self._position_accepted
            )
        )
        previous_num_spec_tokens = self.num_spec_tokens
        acceptance_ratchet = False
        decision_reason = "hold"

        tail_gain_23 = (
            (self._position_accepted[2] + self._position_accepted[3])
            / self._num_drafts
            if self._num_drafts and len(self._position_accepted) >= 4
            else 0.0
        )
        position_4_gain = (
            self._position_accepted[4] / self._num_drafts
            if self._num_drafts and len(self._position_accepted) >= 5
            else 0.0
        )

        # Production GLM policy: k=2 is the safe baseline. Head acceptance
        # only decides whether to probe k=4. Once above baseline, decisions
        # use the unconditional marginal tokens earned by the extra draft
        # positions, avoiding a second p0/p1 gate.
        if self.depth_ladder in ((2, 4), (2, 4, 5)):
            if self.num_spec_tokens == 2:
                head_ratio = (
                    mean_num_accepted_tokens / mean_num_draft_tokens
                    if mean_num_draft_tokens
                    else 0.0
                )
                target_num_spec_tokens = 4 if head_ratio >= 0.85 else 2
                decision_reason = (
                    "probe_k4"
                    if target_num_spec_tokens == 4
                    else "k2_baseline"
                )
                acceptance_ratchet = (
                    target_num_spec_tokens > self.num_spec_tokens
                )
            elif self.num_spec_tokens == 4:
                if (
                    self.depth_ladder == (2, 4, 5)
                    and tail_gain_23 >= 0.70
                ):
                    target_num_spec_tokens = 5
                    decision_reason = "probe_k5"
                    acceptance_ratchet = True
                elif tail_gain_23 >= 0.35:
                    target_num_spec_tokens = 4
                    decision_reason = "k4_hold"
                else:
                    target_num_spec_tokens = 2
                    decision_reason = "k4_tail_reject"
            else:
                if position_4_gain >= 0.15:
                    target_num_spec_tokens = 5
                    decision_reason = "k5_hold"
                elif tail_gain_23 >= 0.35:
                    target_num_spec_tokens = 4
                    decision_reason = "k5_p4_reject"
                else:
                    target_num_spec_tokens = 2
                    decision_reason = "k5_tail_reject"
        else:
            formula_target_num_spec_tokens = min(
                self.max_num_spec_tokens,
                max(1, floor(mean_num_accepted_tokens + 1.5)),
            )
            target_num_spec_tokens = max(
                (
                    depth
                    for depth in self.depth_ladder
                    if depth <= formula_target_num_spec_tokens
                ),
                default=self.depth_ladder[0],
            )
            decision_reason = "formula"

        self.num_spec_tokens = target_num_spec_tokens
        self._reset_window()
        return AcceptanceLengthUpdate(
            previous_num_spec_tokens=previous_num_spec_tokens,
            num_spec_tokens=self.num_spec_tokens,
            mean_num_accepted_tokens=mean_num_accepted_tokens,
            mean_num_draft_tokens=mean_num_draft_tokens,
            raw_target_num_spec_tokens=target_num_spec_tokens,
            acceptance_ratchet=acceptance_ratchet,
            decision_reason=decision_reason,
            position_conditional_acceptance=position_rates,
            observation_window=effective_observation_window,
            tail_gain_23=tail_gain_23,
            position_4_gain=position_4_gain,
        )

    def _reset_window(self) -> None:
        self._num_observation_steps = 0
        self._num_drafts = 0
        self._num_draft_tokens = 0
        self._num_accepted_tokens = 0
        self._position_eligible = [0] * self.max_num_spec_tokens
        self._position_accepted = [0] * self.max_num_spec_tokens
