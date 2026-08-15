"""
self_improvement.py

A lightweight controller that watches the agent's own meta-training curve
and adjusts the agent's *own* hyperparameters (inner learning rate, meta
learning rate, exploration std floor) in response to how training is
going. This is the "self-improving" layer sitting above the meta-RL
core: instead of a human babysitting the learning-rate schedule, the
system monitors its own progress and reacts to plateaus / instabilities.

The controller is intentionally simple and interpretable (a small
rule-based / bandit-style controller rather than a second neural
network) so behaviour stays auditable -- a system that improves itself
should not become a black box in the process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Deque, List
import numpy as np


@dataclass
class SelfImprovementController:
    patience: int = 5             # generations to wait before reacting to a plateau
    min_delta: float = 0.005      # improvement smaller than this counts as "no progress"
    lr_decay: float = 0.7         # shrink learning rates on plateau
    lr_boost: float = 1.15        # grow learning rates on sustained improvement
    boost_streak_needed: int = 3
    min_inner_lr: float = 1e-4
    max_inner_lr: float = 1.0
    min_meta_lr: float = 1e-3
    max_meta_lr: float = 1.0

    history: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    improve_streak: int = 0
    plateau_counter: int = 0
    log: List[str] = field(default_factory=list)

    def observe(self, agent, score: float) -> None:
        """Call once per meta-training generation with the latest score
        (e.g. mean_best_reward). May mutate `agent.inner_lr` / `agent.meta_lr`."""
        improved = len(self.history) > 0 and (score - self.history[-1]) > self.min_delta
        self.history.append(score)

        if improved:
            self.improve_streak += 1
            self.plateau_counter = 0
        else:
            self.improve_streak = 0
            self.plateau_counter += 1

        if self.improve_streak >= self.boost_streak_needed:
            self._scale_learning_rates(agent, self.lr_boost, reason="sustained improvement")
            self.improve_streak = 0

        elif self.plateau_counter >= self.patience:
            self._scale_learning_rates(agent, self.lr_decay, reason="plateau detected")
            self.plateau_counter = 0

    def _scale_learning_rates(self, agent, factor: float, reason: str) -> None:
        old_inner, old_meta = agent.inner_lr, agent.meta_lr
        agent.inner_lr = float(np.clip(agent.inner_lr * factor, self.min_inner_lr, self.max_inner_lr))
        agent.meta_lr = float(np.clip(agent.meta_lr * factor, self.min_meta_lr, self.max_meta_lr))
        msg = (f"[self-improvement] {reason}: "
               f"inner_lr {old_inner:.4f} -> {agent.inner_lr:.4f}, "
               f"meta_lr {old_meta:.4f} -> {agent.meta_lr:.4f}")
        self.log.append(msg)

    def summary(self) -> str:
        if not self.history:
            return "No observations yet."
        return (f"generations={len(self.history)}  "
                f"latest={self.history[-1]:.4f}  "
                f"best={max(self.history):.4f}  "
                f"adjustments={len(self.log)}")
