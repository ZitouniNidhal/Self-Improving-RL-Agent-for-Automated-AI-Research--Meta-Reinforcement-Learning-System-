from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List

import numpy as np


@dataclass
class SelfImprovementController:
    patience: int = 5
    min_delta: float = 0.005
    lr_decay: float = 0.7
    lr_boost: float = 1.15
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
        msg = (
            f"[self-improvement] {reason}: "
            f"inner_lr {old_inner:.4f} -> {agent.inner_lr:.4f}, "
            f"meta_lr {old_meta:.4f} -> {agent.meta_lr:.4f}"
        )
        self.log.append(msg)

    def summary(self) -> str:
        if not self.history:
            return "No observations yet."
        return (
            f"generations={len(self.history)}  "
            f"latest={self.history[-1]:.4f}  "
            f"best={max(self.history):.4f}  "
            f"adjustments={len(self.log)}"
        )
