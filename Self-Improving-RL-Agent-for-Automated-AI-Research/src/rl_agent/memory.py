"""
memory.py

Episodic memory of "research trials". Each trial records the configuration
that was proposed, the observed reward (proxy for model performance), and
book-keeping used to build the next state vector (so the agent can
condition on its own history, similar to how a human researcher
remembers what they already tried).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class Trial:
    state: np.ndarray
    action: np.ndarray
    reward: float
    log_prob: float
    cache: dict = field(repr=False, default_factory=dict)


class EpisodicMemory:
    """Stores the trials of a single task episode and derives state features."""

    def __init__(self, action_dim: int, max_trials: int = 50):
        self.action_dim = action_dim
        self.max_trials = max_trials
        self.trials: List[Trial] = []
        self.best_reward: float = -np.inf
        self.best_action: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.trials.clear()
        self.best_reward = -np.inf
        self.best_action = None

    def state_dim(self) -> int:
        # [best_reward, trials_used_frac, last_action(action_dim), best_action(action_dim)]
        return 2 + 2 * self.action_dim

    def current_state(self) -> np.ndarray:
        last_action = self.trials[-1].action if self.trials else np.zeros(self.action_dim, dtype=np.float32)
        best_action = self.best_action if self.best_action is not None else np.zeros(self.action_dim, dtype=np.float32)
        best_reward = 0.0 if self.best_reward == -np.inf else self.best_reward
        frac_used = len(self.trials) / max(self.max_trials, 1)
        return np.concatenate(
            [[best_reward, frac_used], last_action, best_action]
        ).astype(np.float32)

    def add(self, trial: Trial) -> None:
        self.trials.append(trial)
        if trial.reward > self.best_reward:
            self.best_reward = trial.reward
            self.best_action = trial.action

    def discounted_returns(self, gamma: float = 0.95) -> List[float]:
        """Return-to-go for each trial in the episode (simple REINFORCE target)."""
        returns = []
        running = 0.0
        for trial in reversed(self.trials):
            running = trial.reward + gamma * running
            returns.append(running)
        returns.reverse()
        return returns

    def __len__(self) -> int:
        return len(self.trials)
