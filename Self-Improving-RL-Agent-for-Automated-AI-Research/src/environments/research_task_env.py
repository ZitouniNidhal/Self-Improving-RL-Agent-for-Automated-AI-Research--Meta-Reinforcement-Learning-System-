from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class TaskDistribution:
    config_dim: int = 4
    n_bumps: int = 3
    noise_std: float = 0.05
    seed: int = 0

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def sample_task(self) -> "ResearchTaskEnv":
        rng = self._rng
        centers = rng.uniform(-0.8, 0.8, size=(self.n_bumps, self.config_dim)).astype(np.float32)
        widths = rng.uniform(0.15, 0.5, size=self.n_bumps).astype(np.float32)
        weights = rng.uniform(0.5, 1.0, size=self.n_bumps).astype(np.float32)
        weights /= weights.sum()
        return ResearchTaskEnv(
            config_dim=self.config_dim,
            centers=centers,
            widths=widths,
            weights=weights,
            noise_std=self.noise_std,
            rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
        )


class ResearchTaskEnv:
    def __init__(self, config_dim: int, centers: np.ndarray, widths: np.ndarray,
                 weights: np.ndarray, noise_std: float, rng: np.random.Generator):
        self.config_dim = config_dim
        self.centers = centers
        self.widths = widths
        self.weights = weights
        self.noise_std = noise_std
        self.rng = rng

    def true_reward(self, action: np.ndarray) -> float:
        total = 0.0
        for center, width, weight in zip(self.centers, self.widths, self.weights):
            d2 = np.sum((action - center) ** 2)
            total += weight * np.exp(-d2 / (2 * width ** 2))
        return float(np.clip(total, 0.0, 1.0))

    def step(self, action: np.ndarray) -> float:
        clean = self.true_reward(action)
        noisy = clean + self.rng.normal(0.0, self.noise_std)
        return float(np.clip(noisy, 0.0, 1.0))

    def optimum(self) -> Tuple[np.ndarray, float]:
        best_idx = int(np.argmax(self.weights))
        return self.centers[best_idx], float(self.weights[best_idx])

