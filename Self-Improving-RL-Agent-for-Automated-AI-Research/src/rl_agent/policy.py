"""
policy.py

A small, dependency-free (NumPy only) feed-forward Gaussian policy network.

The policy maps a *state* (features describing the current research trial:
best result so far, number of trials used, last proposed configuration,
etc.) to a distribution over the next *action* (a proposed hyperparameter /
architecture configuration, encoded as a real-valued vector in [-1, 1]^d).

Design goals:
    * No heavy ML framework dependency -> easy to run anywhere, fast to
      install, easy to read end-to-end.
    * Supports both sampling (for exploration during training) and
      analytic gradients of the log-probability (needed for REINFORCE /
      Reptile-style meta-updates).
    * Weights are stored as a flat parameter vector so they can be easily
      cloned, interpolated (Reptile), or serialized.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


def _xavier(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float32)


@dataclass
class PolicyNetwork:
    """A 2-hidden-layer MLP producing a diagonal-Gaussian action distribution."""

    state_dim: int
    action_dim: int
    hidden_dim: int = 32
    seed: int = 0

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.W1 = _xavier(self.state_dim, self.hidden_dim, rng)
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        self.W2 = _xavier(self.hidden_dim, self.hidden_dim, rng)
        self.b2 = np.zeros(self.hidden_dim, dtype=np.float32)
        self.W3 = _xavier(self.hidden_dim, self.action_dim, rng)
        self.b3 = np.zeros(self.action_dim, dtype=np.float32)
        # Learned log std-dev for the Gaussian policy (state-independent,
        # a common simplification that keeps the model small and stable).
        self.log_std = np.full(self.action_dim, -0.5, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Forward pass
    # ------------------------------------------------------------------ #
    def _forward(self, state: np.ndarray) -> Tuple[np.ndarray, dict]:
        z1 = state @ self.W1 + self.b1
        h1 = np.tanh(z1)
        z2 = h1 @ self.W2 + self.b2
        h2 = np.tanh(z2)
        z3 = h2 @ self.W3 + self.b3
        mean = np.tanh(z3)  # actions live in [-1, 1]
        cache = dict(state=state, z1=z1, h1=h1, z2=z2, h2=h2, z3=z3, mean=mean)
        return mean, cache

    def act(self, state: np.ndarray, rng: np.random.Generator, deterministic: bool = False):
        """Sample an action (or return the mean action if deterministic)."""
        mean, cache = self._forward(state)
        std = np.exp(self.log_std)
        if deterministic:
            action = mean
        else:
            action = mean + std * rng.standard_normal(self.action_dim).astype(np.float32)
        action = np.clip(action, -1.0, 1.0)
        log_prob = self._log_prob(action, mean, std)
        return action, log_prob, cache

    @staticmethod
    def _log_prob(action: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
        var = std ** 2
        return float(
            -0.5 * np.sum(((action - mean) ** 2) / var + np.log(2 * np.pi * var))
        )

    # ------------------------------------------------------------------ #
    # REINFORCE gradient of log pi(a|s) w.r.t. every parameter
    # ------------------------------------------------------------------ #
    def grad_log_prob(self, action: np.ndarray, cache: dict) -> "Gradients":
        mean = cache["mean"]
        std = np.exp(self.log_std)
        # d log_prob / d mean = (a - mean) / std^2
        d_mean = (action - mean) / (std ** 2)
        # d mean / d z3 = 1 - tanh(z3)^2
        d_z3 = d_mean * (1 - np.tanh(cache["z3"]) ** 2)

        dW3 = np.outer(cache["h2"], d_z3)
        db3 = d_z3

        d_h2 = d_z3 @ self.W3.T
        d_z2 = d_h2 * (1 - np.tanh(cache["z2"]) ** 2)
        dW2 = np.outer(cache["h1"], d_z2)
        db2 = d_z2

        d_h1 = d_z2 @ self.W2.T
        d_z1 = d_h1 * (1 - np.tanh(cache["z1"]) ** 2)
        dW1 = np.outer(cache["state"], d_z1)
        db1 = d_z1

        # d log_prob / d log_std = ((a-mean)^2 / std^2) - 1
        d_log_std = ((action - mean) ** 2) / (std ** 2) - 1.0

        return Gradients(dW1, db1, dW2, db2, dW3, db3, d_log_std)

    # ------------------------------------------------------------------ #
    # Parameter (de)serialization -- used by the meta-learner for Reptile
    # style interpolation between "task-adapted" and "meta" parameters.
    # ------------------------------------------------------------------ #
    def get_flat_params(self) -> np.ndarray:
        return np.concatenate(
            [
                self.W1.ravel(), self.b1.ravel(),
                self.W2.ravel(), self.b2.ravel(),
                self.W3.ravel(), self.b3.ravel(),
                self.log_std.ravel(),
            ]
        )

    def set_flat_params(self, flat: np.ndarray) -> None:
        shapes = [self.W1.shape, self.b1.shape, self.W2.shape, self.b2.shape,
                  self.W3.shape, self.b3.shape, self.log_std.shape]
        idx = 0
        arrays = []
        for shape in shapes:
            size = int(np.prod(shape))
            arrays.append(flat[idx: idx + size].reshape(shape).astype(np.float32))
            idx += size
        (self.W1, self.b1, self.W2, self.b2,
         self.W3, self.b3, self.log_std) = arrays

    def clone(self) -> "PolicyNetwork":
        clone = PolicyNetwork(self.state_dim, self.action_dim, self.hidden_dim, self.seed)
        clone.set_flat_params(self.get_flat_params().copy())
        return clone

    def apply_gradients(self, grads: "Gradients", lr: float, sign: float = 1.0) -> None:
        """In-place SGD-style update: theta += sign * lr * grad."""
        self.W1 += sign * lr * grads.dW1
        self.b1 += sign * lr * grads.db1
        self.W2 += sign * lr * grads.dW2
        self.b2 += sign * lr * grads.db2
        self.W3 += sign * lr * grads.dW3
        self.b3 += sign * lr * grads.db3
        self.log_std += sign * lr * grads.d_log_std
        np.clip(self.log_std, -3.0, 1.0, out=self.log_std)


@dataclass
class Gradients:
    dW1: np.ndarray
    db1: np.ndarray
    dW2: np.ndarray
    db2: np.ndarray
    dW3: np.ndarray
    db3: np.ndarray
    d_log_std: np.ndarray

    def scale(self, factor: float) -> "Gradients":
        return Gradients(*(factor * g for g in
                            (self.dW1, self.db1, self.dW2, self.db2,
                             self.dW3, self.db3, self.d_log_std)))

    def __add__(self, other: "Gradients") -> "Gradients":
        return Gradients(
            self.dW1 + other.dW1, self.db1 + other.db1,
            self.dW2 + other.dW2, self.db2 + other.db2,
            self.dW3 + other.dW3, self.db3 + other.db3,
            self.d_log_std + other.d_log_std,
        )
