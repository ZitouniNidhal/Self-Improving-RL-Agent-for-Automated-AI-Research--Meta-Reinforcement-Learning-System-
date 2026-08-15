"""
meta_agent.py

The core meta-reinforcement-learning agent.

Two nested loops:

  Inner loop (per task, "fast adaptation"):
      For a single research task (a `ResearchTaskEnv`), run a short
      episode of trials. On each trial the policy proposes a
      configuration, observes a reward, and we accumulate a REINFORCE
      gradient (grad log pi(a|s) * return). At the end of the episode we
      take a few gradient ascent steps *on a cloned copy* of the policy,
      producing "task-adapted" parameters. This is the RL-agent doing
      one small research project and getting better at it in real time.

  Outer loop (across tasks, "meta-learning" -- Reptile):
      After adapting to a batch of sampled tasks, we move the *meta*
      parameters a small step towards the average of the task-adapted
      parameters. Over many tasks this shapes the initial policy into
      one whose few-shot adaptation is fast and effective on *new*,
      unseen research tasks -- i.e. the agent gets better at getting
      better, which is the "self-improving" property this project is
      named for.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Callable, Optional

from .policy import PolicyNetwork, Gradients
from .memory import EpisodicMemory, Trial


@dataclass
class InnerLoopResult:
    adapted_params: np.ndarray
    episode_rewards: List[float]
    best_reward: float


class MetaRLAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 32,
        inner_lr: float = 0.05,
        meta_lr: float = 0.1,
        inner_steps: int = 1,
        trials_per_task: int = 15,
        gamma: float = 0.9,
        seed: int = 0,
    ):
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim, seed)
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.inner_steps = inner_steps
        self.trials_per_task = trials_per_task
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # Inner loop: adapt to ONE task starting from the current meta policy
    # ------------------------------------------------------------------ #
    def adapt_to_task(self, env, deterministic_eval: bool = False) -> InnerLoopResult:
        working_policy = self.policy.clone()
        memory = EpisodicMemory(action_dim=working_policy.action_dim, max_trials=self.trials_per_task)

        for step in range(self.inner_steps):
            memory.reset()
            trials: List[Trial] = []
            for t in range(self.trials_per_task):
                state = memory.current_state()
                action, log_prob, cache = working_policy.act(state, self.rng)
                reward = env.step(action)
                trial = Trial(state=state, action=action, reward=reward, log_prob=log_prob, cache=cache)
                memory.add(trial)
                trials.append(trial)

            returns = memory.discounted_returns(gamma=self.gamma)
            baseline = float(np.mean(returns)) if returns else 0.0

            # Accumulate REINFORCE gradient: sum_t grad log pi(a_t|s_t) * (R_t - baseline)
            accumulated: Optional[Gradients] = None
            for trial, R in zip(trials, returns):
                advantage = R - baseline
                g = working_policy.grad_log_prob(trial.action, trial.cache).scale(advantage)
                accumulated = g if accumulated is None else accumulated + g

            if accumulated is not None:
                n = max(len(trials), 1)
                working_policy.apply_gradients(accumulated.scale(1.0 / n), lr=self.inner_lr, sign=+1.0)

        # Final evaluation episode with the adapted policy (greedy-ish, low noise)
        memory.reset()
        eval_rewards = []
        for t in range(self.trials_per_task):
            state = memory.current_state()
            action, log_prob, cache = working_policy.act(state, self.rng, deterministic=deterministic_eval)
            reward = env.step(action)
            memory.add(Trial(state, action, reward, log_prob, cache))
            eval_rewards.append(reward)

        return InnerLoopResult(
            adapted_params=working_policy.get_flat_params(),
            episode_rewards=eval_rewards,
            best_reward=memory.best_reward,
        )

    # ------------------------------------------------------------------ #
    # Outer loop: Reptile meta-update across a batch of tasks
    # ------------------------------------------------------------------ #
    def meta_train_step(self, tasks: List) -> dict:
        meta_params = self.policy.get_flat_params()
        adapted_batch = []
        best_rewards = []
        mean_rewards = []

        for env in tasks:
            result = self.adapt_to_task(env)
            adapted_batch.append(result.adapted_params)
            best_rewards.append(result.best_reward)
            mean_rewards.append(float(np.mean(result.episode_rewards)))

        avg_adapted = np.mean(adapted_batch, axis=0)
        # Reptile update: theta <- theta + meta_lr * (avg_adapted_theta - theta)
        new_params = meta_params + self.meta_lr * (avg_adapted - meta_params)
        self.policy.set_flat_params(new_params)

        return {
            "mean_best_reward": float(np.mean(best_rewards)),
            "mean_episode_reward": float(np.mean(mean_rewards)),
            "best_reward_std": float(np.std(best_rewards)),
        }

    def evaluate(self, env, deterministic: bool = True) -> InnerLoopResult:
        """Evaluate current meta-policy's few-shot adaptation on a fresh task
        (no gradient applied to `self.policy` -- purely diagnostic)."""
        return self.adapt_to_task(env, deterministic_eval=deterministic)
