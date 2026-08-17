from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from environments.research_task_env import TaskDistribution
from rl_agent.meta_agent import MetaRLAgent
from rl_agent.self_improvement import SelfImprovementController
from utils.logger import ExperimentLogger


@dataclass
class TrainerConfig:
    config_dim: int = 4
    tasks_per_generation: int = 8
    generations: int = 200
    eval_every: int = 10
    eval_tasks: int = 5
    hidden_dim: int = 32
    inner_lr: float = 0.05
    meta_lr: float = 0.1
    inner_steps: int = 1
    trials_per_task: int = 15
    gamma: float = 0.9
    seed: int = 0
    log_dir: str = "runs/default"


class Trainer:
    def __init__(self, config: TrainerConfig):
        self.config = config
        state_dim_probe_action_dim = config.config_dim
        state_dim = 2 + 2 * state_dim_probe_action_dim

        self.agent = MetaRLAgent(
            state_dim=state_dim,
            action_dim=config.config_dim,
            hidden_dim=config.hidden_dim,
            inner_lr=config.inner_lr,
            meta_lr=config.meta_lr,
            inner_steps=config.inner_steps,
            trials_per_task=config.trials_per_task,
            gamma=config.gamma,
            seed=config.seed,
        )
        self.train_tasks = TaskDistribution(config_dim=config.config_dim, seed=config.seed)
        self.eval_tasks_dist = TaskDistribution(config_dim=config.config_dim, seed=config.seed + 999)
        self.controller = SelfImprovementController()
        self.logger = ExperimentLogger(log_dir=config.log_dir)
        self.logger.save_config(config.__dict__)

    def run(self) -> None:
        for gen in range(1, self.config.generations + 1):
            tasks = [self.train_tasks.sample_task() for _ in range(self.config.tasks_per_generation)]
            stats = self.agent.meta_train_step(tasks)
            self.controller.observe(self.agent, stats["mean_best_reward"])

            metrics = {
                "mean_best_reward": stats["mean_best_reward"],
                "mean_episode_reward": stats["mean_episode_reward"],
                "inner_lr": self.agent.inner_lr,
                "meta_lr": self.agent.meta_lr,
            }

            if gen % self.config.eval_every == 0 or gen == self.config.generations:
                eval_score = self.evaluate()
                metrics["eval_best_reward"] = eval_score
                self.logger.save_checkpoint(self.agent.policy, name=f"policy_gen{gen}.npy")

            self.logger.log(gen, metrics)

        self.logger.save_checkpoint(self.agent.policy, name="policy_final.npy")
        self.logger.close()
        print(self.controller.summary())
        for line in self.controller.log[-10:]:
            print(line)

    def evaluate(self) -> float:
        scores = []
        for _ in range(self.config.eval_tasks):
            env = self.eval_tasks_dist.sample_task()
            result = self.agent.evaluate(env)
            scores.append(result.best_reward)
        return float(np.mean(scores))
