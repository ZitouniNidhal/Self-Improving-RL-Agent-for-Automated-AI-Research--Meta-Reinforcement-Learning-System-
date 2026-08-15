"""
quickstart.py

Smallest possible end-to-end example: run a few meta-training generations
on tiny synthetic research tasks and print the improvement curve. Meant
to run in a few seconds on a laptop with no GPU and no external
dependencies beyond NumPy.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rl_agent.meta_agent import MetaRLAgent
from rl_agent.self_improvement import SelfImprovementController
from environments.research_task_env import TaskDistribution


def main():
    config_dim = 3
    state_dim = 2 + 2 * config_dim

    agent = MetaRLAgent(state_dim=state_dim, action_dim=config_dim, trials_per_task=10, seed=1)
    controller = SelfImprovementController(patience=3)
    tasks_dist = TaskDistribution(config_dim=config_dim, seed=1)

    print("Meta-training a self-improving RL agent on synthetic 'AI research' tasks...\n")
    for gen in range(1, 31):
        tasks = [tasks_dist.sample_task() for _ in range(6)]
        stats = agent.meta_train_step(tasks)
        controller.observe(agent, stats["mean_best_reward"])
        print(f"gen {gen:02d}  mean_best_reward={stats['mean_best_reward']:.3f}  "
              f"inner_lr={agent.inner_lr:.4f}  meta_lr={agent.meta_lr:.4f}")

    print("\n" + controller.summary())

    # Try the trained agent on a brand-new, never-before-seen task.
    holdout_env = TaskDistribution(config_dim=config_dim, seed=999).sample_task()
    result = agent.evaluate(holdout_env)
    print(f"\nHeld-out task best reward found in {len(result.episode_rewards)} trials: "
          f"{result.best_reward:.3f}")


if __name__ == "__main__":
    main()
