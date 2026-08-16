
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rl_agent.meta_agent import MetaRLAgent  # noqa: E402
from environments.research_task_env import TaskDistribution  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--config-dim", type=int, default=4)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--trials-per-task", type=int, default=15)
    p.add_argument("--n-tasks", type=int, default=20)
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()

    state_dim = 2 + 2 * args.config_dim
    agent = MetaRLAgent(
        state_dim=state_dim,
        action_dim=args.config_dim,
        hidden_dim=args.hidden_dim,
        trials_per_task=args.trials_per_task,
        seed=args.seed,
    )
    flat = np.load(args.checkpoint)
    agent.policy.set_flat_params(flat)

    dist = TaskDistribution(config_dim=args.config_dim, seed=args.seed)
    best_rewards = []
    for i in range(args.n_tasks):
        env = dist.sample_task()
        result = agent.evaluate(env)
        best_rewards.append(result.best_reward)
        print(f"task {i:02d}: best_reward={result.best_reward:.4f}")

    print("-" * 40)
    print(f"mean best_reward over {args.n_tasks} tasks: {np.mean(best_rewards):.4f} "
          f"(+/- {np.std(best_rewards):.4f})")


if __name__ == "__main__":
    main()
