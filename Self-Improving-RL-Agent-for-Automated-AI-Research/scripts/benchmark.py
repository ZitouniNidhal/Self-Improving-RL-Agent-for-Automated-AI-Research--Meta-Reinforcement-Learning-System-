import argparse
import os
import sys
from statistics import mean, pstdev

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from environments.research_task_env import TaskDistribution
from rl_agent.meta_agent import MetaRLAgent


def evaluate_random_baseline(config_dim: int, n_tasks: int, seed: int, n_samples: int = 256) -> float:
    dist = TaskDistribution(config_dim=config_dim, seed=seed)
    scores = []
    rng = np.random.default_rng(seed)

    for _ in range(n_tasks):
        env = dist.sample_task()
        best = -1.0
        for _ in range(n_samples):
            action = rng.uniform(-1.0, 1.0, size=config_dim).astype(np.float32)
            best = max(best, env.true_reward(action))
        scores.append(best)

    return float(np.mean(scores))


def evaluate_agent(agent: MetaRLAgent, config_dim: int, n_tasks: int, seed: int) -> tuple[float, float]:
    dist = TaskDistribution(config_dim=config_dim, seed=seed)
    scores = []

    for _ in range(n_tasks):
        env = dist.sample_task()
        result = agent.evaluate(env)
        scores.append(float(result.best_reward))

    return float(np.mean(scores)), float(pstdev(scores) if len(scores) > 1 else 0.0)


def train_agent(config_dim: int, hidden_dim: int, trials_per_task: int, generations: int,
                tasks_per_generation: int, seed: int, inner_lr: float, meta_lr: float,
                inner_steps: int, gamma: float) -> MetaRLAgent:
    state_dim = 2 + 2 * config_dim
    agent = MetaRLAgent(
        state_dim=state_dim,
        action_dim=config_dim,
        hidden_dim=hidden_dim,
        inner_lr=inner_lr,
        meta_lr=meta_lr,
        inner_steps=inner_steps,
        trials_per_task=trials_per_task,
        gamma=gamma,
        seed=seed,
    )
    tasks_dist = TaskDistribution(config_dim=config_dim, seed=seed)

    for _ in range(generations):
        tasks = [tasks_dist.sample_task() for _ in range(tasks_per_generation)]
        agent.meta_train_step(tasks)

    return agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the Synthetic AI Research Meta-RL Agent.")
    parser.add_argument("--runs", type=int, default=3, help="Number of independent benchmark runs.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--config-dim", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--trials-per-task", type=int, default=10)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--tasks-per-generation", type=int, default=6)
    parser.add_argument("--inner-lr", type=float, default=0.05)
    parser.add_argument("--meta-lr", type=float, default=0.1)
    parser.add_argument("--inner-steps", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--eval-tasks", type=int, default=20)
    parser.add_argument("--random-samples", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_means = []
    run_stds = []
    baseline = evaluate_random_baseline(
        config_dim=args.config_dim,
        n_tasks=args.eval_tasks,
        seed=args.seed,
        n_samples=args.random_samples,
    )

    print(f"Random baseline (mean held-out reward): {baseline:.4f}")
    print("-" * 60)

    for run in range(args.runs):
        run_seed = args.seed + run
        agent = train_agent(
            config_dim=args.config_dim,
            hidden_dim=args.hidden_dim,
            trials_per_task=args.trials_per_task,
            generations=args.generations,
            tasks_per_generation=args.tasks_per_generation,
            seed=run_seed,
            inner_lr=args.inner_lr,
            meta_lr=args.meta_lr,
            inner_steps=args.inner_steps,
            gamma=args.gamma,
        )
        mean_reward, std_reward = evaluate_agent(agent, args.config_dim, args.eval_tasks, run_seed + 999)
        run_means.append(mean_reward)
        run_stds.append(std_reward)
        print(
            f"Run {run + 1:02d} | mean held-out reward = {mean_reward:.4f} | std = {std_reward:.4f} | "
            f"improvement_vs_baseline = {mean_reward - baseline:.4f}"
        )

    avg_mean = mean(run_means)
    avg_std = mean(run_stds)
    print("-" * 60)
    print(f"Average across {args.runs} runs: mean = {avg_mean:.4f}, std = {avg_std:.4f}")
    print(f"Average gain over random baseline: {avg_mean - baseline:.4f}")


if __name__ == "__main__":
    main()
