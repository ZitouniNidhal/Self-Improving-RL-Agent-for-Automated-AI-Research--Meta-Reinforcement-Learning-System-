#!/usr/bin/env python
"""
Train the self-improving meta-RL agent.

Example:
    python scripts/train.py --generations 300 --tasks-per-generation 10 \
        --config-dim 4 --log-dir runs/exp1
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from training.trainer import Trainer, TrainerConfig  # noqa: E402


def parse_args() -> TrainerConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-dim", type=int, default=4, help="Dimensionality of the hyperparameter/config space.")
    p.add_argument("--tasks-per-generation", type=int, default=8)
    p.add_argument("--generations", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-tasks", type=int, default=5)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--inner-lr", type=float, default=0.05)
    p.add_argument("--meta-lr", type=float, default=0.1)
    p.add_argument("--inner-steps", type=int, default=1)
    p.add_argument("--trials-per-task", type=int, default=15)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-dir", type=str, default="runs/default")
    args = p.parse_args()
    return TrainerConfig(
        config_dim=args.config_dim,
        tasks_per_generation=args.tasks_per_generation,
        generations=args.generations,
        eval_every=args.eval_every,
        eval_tasks=args.eval_tasks,
        hidden_dim=args.hidden_dim,
        inner_lr=args.inner_lr,
        meta_lr=args.meta_lr,
        inner_steps=args.inner_steps,
        trials_per_task=args.trials_per_task,
        gamma=args.gamma,
        seed=args.seed,
        log_dir=args.log_dir,
    )


def main():
    config = parse_args()
    trainer = Trainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
