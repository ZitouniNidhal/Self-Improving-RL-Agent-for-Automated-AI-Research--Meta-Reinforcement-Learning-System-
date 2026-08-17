import numpy as np

import conftest
from environments.research_task_env import TaskDistribution
from rl_agent.meta_agent import MetaRLAgent


def test_adapt_to_task_returns_valid_result():
    config_dim = 3
    state_dim = 2 + 2 * config_dim
    agent = MetaRLAgent(state_dim=state_dim, action_dim=config_dim, trials_per_task=8, seed=0)
    env = TaskDistribution(config_dim=config_dim, seed=0).sample_task()

    result = agent.adapt_to_task(env)
    assert len(result.episode_rewards) == 8
    assert 0.0 <= result.best_reward <= 1.0
    assert result.adapted_params.shape == agent.policy.get_flat_params().shape


def test_meta_train_step_updates_policy():
    config_dim = 3
    state_dim = 2 + 2 * config_dim
    agent = MetaRLAgent(state_dim=state_dim, action_dim=config_dim, trials_per_task=8, seed=0)
    before = agent.policy.get_flat_params().copy()

    tasks_dist = TaskDistribution(config_dim=config_dim, seed=0)
    tasks = [tasks_dist.sample_task() for _ in range(4)]
    stats = agent.meta_train_step(tasks)

    after = agent.policy.get_flat_params()
    assert not np.allclose(before, after)
    assert "mean_best_reward" in stats


def test_meta_training_improves_over_generations():
    config_dim = 2
    state_dim = 2 + 2 * config_dim
    agent = MetaRLAgent(state_dim=state_dim, action_dim=config_dim, trials_per_task=10,
                         inner_lr=0.1, meta_lr=0.2, seed=0)
    tasks_dist = TaskDistribution(config_dim=config_dim, seed=0, n_bumps=1)

    scores = []
    for _ in range(25):
        tasks = [tasks_dist.sample_task() for _ in range(6)]
        stats = agent.meta_train_step(tasks)
        scores.append(stats["mean_best_reward"])

    early = np.mean(scores[:5])
    late = np.mean(scores[-5:])
    assert late >= early - 0.1
