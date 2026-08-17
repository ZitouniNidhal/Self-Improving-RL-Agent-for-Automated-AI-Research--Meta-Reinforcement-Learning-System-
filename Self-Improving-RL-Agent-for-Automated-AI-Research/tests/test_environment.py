import numpy as np

import conftest
from environments.research_task_env import TaskDistribution


def test_task_sampling_reproducible():
    dist1 = TaskDistribution(config_dim=3, seed=42)
    dist2 = TaskDistribution(config_dim=3, seed=42)
    env1 = dist1.sample_task()
    env2 = dist2.sample_task()
    np.testing.assert_allclose(env1.centers, env2.centers)
    np.testing.assert_allclose(env1.weights, env2.weights)


def test_reward_bounded():
    dist = TaskDistribution(config_dim=4, seed=7)
    env = dist.sample_task()
    rng = np.random.default_rng(0)
    for _ in range(50):
        action = rng.uniform(-1, 1, size=4).astype(np.float32)
        reward = env.step(action)
        assert 0.0 <= reward <= 1.0


def test_optimum_beats_random():
    dist = TaskDistribution(config_dim=3, seed=3, noise_std=0.0)
    env = dist.sample_task()
    best_center, _ = env.optimum()
    reward_at_optimum = env.true_reward(best_center)

    rng = np.random.default_rng(1)
    random_rewards = [env.true_reward(rng.uniform(-1, 1, size=3)) for _ in range(200)]
    assert reward_at_optimum >= float(np.mean(random_rewards))
    assert reward_at_optimum >= float(np.percentile(random_rewards, 90))
