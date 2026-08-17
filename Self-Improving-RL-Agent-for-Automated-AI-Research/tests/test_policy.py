import numpy as np

import conftest
from rl_agent.policy import PolicyNetwork


def test_forward_shapes():
    policy = PolicyNetwork(state_dim=6, action_dim=3, hidden_dim=8, seed=0)
    state = np.random.randn(6).astype(np.float32)
    rng = np.random.default_rng(0)
    action, log_prob, cache = policy.act(state, rng)
    assert action.shape == (3,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)
    assert isinstance(log_prob, float)


def test_flat_params_roundtrip():
    policy = PolicyNetwork(state_dim=4, action_dim=2, hidden_dim=6, seed=1)
    flat = policy.get_flat_params()
    other = PolicyNetwork(state_dim=4, action_dim=2, hidden_dim=6, seed=99)
    other.set_flat_params(flat.copy())
    np.testing.assert_allclose(policy.get_flat_params(), other.get_flat_params())


def test_gradients_change_log_prob_direction():
    policy = PolicyNetwork(state_dim=4, action_dim=2, hidden_dim=6, seed=2)
    rng = np.random.default_rng(2)
    state = np.random.randn(4).astype(np.float32)
    action, log_prob_before, cache = policy.act(state, rng, deterministic=True)
    target_action = np.clip(action + 0.3, -1, 1)

    grads = policy.grad_log_prob(target_action, cache)
    policy.apply_gradients(grads, lr=0.01, sign=+1.0)

    _, _, cache2 = policy.act(state, rng, deterministic=True)
    mean_after = cache2["mean"]
    dist_before = np.linalg.norm(cache["mean"] - target_action)
    dist_after = np.linalg.norm(mean_after - target_action)
    assert dist_after <= dist_before + 1e-6
