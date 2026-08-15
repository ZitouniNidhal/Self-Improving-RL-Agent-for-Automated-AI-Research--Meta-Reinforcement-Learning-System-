import conftest  # noqa: F401
from types import SimpleNamespace
from rl_agent.self_improvement import SelfImprovementController


def test_plateau_decays_learning_rates():
    agent = SimpleNamespace(inner_lr=0.1, meta_lr=0.2)
    controller = SelfImprovementController(patience=2, min_delta=0.01)

    # Flat scores -> should trigger a decay after `patience` no-improvement steps.
    for score in [0.5, 0.5, 0.5, 0.5]:
        controller.observe(agent, score)

    assert agent.inner_lr < 0.1
    assert agent.meta_lr < 0.2


def test_sustained_improvement_boosts_learning_rates():
    agent = SimpleNamespace(inner_lr=0.05, meta_lr=0.05)
    controller = SelfImprovementController(boost_streak_needed=2, min_delta=0.01)

    for score in [0.1, 0.3, 0.5, 0.7]:
        controller.observe(agent, score)

    assert agent.inner_lr > 0.05
    assert agent.meta_lr > 0.05


def test_learning_rates_stay_within_bounds():
    agent = SimpleNamespace(inner_lr=0.9, meta_lr=0.9)
    controller = SelfImprovementController(boost_streak_needed=1, min_delta=0.0,
                                            max_inner_lr=1.0, max_meta_lr=1.0)
    for score in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        controller.observe(agent, score)
    assert agent.inner_lr <= 1.0
    assert agent.meta_lr <= 1.0
