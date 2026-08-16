# Self-Improving RL Agent for Automated AI Research
### A Meta-Reinforcement-Learning System

[![tests](https://github.com/ZitouniNidhal/Self-Improving-RL-Agent-for-Automated-AI-Research--Meta-Reinforcement-Learning-System-/actions/workflows/tests.yml/badge.svg)](https://github.com/ZitouniNidhal/Self-Improving-RL-Agent-for-Automated-AI-Research--Meta-Reinforcement-Learning-System-/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](requirements.txt)

A from-scratch, dependency-light (NumPy-only) reference implementation of a
**meta-reinforcement-learning agent that learns how to run AI research
experiments** — proposing configurations (hyperparameters / architecture
choices), observing a performance signal, and getting better at the search
itself over time. A lightweight **self-improvement controller** sits on
top and adapts the agent's own learning rates in response to its recent
progress, so the training process tunes itself rather than relying on a
fixed, hand-picked schedule.

## Why this project

Manually tuning hyperparameters, or the hyperparameters of a
hyperparameter-tuner, is a well-known bottleneck in AI research. This
project explores the idea end-to-end at a small, fully-understandable
scale:

1. **Meta-RL core** — an agent that, instead of learning one policy for
   one environment, learns a policy that adapts *quickly* to new,
   never-before-seen research tasks (few-shot adaptation), using
   [Reptile](https://arxiv.org/abs/1803.02999)-style meta-learning
   around a [REINFORCE](https://link.springer.com/article/10.1007/BF00992696)
   inner loop.
2. **Automated "AI research" environment** — a synthetic but non-trivial
   family of tasks (`ResearchTaskEnv`) standing in for real experiment
   outcomes: a noisy, non-convex performance landscape over a
   configuration space, sampled fresh from a task distribution for every
   training generation and for held-out evaluation.
3. **Self-improvement layer** — a small, auditable rule-based controller
   that watches the agent's own learning curve and reacts to plateaus or
   sustained improvement by shrinking or growing the agent's learning
   rates — the system adjusting *how it learns*, not just *what it
   learns*.

The whole pipeline runs in seconds on a laptop CPU with no GPU and no
deep-learning framework, so it's easy to read top to bottom, extend, and
use as a teaching example or a starting point for a more serious
research-automation system (see [Roadmap](#roadmap)).

## How it works

```
TaskDistribution ──samples──► ResearchTaskEnv (one synthetic research problem)
       ▲                                │
       │                        propose config / observe reward
       │                                ▼
MetaRLAgent  ◄── inner loop (REINFORCE) ── PolicyNetwork (cloned, adapted)
       │
       └── outer loop (Reptile): meta-params ← meta-params + lr·(avg(adapted) − meta-params)
       ▲
       │ observe(score)
SelfImprovementController — grows/shrinks inner_lr & meta_lr based on progress
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for a component-by-component
breakdown.

## Installation

```bash
git clone https://github.com/ZitouniNidhal/Self-Improving-RL-Agent-for-Automated-AI-Research--Meta-Reinforcement-Learning-System-.git
cd Self-Improving-RL-Agent-for-Automated-AI-Research--Meta-Reinforcement-Learning-System-
pip install -r requirements.txt
```

Only [NumPy](https://numpy.org/) is required to train and run the agent.
`matplotlib` and `pytest` (in `requirements-dev.txt`) are only needed for
plotting and running the test suite.

## Quickstart

```bash
python examples/quickstart.py
```

This meta-trains a tiny agent for 30 generations on 3-dimensional
synthetic research tasks and prints the learning curve, including any
learning-rate adjustments made by the self-improvement controller, then
evaluates it on a brand-new held-out task.

## Training a larger agent

```bash
python scripts/train.py \
    --config-dim 4 \
    --tasks-per-generation 10 \
    --generations 300 \
    --trials-per-task 15 \
    --log-dir runs/exp1
```

This writes `runs/exp1/metrics.csv`, periodic policy checkpoints
(`policy_gen*.npy`), a final checkpoint (`policy_final.npy`), and the run
config (`config.json`).

Run `python scripts/train.py --help` for the full list of options
(learning rates, meta-batch size, discount factor, seed, etc.).

## Evaluating a checkpoint

```bash
python scripts/evaluate.py \
    --checkpoint runs/exp1/policy_final.npy \
    --config-dim 4 \
    --n-tasks 20
```

Reports the agent's few-shot best-reward-found on 20 fresh, unseen
research tasks — the metric that matters for a *meta*-RL agent, since the
whole point is generalizing the search strategy to new problems rather
than memorizing one.

## Running the tests

```bash
cd tests
python -m pytest -q
```

## Project structure

```
.
├── src/
│   ├── rl_agent/
│   │   ├── policy.py            # NumPy Gaussian policy network + analytic gradients
│   │   ├── memory.py            # Per-task episodic memory / state features
│   │   ├── meta_agent.py        # Inner-loop REINFORCE + outer-loop Reptile
│   │   └── self_improvement.py  # Rule-based controller that tunes the agent's own LRs
│   ├── environments/
│   │   └── research_task_env.py # Synthetic "AI research" task distribution
│   ├── training/
│   │   └── trainer.py           # Orchestrates generations, logging, checkpoints
│   └── utils/
│       └── logger.py            # CSV metrics + checkpoint I/O
├── scripts/
│   ├── train.py                 # CLI: meta-train an agent
│   └── evaluate.py              # CLI: evaluate a checkpoint on held-out tasks
├── examples/
│   └── quickstart.py            # Minimal end-to-end example
├── tests/                       # pytest suite for every component
├── docs/
│   └── ARCHITECTURE.md
└── .github/workflows/tests.yml  # CI: tests across Python 3.9–3.12
```

## Design choices

* **NumPy-only core.** The policy network, its forward pass, and the
  analytic gradients of `log π(a|s)` are all implemented by hand. This
  keeps the whole learning algorithm inspectable without tracing through
  an autodiff framework, and keeps install/startup fast.
* **Reptile over full MAML.** Reptile only needs first-order gradients
  and is a few lines to implement and verify, while still capturing the
  "learn an initialization that adapts fast" idea central to meta-RL.
* **A rule-based self-improvement controller, not another neural net.**
  Keeping the outer control loop simple and auditable matters: a system
  that adjusts its own training dynamics should stay easy to explain and
  debug, not become a second black box on top of the first.
* **Synthetic but non-trivial tasks.** `ResearchTaskEnv` is not a single
  fixed function — each task is freshly sampled (random bump locations,
  widths, and weights) with observation noise, so the agent genuinely has
  to *search* rather than memorize, and held-out evaluation is a real
  generalization test.

## Roadmap

- [ ] Swap the REINFORCE inner loop for PPO for lower-variance adaptation.
- [ ] Second-order (full MAML) outer loop as an alternative to Reptile.
- [ ] Real hyperparameter-search backend (e.g. wrap `scikit-learn` /
      small PyTorch training jobs as the reward signal) alongside the
      synthetic environment.
- [ ] Richer task distributions (discrete architecture choices, variable
      config dimensionality, multi-fidelity / early-stopping signals).
- [ ] A small dashboard (`matplotlib` / `streamlit`) for browsing runs in
      `runs/`.
- [ ] Distributed / parallel task sampling for larger meta-batches.

Contributions toward any of these are very welcome — see
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

Released under the [MIT License](LICENSE).
