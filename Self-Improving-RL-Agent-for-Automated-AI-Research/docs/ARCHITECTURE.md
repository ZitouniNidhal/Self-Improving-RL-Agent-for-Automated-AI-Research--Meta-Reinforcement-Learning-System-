# Architecture

## Overview

```
                          ┌─────────────────────────────┐
                          │      SelfImprovementController│
                          │  watches training curve,     │
                          │  tunes inner_lr / meta_lr     │
                          └───────────────▲──────────────┘
                                          │ observe(score)
                                          │
┌──────────────┐   sample tasks   ┌───────┴────────┐   Reptile update   ┌───────────────┐
│ TaskDistribution├────────────────►│   MetaRLAgent   ├────────────────────►│ PolicyNetwork │
│ (synthetic AI   │                 │  outer loop     │                     │ (meta params)  │
│  research tasks)│                 └───────┬────────┘                     └───────┬───────┘
└──────────────┘                            │ clone + adapt                        │ clone()
                                             ▼                                      │
                                   ┌───────────────────┐                            │
                                   │  Inner loop (REINFORCE)│◄──────────────────────┘
                                   │  per-task adaptation    │
                                   └──────────┬─────────────┘
                                              │ propose config / observe reward
                                              ▼
                                    ┌───────────────────┐
                                    │ ResearchTaskEnv    │
                                    │ (one research task)│
                                    └────────────────────┘
```

## Components

### `environments/research_task_env.py`
A synthetic stand-in for "running an AI experiment". Each `ResearchTaskEnv`
is a smooth, non-convex reward landscape over a configuration vector in
`[-1, 1]^d` (representing hyperparameters / architecture choices), built
from a random mixture of Gaussian bumps, with observation noise. A
`TaskDistribution` samples new tasks on demand, so the agent is always
meta-trained (and evaluated) on a *distribution* of research problems
rather than a single fixed one.

### `rl_agent/policy.py`
A small NumPy-only feed-forward Gaussian policy (`state -> action
distribution`). Implements its own forward pass and analytic
log-probability gradients, so no autodiff framework is required. Exposes
flat-parameter get/set for cloning and interpolation.

### `rl_agent/memory.py`
Tracks the trials taken within a single task episode (proposed
configuration, observed reward) and derives the state vector fed back
into the policy: best reward so far, fraction of trial budget used, last
proposed configuration, and best configuration found -- giving the agent
the context a human researcher would use ("what have I already tried,
and how well did it work?").

### `rl_agent/meta_agent.py`
Implements the two nested loops:

* **Inner loop** -- for one task, run an episode of `trials_per_task`
  proposals, compute REINFORCE (`grad log pi(a|s) * (return - baseline)`)
  gradients, and apply them to a *cloned* policy (fast adaptation).
* **Outer loop** -- Reptile meta-update: after adapting to a batch of
  sampled tasks, move the meta-policy's parameters a step toward the
  average of the task-adapted parameters. This is what makes the initial
  policy good at *becoming* good quickly on new tasks.

### `rl_agent/self_improvement.py`
A small, auditable rule-based controller that watches the meta-training
score over generations and reacts to plateaus (shrink learning rates) or
sustained improvement (grow learning rates), so the system adapts its
own training dynamics rather than relying on a fixed, hand-tuned
schedule.

### `training/trainer.py` + `scripts/train.py` / `scripts/evaluate.py`
Orchestration, CLI, logging (CSV metrics + `.npy` checkpoints), and a
held-out evaluation routine that measures few-shot adaptation on tasks
never seen during training.

## Why Reptile + REINFORCE instead of PPO/MAML+backprop?

The goal of this repository is to be a clear, hackable reference for the
*meta-RL-for-research-automation* idea rather than a state-of-the-art
benchmark result. Reptile is a first-order meta-learning method that
avoids second-order gradients, and REINFORCE with a simple return
baseline is easy to derive and verify by hand. Both are implemented from
scratch in NumPy so the whole pipeline can be read top to bottom without
tracing through a deep-learning framework. Swapping in PPO for the inner
loop or MAML's second-order update for the outer loop is a natural
extension -- see `docs/ROADMAP.md` in the README.
