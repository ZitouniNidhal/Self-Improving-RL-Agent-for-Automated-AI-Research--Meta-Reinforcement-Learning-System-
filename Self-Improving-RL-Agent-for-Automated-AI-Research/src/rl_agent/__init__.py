"""
rl_agent
========

Core package for the Self-Improving RL Agent for Automated AI Research.

This package implements a meta-reinforcement-learning system whose job is
to *learn how to run AI research experiments*: proposing hyperparameters
and architecture choices, observing the resulting "performance" signal,
and adapting both within a single task (inner loop) and across many tasks
(outer / meta loop). A lightweight self-improvement controller further
tunes the agent's own learning behaviour (exploration, learning rate)
based on its recent trajectory of success.
"""

from .policy import PolicyNetwork
from .memory import EpisodicMemory, Trial
from .meta_agent import MetaRLAgent
from .self_improvement import SelfImprovementController

__all__ = [
    "PolicyNetwork",
    "EpisodicMemory",
    "Trial",
    "MetaRLAgent",
    "SelfImprovementController",
]

__version__ = "0.1.0"
