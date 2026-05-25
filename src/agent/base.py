"""
src/agent/base.py
════════════════════════════════════════════════════════════════════════════
BaseAgent — Abstract base class for all RL agents.
════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseAgent(ABC):
    """Interface that all trading agents must implement."""

    @abstractmethod
    def act(self, obs: np.ndarray, valid_actions: list[int] | None = None,
            greedy: bool = False, **kwargs) -> int:
        """Select an action given observation."""
        ...

    @abstractmethod
    def store(self, obs: np.ndarray, action: int, reward: float,
              next_obs: np.ndarray, done: bool, **kwargs) -> None:
        """Store transition in replay buffer."""
        ...

    @abstractmethod
    def learn(self) -> float | None:
        """Perform one gradient update. Return loss or None if skipped."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Save agent state to disk."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Load agent state from disk."""
        ...

    @abstractmethod
    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        ...

    @abstractmethod
    def decay_lr(self) -> None:
        """Decay learning rate."""
        ...

    @property
    @abstractmethod
    def eps(self) -> float:
        """Current exploration rate."""
        ...

    @property
    @abstractmethod
    def current_lr(self) -> float:
        """Current learning rate."""
        ...
