"""The two evaluated scenarios: still water and a constant horizontal current."""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class CurrentField(ABC):
    @abstractmethod
    def velocity(self, position: np.ndarray, time: float) -> np.ndarray:
        """Return horizontal inertial-frame current velocity in m/s."""


@dataclass(frozen=True)
class NoCurrent(CurrentField):
    def velocity(self, position: np.ndarray, time: float) -> np.ndarray:
        return np.zeros(2, dtype=float)


@dataclass(frozen=True)
class UniformCurrent(CurrentField):
    strength: float = 0.55
    direction: float = np.deg2rad(25.0)

    def velocity(self, position: np.ndarray, time: float) -> np.ndarray:
        return self.strength * np.array(
            [np.cos(self.direction), np.sin(self.direction)], dtype=float
        )


def make_current(mode: str, strength: float) -> CurrentField:
    if mode == "none":
        return NoCurrent()
    if mode == "uniform":
        return UniformCurrent(strength=strength)
    raise ValueError(f"Unsupported current mode {mode!r}; choose 'none' or 'uniform'.")
