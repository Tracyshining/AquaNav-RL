from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class CurrentField(ABC):
    @abstractmethod
    def velocity(self, position: np.ndarray, time: float) -> np.ndarray:
        """Return the inertial-frame current velocity [north, east] in m/s."""


@dataclass(frozen=True)
class NoCurrent(CurrentField):
    def velocity(self, position: np.ndarray, time: float) -> np.ndarray:
        return np.zeros(2, dtype=float)


@dataclass(frozen=True)
class UniformCurrent(CurrentField):
    strength: float = 0.5
    direction: float = np.deg2rad(25.0)

    def velocity(self, position: np.ndarray, time: float) -> np.ndarray:
        return self.strength * np.array(
            [np.cos(self.direction), np.sin(self.direction)], dtype=float
        )


@dataclass(frozen=True)
class VortexCurrent(CurrentField):
    strength: float = 0.55
    center: tuple[float, float] = (0.0, 0.0)
    core_radius: float = 3.0

    def velocity(self, position: np.ndarray, time: float) -> np.ndarray:
        offset = np.asarray(position, dtype=float) - np.asarray(self.center)
        radius = np.linalg.norm(offset)
        if radius < 1e-9:
            return np.zeros(2, dtype=float)
        tangential = np.array([-offset[1], offset[0]]) / radius
        profile = 2.0 * radius * self.core_radius / (
            radius**2 + self.core_radius**2
        )
        return self.strength * profile * tangential


@dataclass(frozen=True)
class OscillatingCurrent(CurrentField):
    strength: float = 0.55
    period: float = 18.0

    def velocity(self, position: np.ndarray, time: float) -> np.ndarray:
        phase = 2.0 * np.pi * time / self.period
        spatial = 0.25 * np.sin(0.16 * position[0])
        direction = phase + spatial
        magnitude = self.strength * (0.7 + 0.3 * np.sin(0.5 * phase))
        return magnitude * np.array([np.cos(direction), np.sin(direction)])


def make_current(mode: str, strength: float) -> CurrentField:
    currents = {
        "none": NoCurrent(),
        "uniform": UniformCurrent(strength=strength),
        "vortex": VortexCurrent(strength=strength),
        "oscillating": OscillatingCurrent(strength=strength),
    }
    if mode not in currents:
        valid = ", ".join(currents)
        raise ValueError(f"Unknown current mode '{mode}'. Choose from: {valid}.")
    return currents[mode]

