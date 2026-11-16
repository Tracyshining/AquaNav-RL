from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class AUVConfig:
    mass: float = 30.0
    yaw_inertia: float = 4.1
    added_mass: tuple[float, float, float] = (5.0, 18.0, 1.2)
    linear_damping: tuple[float, float, float] = (7.0, 18.0, 2.5)
    quadratic_damping: tuple[float, float, float] = (18.0, 36.0, 4.0)
    max_force: tuple[float, float, float] = (75.0, 55.0, 18.0)
    max_speed: tuple[float, float, float] = (2.5, 1.5, 1.8)

    @property
    def mass_matrix(self) -> np.ndarray:
        mx = self.mass + self.added_mass[0]
        my = self.mass + self.added_mass[1]
        mz = self.yaw_inertia + self.added_mass[2]
        return np.diag([mx, my, mz])


@dataclass(frozen=True)
class EnvConfig:
    dt: float = 0.1
    world_size: float = 25.0
    max_steps: int = 800
    goal_radius: float = 0.8
    vehicle_radius: float = 0.35
    start: tuple[float, float, float] = (-10.0, -8.0, 0.0)
    goal: tuple[float, float] = (10.0, 8.0)
    current_mode: str = "none"
    current_strength: float = 0.55
    observation_noise: float = 0.0
    reward_progress: float = 5.0
    reward_time: float = 0.02
    reward_energy: float = 0.002
    reward_collision: float = 100.0
    reward_goal: float = 120.0
    obstacles: tuple[tuple[float, float, float], ...] = field(
        default_factory=lambda: (
            (-3.0, 2.0, 2.0),
            (4.0, -2.0, 2.3),
            (0.0, 7.0, 1.5),
        )
    )
