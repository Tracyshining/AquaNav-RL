from dataclasses import dataclass

import numpy as np

from .dynamics import wrap_angle


class RandomController:
    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        return None

    def act(self, state: np.ndarray, goal: np.ndarray, dt: float) -> np.ndarray:
        return self.rng.uniform(-1.0, 1.0, 3)


@dataclass
class LOSPIDController:
    surge_gain: float = 0.9
    sway_gain: float = 0.15
    yaw_kp: float = 1.8
    yaw_ki: float = 0.06
    yaw_kd: float = 0.35
    desired_speed: float = 1.25

    def __post_init__(self) -> None:
        self.integral = 0.0
        self.previous_error = 0.0

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_error = 0.0

    def act(self, state: np.ndarray, goal: np.ndarray, dt: float) -> np.ndarray:
        delta = np.asarray(goal) - state[:2]
        desired_heading = np.arctan2(delta[1], delta[0])
        heading_error = wrap_angle(desired_heading - state[2])
        self.integral = np.clip(self.integral + heading_error * dt, -2.0, 2.0)
        derivative = (heading_error - self.previous_error) / dt
        self.previous_error = heading_error

        alignment = max(0.0, np.cos(heading_error))
        surge = self.surge_gain * (self.desired_speed * alignment - state[3])
        sway = -self.sway_gain * state[4]
        yaw = (
            self.yaw_kp * heading_error
            + self.yaw_ki * self.integral
            + self.yaw_kd * derivative
        )
        return np.clip(np.array([surge, sway, yaw]), -1.0, 1.0)

