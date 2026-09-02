from dataclasses import dataclass

import numpy as np

from .config import AUVConfig
from .currents import CurrentField


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass
class PlanarAUV:
    config: AUVConfig
    current: CurrentField

    def __post_init__(self) -> None:
        self._mass = self.config.mass_matrix
        self._mass_inv = np.linalg.inv(self._mass)
        self._linear = np.asarray(self.config.linear_damping, dtype=float)
        self._quadratic = np.asarray(self.config.quadratic_damping, dtype=float)
        self._max_force = np.asarray(self.config.max_force, dtype=float)
        self._max_speed = np.asarray(self.config.max_speed, dtype=float)

    def derivative(self, state: np.ndarray, action: np.ndarray, time: float) -> np.ndarray:
        eta = state[:3]
        nu = state[3:]
        psi = eta[2]
        rotation = np.array(
            [[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]]
        )
        current_inertial = self.current.velocity(eta[:2], time)
        current_body = rotation.T @ current_inertial
        relative_velocity = nu.copy()
        relative_velocity[:2] -= current_body

        mx, my, _ = np.diag(self._mass)
        u_r, v_r, _ = relative_velocity
        coriolis = np.array(
            [[0.0, 0.0, -my * v_r], [0.0, 0.0, mx * u_r], [my * v_r, -mx * u_r, 0.0]]
        )
        damping = self._linear + self._quadratic * np.abs(relative_velocity)
        force = np.clip(action, -1.0, 1.0) * self._max_force
        acceleration = self._mass_inv @ (
            force - coriolis @ relative_velocity - damping * relative_velocity
        )
        kinematics = np.array(
            [
                rotation[0] @ nu[:2],
                rotation[1] @ nu[:2],
                nu[2],
            ]
        )
        return np.concatenate([kinematics, acceleration])

    def step(
        self, state: np.ndarray, action: np.ndarray, time: float, dt: float
    ) -> np.ndarray:
        # RK4 is stable enough for the nonlinear damping at the default 0.1 s step.
        k1 = self.derivative(state, action, time)
        k2 = self.derivative(state + 0.5 * dt * k1, action, time + 0.5 * dt)
        k3 = self.derivative(state + 0.5 * dt * k2, action, time + 0.5 * dt)
        k4 = self.derivative(state + dt * k3, action, time + dt)
        next_state = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        next_state[2] = wrap_angle(next_state[2])
        next_state[3:] = np.clip(next_state[3:], -self._max_speed, self._max_speed)
        return next_state

