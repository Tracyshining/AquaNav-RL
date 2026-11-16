from dataclasses import replace

import numpy as np

from .config import AUVConfig, EnvConfig
from .currents import make_current
from .dynamics import PlanarAUV


class AUVNavigationEnv:
    observation_size = 10
    action_size = 3

    def __init__(
        self,
        env_config: EnvConfig | None = None,
        auv_config: AUVConfig | None = None,
    ) -> None:
        self.config = env_config or EnvConfig()
        self.auv_config = auv_config or AUVConfig()
        self.current = make_current(
            self.config.current_mode, self.config.current_strength
        )
        self.vehicle = PlanarAUV(self.auv_config, self.current)
        self.rng = np.random.default_rng()
        self.state = np.zeros(6, dtype=float)
        self.time = 0.0
        self.steps = 0
        self.last_distance = 0.0

    def reset(
        self, seed: int | None = None, randomize: bool = False
    ) -> tuple[np.ndarray, dict]:
        self.rng = np.random.default_rng(seed)
        start = np.asarray(self.config.start, dtype=float)
        if randomize:
            start = start + self.rng.normal(0.0, [0.6, 0.6, 0.15])
        self.state = np.concatenate([start, np.zeros(3, dtype=float)])
        self.time = 0.0
        self.steps = 0
        self.last_distance = self.distance_to_goal
        return self._observation(), self._info(False, False)

    @property
    def distance_to_goal(self) -> float:
        return float(np.linalg.norm(np.asarray(self.config.goal) - self.state[:2]))

    def observation(self) -> np.ndarray:
        return self._observation()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=float)
        if action.shape != (self.action_size,):
            raise ValueError(f"Action must have shape ({self.action_size},).")
        self.state = self.vehicle.step(self.state, action, self.time, self.config.dt)
        self.time += self.config.dt
        self.steps += 1

        distance = self.distance_to_goal
        progress = self.last_distance - distance
        self.last_distance = distance
        collision = self._collision()
        out_of_bounds = np.any(np.abs(self.state[:2]) > self.config.world_size)
        success = distance <= self.config.goal_radius
        terminated = bool(collision or out_of_bounds or success)
        truncated = bool(self.steps >= self.config.max_steps and not terminated)

        reward = (
            self.config.reward_progress * progress
            - self.config.reward_time
            - self.config.reward_energy * float(np.dot(action, action))
            + self.config.reward_goal * float(success)
            - self.config.reward_collision * float(collision or out_of_bounds)
        )
        return self._observation(), reward, terminated, truncated, self._info(
            success, collision, bool(out_of_bounds)
        )

    def with_current(self, mode: str) -> "AUVNavigationEnv":
        return AUVNavigationEnv(replace(self.config, current_mode=mode), self.auv_config)

    def _observation(self) -> np.ndarray:
        goal_delta = np.asarray(self.config.goal) - self.state[:2]
        psi = self.state[2]
        rotation = np.array(
            [[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]]
        )
        current_body = rotation.T @ self.current.velocity(self.state[:2], self.time)
        obs = np.array(
            [
                goal_delta[0] / self.config.world_size,
                goal_delta[1] / self.config.world_size,
                np.cos(psi),
                np.sin(psi),
                self.state[3] / self.auv_config.max_speed[0],
                self.state[4] / self.auv_config.max_speed[1],
                self.state[5] / self.auv_config.max_speed[2],
                current_body[0],
                current_body[1],
                self.distance_to_goal / (2.0 * self.config.world_size),
            ],
            dtype=np.float32,
        )
        if self.config.observation_noise > 0.0:
            obs += self.rng.normal(0.0, self.config.observation_noise, obs.shape)
        return obs

    def _collision(self) -> bool:
        for ox, oy, radius in self.config.obstacles:
            separation = np.linalg.norm(self.state[:2] - np.array([ox, oy]))
            if separation <= radius + self.config.vehicle_radius:
                return True
        return False

    def _info(self, success: bool, collision: bool, out_of_bounds: bool = False) -> dict:
        return {
            "success": success,
            "collision": collision,
            "out_of_bounds": out_of_bounds,
            "distance_to_goal": self.distance_to_goal,
            "time": self.time,
            "state": self.state.copy(),
            "current": self.current.velocity(self.state[:2], self.time),
        }
