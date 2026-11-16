"""Deterministic PPO rollouts and reproducible navigation metrics."""
from dataclasses import dataclass

import numpy as np

from .environment import AUVNavigationEnv


@dataclass
class EpisodeResult:
    seed: int
    states: np.ndarray
    actions: np.ndarray
    currents: np.ndarray
    rewards: np.ndarray
    success: bool
    collision: bool
    out_of_bounds: bool
    timeout: bool
    duration: float
    path_length: float
    control_energy: float
    final_distance: float

    @property
    def steps(self) -> int:
        return len(self.actions)

    @property
    def total_reward(self) -> float:
        return float(self.rewards.sum())

    def metrics(self, scenario: str) -> dict:
        return {
            "scenario": scenario,
            "seed": self.seed,
            "success": int(self.success),
            "collision": int(self.collision),
            "out_of_bounds": int(self.out_of_bounds),
            "timeout": int(self.timeout),
            "steps": self.steps,
            "duration_s": round(self.duration, 4),
            "path_length_m": round(self.path_length, 4),
            "control_energy_proxy": round(self.control_energy, 4),
            "final_distance_m": round(self.final_distance, 4),
            "episode_reward": round(self.total_reward, 4),
        }


def run_episode(env: AUVNavigationEnv, model, seed: int, randomize: bool = True) -> EpisodeResult:
    """Run the supplied PPO policy using the exact episode seed and deterministic actions."""
    obs, _ = env.reset(seed=seed, randomize=randomize)
    states = [env.state.copy()]
    actions = []
    currents = [env.current.velocity(env.state[:2], env.time)]
    rewards = []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        action = np.clip(np.asarray(action, dtype=float).reshape(3), -1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step(action)
        states.append(env.state.copy())
        actions.append(action)
        currents.append(info["current"])
        rewards.append(float(reward))
        if terminated or truncated:
            break

    positions = np.asarray(states)[:, :2]
    actions_array = np.asarray(actions, dtype=float)
    return EpisodeResult(
        seed=seed,
        states=np.asarray(states, dtype=float),
        actions=actions_array,
        currents=np.asarray(currents, dtype=float),
        rewards=np.asarray(rewards, dtype=float),
        success=bool(info["success"]),
        collision=bool(info["collision"]),
        out_of_bounds=bool(info["out_of_bounds"]),
        timeout=bool(truncated and not terminated),
        duration=float(env.time),
        path_length=float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()),
        control_energy=float(np.square(actions_array).sum() * env.config.dt),
        final_distance=env.distance_to_goal,
    )


def evaluate(model, env: AUVNavigationEnv, start_seed: int, episodes: int) -> list[EpisodeResult]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    return [run_episode(env, model, seed=start_seed + i) for i in range(episodes)]


def aggregate(results: list[EpisodeResult], scenario: str) -> dict:
    if not results:
        raise ValueError("Cannot aggregate an empty evaluation")
    successful = [r for r in results if r.success]

    def mean(items, attr):
        return float(np.mean([getattr(r, attr) for r in items])) if items else float("nan")

    return {
        "scenario": scenario,
        "episodes": len(results),
        "success_rate": float(np.mean([r.success for r in results])),
        "collision_rate": float(np.mean([r.collision for r in results])),
        "out_of_bounds_rate": float(np.mean([r.out_of_bounds for r in results])),
        "timeout_rate": float(np.mean([r.timeout for r in results])),
        "mean_reward": mean(results, "total_reward"),
        "mean_final_distance_m": mean(results, "final_distance"),
        "mean_energy_all": mean(results, "control_energy"),
        "mean_path_length_success_m": mean(successful, "path_length"),
        "mean_duration_success_s": mean(successful, "duration"),
        "mean_energy_success": mean(successful, "control_energy"),
    }
