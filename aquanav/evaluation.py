from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .environment import AUVNavigationEnv


@dataclass
class EpisodeResult:
    states: np.ndarray
    actions: np.ndarray
    currents: np.ndarray
    rewards: np.ndarray
    success: bool
    collision: bool
    duration: float
    path_length: float
    control_energy: float
    final_distance: float


def run_episode(
    env: AUVNavigationEnv,
    controller,
    seed: int = 0,
    randomize: bool = False,
) -> EpisodeResult:
    env.reset(seed=seed, randomize=randomize)
    controller.reset()
    states = [env.state.copy()]
    actions: list[np.ndarray] = []
    currents = [env.current.velocity(env.state[:2], env.time)]
    rewards: list[float] = []
    success = False
    collision = False

    while True:
        action = controller.act(env.state.copy(), np.asarray(env.config.goal), env.config.dt)
        _, reward, terminated, truncated, info = env.step(action)
        states.append(env.state.copy())
        actions.append(action)
        currents.append(info["current"])
        rewards.append(reward)
        success = bool(info["success"])
        collision = bool(info["collision"])
        if terminated or truncated:
            break

    state_array = np.asarray(states)
    action_array = np.asarray(actions)
    increments = np.diff(state_array[:, :2], axis=0)
    return EpisodeResult(
        states=state_array,
        actions=action_array,
        currents=np.asarray(currents),
        rewards=np.asarray(rewards),
        success=success,
        collision=collision,
        duration=env.time,
        path_length=float(np.linalg.norm(increments, axis=1).sum()),
        control_energy=float(np.square(action_array).sum() * env.config.dt),
        final_distance=env.distance_to_goal,
    )


def save_summary(results: dict[str, EpisodeResult], path: str | Path) -> None:
    rows = ["scenario,success,collision,duration,path_length,control_energy,final_distance"]
    for name, result in results.items():
        rows.append(
            f"{name},{int(result.success)},{int(result.collision)},"
            f"{result.duration:.3f},{result.path_length:.3f},"
            f"{result.control_energy:.3f},{result.final_distance:.3f}"
        )
    Path(path).write_text("\n".join(rows) + "\n", encoding="utf-8")

