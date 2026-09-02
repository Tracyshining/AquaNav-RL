import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from .config import EnvConfig
from .environment import AUVNavigationEnv
from .evaluation import run_episode, save_summary
from .visualization import animate_episode, plot_benchmark


class PPOController:
    def __init__(self, model: PPO) -> None:
        self.model = model
        self.env: AUVNavigationEnv | None = None

    def attach(self, env: AUVNavigationEnv) -> None:
        self.env = env

    def reset(self) -> None:
        return None

    def act(self, state: np.ndarray, goal: np.ndarray, dt: float) -> np.ndarray:
        if self.env is None:
            raise RuntimeError("Attach an environment before evaluation.")
        observation = self.env.observation()
        action, _ = self.model.predict(observation, deterministic=True)
        return np.asarray(action)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO policy.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--current", default="vortex")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, default=Path("artifacts/ppo"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    env = AUVNavigationEnv(EnvConfig(current_mode=args.current))
    controller = PPOController(PPO.load(args.model))
    controller.attach(env)
    result = run_episode(env, controller, seed=args.seed)
    results = {f"ppo-{args.current}": result}
    save_summary(results, args.output / "summary.csv")
    plot_benchmark(results, env.config, {args.current: env.current}, args.output / "trajectory.png")
    animate_episode(result, env.config, args.output / "trajectory.gif")
    print(
        f"success={result.success} collision={result.collision} "
        f"path={result.path_length:.2f} m energy={result.control_energy:.2f}"
    )


if __name__ == "__main__":
    main()
