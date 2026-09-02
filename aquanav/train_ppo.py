import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from .config import EnvConfig
from .gym_env import GymnasiumAUVEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on AquaNav-RL.")
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--current", default="vortex")
    parser.add_argument("--output", type=Path, default=Path("models/ppo_auv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = EnvConfig(current_mode=args.current, observation_noise=0.005)
    env = Monitor(GymnasiumAUVEnv(config))
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.005,
        verbose=1,
        seed=args.seed,
    )
    model.learn(total_timesteps=args.steps, progress_bar=True)
    model.save(args.output)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Evaluation reward: {mean_reward:.2f} +/- {std_reward:.2f}")


if __name__ == "__main__":
    main()

