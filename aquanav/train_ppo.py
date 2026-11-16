"""Train an unmodified Stable-Baselines3 PPO (MlpPolicy) for one current mode."""
import argparse
import csv
import json
import platform
from dataclasses import asdict
from functools import partial
from pathlib import Path

import gymnasium
import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .config import EnvConfig
from .environment import AUVNavigationEnv
from .evaluation import aggregate, evaluate
from .gym_env import GymnasiumAUVEnv


def make_env(mode: str, rank: int):
    """Top-level worker factory; each worker receives a distinct SB3 reset seed."""
    del rank
    config = EnvConfig(current_mode=mode, observation_noise=0.005)
    # return Monitor(GymnasiumAUVEnv(config))
    # 训练时将奖励缩放至原来的 0.01，减小价值目标的量级；Monitor 和独立评估仍记录原始奖励 (๑•̀ㅂ•́)و✧
    # Scale training rewards by 0.01 to reduce value-target magnitude; Monitor and evaluation retain raw rewards.
    return gymnasium.wrappers.TransformReward(
        Monitor(GymnasiumAUVEnv(config)), lambda reward: 0.01 * reward
    )


def learning_rate(progress_remaining: float) -> float:
    # 随训练进度将学习率从 3e-4 线性减小到 3e-6，降低后期策略漂移 (＾▽＾)
    # Linearly decay the learning rate from 3e-4 to 3e-6 to reduce late-training policy drift.
    return 3e-6 + (3e-4 - 3e-6) * max(0.0, min(1.0, progress_remaining))


class FixedSeedEvaluation(BaseCallback):
    """Compare checkpoints on the same validation seeds; rank by success first."""

    def __init__(self, mode: str, output: Path, every: int, episodes: int, seed: int):
        super().__init__(verbose=0)
        self.env = AUVNavigationEnv(EnvConfig(current_mode=mode, observation_noise=0.0))
        self.output = output
        self.every = every
        self.episodes = episodes
        self.seed = seed
        self.last_step = 0
        self.best_score = None
        self.best_step = 0
        self.degraded_evaluations = 0
        self.stop_requested = False
        self.history_path = output / "evaluation_history.csv"
        self.fields = [
            "timesteps", "episodes", "success_rate", "collision_rate", "out_of_bounds_rate", "timeout_rate",
            "mean_reward", "mean_final_distance_m", "mean_energy_all",
            "mean_path_length_success_m", "mean_duration_success_s", "mean_energy_success",
        ]
        with self.history_path.open("w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=self.fields).writeheader()

    # def assess(self):
    def assess(self, count_degradation: bool = True):
        results = evaluate(self.model, self.env, self.seed, self.episodes)
        stats = aggregate(results, self.env.config.current_mode)
        row = {"timesteps": self.num_timesteps, **{k: stats[k] for k in self.fields[1:]}}
        with self.history_path.open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=self.fields).writerow(row)
        score = (stats["success_rate"], -stats["collision_rate"], stats["mean_reward"])
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_step = self.num_timesteps
            self.model.save(self.output / "best_model")
        print(
            f"[PPO validation] mode={self.env.config.current_mode} "
            f"steps={self.num_timesteps} success={stats['success_rate']:.1%} "
            f"collision={stats['collision_rate']:.1%} reward={stats['mean_reward']:.2f}",
            flush=True,
        )
        self.last_step = self.num_timesteps
        # 达到全成功后，连续三次验证成功率下降则请求早停；完整验证记录继续保留 (๑•̀ㅂ•́)و✧
        # Request early stopping after three consecutive success-rate drops from a fully successful checkpoint.
        if count_degradation:
            self.degraded_evaluations = (
                self.degraded_evaluations + 1
                if self.best_score[0] == 1.0 and stats["success_rate"] < 1.0 else 0
            )
            self.stop_requested = self.degraded_evaluations >= 3
            if self.stop_requested:
                print("[PPO early stop] Validation success declined for three consecutive evaluations.", flush=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_step >= self.every:
            self.assess()
        # return True
        return not self.stop_requested

    def _on_training_end(self) -> None:
        # self.assess()
        # 记录最后一次更新后的真实表现，避免在相同步数上重复累计早停次数 (＾▽＾)
        # Assess the actual last policy without counting a second degradation at the same training step.
        self.assess(count_degradation=False)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train standard PPO in one of two AUV scenarios.")
    parser.add_argument("--current", required=True, choices=("none", "uniform"))
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--vec-env", choices=("subproc", "dummy"), default="subproc")
    parser.add_argument("--eval-every", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=20_000)
    parser.add_argument("--output-dir", type=Path, default=Path("result"))
    return parser.parse_args()


def main() -> None:
    args = get_args()
    if min(args.steps, args.n_envs, args.eval_every, args.eval_episodes) <= 0:
        raise ValueError("steps, n-envs, eval-every and eval-episodes must all be positive")
    if args.seed < 0 or args.eval_seed < 0:
        raise ValueError("seed and eval-seed must be nonnegative")

    torch.set_num_threads(1)
    output = args.output_dir / f"seed_{args.seed}" / args.current
    output.mkdir(parents=True, exist_ok=True)
    config = EnvConfig(current_mode=args.current, observation_noise=0.005)
    (output / "training_config.json").write_text(
        json.dumps(
            {
                "algorithm": "stable_baselines3.PPO",
                "policy": "MlpPolicy",
                "scenario": args.current,
                "environment": asdict(config),
                "requested_timesteps": args.steps,
                "training_seed": args.seed,
                "n_envs": args.n_envs,
                "vectorization": args.vec_env,
                # "n_steps_per_env": 512,
                "n_steps_per_env": 2048,
                "batch_size": 256,
                "n_epochs": 10,
                # "learning_rate": 3e-4,
                "learning_rate": {"schedule": "linear", "initial": 3e-4, "final": 3e-6},
                "training_reward_scale": 0.01,
                "target_kl": 0.02,
                "policy_kwargs": {"log_std_init": -1.0},
                "gamma": 0.995,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "vf_coef": 0.5,
                # "ent_coef": 0.005,
                "ent_coef": 0.0,
                "eval_interval_timesteps": args.eval_every,
                "validation_episodes": args.eval_episodes,
                "validation_start_seed": args.eval_seed,
                "early_stop_degraded_evaluations": 3,
                "early_stop_after_success_rate": 1.0,
                "final_model_selection": "best_validation_checkpoint",
                "versions": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "torch": torch.__version__,
                    "gymnasium": gymnasium.__version__,
                    "stable_baselines3": stable_baselines3.__version__,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    env_fns = [partial(make_env, args.current, i) for i in range(args.n_envs)]
    training_env = (
        SubprocVecEnv(env_fns) if args.vec_env == "subproc" else DummyVecEnv(env_fns)
    )
    try:
        # 使用 SB3 的标准 PPO-Clip；策略概率比及截断目标由库内部计算（对应公式（6）、（7））o(*￣▽￣*)ブ
        # Use standard SB3 PPO-Clip; the library computes the policy ratio and clipped objective (Equations (6) and (7)).
        model = PPO(
            "MlpPolicy",
            training_env,
            # learning_rate=3e-4,
            learning_rate=learning_rate,
            # 用 KL 阈值提前停止单次更新中的剩余优化轮次，限制过大的策略变化 (｀・ω・´)
            # Stop the remaining optimization epochs of an update when its approximate KL becomes too large.
            target_kl=0.02,
            # 降低初始动作标准差至 exp(-1)，减少动作越界裁剪，保留高斯策略探索 (≧▽≦)
            # Initialize the action standard deviation at exp(-1) to reduce clipping while retaining Gaussian exploration.
            policy_kwargs={"log_std_init": -1.0},
            # # 每个并行环境采集 T=512 个时间步，N 个环境合并构成一次更新的数据（对应算法 1）(๑•̀ㅂ•́)و✧
            # # Collect T=512 steps per parallel environment; combine data from N environments for one update (Algorithm 1).
            # n_steps=512,
            # 每个环境采集 T=2048 步，扩大更新批次，降低短轨迹与终止奖励带来的估计波动 (๑•̀ㅂ•́)و✧
            # Collect T=2048 steps per environment to reduce estimation noise from short rollouts and terminal rewards.
            n_steps=2048,
            # 将一次采样得到的数据划分为大小为 M=256 的小批量（对应算法 1）(＾▽＾)
            # Split collected rollout data into minibatches of size M=256 (Algorithm 1).
            batch_size=256,
            # 对同一批采样数据进行 K=10 轮优化（对应算法 1）(ง •_•)ง
            # Optimize the same collected rollout data for K=10 epochs (Algorithm 1).
            n_epochs=10,
            # 折扣系数 γ=0.995 与 GAE 参数 λ=0.95 用于计算 TD 残差和优势估计（对应公式（11）、（12））(≧▽≦)
            # Discount γ=0.995 and GAE parameter λ=0.95 are used to compute TD residuals and advantage estimates (Equations (11) and (12)).
            gamma=0.995,
            gae_lambda=0.95,
            # 将策略概率比的裁剪范围设为 [1-ε, 1+ε]，其中 ε=0.2（对应公式（7））(｀・ω・´)
            # Set the policy-ratio clipping interval to [1-ε, 1+ε], with ε=0.2 (Equation (7)).
            clip_range=0.2,
            # 将价值函数平方误差的权重 c1 设为 0.5（对应公式（9））(｡･∀･)ﾉﾞ
            # Set the value-function squared-error weight c1 to 0.5 (Equation (9)).
            vf_coef=0.5,
            # # 将鼓励策略探索的熵奖励权重 c2 设为 0.005（对应公式（9））(ﾉ◕ヮ◕)ﾉ*:･ﾟ✧
            # # Set the entropy-bonus weight c2 to 0.005 to encourage exploration (Equation (9)).
            # ent_coef=0.005,
            # 缩放奖励后关闭额外熵奖励，避免持续鼓励高方差动作；采样探索仍由高斯策略提供 (ﾉ◕ヮ◕)ﾉ*:･ﾟ✧
            # Disable the extra entropy bonus after reward scaling; the Gaussian policy still provides sampled exploration.
            ent_coef=0.0,
            verbose=1,
            seed=args.seed,
            device="cpu",
        )
        callback = FixedSeedEvaluation(
            args.current, output, args.eval_every, args.eval_episodes, args.eval_seed
        )
        # 交替执行轨迹采样与 K 轮小批量更新，直至达到请求的训练步数（对应算法 1）( •̀ ω •́ )✧
        # Alternate rollout collection with K epochs of minibatch updates until the requested training budget is reached (Algorithm 1).
        model.learn(total_timesteps=args.steps, callback=callback)
        # model.save(output / "final_model")
        # 原始末次训练权重单独保存；最终交付权重恢复为验证集选出的最佳检查点
        # Save the actual last training policy separately; restore the best validation checkpoint for final delivery.
        model.save(output / "last_model")
        selected_model = PPO.load(output / "best_model", device="cpu")
        selected_model.save(output / "final_model")
        (output / "training_result.json").write_text(
            json.dumps(
                {
                    "actual_timesteps": model.num_timesteps,
                    "stopped_early": callback.stop_requested,
                    "selected_timesteps": callback.best_step,
                    "selected_validation_score": callback.best_score,
                    "final_model_source": "best_model.zip",
                    "last_training_model": "last_model.zip",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (output / "actual_timesteps.txt").write_text(f"{model.num_timesteps}\n", encoding="utf-8")
        print(f"Completed {args.current}: {model.num_timesteps} steps; saved to {output}")
    finally:
        training_env.close()


if __name__ == "__main__":
    main()
