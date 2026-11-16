"""Test both independently trained PPO policies on matching held-out seeds."""
import argparse
import csv
from pathlib import Path

from stable_baselines3 import PPO

from .config import EnvConfig
from .environment import AUVNavigationEnv
from .evaluation import aggregate, evaluate
from .visualization import (
    animate_trajectory, plot_learning_curves, plot_summary,
    plot_trajectory, plot_trajectories,
)


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate the two PPO navigation scenarios.")
    parser.add_argument("--model-dir", type=Path, default=Path("result"))
    parser.add_argument("--seed", type=int, default=7, help="Training seed of both saved policies")
    # parser.add_argument("--checkpoint", choices=("best_model", "final_model"), default="best_model")
    parser.add_argument("--checkpoint", choices=("best_model", "final_model", "last_model"), default="best_model")
    parser.add_argument("--start-seed", type=int, default=10_000)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None,
                        help="Evaluation folder (default: result/seed_<seed>)")
    parser.add_argument("--gif-frames", type=int, default=120,
                        help="Maximum frames for each trajectory GIF")
    parser.add_argument("--gif-fps", type=int, default=12,
                        help="Playback frames per second for each trajectory GIF")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No rows to write")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = get_args()
    if args.episodes <= 0 or args.seed < 0 or args.start_seed < 0:
        raise ValueError("episodes must be positive and seeds nonnegative")
    if args.gif_frames < 2 or args.gif_fps <= 0:
        raise ValueError("gif-frames must be >= 2 and gif-fps must be positive")
    if args.output is None:
        args.output = args.model_dir / f"seed_{args.seed}"
    args.output.mkdir(parents=True, exist_ok=True)
    episode_rows = []
    summaries = {}
    representatives = {}
    for mode in ("none", "uniform"):
        model_path = args.model_dir / f"seed_{args.seed}" / mode / f"{args.checkpoint}.zip"
        if not model_path.is_file():
            raise FileNotFoundError(f"PPO checkpoint missing: {model_path}")
        model = PPO.load(model_path, device="cpu")
        env = AUVNavigationEnv(EnvConfig(current_mode=mode, observation_noise=0.0))
        results = evaluate(model, env, args.start_seed, args.episodes)
        episode_rows.extend(r.metrics(mode) for r in results)
        summaries[mode] = aggregate(results, mode)
        successes = [r for r in results if r.success]
        # Representative: median successful path length; otherwise nearest failure.
        representatives[mode] = (
            sorted(successes, key=lambda r: r.path_length)[len(successes) // 2]
            if successes else min(results, key=lambda r: r.final_distance)
        )
        s = summaries[mode]
        print(f"{mode}: success={s['success_rate']:.1%}, "
              f"collision={s['collision_rate']:.1%}, "
              f"out_of_bounds={s['out_of_bounds_rate']:.1%}, "
              f"timeout={s['timeout_rate']:.1%}")

    write_csv(args.output / "episodes.csv", episode_rows)
    write_csv(args.output / "summary.csv", [summaries[n] for n in ("none", "uniform")])
    plot_trajectories(representatives, args.output / "trajectories.png")
    plot_summary(summaries, args.output / "comparison.png")
    plot_learning_curves(
        {
            mode: args.model_dir / f"seed_{args.seed}" / mode / "evaluation_history.csv"
            for mode in ("none", "uniform")
        },
        args.output / "learning_curves.png",
    )
    for mode in ("none", "uniform"):
        plot_trajectory(representatives[mode], mode, args.output / f"trajectory_{mode}.png")
        animate_trajectory(
            representatives[mode], mode, args.output / f"trajectory_{mode}.gif",
            max_frames=args.gif_frames, fps=args.gif_fps,
        )
    print(f"Saved 2-scenario PPO evaluation to {args.output}")


if __name__ == "__main__":
    main()
