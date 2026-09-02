import argparse
from dataclasses import replace
from pathlib import Path

from .config import EnvConfig
from .controllers import LOSPIDController
from .environment import AUVNavigationEnv
from .evaluation import run_episode, save_summary
from .visualization import animate_episode, plot_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AquaNav-RL PID benchmark.")
    parser.add_argument(
        "--currents",
        nargs="+",
        default=["none", "uniform", "vortex", "oscillating"],
        choices=["none", "uniform", "vortex", "oscillating"],
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--animate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base_config = EnvConfig()
    results = {}
    currents = {}

    for mode in args.currents:
        env = AUVNavigationEnv(replace(base_config, current_mode=mode))
        result = run_episode(env, LOSPIDController(), seed=args.seed)
        results[mode] = result
        currents[mode] = env.current
        print(
            f"{mode:12s} success={result.success} collision={result.collision} "
            f"path={result.path_length:.2f} m energy={result.control_energy:.2f}"
        )

    save_summary(results, args.output / "summary.csv")
    plot_benchmark(results, base_config, currents, args.output / "benchmark.png")
    if args.animate:
        preferred = "vortex" if "vortex" in results else next(iter(results))
        animate_episode(results[preferred], base_config, args.output / f"{preferred}.gif")


if __name__ == "__main__":
    main()

