"""Smoke-test result figures and GIF output without running a PPO training job."""
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from aquanav.evaluation import EpisodeResult
from aquanav.visualization import (
    animate_trajectory, plot_learning_curves, plot_summary,
    plot_trajectory, plot_trajectories,
)


def _episode(seed=3, success=True):
    n = 9
    states = np.zeros((n, 6), dtype=float)
    states[:, 0] = np.linspace(-10.0, 10.0, n)
    states[:, 1] = np.linspace(-8.0, 8.0, n)
    states[:, 2] = np.linspace(0.0, 0.7, n)
    return EpisodeResult(
        seed=seed,
        states=states,
        actions=np.zeros((n - 1, 3)),
        currents=np.zeros((n, 2)),
        rewards=np.ones(n - 1),
        success=success,
        collision=False,
        out_of_bounds=False,
        timeout=not success,
        duration=0.8,
        path_length=26.0,
        control_energy=0.0,
        final_distance=0.0 if success else 2.0,
    )


def test_two_trajectory_gifs_and_result_figures(tmp_path):
    reps = {"none": _episode(), "uniform": _episode(seed=4, success=False)}
    for mode in reps:
        png = tmp_path / f"trajectory_{mode}.png"
        gif = tmp_path / f"trajectory_{mode}.gif"
        plot_trajectory(reps[mode], mode, png)
        animate_trajectory(reps[mode], mode, gif, max_frames=6, fps=5)
        assert png.stat().st_size > 1000
        with Image.open(gif) as img:
            assert img.format == "GIF"
            assert img.n_frames >= 2
    plot_trajectories(reps, tmp_path / "trajectories.png")
    assert (tmp_path / "trajectories.png").stat().st_size > 1000


def test_summary_and_learning_curves(tmp_path):
    stats = {
        "none": {"success_rate": 0.0, "collision_rate": 0.5,
                 "mean_path_length_success_m": float("nan"), "mean_energy_all": 6.0},
        "uniform": {"success_rate": 1.0, "collision_rate": 0.0,
                    "mean_path_length_success_m": 30.0, "mean_energy_all": 7.0},
    }
    plot_summary(stats, tmp_path / "comparison.png")
    histories = {}
    for mode in ("none", "uniform"):
        path = tmp_path / f"{mode}.csv"
        histories[mode] = path
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("timesteps", "success_rate", "collision_rate", "mean_reward"))
            writer.writeheader()
            writer.writerow({"timesteps": 2048, "success_rate": 0.3,
                             "collision_rate": 0.2, "mean_reward": -4})
            writer.writerow({"timesteps": 4096, "success_rate": 0.5,
                             "collision_rate": 0.1, "mean_reward": 14})
    plot_learning_curves(histories, tmp_path / "learning_curves.png")
    assert (tmp_path / "comparison.png").stat().st_size > 1000
    assert (tmp_path / "learning_curves.png").stat().st_size > 1000
