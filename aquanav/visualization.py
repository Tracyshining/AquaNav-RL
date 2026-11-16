"""Static figures and GIFs rendered from evaluated PPO episodes."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Polygon
import numpy as np

from .config import EnvConfig
from .currents import make_current
from .evaluation import EpisodeResult

SCENARIOS = ("none", "uniform")
SCENARIO_NAMES = {"none": "No current", "uniform": "Uniform current"}
SCENARIO_COLORS = {"none": "#146c94", "uniform": "#c35e39"}


def _prepare_axes(ax, mode: str, show_legend: bool = True) -> EnvConfig:
    """Draw the same environment in the static and animated trajectories."""
    if mode not in SCENARIOS:
        raise ValueError(f"Unsupported current scenario: {mode}")
    cfg = EnvConfig(current_mode=mode)
    if mode == "uniform":
        current = make_current(mode, cfg.current_strength)
        grid = np.linspace(-12, 12, 8)
        xs, ys = np.meshgrid(grid, grid)
        velocities = np.array([
            current.velocity(np.array([x, y]), 0.0)
            for x, y in zip(xs.ravel(), ys.ravel())
        ]).reshape((*xs.shape, 2))
        ax.quiver(xs, ys, velocities[..., 0], velocities[..., 1],
                  color="#8ab2cf", scale=7, width=0.003, alpha=0.7)
    for ox, oy, radius in cfg.obstacles:
        ax.add_patch(Circle((ox, oy), radius, color="#b04a4a", alpha=0.25))
        ax.add_patch(Circle((ox, oy), radius + cfg.vehicle_radius,
                            fill=False, edgecolor="#b04a4a", linestyle="--", linewidth=0.8))
    ax.scatter(*cfg.goal, color="#f5a623", marker="*", s=190, label="Goal", zorder=6)
    ax.set(xlim=(-15, 15), ylim=(-15, 15), xlabel="x [m]", ylabel="y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)
    if show_legend:
        ax.legend(loc="lower right", fontsize=8)
    return cfg


def _plot_episode(ax, result: EpisodeResult, mode: str) -> None:
    _prepare_axes(ax, mode, show_legend=False)
    positions = result.states[:, :2]
    ax.plot(positions[:, 0], positions[:, 1], color=SCENARIO_COLORS[mode],
            linewidth=2.3, label="PPO path", zorder=5)
    ax.scatter(*positions[0], color="#24272b", marker="o", s=38, label="Start", zorder=6)
    ax.scatter(*positions[-1], color=SCENARIO_COLORS[mode], marker="x", s=45,
               label="End", zorder=7)
    outcome = "success" if result.success else "failure"
    ax.set_title(f"{SCENARIO_NAMES[mode]} | {outcome} | seed {result.seed}")
    ax.legend(loc="lower right", fontsize=8)


def plot_trajectory(result: EpisodeResult, mode: str, output: str | Path) -> None:
    """One high-resolution static PPO path for the given scenario."""
    fig, ax = plt.subplots(figsize=(6.7, 6.1))
    _plot_episode(ax, result, mode)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_trajectories(
    representatives: dict[str, EpisodeResult], output: str | Path
) -> None:
    """Side-by-side comparison using each scenario's evaluated representative path."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7), sharex=True, sharey=True)
    for ax, mode in zip(axes, SCENARIOS):
        _plot_episode(ax, representatives[mode], mode)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _vehicle_vertices(x: float, y: float, psi: float, radius: float) -> np.ndarray:
    """Triangle pointing in the AUV heading direction."""
    local = np.array([[radius * 1.9, 0], [-radius, radius], [-radius, -radius]])
    c, s = np.cos(psi), np.sin(psi)
    rotation = np.array([[c, -s], [s, c]])
    return local @ rotation.T + np.array([x, y])


def animate_trajectory(
    result: EpisodeResult,
    mode: str,
    output: str | Path,
    *,
    max_frames: int = 120,
    fps: int = 12,
) -> None:
    """Export an animated GIF showing progressive PPO path and AUV heading.

    A long episode is uniformly subsampled for reasonable GIF size; first and
    last states are always shown. The path and outcome are from one actual
    deterministic evaluation episode, not from a synthetic animation.
    """
    if len(result.states) < 2:
        raise ValueError("An animation requires at least two states")
    if max_frames < 2 or fps <= 0:
        raise ValueError("max_frames must be >= 2 and fps must be positive")
    frame_count = min(len(result.states), max_frames)
    indices = np.unique(np.rint(np.linspace(0, len(result.states) - 1, frame_count)).astype(int))
    positions = result.states[:, :2]
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    cfg = _prepare_axes(ax, mode, show_legend=False)
    ax.scatter(*positions[0], color="#24272b", marker="o", s=38, label="Start", zorder=6)
    ax.set_title("Navigation", fontfamily="Times New Roman", fontsize=14)
    ax.legend(loc="lower right", fontsize=8)
    (trail,) = ax.plot([], [], color=SCENARIO_COLORS[mode], linewidth=2.5, zorder=5)
    auv = Polygon(_vehicle_vertices(*result.states[0, :3], cfg.vehicle_radius),
                  closed=True, facecolor=SCENARIO_COLORS[mode],
                  edgecolor="white", linewidth=0.8, zorder=9)
    ax.add_patch(auv)
    label = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=9,
                    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    end_text = "success" if result.success else "failure"

    def draw(index: int):
        idx = int(indices[index])
        trail.set_data(positions[:idx + 1, 0], positions[:idx + 1, 1])
        auv.set_xy(_vehicle_vertices(*result.states[idx, :3], cfg.vehicle_radius))
        label.set_text(f"t = {idx * cfg.dt:.1f} s   |   step = {idx}/{result.steps}"
                       + (f"   |   {end_text}" if idx == result.steps else ""))
        return trail, auv, label

    animation = FuncAnimation(fig, draw, frames=len(indices), interval=1000 / fps,
                              blit=False, repeat=False)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        animation.save(str(output), writer=PillowWriter(fps=fps), dpi=90)
    finally:
        plt.close(fig)


def plot_summary(stats: dict[str, dict], output: str | Path) -> None:
    """Compare held-out PPO mission outcomes, successful paths, and energy."""
    names = SCENARIOS
    labels = tuple(SCENARIO_NAMES[n] for n in names)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1))
    x = np.arange(2)
    colors = ("#4478a5", "#cf7e53")
    width = 0.32
    success = [stats[n]["success_rate"] * 100 for n in names]
    collision = [stats[n]["collision_rate"] * 100 for n in names]
    axes[0].bar(x - width / 2, success, width, color=colors[0], label="Success")
    axes[0].bar(x + width / 2, collision, width, color=colors[1], label="Collision")
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Episodes [%]")
    axes[0].set_title("Mission outcomes")
    axes[0].legend(fontsize=8)
    path = [stats[n]["mean_path_length_success_m"] for n in names]
    axes[1].bar(x, path, color=colors[0], width=0.5)
    axes[1].set_title("Path length on successful runs")
    axes[1].set_ylabel("Mean path [m]")
    for xi, value in enumerate(path):
        if not np.isfinite(value):
            axes[1].text(xi, 0.05, "N/A\n(no successes)", ha="center", va="bottom",
                         transform=axes[1].get_xaxis_transform(), fontsize=8)
    energy = [stats[n]["mean_energy_all"] for n in names]
    axes[2].bar(x, energy, color=colors[1], width=0.5)
    axes[2].set_title("Control-energy proxy (all runs)")
    axes[2].set_ylabel("Mean sum(action²) × dt")
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curves(
    histories: dict[str, str | Path], output: str | Path
) -> None:
    """Plot comparable validation curves from training, without additional rollouts."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    keys = ("success_rate", "collision_rate", "mean_reward")
    titles = ("Validation success", "Validation collision", "Validation reward")
    for mode in SCENARIOS:
        with Path(histories[mode]).open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f"No validation points in {histories[mode]}")
        steps = np.array([float(row["timesteps"]) for row in rows])
        for ax, key in zip(axes, keys):
            vals = np.array([float(row[key]) for row in rows])
            if key != "mean_reward":
                vals *= 100
            ax.plot(steps, vals, marker="o", markersize=2.7,
                    linewidth=1.8, label=SCENARIO_NAMES[mode], color=SCENARIO_COLORS[mode])
    for ax, title in zip(axes, titles):
        ax.set(xlabel="Training environment steps", title=title)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes[0].set_ylabel("Success [%]")
    axes[1].set_ylabel("Collision [%]")
    axes[2].set_ylabel("Mean episode reward")
    axes[0].set_ylim(0, 105)
    axes[1].set_ylim(0, 105)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
