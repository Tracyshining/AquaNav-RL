from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from .config import EnvConfig
from .currents import CurrentField
from .evaluation import EpisodeResult


def plot_benchmark(
    results: dict[str, EpisodeResult],
    config: EnvConfig,
    currents: dict[str, CurrentField],
    output_path: str | Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    trajectory_axis, metric_axis = axes

    grid = np.linspace(-config.world_size, config.world_size, 17)
    gx, gy = np.meshgrid(grid, grid)
    current_name = "vortex" if "vortex" in currents else next(iter(currents))
    primary_current = currents[current_name]
    vectors = np.array(
        [primary_current.velocity(np.array([x, y]), 0.0) for x, y in zip(gx.ravel(), gy.ravel())]
    )
    trajectory_axis.quiver(
        gx,
        gy,
        vectors[:, 0].reshape(gx.shape),
        vectors[:, 1].reshape(gy.shape),
        color="#8ecae6",
        alpha=0.6,
        scale=10,
    )

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(results)))
    for color, (name, result) in zip(colors, results.items()):
        trajectory_axis.plot(
            result.states[:, 0], result.states[:, 1], label=name, color=color, linewidth=2.2
        )
    for ox, oy, radius in config.obstacles:
        trajectory_axis.add_patch(Circle((ox, oy), radius, color="#d62828", alpha=0.38))
    trajectory_axis.scatter(*config.start[:2], marker="o", color="black", label="start", zorder=5)
    trajectory_axis.scatter(*config.goal, marker="*", s=180, color="#ffb703", label="goal", zorder=5)
    trajectory_axis.set(
        title=f"AUV trajectories (arrows: {current_name} current)",
        xlabel="North position [m]",
        ylabel="East position [m]",
        xlim=(-config.world_size, config.world_size),
        ylim=(-config.world_size, config.world_size),
        aspect="equal",
    )
    trajectory_axis.grid(alpha=0.2)
    trajectory_axis.legend(loc="upper left", fontsize=8)

    labels = list(results)
    success = [100.0 * float(results[name].success) for name in labels]
    path = [results[name].path_length for name in labels]
    energy = [results[name].control_energy for name in labels]
    x = np.arange(len(labels))
    width = 0.25
    metric_axis.bar(x - width, success, width, label="success [%]")
    metric_axis.bar(x, path, width, label="path [m]")
    metric_axis.bar(x + width, energy, width, label="energy proxy")
    metric_axis.set_title("Episode-level metrics")
    metric_axis.set_xticks(x, labels, rotation=20, ha="right")
    metric_axis.grid(axis="y", alpha=0.2)
    metric_axis.legend()

    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def animate_episode(
    result: EpisodeResult,
    config: EnvConfig,
    output_path: str | Path,
    fps: int = 20,
) -> None:
    from matplotlib.animation import FuncAnimation, PillowWriter

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.set(
        xlim=(-config.world_size, config.world_size),
        ylim=(-config.world_size, config.world_size),
        aspect="equal",
        xlabel="North position [m]",
        ylabel="East position [m]",
    )
    for ox, oy, radius in config.obstacles:
        axis.add_patch(Circle((ox, oy), radius, color="#d62828", alpha=0.4))
    axis.scatter(*config.goal, marker="*", s=180, color="#ffb703")
    path_line, = axis.plot([], [], color="#0077b6", linewidth=2)
    vehicle, = axis.plot([], [], marker="o", color="#023047", markersize=8)

    stride = max(1, len(result.states) // 240)
    frames = result.states[::stride]

    def update(index: int):
        path_line.set_data(frames[: index + 1, 0], frames[: index + 1, 1])
        vehicle.set_data([frames[index, 0]], [frames[index, 1]])
        return path_line, vehicle

    animation = FuncAnimation(figure, update, frames=len(frames), interval=1000 / fps)
    animation.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(figure)
