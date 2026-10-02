"""Benchmark and empirical evaluation module for LINKMEM.

Evaluates how different distance threshold theta (θ) values affect:
- Success rate
- Average episode steps
- Average reward
- Final memory size
- Average memory comparisons per step
- Number of timeouts
- Average nodes created

Across standard environment presets using the existing LearningEngine and Agent.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
# Use Agg backend for headless environments without X11
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from linkmem.core.agent import Agent
from linkmem.core.environment import GridWorld
from linkmem.core.memory import ExperienceMemory
from linkmem.engine.learning_engine import LearningEngine
from linkmem.gui.environments import (
    RandomObstaclesConfig,
    build_preset,
    validate_preset,
)


@dataclass
class BenchmarkConfig:
    """Configuration for empirical benchmark experiment."""

    theta_values: List[float] = field(
        default_factory=lambda: [0.10, 0.20, 0.35, 0.50, 0.75]
    )
    environment_names: List[str] = field(
        default_factory=lambda: [
            "Empty Grid",
            "Simple Maze",
            "Complex Maze",
            "Random Obstacles",
        ]
    )
    episodes_per_run: int = 10
    max_steps_per_episode: int = 100
    max_memory_capacity: int = 100
    random_seed: int = 42
    random_obstacles_count: int = 6
    random_obstacles_seed: int = 0
    results_dir: str = "results"
    csv_filename: str = "benchmark_results.csv"


@dataclass
class BenchmarkRow:
    """Summary metrics for a specific (theta, environment) benchmark run."""

    theta: float
    environment: str
    episodes: int
    success_rate: float
    avg_steps: float
    avg_reward: float
    final_memory_size: int
    avg_comparisons_per_step: float
    timeouts: int
    nodes_created: int
    avg_nodes_created: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "theta": self.theta,
            "environment": self.environment,
            "episodes": self.episodes,
            "success_rate": round(self.success_rate, 4),
            "avg_steps": round(self.avg_steps, 2),
            "avg_reward": round(self.avg_reward, 2),
            "final_memory_size": self.final_memory_size,
            "avg_comparisons_per_step": round(self.avg_comparisons_per_step, 2),
            "timeouts": self.timeouts,
            "nodes_created": self.nodes_created,
            "avg_nodes_created": round(self.avg_nodes_created, 2),
        }


def run_single_benchmark(
    theta: float,
    env_name: str,
    config: Optional[BenchmarkConfig] = None,
) -> BenchmarkRow:
    """Execute multiple episodes for a specific theta and environment preset.

    Uses the existing LearningEngine and Agent.
    """
    cfg = config or BenchmarkConfig()

    # Build and validate environment preset
    if env_name == "Random Obstacles":
        random_cfg = RandomObstaclesConfig(
            width=8,
            height=8,
            obstacle_count=cfg.random_obstacles_count,
            seed=cfg.random_obstacles_seed,
        )
        preset = build_preset("Random Obstacles", random_cfg=random_cfg)
    else:
        preset = build_preset(env_name)

    is_valid, err = validate_preset(preset)
    if not is_valid:
        raise ValueError(f"Environment preset '{env_name}' is invalid: {err}")

    # Construct GridWorld environment
    env = GridWorld(
        width=preset.width,
        height=preset.height,
        start_pos=preset.start_pos,
        goal_pos=preset.goal_pos,
        obstacles=set(preset.obstacles),
        max_steps=cfg.max_steps_per_episode,
    )

    # Initialize ExperienceMemory and Agent
    memory = ExperienceMemory(max_capacity=cfg.max_memory_capacity)
    agent = Agent(
        memory=memory,
        theta=theta,
        seed=cfg.random_seed,
    )

    # Initialize LearningEngine
    engine = LearningEngine(
        env=env,
        memory=memory,
        agent=agent,
        max_memory_size=cfg.max_memory_capacity,
    )

    # Run configured number of episodes
    for _ in range(cfg.episodes_per_run):
        engine.run_episode()

    # Aggregate metrics
    history = engine.history
    num_episodes = len(history)
    successes = sum(1 for ep in history if ep.success)
    timeouts = num_episodes - successes
    success_rate = successes / max(1, num_episodes)
    avg_steps = sum(ep.steps for ep in history) / max(1, num_episodes)
    avg_reward = sum(ep.total_reward for ep in history) / max(1, num_episodes)
    final_memory_size = memory.size

    total_steps = sum(ep.steps for ep in history)
    total_comparisons = memory.total_comparisons
    avg_comps_per_step = total_comparisons / max(1, total_steps)

    nodes_created = max(0, memory._next_id - 1)
    avg_nodes_created = nodes_created / max(1, num_episodes)

    return BenchmarkRow(
        theta=theta,
        environment=env_name,
        episodes=num_episodes,
        success_rate=success_rate,
        avg_steps=avg_steps,
        avg_reward=avg_reward,
        final_memory_size=final_memory_size,
        avg_comparisons_per_step=avg_comps_per_step,
        timeouts=timeouts,
        nodes_created=nodes_created,
        avg_nodes_created=avg_nodes_created,
    )


def save_results_to_csv(results: List[BenchmarkRow], output_path: str | Path) -> None:
    """Save benchmark results to a CSV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "theta",
        "environment",
        "episodes",
        "success_rate",
        "avg_steps",
        "avg_reward",
        "final_memory_size",
        "avg_comparisons_per_step",
        "timeouts",
        "nodes_created",
        "avg_nodes_created",
    ]

    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row.to_dict())


def plot_benchmark_results(
    results: List[BenchmarkRow],
    output_dir: str | Path,
) -> Dict[str, str]:
    """Generate Matplotlib charts visualizing theta impact across environments.

    Generated charts:
    - theta_success_rate.png: Success Rate vs Theta
    - theta_memory_size.png: Final Memory Size vs Theta
    - theta_comparisons.png: Comparisons per Step vs Theta
    - theta_episode_steps.png: Episode Steps vs Theta

    Returns:
        Dictionary mapping chart metric keys to output file paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group rows by environment
    by_env: Dict[str, List[BenchmarkRow]] = {}
    for r in results:
        by_env.setdefault(r.environment, []).append(r)

    # Ensure sorted by theta for clean line plots
    for env_name in by_env:
        by_env[env_name].sort(key=lambda item: item.theta)

    metrics_to_plot = [
        (
            "theta_success_rate.png",
            "Success Rate vs Theta (θ)",
            "Success Rate (fraction)",
            lambda r: r.success_rate,
            (0.0, 1.05),
        ),
        (
            "theta_memory_size.png",
            "Final Memory Size vs Theta (θ)",
            "Number of Nodes in Memory",
            lambda r: r.final_memory_size,
            None,
        ),
        (
            "theta_comparisons.png",
            "Lookup Comparisons per Step vs Theta (θ)",
            "Average Comparisons / Step",
            lambda r: r.avg_comparisons_per_step,
            None,
        ),
        (
            "theta_episode_steps.png",
            "Average Episode Steps vs Theta (θ)",
            "Average Steps to Goal / Timeout",
            lambda r: r.avg_steps,
            None,
        ),
    ]

    color_cycle = ["#2563eb", "#16a34a", "#dc2626", "#d97706", "#9333ea"]
    markers = ["o", "s", "^", "D", "v"]
    saved_files: Dict[str, str] = {}

    for filename, title, ylabel, extractor, ylim in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

        for idx, (env_name, rows) in enumerate(by_env.items()):
            xs = [r.theta for r in rows]
            ys = [extractor(r) for r in rows]
            color = color_cycle[idx % len(color_cycle)]
            marker = markers[idx % len(markers)]
            ax.plot(
                xs,
                ys,
                marker=marker,
                linewidth=2.0,
                markersize=6,
                label=env_name,
                color=color,
            )

        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Distance Threshold θ (theta)", fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="best", frameon=True)
        if ylim is not None:
            ax.set_ylim(*ylim)

        fig.tight_layout()
        filepath = out_dir / filename
        fig.savefig(filepath)
        plt.close(fig)
        saved_files[filename] = str(filepath)

    return saved_files


def run_benchmark(config: Optional[BenchmarkConfig] = None) -> List[BenchmarkRow]:
    """Execute complete empirical benchmark over theta values and environments.

    Saves results to CSV and generates comparison charts.
    """
    cfg = config or BenchmarkConfig()
    results: List[BenchmarkRow] = []

    print("=" * 82)
    print("LINKMEM Empirical Benchmark — Evaluation of Distance Threshold θ")
    print("=" * 82)
    print(f"Theta Values: {cfg.theta_values}")
    print(f"Environments: {cfg.environment_names}")
    print(f"Episodes/Run: {cfg.episodes_per_run} | Max Steps: {cfg.max_steps_per_episode} | Seed: {cfg.random_seed}")
    print("-" * 82)

    total_runs = len(cfg.theta_values) * len(cfg.environment_names)
    run_idx = 0

    for theta in cfg.theta_values:
        for env_name in cfg.environment_names:
            run_idx += 1
            print(f"[{run_idx:>2}/{total_runs:>2}] θ={theta:<4.2f} | Env: {env_name:<18} ... ", end="", flush=True)
            row = run_single_benchmark(theta=theta, env_name=env_name, config=cfg)
            results.append(row)
            print(
                f"Success: {row.success_rate*100:>5.1f}% | "
                f"Avg Steps: {row.avg_steps:>5.1f} | "
                f"Mem: {row.final_memory_size:>2} | "
                f"Comps/Step: {row.avg_comparisons_per_step:>5.1f}"
            )

    # Save CSV
    csv_path = Path(cfg.results_dir) / cfg.csv_filename
    save_results_to_csv(results, csv_path)
    print("-" * 82)
    print(f"Saved benchmark CSV to: {csv_path}")

    # Generate charts
    charts = plot_benchmark_results(results, cfg.results_dir)
    print("Generated benchmark charts:")
    for name, path in charts.items():
        print(f"  - {path}")

    print("=" * 82)
    print("Benchmark complete.")
    print("=" * 82)
    return results
