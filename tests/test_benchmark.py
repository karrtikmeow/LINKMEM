"""Tests for the empirical benchmark module (experiments/benchmark.py)."""

import csv
from pathlib import Path
import pytest

from linkmem.experiments.benchmark import (
    BenchmarkConfig,
    BenchmarkRow,
    plot_benchmark_results,
    run_benchmark,
    run_single_benchmark,
    save_results_to_csv,
)


def test_benchmark_config_defaults():
    """Verify default benchmark configurations."""
    cfg = BenchmarkConfig()
    assert cfg.theta_values == [0.10, 0.20, 0.35, 0.50, 0.75]
    assert "Empty Grid" in cfg.environment_names
    assert "Simple Maze" in cfg.environment_names
    assert "Complex Maze" in cfg.environment_names
    assert "Random Obstacles" in cfg.environment_names
    assert cfg.episodes_per_run == 10
    assert cfg.random_seed == 42


def test_run_single_benchmark_empty_grid():
    """Verify single benchmark execution on Empty Grid."""
    cfg = BenchmarkConfig(episodes_per_run=3, max_steps_per_episode=50)
    row = run_single_benchmark(theta=0.35, env_name="Empty Grid", config=cfg)

    assert isinstance(row, BenchmarkRow)
    assert row.theta == 0.35
    assert row.environment == "Empty Grid"
    assert row.episodes == 3
    assert row.success_rate == 1.0
    assert row.avg_steps > 0
    assert row.final_memory_size > 0
    assert row.timeouts == 0
    assert row.avg_comparisons_per_step >= 0.0
    assert row.nodes_created >= row.final_memory_size


def test_run_single_benchmark_complex_maze():
    """Verify single benchmark execution on Complex Maze."""
    cfg = BenchmarkConfig(episodes_per_run=2, max_steps_per_episode=100)
    row = run_single_benchmark(theta=0.35, env_name="Complex Maze", config=cfg)

    assert row.environment == "Complex Maze"
    assert row.success_rate == 1.0
    assert row.avg_steps > 0
    assert row.final_memory_size > 0


def test_benchmark_deterministic_reproducibility():
    """Identical configurations and seeds must produce identical benchmark results."""
    cfg1 = BenchmarkConfig(episodes_per_run=3, random_seed=42)
    cfg2 = BenchmarkConfig(episodes_per_run=3, random_seed=42)

    r1 = run_single_benchmark(theta=0.35, env_name="Simple Maze", config=cfg1)
    r2 = run_single_benchmark(theta=0.35, env_name="Simple Maze", config=cfg2)

    assert r1.success_rate == r2.success_rate
    assert r1.avg_steps == r2.avg_steps
    assert r1.final_memory_size == r2.final_memory_size
    assert r1.avg_comparisons_per_step == r2.avg_comparisons_per_step
    assert r1.nodes_created == r2.nodes_created


def test_save_results_to_csv(tmp_path: Path):
    """Verify writing benchmark results to CSV format."""
    csv_file = tmp_path / "test_results.csv"
    rows = [
        BenchmarkRow(
            theta=0.35,
            environment="Empty Grid",
            episodes=5,
            success_rate=1.0,
            avg_steps=14.0,
            avg_reward=0.87,
            final_memory_size=6,
            avg_comparisons_per_step=5.7,
            timeouts=0,
            nodes_created=6,
            avg_nodes_created=1.2,
        )
    ]
    save_results_to_csv(rows, csv_file)
    assert csv_file.exists()

    with open(csv_file, mode="r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 1
        assert float(reader[0]["theta"]) == 0.35
        assert reader[0]["environment"] == "Empty Grid"
        assert float(reader[0]["success_rate"]) == 1.0
        assert int(reader[0]["final_memory_size"]) == 6


def test_plot_benchmark_results(tmp_path: Path):
    """Verify generation of all 4 comparison charts."""
    rows = [
        BenchmarkRow(
            theta=0.10,
            environment="Empty Grid",
            episodes=2,
            success_rate=1.0,
            avg_steps=14.0,
            avg_reward=0.87,
            final_memory_size=14,
            avg_comparisons_per_step=13.0,
            timeouts=0,
            nodes_created=14,
            avg_nodes_created=7.0,
        ),
        BenchmarkRow(
            theta=0.50,
            environment="Empty Grid",
            episodes=2,
            success_rate=1.0,
            avg_steps=14.0,
            avg_reward=0.87,
            final_memory_size=6,
            avg_comparisons_per_step=5.7,
            timeouts=0,
            nodes_created=6,
            avg_nodes_created=3.0,
        ),
    ]

    saved = plot_benchmark_results(rows, tmp_path)
    assert "theta_success_rate.png" in saved
    assert "theta_memory_size.png" in saved
    assert "theta_comparisons.png" in saved
    assert "theta_episode_steps.png" in saved

    for filename in [
        "theta_success_rate.png",
        "theta_memory_size.png",
        "theta_comparisons.png",
        "theta_episode_steps.png",
    ]:
        p = tmp_path / filename
        assert p.exists()
        assert p.stat().st_size > 0


def test_run_benchmark_pipeline(tmp_path: Path):
    """Verify run_benchmark end-to-end execution with a lightweight test config."""
    cfg = BenchmarkConfig(
        theta_values=[0.20, 0.50],
        environment_names=["Empty Grid"],
        episodes_per_run=2,
        max_steps_per_episode=30,
        results_dir=str(tmp_path),
        csv_filename="test_bench.csv",
    )
    results = run_benchmark(cfg)
    assert len(results) == 2
    assert (tmp_path / "test_bench.csv").exists()
    assert (tmp_path / "theta_success_rate.png").exists()
    assert (tmp_path / "theta_memory_size.png").exists()
    assert (tmp_path / "theta_comparisons.png").exists()
    assert (tmp_path / "theta_episode_steps.png").exists()
