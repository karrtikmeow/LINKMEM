"""Entry point for LINKMEM: launches interactive GUI by default, or headless demo if --headless."""

import sys
from linkmem.core.environment import GridWorld
from linkmem.core.memory import ExperienceMemory
from linkmem.core.agent import Agent
from linkmem.core.types import NoveltyStrategy, PruningStrategy
from linkmem.engine.learning_engine import LearningEngine
from linkmem.utils.config import AppConfig


def run_headless_demo():
    """Execute a realistic headless training demonstration using actual linked-list memory."""
    print("=" * 78)
    print("LINKMEM — Headless Non-Iterative Linked-List Learning Demonstration")
    print("=" * 78)

    config = AppConfig(
        grid_width=8,
        grid_height=8,
        theta=0.35,
        n_max=60,
        pruning_strategy=PruningStrategy.LOWEST_Q,
        novelty_strategy=NoveltyStrategy.HEURISTIC_TOWARDS_GOAL,
        episodes=5,
        max_steps_per_episode=60,
    )

    obstacles = {(2, 1), (2, 2), (2, 3), (4, 4), (4, 5)}
    env = GridWorld(
        width=config.grid_width,
        height=config.grid_height,
        start_pos=(0, 0),
        goal_pos=(7, 7),
        obstacles=obstacles,
        max_steps=config.max_steps_per_episode,
        reward_goal=config.reward_goal,
        reward_step=config.reward_step,
        reward_collision=config.reward_collision,
    )

    memory = ExperienceMemory(max_capacity=config.n_max)
    agent = Agent(
        memory=memory,
        theta=config.theta,
        novelty_strategy=config.novelty_strategy,
        seed=config.random_seed,
    )
    engine = LearningEngine(
        env=env,
        memory=memory,
        agent=agent,
        config=config,
    )

    print(f"Grid: {config.grid_width}x{config.grid_height} | Theta: {config.theta} | Max Memory: {config.n_max}")
    print(f"Start: {env.start_pos} | Goal: {env.goal_pos} | Obstacles: {len(obstacles)}")
    print("-" * 78)

    for ep in range(1, config.episodes + 1):
        metrics = engine.run_episode()
        print(
            f"Episode {metrics.episode_number:>2} | "
            f"Reward: {metrics.total_reward:>7.2f} | "
            f"Steps: {metrics.steps:>3} | "
            f"Success: {str(metrics.success):>5} | "
            f"Memory: {metrics.memory_size:>2}/{config.n_max} | "
            f"Comparisons: {metrics.lookup_comparisons:>4} | "
            f"Avg Comps/Step: {metrics.average_comparisons_per_lookup:>5.1f}"
        )

    print("-" * 78)
    print("Demonstration completed successfully.")
    print(f"Total global learning steps executed: {engine.global_step}")
    print(f"Total experience nodes stored: {memory.size}")
    print("=" * 78)


def main():
    """Main CLI entry point."""
    if "--benchmark" in sys.argv or "-b" in sys.argv:
        from linkmem.experiments.benchmark import run_benchmark
        run_benchmark()
    elif "--headless" in sys.argv or "-h" in sys.argv:
        run_headless_demo()
    else:
        from linkmem.gui.app import run_gui
        run_gui()


if __name__ == "__main__":
    main()
