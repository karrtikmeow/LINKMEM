"""Application and algorithm configuration dataclass."""

from dataclasses import dataclass, field
from linkmem.core.types import PruningStrategy, NoveltyStrategy, LearningMode


@dataclass
class AppConfig:
    """Holds all simulation, environment, and learning hyperparameters."""

    # Environment dimensions
    grid_width: int = 8
    grid_height: int = 8
    obstacle_density: float = 0.15

    # Memory parameters
    theta: float = 0.35  # Configurable threshold; evaluated empirically in experiments
    n_max: int = 100     # Maximum experience memory capacity
    pruning_strategy: PruningStrategy = PruningStrategy.LOWEST_Q

    # Agent policy
    novelty_strategy: NoveltyStrategy = NoveltyStrategy.HEURISTIC_TOWARDS_GOAL
    exploration_prob: float = 0.0

    # Rewards
    reward_goal: float = 1.0
    reward_step: float = -0.01
    reward_collision: float = -0.5

    # Execution limits
    max_steps_per_episode: int = 100
    episodes: int = 50
    learning_mode: LearningMode = LearningMode.AUTONOMOUS
    
    # Random seed (None for non-deterministic)
    random_seed: int = 42
