"""Unit tests for AppConfig configuration dataclass."""

import pytest
from linkmem.utils.config import AppConfig
from linkmem.core.types import PruningStrategy, NoveltyStrategy, LearningMode


def test_default_config():
    """Verify default hyperparameters align with Phase 1 design."""
    config = AppConfig()
    assert config.grid_width == 8
    assert config.grid_height == 8
    assert config.theta == 0.35
    assert config.n_max == 100
    assert config.pruning_strategy == PruningStrategy.LOWEST_Q
    assert config.novelty_strategy == NoveltyStrategy.HEURISTIC_TOWARDS_GOAL
    assert config.learning_mode == LearningMode.AUTONOMOUS
    assert config.reward_goal == 1.0
    assert config.reward_step == -0.01
    assert config.reward_collision == -0.5


def test_custom_config():
    """Verify overriding hyperparameters."""
    config = AppConfig(
        grid_width=12,
        grid_height=12,
        theta=0.25,
        n_max=50,
        pruning_strategy=PruningStrategy.LFU,
    )
    assert config.grid_width == 12
    assert config.grid_height == 12
    assert config.theta == 0.25
    assert config.n_max == 50
    assert config.pruning_strategy == PruningStrategy.LFU
