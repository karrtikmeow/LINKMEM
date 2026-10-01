"""Pytest fixtures for LINKMEM test suite."""

import sys
import os
import pytest

# Ensure src/ is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from linkmem.core.types import Action, PruningStrategy, NoveltyStrategy
from linkmem.core.memory import ExperienceMemory
from linkmem.core.bucketed_memory import BucketedExperienceMemory
from linkmem.core.environment import GridWorld
from linkmem.core.agent import Agent
from linkmem.core.encoder import StateEncoder


@pytest.fixture
def empty_memory():
    """Provides a fresh, empty ExperienceMemory with capacity 10."""
    return ExperienceMemory(max_capacity=10)


@pytest.fixture
def sample_memory():
    """Provides an ExperienceMemory populated with 4 known states."""
    mem = ExperienceMemory(max_capacity=10)
    # Node 1: near goal, right
    mem.insert_at_head(state=(0.2, 0.0, 0.0, 0.0, 0.0, 0.0), action=Action.RIGHT, q_init=0.8, visit_count=5)
    # Node 2: near goal, up
    mem.insert_at_head(state=(0.0, -0.2, 0.0, 0.0, 0.0, 0.0), action=Action.UP, q_init=0.6, visit_count=3)
    # Node 3: wall on right
    mem.insert_at_head(state=(0.5, 0.5, 0.0, 0.0, 0.0, 1.0), action=Action.DOWN, q_init=-0.4, visit_count=2)
    # Node 4: far away
    mem.insert_at_head(state=(-0.8, -0.8, 1.0, 0.0, 1.0, 0.0), action=Action.RIGHT, q_init=0.1, visit_count=1)
    return mem


@pytest.fixture
def grid_8x8():
    """Provides a clean 8x8 GridWorld without obstacles."""
    return GridWorld(
        width=8,
        height=8,
        start_pos=(0, 0),
        goal_pos=(7, 7),
        obstacles=None,
        max_steps=50,
    )


@pytest.fixture
def grid_with_obstacles():
    """Provides an 8x8 GridWorld with a vertical obstacle barrier."""
    obstacles = {(3, y) for y in range(6)}  # Barrier at x=3, y=0..5, gap at y=6,7
    return GridWorld(
        width=8,
        height=8,
        start_pos=(0, 0),
        goal_pos=(7, 7),
        obstacles=obstacles,
        max_steps=50,
    )


@pytest.fixture
def encoder_8x8():
    """Provides a StateEncoder for an 8x8 grid."""
    return StateEncoder(width=8, height=8)
