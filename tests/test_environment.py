"""Unit tests for GridWorld environment dynamics."""

import pytest
from linkmem.core.types import Action
from linkmem.core.environment import GridWorld


def test_initial_state(grid_8x8):
    """Verify reset state and properties."""
    state = grid_8x8.reset()
    assert len(state) == 6
    assert grid_8x8.agent_pos == (0, 0)
    assert grid_8x8.step_count == 0
    assert grid_8x8.collision_count == 0
    assert not grid_8x8.is_terminal


def test_valid_movement(grid_8x8):
    """Verify standard movement steps correctly."""
    grid_8x8.reset()
    state, reward, is_terminal, info = grid_8x8.step(Action.RIGHT)
    assert grid_8x8.agent_pos == (1, 0)
    assert reward == grid_8x8.reward_step
    assert not is_terminal
    assert not info["collided"]


def test_boundary_collision_bounce(grid_8x8):
    """Verify agent bounces back and receives penalty on wall collision."""
    grid_8x8.reset()  # Agent at (0, 0)
    # Moving UP or LEFT from (0, 0) is out of bounds
    state, reward, is_terminal, info = grid_8x8.step(Action.UP)
    assert grid_8x8.agent_pos == (0, 0)
    assert reward == grid_8x8.reward_collision
    assert info["collided"] is True
    assert grid_8x8.collision_count == 1


def test_obstacle_collision(grid_with_obstacles):
    """Verify agent cannot move into an obstacle cell."""
    grid_with_obstacles.reset()  # Agent at (0, 0)
    # Move to (1, 0), then (2, 0)
    grid_with_obstacles.step(Action.RIGHT)
    grid_with_obstacles.step(Action.RIGHT)
    assert grid_with_obstacles.agent_pos == (2, 0)

    # Obstacle is at (3, 0)
    state, reward, is_terminal, info = grid_with_obstacles.step(Action.RIGHT)
    assert grid_with_obstacles.agent_pos == (2, 0)  # Bounced back
    assert reward == grid_with_obstacles.reward_collision
    assert info["collided"] is True


def test_goal_attainment(grid_8x8):
    """Verify terminal state and reward upon reaching goal."""
    grid_8x8.reset()
    grid_8x8.agent_pos = (6, 7)  # Goal is at (7, 7)
    state, reward, is_terminal, info = grid_8x8.step(Action.RIGHT)

    assert grid_8x8.agent_pos == (7, 7)
    assert reward == grid_8x8.reward_goal
    assert is_terminal is True
    assert info["goal_reached"] is True


def test_timeout_termination():
    """Verify termination when max_steps is reached."""
    env = GridWorld(width=4, height=4, start_pos=(0, 0), goal_pos=(3, 3), max_steps=5)
    env.reset()
    for _ in range(5):
        _, _, is_terminal, info = env.step(Action.RIGHT)
    assert is_terminal is True
    assert info["timed_out"] is True
