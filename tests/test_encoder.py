"""Unit tests for StateEncoder 6D feature vector."""

import pytest
from linkmem.core.encoder import StateEncoder


def test_encoder_normalization_range(encoder_8x8):
    """Verify dx, dy normalization stays strictly within [-1.0, 1.0]."""
    # Agent at top-left (0,0), Goal at bottom-right (7,7)
    s = encoder_8x8.encode(agent_pos=(0, 0), goal_pos=(7, 7), obstacles=set())
    assert len(s) == 6
    assert pytest.approx(s[0], abs=1e-6) == 1.0   # dx = (7-0)/7 = +1.0
    assert pytest.approx(s[1], abs=1e-6) == 1.0   # dy = (7-0)/7 = +1.0

    # Agent at bottom-right (7,7), Goal at top-left (0,0)
    s_rev = encoder_8x8.encode(agent_pos=(7, 7), goal_pos=(0, 0), obstacles=set())
    assert pytest.approx(s_rev[0], abs=1e-6) == -1.0  # dx = (0-7)/7 = -1.0
    assert pytest.approx(s_rev[1], abs=1e-6) == -1.0  # dy = (0-7)/7 = -1.0

    # Agent at goal
    s_same = encoder_8x8.encode(agent_pos=(3, 3), goal_pos=(3, 3), obstacles=set())
    assert s_same[0] == 0.0
    assert s_same[1] == 0.0


def test_encoder_boundary_walls(encoder_8x8):
    """Verify corner and border positions accurately reflect adjacent boundary walls."""
    # Top-left corner (0, 0): UP and LEFT are out-of-bounds
    s = encoder_8x8.encode(agent_pos=(0, 0), goal_pos=(7, 7), obstacles=set())
    # s = [dx, dy, wall_up, wall_down, wall_left, wall_right]
    assert s[2] == 1.0  # wall_up
    assert s[3] == 0.0  # wall_down
    assert s[4] == 1.0  # wall_left
    assert s[5] == 0.0  # wall_right


def test_encoder_obstacle_detection(encoder_8x8):
    """Verify obstacle detection in adjacent cells."""
    # Agent at (2, 2)
    obstacles = {(2, 1), (3, 2)}  # One above (2, 1) and one right (3, 2)
    s = encoder_8x8.encode(agent_pos=(2, 2), goal_pos=(7, 7), obstacles=obstacles)
    assert s[2] == 1.0  # wall_up (due to obstacle)
    assert s[3] == 0.0  # wall_down
    assert s[4] == 0.0  # wall_left
    assert s[5] == 1.0  # wall_right (due to obstacle)
