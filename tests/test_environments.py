"""Tests for environment preset system (Phase 4)."""

import pytest
from linkmem.gui.environments import (
    PRESET_NAMES,
    EnvironmentManager,
    EnvironmentPreset,
    RandomObstaclesConfig,
    build_preset,
    is_path_reachable,
    validate_preset,
)


# ---------------------------------------------------------------------------
# Preset registry
# ---------------------------------------------------------------------------

def test_all_preset_names_exist():
    """build_preset must succeed for each name in PRESET_NAMES."""
    for name in PRESET_NAMES:
        preset = build_preset(name)
        assert preset.name == name


def test_preset_names_list_is_complete():
    assert len(PRESET_NAMES) == 5
    assert "Empty Grid" in PRESET_NAMES
    assert "Simple Maze" in PRESET_NAMES
    assert "Complex Maze" in PRESET_NAMES
    assert "Random Obstacles" in PRESET_NAMES
    assert "Custom Environment" in PRESET_NAMES


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="Unknown preset"):
        build_preset("Nonexistent Preset")


# ---------------------------------------------------------------------------
# Built-in preset validity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "Empty Grid", "Simple Maze", "Complex Maze",
])
def test_builtin_presets_are_valid(name):
    preset = build_preset(name)
    ok, msg = validate_preset(preset)
    assert ok, f"Preset '{name}' invalid: {msg}"


def test_empty_grid_has_no_obstacles():
    preset = build_preset("Empty Grid")
    assert len(preset.obstacles) == 0


# ---------------------------------------------------------------------------
# Random obstacle generation
# ---------------------------------------------------------------------------

def test_random_obstacles_seed_determinism():
    """Same seed must produce identical obstacle sets."""
    cfg = RandomObstaclesConfig(width=8, height=8, obstacle_count=12, seed=99)
    p1 = build_preset("Random Obstacles", random_cfg=cfg)
    p2 = build_preset("Random Obstacles", random_cfg=cfg)
    assert p1.obstacles == p2.obstacles


def test_random_obstacles_different_seeds_differ():
    cfg1 = RandomObstaclesConfig(width=8, height=8, obstacle_count=12, seed=1)
    cfg2 = RandomObstaclesConfig(width=8, height=8, obstacle_count=12, seed=2)
    p1 = build_preset("Random Obstacles", random_cfg=cfg1)
    p2 = build_preset("Random Obstacles", random_cfg=cfg2)
    # With 12 obstacles and 2 different seeds almost certainly differ
    assert p1.obstacles != p2.obstacles


def test_random_obstacles_count_respected():
    cfg = RandomObstaclesConfig(width=8, height=8, obstacle_count=10, seed=7)
    preset = build_preset("Random Obstacles", random_cfg=cfg)
    assert len(preset.obstacles) == 10


def test_random_obstacles_never_block_start_or_goal():
    cfg = RandomObstaclesConfig(width=8, height=8, obstacle_count=62, seed=42)
    preset = build_preset("Random Obstacles", random_cfg=cfg)
    assert (0, 0) not in preset.obstacles
    assert (7, 7) not in preset.obstacles


def test_random_preset_is_valid():
    cfg = RandomObstaclesConfig(width=8, height=8, obstacle_count=6, seed=0)
    preset = build_preset("Random Obstacles", random_cfg=cfg)
    ok, msg = validate_preset(preset)
    assert ok, f"Random preset invalid: {msg}"



# ---------------------------------------------------------------------------
# Path reachability (BFS)
# ---------------------------------------------------------------------------

def test_open_grid_reachable():
    assert is_path_reachable(4, 4, (0, 0), (3, 3), set()) is True


def test_fully_blocked_not_reachable():
    # Wall of obstacles blocking (0,1) to (4,1)
    obs = {(x, 1) for x in range(4)}
    assert is_path_reachable(4, 4, (0, 0), (0, 2), obs) is False


def test_goal_unreachable_when_surrounded():
    obs = {(2, 1), (3, 2), (2, 3), (1, 2)}
    assert is_path_reachable(4, 4, (0, 0), (2, 2), obs) is False


def test_same_start_and_goal_reachable():
    assert is_path_reachable(4, 4, (1, 1), (1, 1), set()) is True


# ---------------------------------------------------------------------------
# validate_preset edge cases
# ---------------------------------------------------------------------------

def test_validate_start_in_obstacle_fails():
    p = EnvironmentPreset(
        name="Test", description="",
        width=4, height=4,
        start_pos=(1, 1), goal_pos=(3, 3),
        obstacles={(1, 1)},
    )
    ok, msg = validate_preset(p)
    assert not ok
    assert "Start" in msg


def test_validate_goal_in_obstacle_fails():
    p = EnvironmentPreset(
        name="Test", description="",
        width=4, height=4,
        start_pos=(0, 0), goal_pos=(3, 3),
        obstacles={(3, 3)},
    )
    ok, msg = validate_preset(p)
    assert not ok
    assert "Goal" in msg


def test_validate_start_equals_goal_fails():
    p = EnvironmentPreset(
        name="Test", description="",
        width=4, height=4,
        start_pos=(0, 0), goal_pos=(0, 0),
        obstacles=set(),
    )
    ok, msg = validate_preset(p)
    assert not ok
    assert "identical" in msg


def test_validate_no_path_fails():
    # Completely walls off goal cell
    obs = {(2, 1), (3, 2), (2, 3), (1, 2)}
    p = EnvironmentPreset(
        name="Test", description="",
        width=4, height=4,
        start_pos=(0, 0), goal_pos=(2, 2),
        obstacles=obs,
    )
    ok, msg = validate_preset(p)
    assert not ok
    assert "path" in msg.lower()


# ---------------------------------------------------------------------------
# EnvironmentManager
# ---------------------------------------------------------------------------

def test_env_manager_default_preset():
    mgr = EnvironmentManager()
    assert mgr.preset_name == "Simple Maze"


def test_env_manager_select_empty_grid():
    mgr = EnvironmentManager()
    ok, msg = mgr.select_preset("Empty Grid")
    assert ok, msg
    assert mgr.preset_name == "Empty Grid"


def test_env_manager_select_complex_maze():
    mgr = EnvironmentManager()
    ok, msg = mgr.select_preset("Complex Maze")
    assert ok, msg


def test_env_manager_apply_random_config():
    mgr = EnvironmentManager()
    # Apply a reachable random config (seed=77 count=8 is verified reachable)
    cfg = RandomObstaclesConfig(width=8, height=8, obstacle_count=8, seed=77)
    ok, msg = mgr.apply_random_config(cfg)
    assert ok, msg
    # Now select random obstacles to activate the config
    ok2, msg2 = mgr.select_preset("Random Obstacles")
    assert ok2, msg2
    assert len(mgr.active_preset.obstacles) == 8



def test_env_manager_build_gridworld():
    """build_gridworld returns a real GridWorld with matching dimensions."""
    from linkmem.core.environment import GridWorld

    mgr = EnvironmentManager()
    mgr.select_preset("Empty Grid")
    env = mgr.build_gridworld(max_steps=50)
    assert isinstance(env, GridWorld)
    assert env.width == 8
    assert env.height == 8
    assert len(env.obstacles) == 0


def test_env_manager_set_custom_preset():
    mgr = EnvironmentManager()
    custom = EnvironmentPreset(
        name="Custom Environment",
        description="test",
        width=6, height=6,
        start_pos=(0, 0), goal_pos=(5, 5),
        obstacles={(2, 1), (2, 2)},
    )
    ok, msg = mgr.set_custom_preset(custom)
    assert ok, msg
    assert mgr.preset_name == "Custom Environment"
    assert mgr.active_preset.width == 6


def test_env_switch_resets_path_validity():
    """Switching environments should always produce a valid (reachable) config."""
    mgr = EnvironmentManager()
    for name in ["Empty Grid", "Simple Maze", "Complex Maze"]:
        mgr.select_preset(name)
        ok, msg = validate_preset(mgr.active_preset)
        assert ok, f"After switching to '{name}': {msg}"
