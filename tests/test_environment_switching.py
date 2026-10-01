"""Comprehensive tests for environment switching and simulation rebuilding."""

import time
import tkinter as tk
import pytest

from linkmem.core.types import Action, LearningMode
from linkmem.core.environment import GridWorld
from linkmem.core.memory import ExperienceMemory
from linkmem.core.agent import Agent
from linkmem.engine.learning_engine import LearningEngine
from linkmem.gui.controller import SimulationController
from linkmem.gui.environments import (
    EnvironmentManager,
    EnvironmentPreset,
    RandomObstaclesConfig,
    build_preset,
    is_path_reachable,
    validate_preset,
)
from linkmem.gui.app import MainWindow


@pytest.fixture
def tk_app():
    """Provides an active MainWindow instance (skips if display unavailable)."""
    try:
        root = tk.Tk()
        root.withdraw()
    except Exception as e:
        pytest.skip(f"Tkinter display not available: {e}")
    app = MainWindow(root)
    root.update()
    yield app
    try:
        app._on_close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 1-6. Environment Preset Creation & Validation Tests
# ---------------------------------------------------------------------------

def test_1_empty_grid_creation():
    preset = build_preset("Empty Grid")
    assert preset.width == 8 and preset.height == 8
    assert len(preset.obstacles) == 0
    assert preset.start_pos == (0, 0)
    assert preset.goal_pos == (7, 7)


def test_2_simple_maze_creation():
    preset = build_preset("Simple Maze")
    assert preset.width == 8 and preset.height == 8
    assert len(preset.obstacles) > 0
    assert is_path_reachable(8, 8, preset.start_pos, preset.goal_pos, preset.obstacles)


def test_3_complex_maze_creation():
    preset = build_preset("Complex Maze")
    assert preset.width == 8 and preset.height == 8
    assert len(preset.obstacles) >= 15
    assert is_path_reachable(8, 8, preset.start_pos, preset.goal_pos, preset.obstacles)


def test_4_random_environment_creation():
    cfg = RandomObstaclesConfig(width=8, height=8, obstacle_count=6, seed=0)
    preset = build_preset("Random Obstacles", random_cfg=cfg)
    assert len(preset.obstacles) == 6
    assert preset.start_pos not in preset.obstacles
    assert preset.goal_pos not in preset.obstacles
    assert is_path_reachable(8, 8, preset.start_pos, preset.goal_pos, preset.obstacles)


def test_5_custom_environment_creation():
    preset = build_preset("Custom Environment")
    assert preset.name == "Custom Environment"
    assert preset.width == 8 and preset.height == 8
    ok, err = validate_preset(preset)
    assert ok


def test_6_bfs_validation():
    # Valid open path
    assert is_path_reachable(4, 4, (0, 0), (3, 3), set()) is True
    # Fully blocked wall
    blocked_obstacles = {(x, 1) for x in range(4)}
    assert is_path_reachable(4, 4, (0, 0), (3, 3), blocked_obstacles) is False


# ---------------------------------------------------------------------------
# 7-15. Environment Switching & Lifecycle Invariants (MainWindow)
# ---------------------------------------------------------------------------

def test_7_to_12_environment_switching_replaces_all_components(tk_app):
    """Test 7: Environment switching
    Test 8: Memory reset after environment switching
    Test 9: Episode reset after environment switching
    Test 10: Controller replacement
    Test 11: Engine replacement
    Test 12: Old controller does not continue
    """
    app = tk_app

    # Verify initial Simple Maze loaded
    assert app.active_preset.name == "Simple Maze"
    old_env = app.env
    old_engine = app.engine
    old_controller = app.controller
    old_memory = app.memory

    # Run a few steps to populate memory & increment steps
    app.controller.set_mode(LearningMode.STEP_BY_STEP)
    app.controller.step()
    app.controller.step()
    app._poll_queue()
    app.root.update()

    assert app.memory.size > 0
    assert app.engine.global_step == 2

    # Switch to Empty Grid
    ok = app.load_environment("Empty Grid")
    assert ok is True
    app.root.update()

    # 10 & 11: Controller and Engine replacement
    assert app.engine is not old_engine
    assert app.controller is not old_controller
    assert app.env is not old_env
    assert app.memory is not old_memory

    # Invariants:
    assert app.engine.env is app.env
    assert app.engine.memory is app.memory
    assert app.controller.engine is app.engine
    assert app.active_preset.name == "Empty Grid"
    assert len(app.env.obstacles) == 0

    # 8 & 9: Memory reset & episode reset
    assert app.memory.size == 0
    assert app.engine.global_step == 0
    assert app.engine.current_episode == 0
    assert app.env.agent_pos == (0, 0)

    # 12: Old controller is stopped
    assert not old_controller.is_running


def test_13_autonomous_mode_after_switching(tk_app):
    """Test 13: Autonomous mode runs cleanly on the new environment."""
    app = tk_app
    app.load_environment("Empty Grid")
    app.root.update()

    app.controller.set_speed(50.0)
    app.controller.start()
    time.sleep(0.1)

    assert app.controller.is_running
    app._poll_queue()
    app.root.update()

    assert app.engine.global_step > 0
    assert app.env.width == 8
    assert len(app.env.obstacles) == 0

    app.controller.pause()
    time.sleep(0.04)
    assert not app.controller.is_running


def test_14_step_mode_after_switching(tk_app):
    """Test 14: Step mode works reliably after switching."""
    app = tk_app
    app.load_environment("Complex Maze")
    app.root.update()

    assert app.active_preset.name == "Complex Maze"
    assert len(app.env.obstacles) > 0

    app.controller.set_mode(LearningMode.STEP_BY_STEP)
    event = app.controller.step()
    assert event is not None
    assert event.timestep == 1
    assert app.memory.size == 1


def test_15_to_19_repeated_environment_switching(tk_app):
    """Test 15: Repeated environment switching across all combinations
    Test 16: Custom → Simple Maze
    Test 17: Simple Maze → Custom
    Test 18: Random → Complex
    Test 19: Complex → Empty
    """
    app = tk_app

    # Simple Maze -> Custom
    custom_preset = EnvironmentPreset(
        name="Custom Environment",
        description="test custom",
        width=8, height=8,
        start_pos=(0, 0), goal_pos=(7, 7),
        obstacles={(1, 1), (1, 2), (1, 3)},
    )
    app.env_manager.set_custom_preset(custom_preset)
    ok = app.load_environment(custom_preset)
    assert ok is True
    assert app.active_preset.name == "Custom Environment"
    assert (1, 1) in app.env.obstacles

    # Custom -> Simple Maze
    ok = app.load_environment("Simple Maze")
    assert ok is True
    assert app.active_preset.name == "Simple Maze"
    assert (2, 0) in app.env.obstacles
    assert (1, 1) not in app.env.obstacles

    # Simple Maze -> Random
    cfg = RandomObstaclesConfig(width=8, height=8, obstacle_count=6, seed=0)
    app.env_manager.apply_random_config(cfg)
    ok = app.load_environment("Random Obstacles")
    assert ok is True
    assert app.active_preset.name == "Random Obstacles"
    assert len(app.env.obstacles) == 6

    # Random -> Complex Maze
    ok = app.load_environment("Complex Maze")
    assert ok is True
    assert app.active_preset.name == "Complex Maze"
    assert len(app.env.obstacles) >= 15

    # Complex -> Empty
    ok = app.load_environment("Empty Grid")
    assert ok is True
    assert app.active_preset.name == "Empty Grid"
    assert len(app.env.obstacles) == 0


def test_20_reset_does_not_change_environment(tk_app):
    """Test 20: Reset clears state but PRESERVES the selected environment."""
    app = tk_app

    # Load Complex Maze
    app.load_environment("Complex Maze")
    assert app.active_preset.name == "Complex Maze"
    active_env_ref = app.env

    # Run steps to populate memory
    app.controller.set_mode(LearningMode.STEP_BY_STEP)
    app.controller.step()
    app.controller.step()
    assert app.memory.size > 0

    # Reset simulation
    app.reset_simulation()
    app.root.update()

    # Environment MUST NOT change
    assert app.active_preset.name == "Complex Maze"
    assert app.env is active_env_ref
    assert len(app.env.obstacles) >= 15

    # But memory and agent state ARE reset
    assert app.memory.size == 0
    assert app.engine.global_step == 0
    assert app.env.agent_pos == (0, 0)


def test_debug_banner_reflects_actual_gridworld(tk_app):
    """Verify debug banner text uses live properties from self.env."""
    app = tk_app
    app.load_environment("Simple Maze")
    app.root.update()

    banner_text = app._lbl_debug_env.cget("text")
    assert "Simple Maze" in banner_text
    assert f"{app.env.width} × {app.env.height}" in banner_text
    assert str(app.env.start_pos) in banner_text
    assert str(app.env.goal_pos) in banner_text
    assert f"Obstacles: {len(app.env.obstacles)}" in banner_text
