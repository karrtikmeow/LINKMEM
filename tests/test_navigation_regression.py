"""Regression tests for agent cycle prevention, loop escape, and navigation progress."""

import pytest
from linkmem.core.types import Action, NoveltyStrategy
from linkmem.core.environment import GridWorld
from linkmem.core.memory import ExperienceMemory
from linkmem.core.agent import Agent
from linkmem.engine.learning_engine import LearningEngine
from linkmem.gui.environments import build_preset, is_path_reachable


def test_a_simple_reachable_environment():
    """Requirement 9A: Agent reaches goal in a simple reachable environment."""
    preset = build_preset("Simple Maze")
    env = GridWorld(
        width=preset.width,
        height=preset.height,
        start_pos=preset.start_pos,
        goal_pos=preset.goal_pos,
        obstacles=set(preset.obstacles),
        max_steps=50,
    )
    memory = ExperienceMemory(max_capacity=50)
    agent = Agent(memory=memory, theta=0.35, seed=42)
    engine = LearningEngine(env=env, memory=memory, agent=agent)

    metrics = engine.run_episode()
    assert metrics.success is True
    assert metrics.steps < 50
    assert metrics.total_reward > 0.0


def test_b_complex_maze_no_indefinite_loop_and_reaches_goal():
    """Requirement 9B: Complex Maze agent does NOT get trapped in 2-state/4-state loop and reaches goal."""
    preset = build_preset("Complex Maze")
    assert is_path_reachable(8, 8, preset.start_pos, preset.goal_pos, preset.obstacles)

    env = GridWorld(
        width=preset.width,
        height=preset.height,
        start_pos=preset.start_pos,
        goal_pos=preset.goal_pos,
        obstacles=set(preset.obstacles),
        max_steps=100,
    )
    memory = ExperienceMemory(max_capacity=100)
    agent = Agent(memory=memory, theta=0.35, seed=42)
    engine = LearningEngine(env=env, memory=memory, agent=agent)

    positions = []
    # Track positions during episode 1
    engine.reset()
    while not engine.is_terminal and engine.episode_step_count < 100:
        event = engine.run_step()
        positions.append(event.next_agent_pos)

    # Agent must reach the goal in episode 1
    assert env.agent_pos == preset.goal_pos
    assert engine.history[-1].success is True

    # Verify no infinite 2-state oscillation in the trajectory
    # (i.e. we don't have the same 2 states repeating 6 times in a row)
    for i in range(len(positions) - 12):
        chunk = positions[i : i + 12]
        # An oscillation A, B, A, B, A, B, A, B, A, B, A, B has len(set) == 2
        assert len(set(chunk)) > 2, f"Detected 2-state oscillation loop at steps {i}-{i+12}: {chunk}"


def test_c_blocked_shortest_direction_backtracks_and_explores():
    """Requirement 9C: When shortest-looking direction is blocked (corridor dead-end),
    agent backtracks and discovers the alternative open route.
    """
    # Start (0,0), Goal (4,2).
    # Corridor along row 0: (0,0) -> (1,0) -> (2,0) -> (3,0) blocked at (4,0).
    # Wall below: (1,1), (2,1), (3,1).
    # Must backtrack left to (0,0) and go down (0,1)->(0,2)->...->(4,2).
    obstacles = {(4, 0), (1, 1), (2, 1), (3, 1)}
    env = GridWorld(width=5, height=5, start_pos=(0, 0), goal_pos=(4, 2), obstacles=obstacles, max_steps=60)
    memory = ExperienceMemory(max_capacity=50)
    agent = Agent(memory=memory, theta=0.35, seed=42)
    engine = LearningEngine(env=env, memory=memory, agent=agent)

    metrics = engine.run_episode()
    assert metrics.success is True
    assert metrics.steps < 30


def test_d_repeated_state_action_triggers_alternative_exploration():
    """Requirement 9D: Repeatedly selecting the same action from a state penalizes it
    and triggers exploration of alternative legal actions.
    """
    memory = ExperienceMemory(max_capacity=10)
    agent = Agent(memory=memory, theta=0.35, seed=123)

    # State with open right and open down: dx=0.5, dy=0.5, wall_u=1, wall_d=0, wall_l=1, wall_r=0
    state = (0.5, 0.5, 1.0, 0.0, 1.0, 0.0)

    # First call: picks an action (e.g. RIGHT or DOWN)
    action1, _, _, _, _ = agent.select_action(state)

    # Simulate coming back to this same state multiple times
    actions_taken = [action1]
    for _ in range(5):
        act, _, _, _, _ = agent.select_action(state)
        actions_taken.append(act)

    # Because legal actions include both RIGHT and DOWN, repeated visits must explore both!
    distinct_actions = set(actions_taken)
    assert len(distinct_actions) >= 2, "Agent failed to explore alternative legal action after repeat visits"
    assert Action.RIGHT in distinct_actions or Action.DOWN in distinct_actions


def test_memory_learns_and_improves_across_episodes():
    """Verify that after learning the path in episode 1, subsequent episodes
    use ExperienceMemory to navigate in fewer steps.
    """
    preset = build_preset("Complex Maze")
    env = GridWorld(
        width=preset.width,
        height=preset.height,
        start_pos=preset.start_pos,
        goal_pos=preset.goal_pos,
        obstacles=set(preset.obstacles),
        max_steps=100,
    )
    memory = ExperienceMemory(max_capacity=100)
    agent = Agent(memory=memory, theta=0.35, seed=42)
    engine = LearningEngine(env=env, memory=memory, agent=agent)

    # Episode 1: Initial exploration
    ep1 = engine.run_episode()
    assert ep1.success is True

    # Episode 2: Leverages stored experience in memory
    ep2 = engine.run_episode()
    assert ep2.success is True
    # Episode 2 should be at least as fast or faster than Episode 1
    assert ep2.steps <= ep1.steps
