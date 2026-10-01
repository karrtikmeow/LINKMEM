"""Comprehensive unit and integration tests for Phase 2 LearningEngine."""

import pytest
from linkmem.core.types import Action, NoveltyStrategy, PruningStrategy
from linkmem.core.environment import GridWorld
from linkmem.core.memory import ExperienceMemory
from linkmem.core.agent import Agent
from linkmem.engine.learning_engine import LearningEngine
from linkmem.engine.events import StepEvent
from linkmem.engine.metrics import EpisodeMetrics
from linkmem.utils.config import AppConfig


@pytest.fixture
def basic_engine():
    """Provides a fresh LearningEngine on a clean 6x6 grid."""
    env = GridWorld(width=6, height=6, start_pos=(0, 0), goal_pos=(5, 5), max_steps=30)
    memory = ExperienceMemory(max_capacity=50)
    agent = Agent(memory=memory, theta=0.35, novelty_strategy=NoveltyStrategy.HEURISTIC_TOWARDS_GOAL)
    return LearningEngine(env=env, memory=memory, agent=agent)


def test_engine_initialization(basic_engine):
    """Verify engine starts with clean state counters and history."""
    assert basic_engine.global_step == 0
    assert basic_engine.current_episode == 0
    assert basic_engine.episode_step_count == 0
    assert basic_engine.episode_reward == 0.0
    assert basic_engine.history == []
    assert not basic_engine.is_terminal
    assert not basic_engine.is_stopped


def test_engine_reset(basic_engine):
    """Verify reset initializes the environment and increments episode."""
    state = basic_engine.reset()
    assert len(state) == 6
    assert basic_engine.current_episode == 1
    assert basic_engine.episode_step_count == 0
    assert basic_engine.episode_reward == 0.0
    assert basic_engine.env.agent_pos == (0, 0)


def test_one_learning_step(basic_engine):
    """Verify executing one learning step generates a valid StepEvent."""
    basic_engine.reset()
    event = basic_engine.run_step()

    assert isinstance(event, StepEvent)
    assert event.timestep == 1
    assert event.episode == 1
    assert event.agent_pos == (0, 0)
    assert basic_engine.global_step == 1
    assert basic_engine.episode_step_count == 1
    assert event.memory_size == basic_engine.memory.size
    assert not event.is_match  # First step must be novel state


def test_q_update_and_visit_count_in_engine():
    """Verify that engine applies the sample-average Q update and visit increment."""
    env = GridWorld(width=4, height=4, start_pos=(0, 0), goal_pos=(3, 3), max_steps=10)
    memory = ExperienceMemory(max_capacity=20)
    agent = Agent(memory=memory, theta=0.35)
    engine = LearningEngine(env=env, memory=memory, agent=agent)

    engine.reset()
    event = engine.run_step()

    # The active node must have visit_count 1 and Q equal to the reward
    node = memory.head
    assert node is not None
    assert node.visit_count == 1
    assert node.q_value == event.reward
    assert event.q_after == event.reward
    assert event.q_before == 0.0


def test_successful_goal_episode():
    """Verify agent can navigate to goal and record a successful episode."""
    # Small 3x3 grid where goal is at (2, 0), 2 steps right
    env = GridWorld(width=3, height=3, start_pos=(0, 0), goal_pos=(2, 0), max_steps=10)
    memory = ExperienceMemory(max_capacity=20)
    agent = Agent(memory=memory, theta=0.35, novelty_strategy=NoveltyStrategy.HEURISTIC_TOWARDS_GOAL)
    engine = LearningEngine(env=env, memory=memory, agent=agent)

    metrics = engine.run_episode()

    assert metrics.success is True
    assert metrics.steps <= 10
    assert metrics.total_reward > 0.0  # Reached goal (+1.0 minus step costs)
    assert len(engine.history) == 1
    assert engine.history[0] is metrics


def test_max_step_termination():
    """Verify episode terminates when max_steps limit is reached without goal."""
    # Obstacle blocking goal completely
    obstacles = {(1, 0), (0, 1)}
    env = GridWorld(width=3, height=3, start_pos=(0, 0), goal_pos=(2, 2), obstacles=obstacles, max_steps=5)
    memory = ExperienceMemory(max_capacity=20)
    agent = Agent(memory=memory, theta=0.35, novelty_strategy=NoveltyStrategy.HEURISTIC_TOWARDS_GOAL)
    engine = LearningEngine(env=env, memory=memory, agent=agent)

    metrics = engine.run_episode()

    assert metrics.success is False
    assert metrics.steps == 5
    assert metrics.collision_count > 0


def test_reward_accumulation():
    """Verify accumulated episode reward matches sum of individual step rewards."""
    env = GridWorld(width=5, height=5, start_pos=(0, 0), goal_pos=(4, 4), max_steps=10)
    memory = ExperienceMemory(max_capacity=30)
    agent = Agent(memory=memory, theta=0.35)

    collected_step_rewards = []

    def on_step_callback(event: StepEvent):
        collected_step_rewards.append(event.reward)

    engine = LearningEngine(env=env, memory=memory, agent=agent, on_step=on_step_callback)
    metrics = engine.run_episode()

    assert pytest.approx(metrics.total_reward, rel=1e-5) == sum(collected_step_rewards)
    assert len(collected_step_rewards) == metrics.steps


def test_memory_growth_and_reuse():
    """Verify novel states grow memory, and repeating identical episodes reuses nodes."""
    env = GridWorld(width=4, height=4, start_pos=(0, 0), goal_pos=(3, 0), max_steps=10)
    memory = ExperienceMemory(max_capacity=50)
    agent = Agent(memory=memory, theta=0.35, novelty_strategy=NoveltyStrategy.HEURISTIC_TOWARDS_GOAL)
    engine = LearningEngine(env=env, memory=memory, agent=agent)

    # Episode 1: creates experiences
    metrics1 = engine.run_episode()
    mem_size_after_ep1 = memory.size
    assert mem_size_after_ep1 > 0

    # Episode 2: identical path should reuse nodes
    reused_matches = []

    def track_reuse(event: StepEvent):
        if event.is_match:
            reused_matches.append(event.matched_node_id)

    engine.on_step = track_reuse
    metrics2 = engine.run_episode()

    # In episode 2, matched experiences must have been reused
    assert len(reused_matches) > 0
    # Visit counts of reused nodes should have increased
    for node in memory:
        if node.node_id in reused_matches:
            assert node.visit_count >= 2


def test_pruning_integration_in_engine():
    """Verify that exceeding max_memory_size triggers pruning policy."""
    env = GridWorld(width=6, height=6, start_pos=(0, 0), goal_pos=(5, 5), max_steps=30)
    max_cap = 5
    memory = ExperienceMemory(max_capacity=max_cap)
    agent = Agent(memory=memory, theta=0.01)  # small theta forces lots of novel nodes
    engine = LearningEngine(
        env=env,
        memory=memory,
        agent=agent,
        pruning_strategy=PruningStrategy.LOWEST_Q,
        max_memory_size=max_cap,
    )

    pruned_events = []

    def on_step(event: StepEvent):
        if event.pruned_node_ids:
            pruned_events.extend(event.pruned_node_ids)

    engine.on_step = on_step
    engine.run_episode()

    assert len(pruned_events) > 0
    # Memory size must strictly never exceed max_cap
    assert memory.size <= max_cap


def test_theta_behavior():
    """Verify theta parameter directly controls matching vs novelty."""
    # When theta = 0.0, any difference in state creates a new node
    mem1 = ExperienceMemory(max_capacity=10)
    mem1.insert_at_head((0.5, 0.5, 0, 0, 0, 0), Action.UP, visit_count=1)
    agent_strict = Agent(memory=mem1, theta=0.0)

    # Slight perturbation (dist = 0.05)
    query = (0.55, 0.5, 0, 0, 0, 0)
    _, _, _, is_match_strict, _ = agent_strict.select_action(query)
    assert not is_match_strict

    # When theta is generous (0.5), it matches
    mem2 = ExperienceMemory(max_capacity=10)
    mem2.insert_at_head((0.5, 0.5, 0, 0, 0, 0), Action.UP, visit_count=1)
    agent_loose = Agent(memory=mem2, theta=0.5)
    _, _, _, is_match_loose, _ = agent_loose.select_action(query)
    assert is_match_loose


def test_metrics_correctness():
    """Verify EpisodeMetrics accurately reflects all step counters."""
    env = GridWorld(width=4, height=4, start_pos=(0, 0), goal_pos=(3, 3), max_steps=10)
    memory = ExperienceMemory(max_capacity=20)
    agent = Agent(memory=memory, theta=0.35)
    engine = LearningEngine(env=env, memory=memory, agent=agent)

    metrics = engine.run_episode()

    assert metrics.steps > 0
    assert metrics.lookup_operations == metrics.steps
    assert metrics.lookup_comparisons >= metrics.lookup_operations
    assert metrics.average_comparisons_per_lookup >= 0.0
    
    d = metrics.to_dict()
    assert d["episode_number"] == 1
    assert d["steps"] == metrics.steps
    assert "avg_comparisons_per_lookup" in d


def test_deterministic_execution_with_fixed_seed():
    """Verify repeated runs with identical seeds and heuristics produce identical trajectories."""
    def run_deterministic():
        env = GridWorld(width=6, height=6, start_pos=(0, 0), goal_pos=(5, 5), max_steps=20)
        memory = ExperienceMemory(max_capacity=50)
        agent = Agent(memory=memory, theta=0.35, novelty_strategy=NoveltyStrategy.HEURISTIC_TOWARDS_GOAL, seed=42)
        engine = LearningEngine(env=env, memory=memory, agent=agent)
        return engine.run_episode()

    run1 = run_deterministic()
    run2 = run_deterministic()

    assert run1.steps == run2.steps
    assert pytest.approx(run1.total_reward) == run2.total_reward
    assert run1.success == run2.success
    assert run1.memory_size == run2.memory_size


def test_run_episodes_batch():
    """Verify run_episodes runs multiple episodes and returns full batch metrics."""
    env = GridWorld(width=5, height=5, start_pos=(0, 0), goal_pos=(4, 4), max_steps=15)
    memory = ExperienceMemory(max_capacity=30)
    agent = Agent(memory=memory, theta=0.35)
    engine = LearningEngine(env=env, memory=memory, agent=agent)

    results = engine.run_episodes(num_episodes=4)

    assert len(results) == 4
    assert [r.episode_number for r in results] == [1, 2, 3, 4]
    assert len(engine.history) == 4


def test_cannot_step_terminal_environment_without_reset(basic_engine):
    """Verify calling run_step on a terminal engine raises RuntimeError."""
    basic_engine.reset()
    basic_engine.env.is_terminal = True
    with pytest.raises(RuntimeError, match="Cannot step a terminal environment"):
        basic_engine.run_step()
