"""Unit tests for Agent decision making and novel state handling."""

import pytest
from linkmem.core.types import Action, NoveltyStrategy
from linkmem.core.memory import ExperienceMemory
from linkmem.core.agent import Agent


def test_agent_novel_state_creation():
    """Verify that visiting an unmapped state prepends a new ExperienceNode in O(1)."""
    memory = ExperienceMemory(max_capacity=10)
    agent = Agent(
        memory=memory,
        theta=0.35,
        novelty_strategy=NoveltyStrategy.HEURISTIC_TOWARDS_GOAL,
        exploration_prob=0.0,
    )

    # Agent is at top-left, goal is bottom-right: dx=1.0, dy=1.0, wall_u=1, wall_l=1
    state = (1.0, 1.0, 1.0, 0.0, 1.0, 0.0)
    action, active_node, dist, is_match, trace = agent.select_action(state)

    assert not is_match
    assert memory.size == 1
    assert memory.head is active_node
    # Novel action should move towards goal (RIGHT or DOWN), avoiding walls
    assert action in (Action.RIGHT, Action.DOWN)
    assert active_node.action == action
    assert active_node.visit_count == 0  # Pre-update state


def test_agent_match_reuse():
    """Verify that visiting a state within distance <= theta reuses the stored node and action."""
    memory = ExperienceMemory(max_capacity=10)
    stored_state = (0.5, 0.5, 0.0, 0.0, 0.0, 0.0)
    reusable_node = memory.insert_at_head(stored_state, Action.DOWN, q_init=0.9, visit_count=5)

    agent = Agent(
        memory=memory,
        theta=0.35,
        novelty_strategy=NoveltyStrategy.HEURISTIC_TOWARDS_GOAL,
        exploration_prob=0.0,
    )

    # Query state very close (distance = 0.1 <= 0.35)
    query_state = (0.55, 0.5, 0.0, 0.0, 0.0, 0.0)
    action, active_node, dist, is_match, trace = agent.select_action(query_state)

    assert is_match is True
    assert active_node is reusable_node
    assert action == Action.DOWN
    # No new node created
    assert memory.size == 1


def test_agent_novelty_random_strategy():
    """Verify random exploration novelty strategy picks valid actions."""
    memory = ExperienceMemory()
    agent = Agent(
        memory=memory,
        theta=0.35,
        novelty_strategy=NoveltyStrategy.RANDOM_EXPLORATION,
        seed=123,
    )
    state = (0.2, -0.4, 0, 0, 0, 0)
    action, _, _, is_match, _ = agent.select_action(state)
    assert isinstance(action, Action)
    assert not is_match
