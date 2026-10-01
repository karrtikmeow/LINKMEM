"""Unit tests for ExperienceNode and non-iterative sample-average learning."""

import pytest
from linkmem.core.node import ExperienceNode
from linkmem.core.types import Action


def test_node_creation():
    """Verify initial properties of an ExperienceNode."""
    node = ExperienceNode(
        node_id=1,
        state=(0.5, -0.5, 0.0, 0.0, 0.0, 0.0),
        action=Action.RIGHT,
        q_value=0.0,
        visit_count=1,
    )
    assert node.node_id == 1
    assert node.action == Action.RIGHT
    assert node.q_value == 0.0
    assert node.visit_count == 1
    assert node.next is None


def test_non_iterative_sample_average_math():
    """Verify Q_new = Q_old + (reward - Q_old) / n matches true arithmetic mean."""
    node = ExperienceNode(
        node_id=1,
        state=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        action=Action.UP,
        q_value=0.0,
        visit_count=0,
    )

    rewards = [1.0, 0.5, -0.5, 1.0, 0.0]
    cumulative = 0.0

    for i, r in enumerate(rewards, start=1):
        cumulative += r
        expected_mean = cumulative / i
        updated_q = node.update_q(r)
        
        assert node.visit_count == i
        assert pytest.approx(updated_q, rel=1e-6) == expected_mean
        assert pytest.approx(node.q_value, rel=1e-6) == expected_mean


def test_node_serialization():
    """Verify serialization to and from dictionary."""
    node = ExperienceNode(
        node_id=42,
        state=(0.1, 0.2, 1.0, 0.0, 0.0, 1.0),
        action=Action.DOWN,
        q_value=0.75,
        visit_count=8,
    )
    d = node.to_dict()
    assert d["node_id"] == 42
    assert d["action"] == "DOWN"
    assert d["q_value"] == 0.75
    assert d["visit_count"] == 8
    assert d["state"] == [0.1, 0.2, 1.0, 0.0, 0.0, 1.0]

    reconstructed = ExperienceNode.from_dict(d)
    assert reconstructed.node_id == node.node_id
    assert reconstructed.action == node.action
    assert reconstructed.q_value == node.q_value
    assert reconstructed.visit_count == node.visit_count
    assert reconstructed.state == node.state


def test_no_bellman_or_epochs():
    """Verify that Bellman variables (gamma, discount, target, epochs) are absent."""
    node = ExperienceNode(1, (0, 0, 0, 0, 0, 0), Action.UP)
    assert not hasattr(node, "gamma")
    assert not hasattr(node, "discount")
    assert not hasattr(node, "epoch")
    assert not hasattr(node, "learning_rate")

