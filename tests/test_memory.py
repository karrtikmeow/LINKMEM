"""Unit tests for ExperienceMemory singly linked list operations."""

import pytest
from linkmem.core.types import Action
from linkmem.core.memory import ExperienceMemory


def test_insert_at_head_maintains_order(empty_memory):
    """Verify O(1) head insertion correctly links nodes."""
    node1 = empty_memory.insert_at_head((0.1, 0.1, 0, 0, 0, 0), Action.UP, 0.5)
    assert empty_memory.size == 1
    assert empty_memory.head is node1
    assert node1.next is None

    node2 = empty_memory.insert_at_head((0.2, 0.2, 0, 0, 0, 0), Action.DOWN, 0.7)
    assert empty_memory.size == 2
    assert empty_memory.head is node2
    assert node2.next is node1
    assert node1.next is None


def test_find_nearest_empty_memory(empty_memory):
    """Verify searching an empty memory returns expected sentinel values."""
    node, dist, comparisons, trace = empty_memory.find_nearest((0.0, 0.0, 0, 0, 0, 0))
    assert node is None
    assert dist == float("inf")
    assert comparisons == 0
    assert trace == []


def test_find_nearest_exact_match(sample_memory):
    """Verify nearest-node search finds an exact or closest state."""
    # Query identical to sample node 1 (inserted first, so near tail): (0.2, 0.0, 0, 0, 0, 0)
    query = (0.2, 0.0, 0.0, 0.0, 0.0, 0.0)
    node, dist, comparisons, trace = sample_memory.find_nearest(query)
    
    assert node is not None
    assert pytest.approx(dist, abs=1e-6) == 0.0
    assert node.action == Action.RIGHT
    # All nodes must have been inspected (O(n) linear scan)
    assert comparisons == sample_memory.size
    assert len(trace) == sample_memory.size


def test_telemetry_counters(empty_memory):
    """Verify lookups and comparisons counters accumulate accurately."""
    assert empty_memory.average_comparisons == 0.0

    empty_memory.insert_at_head((1, 0, 0, 0, 0, 0), Action.UP)
    empty_memory.insert_at_head((2, 0, 0, 0, 0, 0), Action.DOWN)
    empty_memory.insert_at_head((3, 0, 0, 0, 0, 0), Action.LEFT)
    # Size is 3

    empty_memory.find_nearest((0, 0, 0, 0, 0, 0))
    assert empty_memory.total_lookups == 1
    assert empty_memory.total_comparisons == 3
    assert empty_memory.average_comparisons == 3.0

    empty_memory.find_nearest((1, 0, 0, 0, 0, 0))
    assert empty_memory.total_lookups == 2
    assert empty_memory.total_comparisons == 6
    assert empty_memory.average_comparisons == 3.0


def test_remove_node_head(sample_memory):
    """Verify removal of head node."""
    initial_size = sample_memory.size
    head_node = sample_memory.head
    second_node = head_node.next

    assert sample_memory.remove_node(head_node) is True
    assert sample_memory.size == initial_size - 1
    assert sample_memory.head is second_node


def test_remove_node_middle(sample_memory):
    """Verify removal of a middle node."""
    initial_size = sample_memory.size
    nodes = sample_memory.to_list()
    target = nodes[1]  # Middle element

    assert sample_memory.remove_node(target) is True
    assert sample_memory.size == initial_size - 1
    remaining_ids = [n.node_id for n in sample_memory.to_list()]
    assert target.node_id not in remaining_ids


def test_remove_nonexistent_node(empty_memory):
    """Verify removal on empty memory or nonexistent node returns False."""
    assert empty_memory.remove_by_id(999) is False


def test_clear_memory(sample_memory):
    """Verify clear empties the list and resets counters."""
    sample_memory.clear()
    assert sample_memory.size == 0
    assert sample_memory.head is None
    assert sample_memory.total_lookups == 0
    assert sample_memory.total_comparisons == 0
