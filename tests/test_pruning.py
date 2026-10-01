"""Unit tests for PruningPolicy (LOWEST_Q primary and LFU secondary)."""

import pytest
from linkmem.core.types import Action, PruningStrategy
from linkmem.core.memory import ExperienceMemory
from linkmem.core.pruning import PruningPolicy


def test_pruning_within_capacity(sample_memory):
    """Verify pruning does nothing when size <= target_size."""
    initial_size = sample_memory.size
    evicted = PruningPolicy.prune(sample_memory, target_size=initial_size + 2)
    assert evicted == []
    assert sample_memory.size == initial_size


def test_lowest_q_pruning():
    """Verify LOWEST_Q evicts the node with minimum Q-value."""
    mem = ExperienceMemory(max_capacity=3)
    n1 = mem.insert_at_head((0, 0, 0, 0, 0, 0), Action.UP, q_init=0.9, visit_count=1)
    n2 = mem.insert_at_head((0, 0, 0, 0, 0, 0), Action.DOWN, q_init=-0.8, visit_count=5)  # Lowest Q!
    n3 = mem.insert_at_head((0, 0, 0, 0, 0, 0), Action.LEFT, q_init=0.2, visit_count=3)

    assert mem.size == 3
    # Prune to target size 2
    evicted = PruningPolicy.prune(mem, strategy=PruningStrategy.LOWEST_Q, target_size=2)
    assert evicted == [n2.node_id]
    assert mem.size == 2

    # Verify n2 is no longer in memory
    remaining_ids = [n.node_id for n in mem.to_list()]
    assert n2.node_id not in remaining_ids
    assert n1.node_id in remaining_ids
    assert n3.node_id in remaining_ids


def test_lfu_pruning():
    """Verify LFU evicts the node with lowest visit_count."""
    mem = ExperienceMemory(max_capacity=3)
    n1 = mem.insert_at_head((0, 0, 0, 0, 0, 0), Action.UP, q_init=0.5, visit_count=10)
    n2 = mem.insert_at_head((0, 0, 0, 0, 0, 0), Action.DOWN, q_init=0.1, visit_count=1)   # Lowest visits!
    n3 = mem.insert_at_head((0, 0, 0, 0, 0, 0), Action.LEFT, q_init=0.7, visit_count=4)

    assert mem.size == 3
    evicted = PruningPolicy.prune(mem, strategy=PruningStrategy.LFU, target_size=2)
    assert evicted == [n2.node_id]
    assert mem.size == 2
    remaining_ids = [n.node_id for n in mem.to_list()]
    assert n2.node_id not in remaining_ids
