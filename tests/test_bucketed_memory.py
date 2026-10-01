"""Unit tests for BucketedExperienceMemory benchmark comparator."""

import pytest
from linkmem.core.types import Action
from linkmem.core.memory import ExperienceMemory
from linkmem.core.bucketed_memory import BucketedExperienceMemory


def test_bucketed_insert_and_find():
    """Verify insertion and retrieval from bucketed memory."""
    bmem = BucketedExperienceMemory(bucket_resolution=4)
    assert bmem.size == 0

    s1 = (0.5, 0.5, 0.0, 0.0, 0.0, 0.0)
    bmem.insert_at_head(s1, Action.RIGHT, 1.0)
    assert bmem.size == 1

    node, dist, comparisons, trace = bmem.find_nearest(s1)
    assert node is not None
    assert pytest.approx(dist, abs=1e-6) == 0.0
    assert node.action == Action.RIGHT
    assert comparisons >= 1


def test_bucketed_vs_linked_comparisons():
    """Verify bucketed memory performs significantly fewer comparisons on distributed data."""
    lmem = ExperienceMemory()
    bmem = BucketedExperienceMemory(bucket_resolution=5)

    # Insert 40 diverse states
    for i in range(40):
        # Evenly spread across normalized dx, dy space [-1, 1]
        dx = -1.0 + (i % 8) * 0.25
        dy = -1.0 + (i // 8) * 0.4
        wall_flag = (i % 2, (i // 2) % 2, (i // 4) % 2, (i // 8) % 2)
        state = (dx, dy, float(wall_flag[0]), float(wall_flag[1]), float(wall_flag[2]), float(wall_flag[3]))
        
        lmem.insert_at_head(state, Action.UP)
        bmem.insert_at_head(state, Action.UP)

    assert lmem.size == 40
    assert bmem.size == 40

    query = (0.25, 0.2, 0.0, 0.0, 0.0, 0.0)
    _, _, l_comps, _ = lmem.find_nearest(query)
    _, _, b_comps, _ = bmem.find_nearest(query)

    # Pure linked list always scans exactly all N elements
    assert l_comps == 40
    # Bucketed memory should scan only candidate buckets (substantially fewer than 40)
    assert b_comps < l_comps


def test_bucketed_remove_node():
    """Verify removal from bucketed memory."""
    bmem = BucketedExperienceMemory()
    s = (0.1, -0.2, 0, 1, 0, 0)
    node = bmem.insert_at_head(s, Action.DOWN)
    assert bmem.size == 1

    assert bmem.remove_node(node) is True
    assert bmem.size == 0
    assert bmem.find_nearest(s)[0] is None
