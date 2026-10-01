"""Unit tests for DistanceCalculator."""

import math
import pytest
from linkmem.core.distance import DistanceCalculator


def test_euclidean_distance():
    """Verify standard Euclidean distance computation."""
    s1 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    s2 = (3.0, 4.0, 0.0, 0.0, 0.0, 0.0)
    assert DistanceCalculator.euclidean(s1, s2) == 5.0

    # Identical vectors
    assert DistanceCalculator.euclidean(s1, s1) == 0.0


def test_manhattan_distance():
    """Verify Manhattan distance computation."""
    s1 = (1.0, 2.0, 0.0, 1.0, 0.0, 0.0)
    s2 = (4.0, 6.0, 1.0, 0.0, 0.0, 0.0)
    # |1-4| + |2-6| + |0-1| + |1-0| = 3 + 4 + 1 + 1 = 9.0
    assert DistanceCalculator.manhattan(s1, s2) == 9.0


def test_distance_dimension_mismatch():
    """Verify ValueError is raised when comparing vectors of unequal length."""
    s1 = (1.0, 2.0)
    s2 = (1.0, 2.0, 3.0)
    with pytest.raises(ValueError):
        DistanceCalculator.euclidean(s1, s2)
    with pytest.raises(ValueError):
        DistanceCalculator.manhattan(s1, s2)
