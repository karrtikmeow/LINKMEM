"""Distance metrics for comparing state feature vectors."""

import math
from typing import Callable
from linkmem.core.types import StateVector

DistanceMetric = Callable[[StateVector, StateVector], float]


class DistanceCalculator:
    """Provides pure distance metric calculations between state vectors."""

    @staticmethod
    def euclidean(s1: StateVector, s2: StateVector) -> float:
        """Compute standard Euclidean distance L2 between two vectors."""
        if len(s1) != len(s2):
            raise ValueError(f"Vector dimensions mismatch: {len(s1)} vs {len(s2)}")
        total = 0.0
        for a, b in zip(s1, s2):
            diff = a - b
            total += diff * diff
        return math.sqrt(total)

    @staticmethod
    def manhattan(s1: StateVector, s2: StateVector) -> float:
        """Compute Manhattan distance L1 between two vectors."""
        if len(s1) != len(s2):
            raise ValueError(f"Vector dimensions mismatch: {len(s1)} vs {len(s2)}")
        total = 0.0
        for a, b in zip(s1, s2):
            total += abs(a - b)
        return total
