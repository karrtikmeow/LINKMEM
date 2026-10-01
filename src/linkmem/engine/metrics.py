"""Metrics tracking and data containers for LINKMEM learning runs."""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class EpisodeMetrics:
    """Empirical measurements recorded during a single simulation episode."""

    episode_number: int
    total_reward: float
    steps: int
    success: bool
    collision_count: int
    memory_size: int
    lookup_operations: int
    lookup_comparisons: int

    @property
    def average_comparisons_per_lookup(self) -> float:
        """Average number of node comparisons per nearest-node lookup in this episode."""
        if self.lookup_operations == 0:
            return 0.0
        return self.lookup_comparisons / self.lookup_operations

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary representation."""
        return {
            "episode_number": self.episode_number,
            "total_reward": round(self.total_reward, 4),
            "steps": self.steps,
            "success": self.success,
            "collision_count": self.collision_count,
            "memory_size": self.memory_size,
            "lookup_operations": self.lookup_operations,
            "lookup_comparisons": self.lookup_comparisons,
            "avg_comparisons_per_lookup": round(self.average_comparisons_per_lookup, 2),
        }
