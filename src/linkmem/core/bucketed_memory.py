"""BucketedExperienceMemory: Optimized spatial-partitioning comparator for linked-list memory."""

from typing import Optional, Tuple, List, Dict
from linkmem.core.types import Action, StateVector
from linkmem.core.node import ExperienceNode
from linkmem.core.distance import DistanceCalculator, DistanceMetric


class BucketedExperienceMemory:
    """Spatially bucketed experience memory used as an algorithmic benchmark against pure ExperienceMemory.
    
    Partitions the continuous normalized state vector space into discrete buckets
    (e.g., discretizing (norm_dx_goal, norm_dy_goal) and wall flags).
    Instead of scanning all N nodes, it scans candidate buckets, reducing comparisons to O(N / K).
    """

    def __init__(self, bucket_resolution: int = 4, max_capacity: int = 100):
        self.bucket_resolution: int = bucket_resolution
        self.max_capacity: int = max_capacity
        self._size: int = 0
        self._next_id: int = 1
        
        # Mapping from bucket key -> list of ExperienceNode
        self.buckets: Dict[Tuple[int, int], List[ExperienceNode]] = {}
        
        # Telemetry counters
        self.total_lookups: int = 0
        self.total_comparisons: int = 0

    @property
    def size(self) -> int:
        return self._size

    def __len__(self) -> int:
        return self._size

    @property
    def average_comparisons(self) -> float:
        if self.total_lookups == 0:
            return 0.0
        return self.total_comparisons / self.total_lookups

    def _get_bucket_key(self, state: StateVector) -> Tuple[int, int]:
        """Discretize continuous (dx, dy) state into integer bucket coordinates."""
        dx, dy = state[0], state[1]
        bx = int(math_floor((dx + 1.0) / 2.0 * self.bucket_resolution))
        by = int(math_floor((dy + 1.0) / 2.0 * self.bucket_resolution))
        bx = max(0, min(self.bucket_resolution - 1, bx))
        by = max(0, min(self.bucket_resolution - 1, by))
        return (bx, by)

    def insert_at_head(
        self,
        state: StateVector,
        action: Action,
        q_init: float = 0.0,
        visit_count: int = 1,
    ) -> ExperienceNode:
        """Insert a node into its corresponding spatial bucket."""
        key = self._get_bucket_key(state)
        node = ExperienceNode(
            node_id=self._next_id,
            state=state,
            action=action,
            q_value=q_init,
            visit_count=visit_count,
            next=None,
        )
        self._next_id += 1
        if key not in self.buckets:
            self.buckets[key] = []
        self.buckets[key].insert(0, node)
        self._size += 1
        return node

    def find_nearest(
        self,
        query_state: StateVector,
        metric: Optional[DistanceMetric] = None,
    ) -> Tuple[Optional[ExperienceNode], float, int, List[Tuple[int, float]]]:
        """Search for nearest node by inspecting the target bucket and adjacent buckets."""
        if metric is None:
            metric = DistanceCalculator.euclidean

        self.total_lookups += 1
        if self._size == 0:
            return None, float("inf"), 0, []

        key = self._get_bucket_key(query_state)
        bx, by = key

        # Candidate buckets: check adjacent spatial cells (radius 1)
        candidate_keys = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx, ny = bx + dx, by + dy
                if 0 <= nx < self.bucket_resolution and 0 <= ny < self.bucket_resolution:
                    candidate_keys.append((nx, ny))

        best_node: Optional[ExperienceNode] = None
        min_dist: float = float("inf")
        comparisons: int = 0
        trace: List[Tuple[int, float]] = []

        # First search candidate buckets
        for k in candidate_keys:
            if k in self.buckets:
                for node in self.buckets[k]:
                    dist = metric(query_state, node.state)
                    comparisons += 1
                    trace.append((node.node_id, dist))
                    if dist < min_dist:
                        min_dist = dist
                        best_node = node

        # Fallback: if no candidates in adjacent cells, search remaining buckets
        if best_node is None:
            for k, node_list in self.buckets.items():
                if k not in candidate_keys:
                    for node in node_list:
                        dist = metric(query_state, node.state)
                        comparisons += 1
                        trace.append((node.node_id, dist))
                        if dist < min_dist:
                            min_dist = dist
                            best_node = node

        self.total_comparisons += comparisons
        return best_node, min_dist, comparisons, trace

    def remove_node(self, target_node: ExperienceNode) -> bool:
        """Remove target node from its bucket."""
        key = self._get_bucket_key(target_node.state)
        if key in self.buckets:
            node_list = self.buckets[key]
            for i, n in enumerate(node_list):
                if n.node_id == target_node.node_id:
                    node_list.pop(i)
                    self._size -= 1
                    if not node_list:
                        del self.buckets[key]
                    return True
        return False

    def to_list(self) -> List[ExperienceNode]:
        """Return all nodes across all buckets."""
        result = []
        for nodes in self.buckets.values():
            result.extend(nodes)
        return result

    def clear(self) -> None:
        self.buckets.clear()
        self._size = 0
        self.total_lookups = 0
        self.total_comparisons = 0


def math_floor(x: float) -> int:
    """Helper floor function."""
    import math
    return math.floor(x)
