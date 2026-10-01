"""ExperienceMemory: Singly linked list memory for experience nodes."""

from typing import Optional, Tuple, List, Callable
from linkmem.core.types import Action, StateVector
from linkmem.core.node import ExperienceNode
from linkmem.core.distance import DistanceCalculator, DistanceMetric


class ExperienceMemory:
    """Explicit singly linked list storing ExperienceNode instances.
    
    This is the core data structure of LINKMEM. Lookup performs an O(n) linear scan
    through the pointer chain, while insertion prepends to the head in O(1).
    """

    def __init__(self, max_capacity: int = 100):
        self.head: Optional[ExperienceNode] = None
        self._size: int = 0
        self.max_capacity: int = max_capacity
        self._next_id: int = 1
        
        # Operational telemetry counters
        self.total_lookups: int = 0
        self.total_comparisons: int = 0

    @property
    def size(self) -> int:
        """Current number of nodes in memory."""
        return self._size

    def __len__(self) -> int:
        return self._size

    @property
    def average_comparisons(self) -> float:
        """Empirical average comparisons performed per nearest-node lookup."""
        if self.total_lookups == 0:
            return 0.0
        return self.total_comparisons / self.total_lookups

    def insert_at_head(
        self,
        state: StateVector,
        action: Action,
        q_init: float = 0.0,
        visit_count: int = 1,
    ) -> ExperienceNode:
        """Prepend a new experience node at the head of the list in O(1).
        
        Args:
            state: Encoded normalized feature vector.
            action: Selected action.
            q_init: Initial Q-value estimate.
            visit_count: Initial visit count (default 1).
            
        Returns:
            The newly created and linked ExperienceNode.
        """
        node = ExperienceNode(
            node_id=self._next_id,
            state=state,
            action=action,
            q_value=q_init,
            visit_count=visit_count,
            next=self.head,
        )
        self._next_id += 1
        self.head = node
        self._size += 1
        return node

    def find_nearest(
        self,
        query_state: StateVector,
        metric: Optional[DistanceMetric] = None,
    ) -> Tuple[Optional[ExperienceNode], float, int, List[Tuple[int, float]]]:
        """Perform an O(n) linear search for the nearest stored state.
        
        Args:
            query_state: Current state vector to search.
            metric: Distance calculation function (defaults to Euclidean).
            
        Returns:
            A tuple of:
            - best_node: ExperienceNode with smallest distance (or None if empty)
            - min_distance: Float minimum distance (inf if empty)
            - comparisons: Number of node comparisons performed
            - search_trace: List of (node_id, distance) pairs in traversal order
        """
        if metric is None:
            metric = DistanceCalculator.euclidean

        self.total_lookups += 1
        
        if self.head is None:
            return None, float("inf"), 0, []

        best_node: Optional[ExperienceNode] = None
        min_dist: float = float("inf")
        comparisons: int = 0
        trace: List[Tuple[int, float]] = []

        curr: Optional[ExperienceNode] = self.head
        while curr is not None:
            dist = metric(query_state, curr.state)
            comparisons += 1
            trace.append((curr.node_id, dist))
            
            if dist < min_dist:
                min_dist = dist
                best_node = curr
                
            curr = curr.next

        self.total_comparisons += comparisons
        return best_node, min_dist, comparisons, trace

    def remove_node(self, target_node: ExperienceNode) -> bool:
        """Remove a specific node from the linked list via pointer bypass in O(n).
        
        Args:
            target_node: The node instance to remove.
            
        Returns:
            True if removed, False if node was not found.
        """
        if self.head is None:
            return False

        # If head is target
        if self.head is target_node or self.head.node_id == target_node.node_id:
            self.head = self.head.next
            self._size -= 1
            return True

        prev = self.head
        curr = self.head.next
        while curr is not None:
            if curr is target_node or curr.node_id == target_node.node_id:
                prev.next = curr.next
                self._size -= 1
                return True
            prev = curr
            curr = curr.next

        return False

    def remove_by_id(self, node_id: int) -> bool:
        """Remove node matching node_id in O(n)."""
        if self.head is None:
            return False

        if self.head.node_id == node_id:
            self.head = self.head.next
            self._size -= 1
            return True

        prev = self.head
        curr = self.head.next
        while curr is not None:
            if curr.node_id == node_id:
                prev.next = curr.next
                self._size -= 1
                return True
            prev = curr
            curr = curr.next

        return False

    def to_list(self) -> List[ExperienceNode]:
        """Convert linked list elements to a Python list in head-to-tail order."""
        result: List[ExperienceNode] = []
        curr = self.head
        while curr is not None:
            result.append(curr)
            curr = curr.next
        return result

    def clear(self) -> None:
        """Clear all nodes from memory and reset counters."""
        self.head = None
        self._size = 0
        self.total_lookups = 0
        self.total_comparisons = 0

    def __iter__(self):
        """Allow iteration across all nodes in the list."""
        curr = self.head
        while curr is not None:
            yield curr
            curr = curr.next
