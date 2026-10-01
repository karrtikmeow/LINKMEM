"""PruningPolicy: Memory eviction algorithms for ExperienceMemory."""

from typing import List, Optional
from linkmem.core.types import PruningStrategy
from linkmem.core.memory import ExperienceMemory
from linkmem.core.node import ExperienceNode


class PruningPolicy:
    """Enforces memory capacity constraints by evicting nodes in O(n)."""

    @staticmethod
    def prune(
        memory: ExperienceMemory,
        strategy: PruningStrategy = PruningStrategy.LOWEST_Q,
        target_size: Optional[int] = None,
    ) -> List[int]:
        """Prune nodes until memory.size <= target_size.
        
        Args:
            memory: The ExperienceMemory linked list instance.
            strategy: LOWEST_Q (primary) or LFU (secondary benchmark).
            target_size: Desired capacity (defaults to memory.max_capacity).
            
        Returns:
            List of node_ids that were evicted.
        """
        if target_size is None:
            target_size = memory.max_capacity

        evicted_ids: List[int] = []

        while memory.size > target_size and memory.head is not None:
            victim = PruningPolicy._find_victim(memory, strategy)
            if victim is not None:
                evicted_ids.append(victim.node_id)
                memory.remove_node(victim)
            else:
                break

        return evicted_ids

    @staticmethod
    def _find_victim(
        memory: ExperienceMemory,
        strategy: PruningStrategy,
    ) -> Optional[ExperienceNode]:
        """Scan the linked list in O(n) to find the eviction candidate."""
        if memory.head is None:
            return None

        best_candidate: Optional[ExperienceNode] = memory.head
        curr: Optional[ExperienceNode] = memory.head.next

        if strategy == PruningStrategy.LOWEST_Q:
            min_q = memory.head.q_value
            while curr is not None:
                if curr.q_value < min_q:
                    min_q = curr.q_value
                    best_candidate = curr
                curr = curr.next
            return best_candidate

        elif strategy == PruningStrategy.LFU:
            min_visits = memory.head.visit_count
            while curr is not None:
                if curr.visit_count < min_visits:
                    min_visits = curr.visit_count
                    best_candidate = curr
                curr = curr.next
            return best_candidate

        else:
            raise ValueError(f"Unsupported pruning strategy: {strategy}")
