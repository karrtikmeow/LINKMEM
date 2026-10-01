"""ExperienceNode: Singly linked list node storing state-action exemplars."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
from linkmem.core.types import Action, StateVector


@dataclass
class ExperienceNode:
    """A single experience node in the linked-list memory.
    
    Attributes:
        node_id: Unique monotonic identifier for debugging/visualization.
        state: Encoded normalized feature vector.
        action: The action associated with this experience.
        q_value: Estimated quality/value computed by sample-average formula.
        visit_count: Number of times this node has been visited/updated (n >= 1).
        next: Pointer to the next ExperienceNode in the linked list (or None).
    """

    node_id: int
    state: StateVector
    action: Action
    q_value: float = 0.0
    visit_count: int = 1
    next: Optional[ExperienceNode] = None

    def update_q(self, reward: float) -> float:
        """Update Q-value using non-iterative sample-average update.
        
        Formula:
            n <- n + 1
            Q <- Q + (reward - Q) / n
            
        Args:
            reward: Immediate scalar reward received from environment.
            
        Returns:
            The newly updated Q-value.
        """
        self.visit_count += 1
        self.q_value += (reward - self.q_value) / self.visit_count
        return self.q_value

    def to_dict(self) -> Dict[str, Any]:
        """Convert node attributes to dictionary representation."""
        return {
            "node_id": self.node_id,
            "state": list(self.state),
            "action": self.action.name,
            "q_value": round(self.q_value, 4),
            "visit_count": self.visit_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExperienceNode:
        """Construct an ExperienceNode from a dictionary."""
        return cls(
            node_id=int(data["node_id"]),
            state=tuple(float(x) for x in data["state"]),
            action=Action.from_str(data["action"]),
            q_value=float(data.get("q_value", 0.0)),
            visit_count=int(data.get("visit_count", 1)),
            next=None,
        )

    def __repr__(self) -> str:
        return (
            f"ExperienceNode(id={self.node_id}, action={self.action.name}, "
            f"Q={self.q_value:.3f}, visits={self.visit_count})"
        )
