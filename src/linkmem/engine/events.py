"""Data structures for engine observation events and step telemetry."""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional
from linkmem.core.types import Action, StateVector


@dataclass(frozen=True)
class StepEvent:
    """Immutable event payload generated after each learning step.
    
    Exposes complete internal decision and memory state for inspection and future GUI display.
    """
    timestep: int
    episode: int
    agent_pos: Tuple[int, int]
    encoded_state: StateVector
    matched_node_id: Optional[int]
    search_trace: List[Tuple[int, float]]
    nearest_distance: float
    is_match: bool
    theta: float
    action: Action
    reward: float
    next_agent_pos: Tuple[int, int]
    next_encoded_state: StateVector
    q_before: float
    q_after: float
    visit_count: int
    memory_size: int
    is_terminal: bool
    collided: bool
    goal_reached: bool
    timed_out: bool
    pruned_node_ids: List[int] = field(default_factory=list)
