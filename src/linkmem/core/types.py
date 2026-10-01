"""Enumerations and shared type definitions for LINKMEM."""

from enum import Enum, auto
from typing import Tuple

StateVector = Tuple[float, ...]


class Action(Enum):
    """Discrete cardinal navigation actions for the GridWorld agent."""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    @property
    def delta(self) -> Tuple[int, int]:
        """Return (dx, dy) movement delta for the grid coordinate space."""
        if self == Action.UP:
            return (0, -1)
        if self == Action.DOWN:
            return (0, 1)
        if self == Action.LEFT:
            return (-1, 0)
        if self == Action.RIGHT:
            return (1, 0)
        raise ValueError(f"Unknown action: {self}")

    @classmethod
    def from_str(cls, name: str) -> "Action":
        """Convert string to Action enum."""
        return cls[name.upper()]


class CellType(Enum):
    """GridWorld cell classifications."""
    EMPTY = 0
    OBSTACLE = 1
    START = 2
    GOAL = 3


class PruningStrategy(Enum):
    """Memory eviction strategies when capacity exceeds N_max."""
    LOWEST_Q = auto()  # Primary strategy: remove node with minimum Q value
    LFU = auto()       # Secondary benchmark: remove node with minimum visit count


class LearningMode(Enum):
    """Execution modes for simulation interaction."""
    AUTONOMOUS = auto()
    MANUAL = auto()
    STEP_BY_STEP = auto()


class NoveltyStrategy(Enum):
    """Action selection strategy when state distance > threshold theta."""
    HEURISTIC_TOWARDS_GOAL = auto()
    RANDOM_EXPLORATION = auto()
