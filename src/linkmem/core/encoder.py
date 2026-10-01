"""StateEncoder: Transforms raw grid coordinates into a 6D normalized feature vector."""

from typing import Tuple, Set
from linkmem.core.types import StateVector


class StateEncoder:
    """Encodes raw GridWorld state into the simplified 6D invariant feature vector:
    
    [
        normalized_dx_goal,
        normalized_dy_goal,
        wall_up,
        wall_down,
        wall_left,
        wall_right
    ]
    
    Where:
        - normalized_dx_goal = (x_goal - x_agent) / max(1, width - 1) in [-1.0, 1.0]
        - normalized_dy_goal = (y_goal - y_agent) / max(1, height - 1) in [-1.0, 1.0]
        - wall_* in {0.0, 1.0} indicates whether moving in that direction hits a wall or obstacle.
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self._max_dx = max(1, width - 1)
        self._max_dy = max(1, height - 1)

    def encode(
        self,
        agent_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
        obstacles: Set[Tuple[int, int]],
    ) -> StateVector:
        """Encode raw positions and obstacles into the 6D feature vector.
        
        Args:
            agent_pos: (x, y) coordinates of the agent.
            goal_pos: (x, y) coordinates of the goal.
            obstacles: Set of (x, y) obstacle cell coordinates.
            
        Returns:
            A 6-element tuple of floats.
        """
        ax, ay = agent_pos
        gx, gy = goal_pos

        norm_dx = (gx - ax) / self._max_dx
        norm_dy = (gy - ay) / self._max_dy

        # Check immediate adjacent cells (cardinal directions)
        wall_up = 1.0 if (ay - 1 < 0 or (ax, ay - 1) in obstacles) else 0.0
        wall_down = 1.0 if (ay + 1 >= self.height or (ax, ay + 1) in obstacles) else 0.0
        wall_left = 1.0 if (ax - 1 < 0 or (ax - 1, ay) in obstacles) else 0.0
        wall_right = 1.0 if (ax + 1 >= self.width or (ax + 1, ay) in obstacles) else 0.0

        return (
            float(norm_dx),
            float(norm_dy),
            wall_up,
            wall_down,
            wall_left,
            wall_right,
        )
