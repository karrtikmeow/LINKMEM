"""GridWorld: 2D discrete environment simulation for LINKMEM."""

import random
from typing import Tuple, Set, Optional, Dict, Any
from linkmem.core.types import Action, CellType
from linkmem.core.encoder import StateEncoder


class GridWorld:
    """Discrete 2D GridWorld environment for agent navigation.
    
    Coordinates use standard (x, y) with:
        x in [0, width - 1] (0 is leftmost, width-1 is rightmost)
        y in [0, height - 1] (0 is top, height-1 is bottom)
    """

    def __init__(
        self,
        width: int = 8,
        height: int = 8,
        start_pos: Tuple[int, int] = (0, 0),
        goal_pos: Tuple[int, int] = (7, 7),
        obstacles: Optional[Set[Tuple[int, int]]] = None,
        max_steps: int = 100,
        reward_goal: float = 1.0,
        reward_step: float = -0.01,
        reward_collision: float = -0.5,
    ):
        self.width = width
        self.height = height
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.obstacles: Set[Tuple[int, int]] = set(obstacles) if obstacles else set()
        self.max_steps = max_steps
        
        self.reward_goal = reward_goal
        self.reward_step = reward_step
        self.reward_collision = reward_collision
        
        self.encoder = StateEncoder(width, height)
        
        # State variables
        self.agent_pos: Tuple[int, int] = start_pos
        self.step_count: int = 0
        self.collision_count: int = 0
        self.is_terminal: bool = False

    def is_valid_cell(self, x: int, y: int) -> bool:
        """Check if (x, y) is within bounds and not an obstacle."""
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        if (x, y) in self.obstacles:
            return False
        return True

    def reset(self) -> Tuple[float, ...]:
        """Reset the environment to start position.
        
        Returns:
            The initial 6D encoded state vector.
        """
        self.agent_pos = self.start_pos
        self.step_count = 0
        self.collision_count = 0
        self.is_terminal = False
        return self.get_state()

    def get_state(self) -> Tuple[float, ...]:
        """Return the current 6D encoded state vector."""
        return self.encoder.encode(self.agent_pos, self.goal_pos, self.obstacles)

    def step(self, action: Action) -> Tuple[Tuple[float, ...], float, bool, Dict[str, Any]]:
        """Execute one action transition.
        
        Args:
            action: Action enum to execute.
            
        Returns:
            A tuple of (next_state_vector, reward, is_terminal, info_dict)
        """
        if self.is_terminal:
            return self.get_state(), 0.0, True, {"reason": "already_terminal"}

        self.step_count += 1
        dx, dy = action.delta
        target_x = self.agent_pos[0] + dx
        target_y = self.agent_pos[1] + dy

        collided = False
        if not self.is_valid_cell(target_x, target_y):
            # Collision: agent bounces back to original position
            collided = True
            self.collision_count += 1
            reward = self.reward_collision
        else:
            # Valid movement
            self.agent_pos = (target_x, target_y)
            if self.agent_pos == self.goal_pos:
                reward = self.reward_goal
                self.is_terminal = True
            else:
                reward = self.reward_step

        # Timeout termination check
        if self.step_count >= self.max_steps and not self.is_terminal:
            self.is_terminal = True

        info = {
            "collided": collided,
            "goal_reached": self.agent_pos == self.goal_pos,
            "timed_out": self.step_count >= self.max_steps and self.agent_pos != self.goal_pos,
            "agent_pos": self.agent_pos,
            "step_count": self.step_count,
        }

        return self.get_state(), reward, self.is_terminal, info

    @classmethod
    def create_with_density(
        cls,
        width: int = 8,
        height: int = 8,
        density: float = 0.2,
        seed: Optional[int] = None,
        start_pos: Tuple[int, int] = (0, 0),
        goal_pos: Optional[Tuple[int, int]] = None,
    ) -> "GridWorld":
        """Factory method to generate a GridWorld with random obstacles."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()

        if goal_pos is None:
            goal_pos = (width - 1, height - 1)

        obstacles: Set[Tuple[int, int]] = set()
        total_cells = width * height
        num_obstacles = int(total_cells * density)

        all_cells = [
            (x, y)
            for x in range(width)
            for y in range(height)
            if (x, y) != start_pos and (x, y) != goal_pos
        ]
        
        sampled = rng.sample(all_cells, min(num_obstacles, len(all_cells)))
        obstacles.update(sampled)

        return cls(
            width=width,
            height=height,
            start_pos=start_pos,
            goal_pos=goal_pos,
            obstacles=obstacles,
        )
