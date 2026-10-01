"""Agent: Decision making and memory interaction for LINKMEM."""

import random
from typing import Optional, Tuple, List
from linkmem.core.types import Action, StateVector, NoveltyStrategy
from linkmem.core.memory import ExperienceMemory
from linkmem.core.node import ExperienceNode
from linkmem.core.distance import DistanceCalculator, DistanceMetric


class Agent:
    """Non-iterative learning agent navigating via linked-list experience memory."""

    def __init__(
        self,
        memory: ExperienceMemory,
        theta: float = 0.35,
        novelty_strategy: NoveltyStrategy = NoveltyStrategy.HEURISTIC_TOWARDS_GOAL,
        exploration_prob: float = 0.0,
        distance_metric: Optional[DistanceMetric] = None,
        seed: Optional[int] = None,
    ):
        self.memory = memory
        self.theta = theta
        self.novelty_strategy = novelty_strategy
        self.exploration_prob = exploration_prob
        self.distance_metric = distance_metric or DistanceCalculator.euclidean
        self.rng = random.Random(seed)

    def select_action(
        self,
        state: StateVector,
    ) -> Tuple[Action, ExperienceNode, float, bool, List[Tuple[int, float]]]:
        """Execute the perception and decision phase.
        
        Steps:
            1. Search linked-list memory for nearest state.
            2. If distance <= theta: reuse stored node and action.
            3. Otherwise: create a new ExperienceNode at list head using novelty strategy.
            
        Args:
            state: 6D encoded state vector.
            
        Returns:
            Tuple of:
            - action: Chosen Action
            - active_node: The ExperienceNode (reused or newly prepended)
            - distance: Distance to nearest node in memory (or inf if was empty)
            - is_match: True if distance <= theta (reuse), False if new node created
            - trace: List of (node_id, distance) searched
        """
        nearest_node, min_dist, comparisons, trace = self.memory.find_nearest(
            state, metric=self.distance_metric
        )

        if nearest_node is not None and min_dist <= self.theta:
            # MATCH FOUND: Reuse stored experience node
            is_match = True
            active_node = nearest_node
            
            # Optional exploration does not change the learning update rule
            if self.exploration_prob > 0.0 and self.rng.random() < self.exploration_prob:
                action = self.rng.choice(list(Action))
            else:
                action = nearest_node.action
        else:
            # NOVEL STATE: Select action via configured strategy and prepend to memory
            is_match = False
            action = self._select_novel_action(state)
            
            # Initialize with visit_count=0 so first update_q() sets visit_count=1 and Q=reward
            active_node = self.memory.insert_at_head(
                state=state,
                action=action,
                q_init=0.0,
                visit_count=0,
            )

        return action, active_node, min_dist, is_match, trace

    def _select_novel_action(self, state: StateVector) -> Action:
        """Choose action for an unvisited/novel state vector."""
        if self.novelty_strategy == NoveltyStrategy.RANDOM_EXPLORATION:
            return self.rng.choice(list(Action))

        # NoveltyStrategy.HEURISTIC_TOWARDS_GOAL:
        # state = [norm_dx_goal, norm_dy_goal, wall_u, wall_d, wall_l, wall_r]
        norm_dx, norm_dy = state[0], state[1]
        wall_u, wall_d, wall_l, wall_r = state[2], state[3], state[4], state[5]

        # Prioritize axis with larger distance to goal
        horizontal_first = abs(norm_dx) >= abs(norm_dy)

        candidate_actions: List[Action] = []
        if horizontal_first:
            if norm_dx > 0 and wall_r == 0.0:
                candidate_actions.append(Action.RIGHT)
            elif norm_dx < 0 and wall_l == 0.0:
                candidate_actions.append(Action.LEFT)
                
            if norm_dy > 0 and wall_d == 0.0:
                candidate_actions.append(Action.DOWN)
            elif norm_dy < 0 and wall_u == 0.0:
                candidate_actions.append(Action.UP)
        else:
            if norm_dy > 0 and wall_d == 0.0:
                candidate_actions.append(Action.DOWN)
            elif norm_dy < 0 and wall_u == 0.0:
                candidate_actions.append(Action.UP)
                
            if norm_dx > 0 and wall_r == 0.0:
                candidate_actions.append(Action.RIGHT)
            elif norm_dx < 0 and wall_l == 0.0:
                candidate_actions.append(Action.LEFT)

        if candidate_actions:
            return candidate_actions[0]

        # Fallback: Pick any non-blocked cardinal direction
        unblocked: List[Action] = []
        if wall_u == 0.0:
            unblocked.append(Action.UP)
        if wall_d == 0.0:
            unblocked.append(Action.DOWN)
        if wall_l == 0.0:
            unblocked.append(Action.LEFT)
        if wall_r == 0.0:
            unblocked.append(Action.RIGHT)

        if unblocked:
            return self.rng.choice(unblocked)

        # Extreme fallback (surrounded)
        return self.rng.choice(list(Action))
