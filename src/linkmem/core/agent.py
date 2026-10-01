"""Agent: Decision making, memory interaction, and cycle prevention for LINKMEM."""

from __future__ import annotations

import collections
import os
import random
from typing import Dict, List, Optional, Tuple, Any

from linkmem.core.types import Action, NoveltyStrategy, StateVector
from linkmem.core.memory import ExperienceMemory
from linkmem.core.node import ExperienceNode
from linkmem.core.distance import DistanceCalculator, DistanceMetric


class Agent:
    """Non-iterative learning agent navigating via linked-list experience memory.

    Incorporates episode-local cycle detection and exploration penalties to prevent
    indefinite oscillation (e.g. A <-> B ping-ponging or 4-state local loops)
    while preserving the core LINKMEM single-pass sample-average memory model.
    """

    def __init__(
        self,
        memory: ExperienceMemory,
        theta: float = 0.35,
        novelty_strategy: NoveltyStrategy = NoveltyStrategy.HEURISTIC_TOWARDS_GOAL,
        exploration_prob: float = 0.0,
        distance_metric: Optional[DistanceMetric] = None,
        seed: Optional[int] = None,
        debug: bool = False,
    ):
        self.memory = memory
        self.theta = theta
        self.novelty_strategy = novelty_strategy
        self.exploration_prob = exploration_prob
        self.distance_metric = distance_metric or DistanceCalculator.euclidean
        self.rng = random.Random(seed)
        self.debug = debug or (os.environ.get("LINKMEM_DEBUG") == "1")

        # Episode-local navigation state
        self.reset_episode()

        # Debug inspection payload of last decision
        self.last_decision_debug: Dict[str, Any] = {}

    def reset_episode(self) -> None:
        """Reset episode-local navigation tracking state.

        Called at the start of each episode or on simulation reset.
        Preserves persistent ExperienceMemory.
        """
        self.visited_states: Dict[Tuple[float, ...], int] = collections.defaultdict(int)
        self.state_action_visits: Dict[Tuple[Tuple[float, ...], Action], int] = collections.defaultdict(int)
        self.recent_states: collections.deque[Tuple[float, ...]] = collections.deque(maxlen=16)
        self.last_action: Optional[Action] = None
        self.consecutive_cycles: int = 0
        self.cycle_count: int = 0

    def select_action(
        self,
        state: StateVector,
    ) -> Tuple[Action, ExperienceNode, float, bool, List[Tuple[int, float]]]:
        """Execute the perception, memory lookup, and decision phase.

        Steps:
            1. Search linked-list memory for nearest state exemplar.
            2. Determine legal actions (unblocked by walls).
            3. Detect short cycles (e.g. 2-state oscillation or local loop).
            4. Score candidate legal actions:
               - Goal alignment (heuristic towards goal)
               - Memory bonus (if matched and Q-value is viable)
               - Visit penalty (prefer unexplored actions from this state)
               - Reversal penalty (avoid immediate ping-pong bounce)
               - Cycle penalty (break active oscillation loops)
            5. Resolve action and update/insert ExperienceNode in memory.
            6. Track episode-local navigation history.

        Args:
            state: 6D encoded state vector [dx_norm, dy_norm, wall_u, wall_d, wall_l, wall_r].

        Returns:
            Tuple of:
            - action: Chosen Action
            - active_node: ExperienceNode (reused/updated or newly prepended)
            - distance: Distance to nearest node in memory
            - is_match: True if nearest_node distance <= theta
            - trace: List of (node_id, distance) searched
        """
        nearest_node, min_dist, comparisons, trace = self.memory.find_nearest(
            state, metric=self.distance_metric
        )
        is_match = (nearest_node is not None and min_dist <= self.theta)

        norm_dx, norm_dy = state[0], state[1]
        wall_u, wall_d, wall_l, wall_r = state[2], state[3], state[4], state[5]

        # 1. Identify legal actions (not blocked by adjacent walls)
        legal_actions: List[Action] = []
        if wall_u == 0.0:
            legal_actions.append(Action.UP)
        if wall_d == 0.0:
            legal_actions.append(Action.DOWN)
        if wall_l == 0.0:
            legal_actions.append(Action.LEFT)
        if wall_r == 0.0:
            legal_actions.append(Action.RIGHT)

        if not legal_actions:
            # Trapped / surrounded fallback
            legal_actions = list(Action)

        # Discretized state key for episode-local tracking
        state_key = tuple(round(v, 4) for v in state)

        # 2. Cycle Detection (2-state oscillation or recent state recurrence)
        is_cycle = False
        cycle_period = 0
        if len(self.recent_states) >= 2 and self.recent_states[-2] == state_key:
            is_cycle = True
            cycle_period = 2
        elif state_key in self.recent_states:
            is_cycle = True
            # Find period
            for idx, prev_st in enumerate(reversed(self.recent_states)):
                if prev_st == state_key and idx > 0:
                    cycle_period = idx + 1
                    break

        if is_cycle:
            self.consecutive_cycles += 1
            self.cycle_count += 1
        else:
            self.consecutive_cycles = max(0, self.consecutive_cycles - 1)

        # 3. Action selection
        # Fast path for RANDOM_EXPLORATION on novel state or exploration_prob
        if (
            self.novelty_strategy == NoveltyStrategy.RANDOM_EXPLORATION and not is_match
        ) or (self.exploration_prob > 0.0 and self.rng.random() < self.exploration_prob):
            action = self.rng.choice(legal_actions)
            best_score = 0.0
        else:
            action, best_score = self._score_and_pick_action(
                legal_actions=legal_actions,
                state_key=state_key,
                norm_dx=norm_dx,
                norm_dy=norm_dy,
                is_match=is_match,
                nearest_node=nearest_node,
                is_cycle=is_cycle,
            )

        # 4. Memory exemplar management
        if is_match and nearest_node is not None:
            active_node = nearest_node
            # If the selected action diverged from stored action due to cycle or visit penalty,
            # update the stored action so memory learns the non-cycling escape route.
            if action != nearest_node.action and (
                is_cycle or self.state_action_visits[(state_key, nearest_node.action)] > 0
            ):
                nearest_node.action = action
                nearest_node.q_value = 0.0
                nearest_node.visit_count = 0
        else:
            active_node = self.memory.insert_at_head(
                state=state,
                action=action,
                q_init=0.0,
                visit_count=0,
            )

        # 5. Update episode navigation history
        self.visited_states[state_key] += 1
        self.state_action_visits[(state_key, action)] += 1
        self.recent_states.append(state_key)
        self.last_action = action

        # 6. Debug logging / telemetry payload
        self.last_decision_debug = {
            "state": state_key[:2],
            "legal_actions": [a.name for a in legal_actions],
            "selected_action": action.name,
            "memory_match": is_match,
            "action_score": round(best_score, 4),
            "visited_count": self.visited_states[state_key],
            "cycle_detected": is_cycle,
            "cycle_period": cycle_period,
        }

        if self.debug:
            print(
                f"STATE={state_key[:2]} "
                f"LEGAL={[a.name for a in legal_actions]} "
                f"SELECTED={action.name} "
                f"MEMORY_MATCH={is_match} "
                f"SCORE={best_score:.2f} "
                f"VISITS={self.visited_states[state_key]} "
                f"CYCLE_DETECTED={is_cycle}"
            )

        return action, active_node, min_dist, is_match, trace

    def _score_and_pick_action(
        self,
        legal_actions: List[Action],
        state_key: Tuple[float, ...],
        norm_dx: float,
        norm_dy: float,
        is_match: bool,
        nearest_node: Optional[ExperienceNode],
        is_cycle: bool,
    ) -> Tuple[Action, float]:
        """Score all candidate legal actions and select the optimal choice."""
        opposite = {
            Action.UP: Action.DOWN,
            Action.DOWN: Action.UP,
            Action.LEFT: Action.RIGHT,
            Action.RIGHT: Action.LEFT,
        }

        best_action = legal_actions[0]
        best_score = -float("inf")

        for a in legal_actions:
            score = 0.0

            # 1. Goal alignment heuristic
            dx_dir = 1.0 if a == Action.RIGHT else (-1.0 if a == Action.LEFT else 0.0)
            dy_dir = 1.0 if a == Action.DOWN else (-1.0 if a == Action.UP else 0.0)
            goal_alignment = dx_dir * norm_dx + dy_dir * norm_dy
            score += 1.0 * goal_alignment

            # 2. Memory experience signal
            if is_match and nearest_node is not None and a == nearest_node.action:
                if nearest_node.q_value < -0.1:
                    # Penalize previously recorded collisions / negative outcomes
                    score += nearest_node.q_value
                else:
                    # Reusable positive/neutral experience exemplar
                    score += max(0.5, nearest_node.q_value)


            # 3. Episode-local visit penalty (prefer unexplored legal actions)
            visits = self.state_action_visits[(state_key, a)]
            score -= visits * 2.0

            # 4. Immediate reversal penalty (avoid immediate ping-pong bounce if alternatives exist)
            if self.last_action and a == opposite.get(self.last_action) and len(legal_actions) > 1:
                min_other_visits = min(
                    self.state_action_visits[(state_key, other)]
                    for other in legal_actions
                    if other != a
                )
                if visits >= min_other_visits:
                    score -= 1.2

            # 5. Cycle penalty (heavily penalize repeating actions in an active loop)
            if is_cycle and visits > 0:
                score -= 3.0 * self.consecutive_cycles

            # Deterministic, seeded jitter for tie-breaking
            score += self.rng.uniform(0.0, 0.01)

            if score > best_score:
                best_score = score
                best_action = a

        return best_action, best_score
