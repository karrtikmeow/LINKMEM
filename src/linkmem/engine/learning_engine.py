"""LearningEngine: Coordinates non-iterative experience memory learning loop."""

from typing import Optional, Callable, List, Tuple
from linkmem.core.types import Action, PruningStrategy, StateVector
from linkmem.core.environment import GridWorld
from linkmem.core.memory import ExperienceMemory
from linkmem.core.agent import Agent
from linkmem.core.pruning import PruningPolicy
from linkmem.engine.events import StepEvent
from linkmem.engine.metrics import EpisodeMetrics
from linkmem.utils.config import AppConfig


class LearningEngine:
    """Non-iterative reinforcement learning execution engine.
    
    Orchestrates the environment, state encoder, agent policy, linked-list memory,
    and single-pass sample-average Q updates. Fully decoupled from any UI framework.
    """

    def __init__(
        self,
        env: GridWorld,
        memory: ExperienceMemory,
        agent: Agent,
        config: Optional[AppConfig] = None,
        pruning_strategy: Optional[PruningStrategy] = None,
        max_memory_size: Optional[int] = None,
        on_step: Optional[Callable[[StepEvent], None]] = None,
        on_episode_end: Optional[Callable[[EpisodeMetrics], None]] = None,
    ):
        self.env = env
        self.memory = memory
        self.agent = agent
        self.config = config or AppConfig()

        self.pruning_strategy = (
            pruning_strategy
            if pruning_strategy is not None
            else self.config.pruning_strategy
        )
        self.max_memory_size = (
            max_memory_size
            if max_memory_size is not None
            else self.config.n_max
        )

        self.on_step = on_step
        self.on_episode_end = on_episode_end

        # Execution state
        self.global_step: int = 0
        self.current_episode: int = 0
        self.episode_step_count: int = 0
        self.episode_reward: float = 0.0
        self.episode_collisions: int = 0
        self.episode_lookups: int = 0
        self.episode_comparisons: int = 0
        self._stopped: bool = False

        # History of completed episode metrics
        self.history: List[EpisodeMetrics] = []

    @property
    def is_terminal(self) -> bool:
        """True if the current environment episode has terminated."""
        return self.env.is_terminal

    @property
    def is_stopped(self) -> bool:
        """True if execution stop has been requested."""
        return self._stopped

    def stop(self) -> None:
        """Signal the engine to halt execution loops."""
        self._stopped = True

    def reset(self) -> StateVector:
        """Reset the environment for a new episode and initialize episode tracking.
        
        Returns:
            The initial 6D encoded state vector.
        """
        self.current_episode += 1
        self.episode_step_count = 0
        self.episode_reward = 0.0
        self.episode_collisions = 0
        self.episode_lookups = 0
        self.episode_comparisons = 0
        self._stopped = False
        return self.env.reset()

    def run_step(self, manual_action: Optional[Action] = None) -> StepEvent:
        """Execute one complete non-iterative learning step.
        
        Steps:
            1. Sense current state.
            2. Search memory & select action.
            3. Execute action in GridWorld.
            4. Update node Q-value: Q <- Q + (reward - Q) / n.
            5. Check memory capacity & prune if size > max_memory_size.
            6. Track metrics & dispatch step event.
            
        Args:
            manual_action: Optional forced action override (for manual control mode).
            
        Returns:
            StepEvent containing comprehensive transition and decision telemetry.
        """
        if self.is_terminal:
            raise RuntimeError("Cannot step a terminal environment; call reset() first.")

        agent_pos_before = self.env.agent_pos
        state_vector_before = self.env.get_state()

        # Telemetry tracking for memory lookup
        lookups_before = self.memory.total_lookups
        comparisons_before = self.memory.total_comparisons

        # Action selection and node resolution
        if manual_action is not None:
            # Manual mode override: find nearest node for state, or create novel node
            nearest_node, min_dist, comps, trace = self.memory.find_nearest(
                state_vector_before, metric=self.agent.distance_metric
            )
            action = manual_action
            if nearest_node is not None and min_dist <= self.agent.theta:
                is_match = True
                active_node = nearest_node
            else:
                is_match = False
                active_node = self.memory.insert_at_head(
                    state=state_vector_before,
                    action=action,
                    q_init=0.0,
                    visit_count=0,
                )
        else:
            action, active_node, min_dist, is_match, trace = self.agent.select_action(
                state_vector_before
            )

        step_lookups = self.memory.total_lookups - lookups_before
        step_comparisons = self.memory.total_comparisons - comparisons_before

        # Execute selected action in the environment
        next_state_vector, reward, is_terminal, info = self.env.step(action)

        # Sample-average Q-value update: Q_new = Q_old + (reward - Q_old) / n
        q_before = active_node.q_value
        q_after = active_node.update_q(reward)
        visit_count = active_node.visit_count

        # Memory capacity management: Prune if size > max_memory_size
        pruned_ids: List[int] = []
        if self.memory.size > self.max_memory_size:
            pruned_ids = PruningPolicy.prune(
                self.memory,
                strategy=self.pruning_strategy,
                target_size=self.max_memory_size,
            )

        # Update step counters
        self.global_step += 1
        self.episode_step_count += 1
        self.episode_reward += reward
        if info.get("collided", False):
            self.episode_collisions += 1
        self.episode_lookups += step_lookups
        self.episode_comparisons += step_comparisons

        # Construct immutable step event
        event = StepEvent(
            timestep=self.global_step,
            episode=self.current_episode,
            agent_pos=agent_pos_before,
            encoded_state=state_vector_before,
            matched_node_id=active_node.node_id if active_node else None,
            search_trace=trace,
            nearest_distance=min_dist,
            is_match=is_match,
            theta=self.agent.theta,
            action=action,
            reward=reward,
            next_agent_pos=self.env.agent_pos,
            next_encoded_state=next_state_vector,
            q_before=q_before,
            q_after=q_after,
            visit_count=visit_count,
            memory_size=self.memory.size,
            is_terminal=is_terminal,
            collided=info.get("collided", False),
            goal_reached=info.get("goal_reached", False),
            timed_out=info.get("timed_out", False),
            pruned_node_ids=pruned_ids,
        )

        # Dispatch step callback if registered
        if self.on_step:
            self.on_step(event)

        # If terminal, finalize episode metrics
        if is_terminal:
            self._finalize_episode(success=info.get("goal_reached", False))

        return event

    def run_episode(self, max_steps: Optional[int] = None) -> EpisodeMetrics:
        """Run an entire episode from start to termination.
        
        Args:
            max_steps: Optional override for maximum steps in this episode.
            
        Returns:
            The finalized EpisodeMetrics record.
        """
        # Reset if currently at terminal state or beginning
        if self.is_terminal or self.current_episode == 0:
            self.reset()

        limit = max_steps or self.env.max_steps

        while not self.is_terminal and not self._stopped:
            if self.episode_step_count >= limit:
                # Force timeout if external limit reached
                self.env.is_terminal = True
                self._finalize_episode(success=False)
                break
            self.run_step()

        return self.history[-1]

    def run_episodes(
        self,
        num_episodes: int,
        max_steps: Optional[int] = None,
    ) -> List[EpisodeMetrics]:
        """Run a batch of consecutive learning episodes.
        
        Args:
            num_episodes: Number of episodes to execute.
            max_steps: Optional step limit per episode.
            
        Returns:
            List of EpisodeMetrics collected during this batch run.
        """
        batch_metrics: List[EpisodeMetrics] = []
        for _ in range(num_episodes):
            if self._stopped:
                break
            metrics = self.run_episode(max_steps=max_steps)
            batch_metrics.append(metrics)
        return batch_metrics

    def _finalize_episode(self, success: bool) -> None:
        """Record and store the completed episode metrics."""
        metrics = EpisodeMetrics(
            episode_number=self.current_episode,
            total_reward=self.episode_reward,
            steps=self.episode_step_count,
            success=success,
            collision_count=self.episode_collisions,
            memory_size=self.memory.size,
            lookup_operations=self.episode_lookups,
            lookup_comparisons=self.episode_comparisons,
        )
        self.history.append(metrics)
        if self.on_episode_end:
            self.on_episode_end(metrics)
