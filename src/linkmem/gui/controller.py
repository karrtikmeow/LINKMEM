"""SimulationController: Thread-safe orchestration between worker thread and GUI."""

import time
import threading
import queue
from typing import Optional, Tuple, Any
from linkmem.core.types import Action, LearningMode
from linkmem.engine.learning_engine import LearningEngine
from linkmem.engine.events import StepEvent
from linkmem.engine.metrics import EpisodeMetrics


class SimulationController:
    """Manages execution flow of the LearningEngine via a background worker thread.
    
    All communication with the Tkinter UI is achieved asynchronously through queue.Queue.
    """

    def __init__(self, engine: LearningEngine, update_queue: Optional[queue.Queue] = None):
        self.engine = engine
        self.queue = update_queue if update_queue is not None else queue.Queue()

        self.mode: LearningMode = LearningMode.AUTONOMOUS
        self.step_delay: float = 0.08  # Default ~12 steps per second

        self._worker: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._lock = threading.Lock()

        # Connect engine callbacks to push directly into queue
        self.engine.on_step = self._handle_engine_step
        self.engine.on_episode_end = self._handle_engine_episode_end

    def _handle_engine_step(self, event: StepEvent) -> None:
        """Push StepEvent to the GUI update queue."""
        self.queue.put(("step", event))

    def _handle_engine_episode_end(self, metrics: EpisodeMetrics) -> None:
        """Push EpisodeMetrics to the GUI update queue."""
        self.queue.put(("episode_complete", metrics))

    @property
    def is_running(self) -> bool:
        """True if the worker thread is active and not paused."""
        return (
            self._worker is not None
            and self._worker.is_alive()
            and not self._pause_event.is_set()
            and not self._stop_event.is_set()
        )

    @property
    def is_paused(self) -> bool:
        """True if worker thread is running but currently paused."""
        return (
            self._worker is not None
            and self._worker.is_alive()
            and self._pause_event.is_set()
        )

    def set_mode(self, mode: LearningMode) -> None:
        """Change the simulation mode (Autonomous, Manual, Step-by-Step)."""
        with self._lock:
            if self.mode != mode:
                # If running autonomously, pause when switching away
                if self.is_running and mode != LearningMode.AUTONOMOUS:
                    self.pause()
                self.mode = mode
                self.queue.put(("mode_changed", mode))

    def set_speed(self, steps_per_second: float) -> None:
        """Update delay between steps in autonomous mode (1 to 50 Hz)."""
        rate = max(0.5, min(100.0, float(steps_per_second)))
        self.step_delay = 1.0 / rate

    def update_parameters(self, theta: Optional[float] = None, n_max: Optional[int] = None) -> Tuple[bool, str]:
        """Validate and apply live hyperparameter changes.
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        with self._lock:
            if theta is not None:
                if theta < 0.0 or theta > 5.0:
                    return False, "Theta must be between 0.0 and 5.0."
                self.engine.agent.theta = float(theta)

            if n_max is not None:
                if n_max < 2 or n_max > 5000:
                    return False, "N_max must be between 2 and 5000."
                self.engine.max_memory_size = int(n_max)
                self.engine.memory.max_capacity = int(n_max)

            return True, "Parameters updated successfully."

    def start(self) -> None:
        """Start or resume autonomous simulation execution."""
        with self._lock:
            if self.mode != LearningMode.AUTONOMOUS:
                self.mode = LearningMode.AUTONOMOUS
                self.queue.put(("mode_changed", self.mode))

            if self._worker is not None and self._worker.is_alive():
                # Resume from paused state
                self._pause_event.clear()
                self.queue.put(("status", "RUNNING"))
                return

            # Start fresh worker thread
            self._stop_event.clear()
            self._pause_event.clear()
            self._worker = threading.Thread(target=self._run_autonomous_loop, daemon=True)
            self._worker.start()
            self.queue.put(("status", "RUNNING"))

    def pause(self) -> None:
        """Pause autonomous simulation execution."""
        self._pause_event.set()
        self.queue.put(("status", "PAUSED"))

    def step(self) -> Optional[StepEvent]:
        """Execute exactly one step (for Step-by-Step mode)."""
        with self._lock:
            if self.is_running:
                self.pause()

            if self.engine.is_terminal or self.engine.current_episode == 0:
                self.engine.reset()
                self.queue.put(("reset", None))

            event = self.engine.run_step()
            self.queue.put(("status", "STEPPED"))
            return event

    def manual_step(self, action: Action) -> Optional[StepEvent]:
        """Execute one step driven by user keyboard input (Manual mode)."""
        with self._lock:
            if self.is_running:
                self.pause()

            if self.engine.is_terminal or self.engine.current_episode == 0:
                self.engine.reset()
                self.queue.put(("reset", None))

            event = self.engine.run_step(manual_action=action)
            self.queue.put(("status", f"MANUAL {action.name}"))
            return event

    def reset(self) -> None:
        """Stop worker, reset environment and engine, and notify GUI."""
        self.stop_worker()
        with self._lock:
            self.engine.reset()
            self.queue.put(("reset", None))
            self.queue.put(("status", "READY"))

    def stop_worker(self) -> None:
        """Cooperatively stop and join the worker thread."""
        self._stop_event.set()
        self._pause_event.clear()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=0.3)
        self._worker = None

    def _run_autonomous_loop(self) -> None:
        """Background thread target loop running continuous episodes."""
        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                time.sleep(0.05)
                continue

            try:
                with self._lock:
                    if self.engine.is_terminal or self.engine.current_episode == 0:
                        self.engine.reset()

                    self.engine.run_step()

                if self.step_delay > 0:
                    time.sleep(self.step_delay)

            except Exception as ex:
                self.queue.put(("error", str(ex)))
                self.queue.put(("status", "ERROR"))
                break

        self.queue.put(("status", "STOPPED"))
