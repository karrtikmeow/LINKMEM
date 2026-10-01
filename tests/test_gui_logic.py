"""Unit and logic tests for Phase 3 GUI controller, queue messaging, and parameter validation."""

import time
import queue
import pytest

from linkmem.core.types import Action, LearningMode
from linkmem.core.environment import GridWorld
from linkmem.core.memory import ExperienceMemory
from linkmem.core.agent import Agent
from linkmem.engine.learning_engine import LearningEngine
from linkmem.gui.controller import SimulationController
from linkmem.engine.events import StepEvent


@pytest.fixture
def controller():
    """Provides a fresh SimulationController connected to an engine."""
    env = GridWorld(width=5, height=5, start_pos=(0, 0), goal_pos=(4, 4), max_steps=20)
    memory = ExperienceMemory(max_capacity=30)
    agent = Agent(memory=memory, theta=0.35)
    engine = LearningEngine(env=env, memory=memory, agent=agent)
    q = queue.Queue()
    ctrl = SimulationController(engine, update_queue=q)
    yield ctrl
    ctrl.stop_worker()


def test_controller_initialization(controller):
    """Verify initial controller states."""
    assert controller.mode == LearningMode.AUTONOMOUS
    assert not controller.is_running
    assert not controller.is_paused
    assert controller.step_delay > 0.0


def test_mode_switching(controller):
    """Verify changing modes posts events to queue."""
    controller.set_mode(LearningMode.MANUAL)
    assert controller.mode == LearningMode.MANUAL

    msg_type, payload = controller.queue.get_nowait()
    assert msg_type == "mode_changed"
    assert payload == LearningMode.MANUAL


def test_parameter_validation(controller):
    """Verify parameter bounds checking prevents invalid values."""
    # Valid parameters
    success, msg = controller.update_parameters(theta=0.25, n_max=50)
    assert success is True
    assert controller.engine.agent.theta == 0.25
    assert controller.engine.max_memory_size == 50
    assert controller.engine.memory.max_capacity == 50

    # Negative theta
    success, msg = controller.update_parameters(theta=-0.1)
    assert success is False
    assert "Theta must be between" in msg

    # Theta > 5.0
    success, msg = controller.update_parameters(theta=6.0)
    assert success is False

    # N_max < 2
    success, msg = controller.update_parameters(n_max=1)
    assert success is False
    assert "N_max must be between" in msg


def test_step_execution_via_controller(controller):
    """Verify single-stepping through the controller."""
    event = controller.step()
    assert isinstance(event, StepEvent)
    assert event.timestep == 1

    # Check queue messages: ("reset", None) from first step auto-reset + ("step", event) + ("status", "STEPPED")
    events = []
    while not controller.queue.empty():
        events.append(controller.queue.get_nowait())

    types = [e[0] for e in events]
    assert "step" in types
    assert "status" in types


def test_manual_step_via_controller(controller):
    """Verify manual step forces the requested action."""
    controller.set_mode(LearningMode.MANUAL)
    # Drain mode change event
    while not controller.queue.empty():
        controller.queue.get_nowait()

    event = controller.manual_step(Action.RIGHT)
    assert isinstance(event, StepEvent)
    assert event.action == Action.RIGHT
    assert event.agent_pos == (0, 0)
    assert event.next_agent_pos == (1, 0)


def test_controller_reset(controller):
    """Verify controller reset cleans state and posts reset event."""
    controller.step()
    controller.reset()

    assert controller.engine.env.agent_pos == (0, 0)
    assert controller.engine.episode_step_count == 0

    events = []
    while not controller.queue.empty():
        events.append(controller.queue.get_nowait())

    types = [e[0] for e in events]
    assert "reset" in types
    assert "status" in types


def test_speed_control(controller):
    """Verify speed slider correctly calculates step delay."""
    controller.set_speed(10.0)  # 10 Hz
    assert pytest.approx(controller.step_delay, abs=1e-4) == 0.1

    controller.set_speed(50.0)  # 50 Hz
    assert pytest.approx(controller.step_delay, abs=1e-4) == 0.02


def test_worker_thread_lifecycle(controller):
    """Verify starting, pausing, and stopping background worker."""
    controller.set_speed(50.0)  # Fast ticks
    controller.start()
    time.sleep(0.08)

    assert controller.is_running
    assert not controller.is_paused

    controller.pause()
    time.sleep(0.04)
    assert controller.is_paused
    assert not controller.is_running

    controller.stop_worker()
    time.sleep(0.04)
    assert not controller.is_running
    assert not controller.is_paused
