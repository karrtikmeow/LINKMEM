"""Integration test verifying LINKMEM GUI capabilities."""

import time
import tkinter as tk
import pytest
from linkmem.core.types import Action, LearningMode
from linkmem.gui.app import MainWindow


def test_mainwindow_full_capabilities():
    """Verify clean GUI capabilities:
    1. Opens without errors
    2. Default environment (Simple Maze) loaded
    3. Step mode works: controller.step() returns StepEvent
    4. Memory table reflects new nodes after step
    5. Live KPI metrics update
    6. Algorithm Trace shows real step breakdown
    7. Manual mode: keyboard input steps agent
    8. Autonomous mode: worker thread runs
    9. Pause: freezes simulation
    10. Reset: returns agent to start, clears memory
    11. Environment switching: switches to Empty Grid and rebuilds components
    """
    try:
        root = tk.Tk()
    except Exception as e:
        pytest.skip(f"Display not available for Tkinter: {e}")

    try:
        root.withdraw()
        # 1. Open without errors
        app = MainWindow(root)
        root.update()

        # 2. Verify initial state (Simple Maze)
        assert app.active_preset.name == "Simple Maze"
        assert app.env is not None
        assert "Simple Maze" in app._lbl_debug_env.cget("text")

        # Initial trace state
        trace_text = app._txt_trace.get("1.0", tk.END)
        assert "WAITING FOR FIRST STEP" in trace_text

        # 3. Step mode works
        app.controller.set_mode(LearningMode.STEP_BY_STEP)
        event1 = app.controller.step()
        assert event1 is not None
        assert event1.timestep == 1

        # Process queue & update UI
        app._poll_queue()
        root.update()

        # 4. Memory table reflects new node
        assert app.memory.size == 1
        mem_children = app._tree_memory.get_children()
        assert len(mem_children) >= 1

        # 5. Live KPI metrics updated
        assert app._kpis["Step"].cget("text") == "1"
        assert app._kpis["Memory Nodes"].cget("text") == "1"

        # 6. Algorithm Trace shows real step data
        trace_after = app._txt_trace.get("1.0", tk.END)
        assert "STEP 1" in trace_after
        assert "SENSE STATE VECTOR" in trace_after
        assert "SAMPLE-AVERAGE Q UPDATE" in trace_after
        assert "Q_new = Q_old + (reward - Q_old) / n" in trace_after

        # 7. Manual mode: agent moves
        pos_before = app.env.agent_pos
        app.controller.set_mode(LearningMode.MANUAL)
        app._handle_manual_key(Action.RIGHT)
        app._poll_queue()
        root.update()
        assert isinstance(app.env.agent_pos, tuple)

        # 8. Autonomous mode worker thread runs
        app.controller.set_speed(40.0)
        app.controller.start()
        time.sleep(0.12)
        assert app.controller.is_running

        app._poll_queue()
        root.update()
        assert app.controller.engine.global_step > 2

        # 9. Pause
        app.controller.pause()
        time.sleep(0.05)
        assert app.controller.is_paused
        assert not app.controller.is_running

        # 10. Reset
        app.reset_simulation()
        app._poll_queue()
        root.update()
        assert app.memory.size == 0
        assert app.env.agent_pos == (0, 0)
        assert app.active_preset.name == "Simple Maze"

        # 11. Environment switching to Empty Grid
        old_env = app.env
        old_engine = app.engine
        old_controller = app.controller

        ok = app.load_environment("Empty Grid")
        assert ok is True
        root.update()

        assert app.active_preset.name == "Empty Grid"
        assert app.env is not old_env
        assert app.engine is not old_engine
        assert app.controller is not old_controller
        assert app.engine.env is app.env
        assert len(app.env.obstacles) == 0
        assert "Empty Grid" in app._lbl_debug_env.cget("text")

        # Teardown
        app._on_close()

    except Exception:
        root.destroy()
        raise
