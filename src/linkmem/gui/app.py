"""MainWindow: Clean, robust, single-source-of-truth GUI for LINKMEM."""

from __future__ import annotations

import queue
import tkinter as tk
from tkinter import ttk
from typing import Optional, Union

from linkmem.core.types import Action, LearningMode
from linkmem.core.environment import GridWorld
from linkmem.core.memory import ExperienceMemory
from linkmem.core.agent import Agent
from linkmem.engine.learning_engine import LearningEngine
from linkmem.engine.events import StepEvent
from linkmem.engine.metrics import EpisodeMetrics
from linkmem.gui.controller import SimulationController
from linkmem.gui.environments import (
    PRESET_NAMES,
    EnvironmentManager,
    EnvironmentPreset,
    RandomObstaclesConfig,
    build_preset,
    validate_preset,
)
from linkmem.gui.theme import (
    apply_theme,
    COLOR_BG_CARD,
    COLOR_BG_MAIN,
    COLOR_BORDER,
    COLOR_DANGER,
    COLOR_PRIMARY,
    COLOR_SUCCESS,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
    COLOR_WARNING,
    FONT_BODY,
    FONT_BODY_BOLD,
    FONT_CODE,
    FONT_SMALL,
    FONT_SUBTITLE,
    FONT_TITLE,
)
from linkmem.gui.views.grid_view import GridView
from linkmem.gui.views.custom_env_editor import CustomEnvEditor
from linkmem.gui.views.settings_dialog import SettingsDialog
from linkmem.utils.config import AppConfig


class MainWindow:
    """Master application window for LINKMEM.

    Architecture:
        Single clear source of truth:
        - self.env: current GridWorld
        - self.memory: current ExperienceMemory
        - self.agent: current Agent
        - self.engine: current LearningEngine
        - self.controller: current SimulationController

        When the environment is switched, ALL FIVE are cleanly recreated.
        The old simulation worker is stopped and joined before new creation.
    """

    def __init__(self, root: tk.Tk, config: Optional[AppConfig] = None):
        self.root = root
        self.config = config or AppConfig()

        self.root.title("LINKMEM — Interactive Non-Iterative Learning")
        self.root.geometry("1180x800")
        self.root.minsize(980, 680)
        self.root.configure(background=COLOR_BG_MAIN)

        apply_theme(self.root)

        # Simulation configuration state
        self.speed_hz: float = 12.0
        self.generation: int = 0
        self.env_manager = EnvironmentManager()
        self.active_preset: EnvironmentPreset = self.env_manager.active_preset

        # Single source of truth simulation objects (initialized in load_environment)
        self.env: Optional[GridWorld] = None
        self.memory: Optional[ExperienceMemory] = None
        self.agent: Optional[Agent] = None
        self.engine: Optional[LearningEngine] = None
        self.controller: Optional[SimulationController] = None

        # Cumulative metrics across episodes in current environment session
        self.total_episodes: int = 0
        self.total_successes: int = 0

        # Build UI layout
        self._build_ui()
        self._bind_keys()

        # Initial environment load (Simple Maze default)
        self.load_environment(self.active_preset)

        # Register window close and start polling queue
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(20, self._poll_queue)

    # ------------------------------------------------------------------
    # Environment Loading & Simulation Lifecycle (CORE ARCHITECTURE)
    # ------------------------------------------------------------------

    def load_environment(self, preset_or_name: Union[str, EnvironmentPreset]) -> bool:
        """Atomically discard old simulation and instantiate a new environment pipeline.

        Follows strict sequence:
        1. Stop old background worker thread and disconnect callbacks
        2. Resolve and validate new EnvironmentPreset
        3. Create new GridWorld
        4. Create new ExperienceMemory
        5. Create new Agent
        6. Create new LearningEngine
        7. Create new SimulationController
        8. Reset GUI metrics and tables
        9. Update debug verification banner directly from self.env
        10. Refresh GridView
        """
        # 1. Stop old worker thread
        if self.controller is not None:
            self.controller.stop_worker()
            if self.engine is not None:
                self.engine.on_step = None
                self.engine.on_episode_end = None

        # 2. Resolve preset
        if isinstance(preset_or_name, str):
            if preset_or_name == "Random Obstacles":
                preset = build_preset("Random Obstacles", random_cfg=self.env_manager.random_config)
            elif preset_or_name == "Custom Environment":
                preset = self.env_manager._custom_preset
            else:
                preset = build_preset(preset_or_name)
        else:
            preset = preset_or_name

        ok, err = validate_preset(preset)
        if not ok:
            self._set_status(f"ERROR: {err}", is_error=True)
            return False

        self.active_preset = preset
        self.generation += 1

        # 3. Create fresh GridWorld
        self.env = GridWorld(
            width=preset.width,
            height=preset.height,
            start_pos=preset.start_pos,
            goal_pos=preset.goal_pos,
            obstacles=set(preset.obstacles),
            max_steps=self.config.max_steps_per_episode,
            reward_goal=self.config.reward_goal,
            reward_step=self.config.reward_step,
            reward_collision=self.config.reward_collision,
        )

        # 4. Create fresh ExperienceMemory
        self.memory = ExperienceMemory(max_capacity=self.config.n_max)

        # 5. Create fresh Agent
        self.agent = Agent(
            memory=self.memory,
            theta=self.config.theta,
            novelty_strategy=self.config.novelty_strategy,
            seed=self.config.random_seed,
        )

        # 6. Create fresh LearningEngine
        self.engine = LearningEngine(
            env=self.env,
            memory=self.memory,
            agent=self.agent,
            config=self.config,
            pruning_strategy=self.config.pruning_strategy,
            max_memory_size=self.config.n_max,
        )

        # 7. Create fresh SimulationController
        self.controller = SimulationController(self.engine)
        self.controller.set_speed(self.speed_hz)

        # Sync mode
        current_mode = LearningMode[self._mode_var.get()] if hasattr(self, "_mode_var") else LearningMode.AUTONOMOUS
        self.controller.set_mode(current_mode)

        # 8. Reset GUI statistics and logs
        self.total_episodes = 0
        self.total_successes = 0
        self._clear_gui_stats_and_tables()

        # 9. Update debug banner directly from self.env attributes
        self._update_debug_banner()

        # 10. Update GridView
        if hasattr(self, "grid_view") and self.grid_view is not None:
            self.grid_view.set_environment(self.env)

        # 11. Sync environment dropdown value
        if hasattr(self, "_env_combo"):
            self._env_combo.set(preset.name)
            self._update_env_extra_controls(preset.name)

        self._set_status(f"READY — {preset.name} loaded")
        return True

    def reset_simulation(self) -> None:
        """Reset the simulation for the CURRENT environment without changing environments.

        Guarantees:
        - Current environment remains identical
        - Memory is cleared
        - Engine is reset to step 0, episode 0
        - Agent returns to start
        - Statistics are cleared
        - Simulation is stopped
        - Status is READY
        """
        if self.controller is not None:
            self.controller.stop_worker()

        if self.memory is not None:
            self.memory.clear()

        if self.engine is not None:
            self.engine.reset()
            self.engine.history.clear()
            self.engine.global_step = 0
            self.engine.current_episode = 0
            self.engine.episode_step_count = 0
            self.engine.episode_reward = 0.0
            self.engine.episode_collisions = 0
            self.engine.episode_lookups = 0
            self.engine.episode_comparisons = 0

        if self.env is not None:
            self.env.reset()

        self.total_episodes = 0
        self.total_successes = 0
        self._clear_gui_stats_and_tables()

        if hasattr(self, "grid_view") and self.grid_view is not None and self.env is not None:
            self.grid_view.set_environment(self.env)

        self._set_status(f"READY — {self.active_preset.name} reset")

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # ── 1. Header Bar ─────────────────────────────────────────────
        header = ttk.Frame(self.root)
        header.pack(fill=tk.X, padx=12, pady=(10, 4))

        title_box = ttk.Frame(header)
        title_box.pack(side=tk.LEFT)
        ttk.Label(title_box, text="LINKMEM", font=FONT_TITLE, foreground=COLOR_PRIMARY).pack(side=tk.LEFT)
        ttk.Label(
            title_box,
            text="  —  Non-Iterative Learning using Experience Memory",
            font=FONT_SMALL,
            foreground=COLOR_TEXT_SECONDARY,
        ).pack(side=tk.LEFT, pady=(4, 0))

        ttk.Button(header, text="⚙ Settings", command=self._open_settings).pack(side=tk.RIGHT)

        # ── 2. Environment Selector Bar ───────────────────────────────
        env_bar = ttk.Frame(self.root)
        env_bar.pack(fill=tk.X, padx=12, pady=(4, 4))

        ttk.Label(env_bar, text="Environment:", font=FONT_BODY_BOLD).pack(side=tk.LEFT, padx=(0, 6))

        self._env_combo = ttk.Combobox(
            env_bar,
            values=PRESET_NAMES,
            state="readonly",
            width=20,
            font=FONT_BODY,
        )
        self._env_combo.set(self.active_preset.name)
        self._env_combo.pack(side=tk.LEFT, padx=(0, 6))
        self._env_combo.bind("<<ComboboxSelected>>", self._on_env_combo_selected)

        ttk.Button(env_bar, text="Apply Environment", command=self._on_apply_env_clicked).pack(
            side=tk.LEFT, padx=(0, 8)
        )

        # Inline controls for Random Obstacles
        self._random_frame = ttk.Frame(env_bar)
        ttk.Label(self._random_frame, text="Obstacles:", font=FONT_SMALL).pack(side=tk.LEFT, padx=(6, 2))
        self._rand_count_var = tk.StringVar(value="6")
        ttk.Entry(self._random_frame, textvariable=self._rand_count_var, width=4).pack(side=tk.LEFT)

        ttk.Label(self._random_frame, text="Seed:", font=FONT_SMALL).pack(side=tk.LEFT, padx=(6, 2))
        self._rand_seed_var = tk.StringVar(value="0")
        ttk.Entry(self._random_frame, textvariable=self._rand_seed_var, width=4).pack(side=tk.LEFT)

        ttk.Button(self._random_frame, text="Generate", command=self._on_generate_random).pack(
            side=tk.LEFT, padx=(4, 0)
        )

        # Button for Custom Environment
        self._custom_btn = ttk.Button(
            env_bar,
            text="✏ Edit Custom Environment",
            command=self._on_open_custom_editor,
        )

        # ── 3. Verification & Debug Banner ────────────────────────────
        debug_frame = tk.Frame(self.root, background="#e9ecef", highlightthickness=1, highlightbackground=COLOR_BORDER)
        debug_frame.pack(fill=tk.X, padx=12, pady=(2, 6))

        self._lbl_debug_env = tk.Label(
            debug_frame,
            text="Current Environment: — | Grid: — | Start: — | Goal: —",
            font=FONT_SMALL,
            background="#e9ecef",
            foreground="#495057",
            anchor=tk.W,
            padx=8,
            pady=4,
        )
        self._lbl_debug_env.pack(fill=tk.X)

        # ── 4. Main Body: GridWorld (Left) + Dashboard (Right) ─────────
        body = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

        # Left Column: Grid World
        left_frame = ttk.Frame(body)
        body.add(left_frame, weight=5)

        # Placeholder GridView (updated immediately in load_environment)
        dummy_env = GridWorld(width=8, height=8)
        self.grid_view = GridView(left_frame, dummy_env)
        self.grid_view.pack(fill=tk.BOTH, expand=True)

        # Right Column: Dashboard & Details
        right_frame = ttk.Frame(body)
        body.add(right_frame, weight=6)

        # Top of right: KPI cards
        kpi_frame = ttk.LabelFrame(right_frame, text="Live Telemetry", padding=6)
        kpi_frame.pack(fill=tk.X, pady=(0, 6))

        self._kpis: dict[str, ttk.Label] = {}
        kpi_specs = [
            ("Episode", "0", 0, 0),
            ("Step", "0", 0, 1),
            ("Last Reward", "0.00", 0, 2),
            ("Ep Reward", "0.00", 0, 3),
            ("Memory Nodes", "0", 1, 0),
            ("Success Rate", "0.0%", 1, 1),
            ("Avg Comps", "0.0", 1, 2),
            ("Status", "READY", 1, 3),
        ]
        for col in range(4):
            kpi_frame.columnconfigure(col, weight=1)

        for name, initial_val, r, c in kpi_specs:
            box = ttk.Frame(kpi_frame, padding=2)
            box.grid(row=r, column=c, padx=3, pady=2, sticky="nsew")
            ttk.Label(box, text=name, font=FONT_SMALL, foreground=COLOR_TEXT_SECONDARY).pack(anchor=tk.W)
            lbl = ttk.Label(box, text=initial_val, font=FONT_BODY_BOLD)
            lbl.pack(anchor=tk.W)
            self._kpis[name] = lbl

        # Bottom of right: Details Notebook
        self._notebook = ttk.Notebook(right_frame)
        self._notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Recent Episodes Table
        tab_episodes = ttk.Frame(self._notebook)
        self._notebook.add(tab_episodes, text="Recent Episodes")

        ep_cols = ("ep", "steps", "reward", "mem", "outcome")
        self._tree_episodes = ttk.Treeview(tab_episodes, columns=ep_cols, show="headings", height=8)
        self._tree_episodes.heading("ep", text="Ep #")
        self._tree_episodes.heading("steps", text="Steps")
        self._tree_episodes.heading("reward", text="Reward")
        self._tree_episodes.heading("mem", text="Memory")
        self._tree_episodes.heading("outcome", text="Outcome")

        self._tree_episodes.column("ep", width=55, anchor=tk.CENTER)
        self._tree_episodes.column("steps", width=65, anchor=tk.CENTER)
        self._tree_episodes.column("reward", width=85, anchor=tk.CENTER)
        self._tree_episodes.column("mem", width=75, anchor=tk.CENTER)
        self._tree_episodes.column("outcome", width=95, anchor=tk.CENTER)

        self._tree_episodes.tag_configure("success", foreground=COLOR_SUCCESS)
        self._tree_episodes.tag_configure("timeout", foreground=COLOR_DANGER)

        sb_ep = ttk.Scrollbar(tab_episodes, orient=tk.VERTICAL, command=self._tree_episodes.yview)
        self._tree_episodes.configure(yscrollcommand=sb_ep.set)
        self._tree_episodes.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb_ep.pack(side=tk.RIGHT, fill=tk.Y)

        # Tab 2: Memory Nodes Table
        tab_memory = ttk.Frame(self._notebook)
        self._notebook.add(tab_memory, text="Memory Nodes")

        mem_cols = ("id", "action", "q", "visits", "state")
        self._tree_memory = ttk.Treeview(tab_memory, columns=mem_cols, show="headings", height=8)
        self._tree_memory.heading("id", text="Node ID")
        self._tree_memory.heading("action", text="Action")
        self._tree_memory.heading("q", text="Q-Value")
        self._tree_memory.heading("visits", text="Visits")
        self._tree_memory.heading("state", text="State Vector")

        self._tree_memory.column("id", width=65, anchor=tk.CENTER)
        self._tree_memory.column("action", width=75, anchor=tk.CENTER)
        self._tree_memory.column("q", width=80, anchor=tk.CENTER)
        self._tree_memory.column("visits", width=60, anchor=tk.CENTER)
        self._tree_memory.column("state", width=220, anchor=tk.W)

        sb_mem = ttk.Scrollbar(tab_memory, orient=tk.VERTICAL, command=self._tree_memory.yview)
        self._tree_memory.configure(yscrollcommand=sb_mem.set)
        self._tree_memory.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb_mem.pack(side=tk.RIGHT, fill=tk.Y)

        # Tab 3: Algorithm Trace
        tab_trace = ttk.Frame(self._notebook)
        self._notebook.add(tab_trace, text="Algorithm Trace")

        self._txt_trace = tk.Text(
            tab_trace,
            wrap=tk.WORD,
            font=FONT_CODE,
            background=COLOR_BG_CARD,
            foreground=COLOR_TEXT_PRIMARY,
            padx=8,
            pady=8,
            height=8,
        )
        sb_trace = ttk.Scrollbar(tab_trace, orient=tk.VERTICAL, command=self._txt_trace.yview)
        self._txt_trace.configure(yscrollcommand=sb_trace.set)
        self._txt_trace.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb_trace.pack(side=tk.RIGHT, fill=tk.Y)
        self._txt_trace.insert(
            tk.END,
            "WAITING FOR FIRST STEP\n\n"
            "Press 'Step' or 'Start' to observe the non-iterative learning loop.\n"
            "The trace will show:\n"
            "1. Sense State\n"
            "2. Search Memory (O(n) linear scan)\n"
            "3. Distance vs θ (Match Reuse vs Novel State)\n"
            "4. Action Selection\n"
            "5. Reward Feedback\n"
            "6. Sample-Average Q Update: Q <- Q + (r - Q)/n\n",
        )
        self._txt_trace.config(state=tk.DISABLED)

        # ── 5. Bottom Controls Bar ────────────────────────────────────
        ctrl_bar = ttk.Frame(self.root, padding=6)
        ctrl_bar.pack(fill=tk.X, padx=12, pady=(4, 8))

        # Mode Selector
        self._mode_var = tk.StringVar(value=LearningMode.AUTONOMOUS.name)
        mode_box = ttk.Frame(ctrl_bar)
        mode_box.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(mode_box, text="Mode:", font=FONT_BODY_BOLD).pack(side=tk.LEFT, padx=(0, 4))
        for mode in [LearningMode.AUTONOMOUS, LearningMode.STEP_BY_STEP, LearningMode.MANUAL]:
            rb = ttk.Radiobutton(
                mode_box,
                text=mode.name.replace("_", " ").title(),
                value=mode.name,
                variable=self._mode_var,
                command=self._on_mode_radio_changed,
            )
            rb.pack(side=tk.LEFT, padx=3)


        ttk.Separator(ctrl_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        # Action Buttons
        btn_box = ttk.Frame(ctrl_bar)
        btn_box.pack(side=tk.LEFT, padx=(4, 10))

        ttk.Button(btn_box, text="▶ Start", style="Primary.TButton", command=self._on_start_clicked).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Button(btn_box, text="⏸ Pause", command=self._on_pause_clicked).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_box, text="⏭ Step", command=self._on_step_clicked).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_box, text="↺ Reset", command=self._on_reset_clicked).pack(side=tk.LEFT, padx=3)

        ttk.Separator(ctrl_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        # Speed control
        speed_box = ttk.Frame(ctrl_bar)
        speed_box.pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(speed_box, text="Speed:", font=FONT_SMALL).pack(side=tk.LEFT, padx=(0, 2))
        self._speed_var = tk.StringVar(value="12")
        speed_entry = ttk.Entry(speed_box, textvariable=self._speed_var, width=4)
        speed_entry.pack(side=tk.LEFT, padx=2)
        ttk.Label(speed_box, text="Hz", font=FONT_SMALL).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(speed_box, text="Set", command=self._on_speed_set).pack(side=tk.LEFT)

        # Status text on far right
        self._lbl_status = ttk.Label(ctrl_bar, text="● READY", font=FONT_BODY_BOLD, foreground=COLOR_SUCCESS)
        self._lbl_status.pack(side=tk.RIGHT, padx=6)

    # ------------------------------------------------------------------
    # Keyboard Bindings (Manual Mode)
    # ------------------------------------------------------------------

    def _bind_keys(self) -> None:
        self.root.bind("<Up>", lambda e: self._handle_manual_key(Action.UP))
        self.root.bind("<Down>", lambda e: self._handle_manual_key(Action.DOWN))
        self.root.bind("<Left>", lambda e: self._handle_manual_key(Action.LEFT))
        self.root.bind("<Right>", lambda e: self._handle_manual_key(Action.RIGHT))
        self.root.bind("<w>", lambda e: self._handle_manual_key(Action.UP))
        self.root.bind("<s>", lambda e: self._handle_manual_key(Action.DOWN))
        self.root.bind("<a>", lambda e: self._handle_manual_key(Action.LEFT))
        self.root.bind("<d>", lambda e: self._handle_manual_key(Action.RIGHT))

    def _handle_manual_key(self, action: Action) -> None:
        if self.controller is not None and self.controller.mode == LearningMode.MANUAL:
            self.controller.manual_step(action)

    # ------------------------------------------------------------------
    # Button & Interaction Event Handlers
    # ------------------------------------------------------------------

    def _on_start_clicked(self) -> None:
        if self.controller is not None:
            self.controller.start()

    def _on_pause_clicked(self) -> None:
        if self.controller is not None:
            self.controller.pause()

    def _on_step_clicked(self) -> None:
        if self.controller is not None:
            self.controller.step()

    def _on_reset_clicked(self) -> None:
        self.reset_simulation()

    def _on_mode_radio_changed(self) -> None:
        if self.controller is not None:
            mode = LearningMode[self._mode_var.get()]
            self.controller.set_mode(mode)


    def _on_speed_set(self) -> None:
        try:
            val = float(self._speed_var.get().strip())
            val = max(0.5, min(100.0, val))
            self.speed_hz = val
            if self.controller is not None:
                self.controller.set_speed(val)
        except ValueError:
            pass

    def _on_env_combo_selected(self, event=None) -> None:
        selected = self._env_combo.get()
        self._update_env_extra_controls(selected)

    def _update_env_extra_controls(self, name: str) -> None:
        self._random_frame.pack_forget()
        self._custom_btn.pack_forget()

        if name == "Random Obstacles":
            self._random_frame.pack(side=tk.LEFT, padx=6)
        elif name == "Custom Environment":
            self._custom_btn.pack(side=tk.LEFT, padx=6)

    def _on_apply_env_clicked(self) -> None:
        selected = self._env_combo.get()
        if selected == "Random Obstacles":
            self._on_generate_random()
        elif selected == "Custom Environment":
            self._on_open_custom_editor()
        else:
            self.load_environment(selected)

    def _on_generate_random(self) -> None:
        try:
            count = int(self._rand_count_var.get().strip())
            seed_str = self._rand_seed_var.get().strip()
            seed = int(seed_str) if seed_str else None
        except ValueError:
            self._set_status("ERROR: Invalid count or seed.", is_error=True)
            return

        cfg = RandomObstaclesConfig(width=8, height=8, obstacle_count=count, seed=seed)
        ok, msg = self.env_manager.apply_random_config(cfg)
        if not ok:
            self._set_status(f"ERROR: {msg}", is_error=True)
            return

        self._env_combo.set("Random Obstacles")
        self.load_environment("Random Obstacles")

    def _on_open_custom_editor(self) -> None:
        editor_preset = self.env_manager._custom_preset
        CustomEnvEditor(
            self.root,
            initial_preset=editor_preset,
            on_apply=self._on_custom_env_applied,
        )

    def _on_custom_env_applied(self, preset: EnvironmentPreset) -> None:
        ok, msg = self.env_manager.set_custom_preset(preset)
        if not ok:
            self._set_status(f"ERROR: {msg}", is_error=True)
            return
        self._env_combo.set("Custom Environment")
        self.load_environment(preset)

    def _open_settings(self) -> None:
        SettingsDialog(
            parent=self.root,
            current_theta=self.agent.theta if self.agent else self.config.theta,
            current_n_max=self.engine.max_memory_size if self.engine else self.config.n_max,
            current_speed=self.speed_hz,
            current_max_steps=self.env.max_steps if self.env else self.config.max_steps_per_episode,
            current_seed=self.config.random_seed,
            on_apply=self._on_settings_applied,
        )

    def _on_settings_applied(
        self,
        theta: float,
        n_max: int,
        speed: float,
        max_steps: int,
        seed: Optional[int],
    ) -> None:
        self.speed_hz = speed
        self._speed_var.set(str(int(speed)))
        if self.controller is not None:
            self.controller.update_parameters(theta=theta, n_max=n_max)
            self.controller.set_speed(speed)
        if self.env is not None:
            self.env.max_steps = max_steps
        self.config.random_seed = seed
        self._set_status("Settings applied.")

    # ------------------------------------------------------------------
    # Queue Polling and Event Handling
    # ------------------------------------------------------------------

    def _poll_queue(self) -> None:
        if self.controller is None:
            self.root.after(20, self._poll_queue)
            return

        events_processed = 0
        while events_processed < 50:
            try:
                msg_type, payload = self.controller.queue.get_nowait()
                events_processed += 1
            except queue.Empty:
                break

            if msg_type == "step":
                self._handle_step_event(payload)
            elif msg_type == "episode_complete":
                self._handle_episode_complete(payload)
            elif msg_type == "status":
                self._set_status(payload)
            elif msg_type == "mode_changed":
                self._mode_var.set(payload.name)

            elif msg_type == "reset":
                if self.grid_view and self.env:
                    self.grid_view.update_agent_position(self.env.agent_pos)
            elif msg_type == "error":
                self._set_status(f"ERROR: {payload}", is_error=True)

        self.root.after(20, self._poll_queue)

    def _handle_step_event(self, event: StepEvent) -> None:
        # 1. Update GridView
        if self.grid_view is not None:
            self.grid_view.update_agent_position(event.next_agent_pos, action=event.action)

        # 2. Update KPI Tiles
        if self.engine is not None and self.memory is not None:
            self._kpis["Episode"].config(text=str(event.episode))
            self._kpis["Step"].config(text=str(self.engine.episode_step_count))
            self._kpis["Last Reward"].config(text=f"{event.reward:+.2f}")
            self._kpis["Ep Reward"].config(text=f"{self.engine.episode_reward:+.2f}")
            self._kpis["Memory Nodes"].config(text=str(event.memory_size))
            rate = (self.total_successes / max(1, self.total_episodes)) * 100 if self.total_episodes > 0 else 0.0
            self._kpis["Success Rate"].config(text=f"{rate:.1f}%")
            self._kpis["Avg Comps"].config(text=f"{self.memory.average_comparisons:.1f}")

        # 3. Update Algorithm Trace
        self._update_algorithm_trace(event)

        # 4. Refresh Memory Nodes Table
        self._refresh_memory_table()

    def _handle_episode_complete(self, metrics: EpisodeMetrics) -> None:
        self.total_episodes += 1
        if metrics.success:
            self.total_successes += 1

        # Insert at top of Recent Episodes Treeview
        tag = "success" if metrics.success else "timeout"
        self._tree_episodes.insert(
            "",
            0,
            values=(
                metrics.episode_number,
                metrics.steps,
                f"{metrics.total_reward:+.2f}",
                metrics.memory_size,
                "SUCCESS" if metrics.success else "TIMEOUT",
            ),
            tags=(tag,),
        )

        rate = (self.total_successes / max(1, self.total_episodes)) * 100
        self._kpis["Success Rate"].config(text=f"{rate:.1f}%")

    def _update_algorithm_trace(self, event: StepEvent) -> None:
        text = (
            f"=== STEP {event.timestep}  (Episode {event.episode}, Local Step {self.engine.episode_step_count if self.engine else '?'}) ===\n\n"
            f"1. SENSE STATE VECTOR (6D):\n"
            f"   dx_goal={event.encoded_state[0]:+.3f}, dy_goal={event.encoded_state[1]:+.3f}\n"
            f"   Walls: Up={event.encoded_state[2]:.0f}, Down={event.encoded_state[3]:.0f}, "
            f"Left={event.encoded_state[4]:.0f}, Right={event.encoded_state[5]:.0f}\n\n"
            f"2. SEARCH MEMORY (O(n) linear scan through singly linked list):\n"
            f"   Searched nodes: {len(event.search_trace)} comparisons\n"
            f"   Nearest stored distance: {event.nearest_distance:.4f}\n\n"
            f"3. DECISION CRITERION:\n"
            f"   Distance ({event.nearest_distance:.4f}) {'<=' if event.is_match else '>'} θ ({event.theta:.2f})\n"
            f"   -> {'MATCH REUSE: Reused Node #' + str(event.matched_node_id) if event.is_match else 'NOVEL STATE: Prepended fresh node at head'}\n\n"
            f"4. ACTION SELECTION:\n"
            f"   Action: {event.action.name} (move agent to {event.next_agent_pos})\n\n"
            f"5. ENVIRONMENT FEEDBACK:\n"
            f"   Reward: {event.reward:+.2f}  |  Collided: {event.collided}  |  Goal Reached: {event.goal_reached}\n\n"
            f"6. SAMPLE-AVERAGE Q UPDATE (no Bellman, no discount, no epochs):\n"
            f"   Q_new = Q_old + (reward - Q_old) / n\n"
            f"   Q_new = {event.q_before:+.4f} + ({event.reward:+.2f} - ({event.q_before:+.4f})) / {event.visit_count}\n"
            f"   Q_new = {event.q_after:+.4f}  (visit count = {event.visit_count})\n"
        )
        self._txt_trace.config(state=tk.NORMAL)
        self._txt_trace.delete("1.0", tk.END)
        self._txt_trace.insert(tk.END, text)
        self._txt_trace.config(state=tk.DISABLED)

    def _refresh_memory_table(self) -> None:
        if self.memory is None:
            return

        self._tree_memory.delete(*self._tree_memory.get_children())
        curr = self.memory.head
        count = 0
        while curr is not None and count < 100:
            st = curr.state
            st_str = f"[{st[0]:+.2f}, {st[1]:+.2f}, {int(st[2])},{int(st[3])},{int(st[4])},{int(st[5])}]"
            self._tree_memory.insert(
                "",
                tk.END,
                values=(
                    f"#{curr.node_id:03d}",
                    curr.action.name,
                    f"{curr.q_value:+.3f}",
                    curr.visit_count,
                    st_str,
                ),
            )
            curr = curr.next
            count += 1

    def _clear_gui_stats_and_tables(self) -> None:
        for name in ["Episode", "Step", "Last Reward", "Ep Reward", "Memory Nodes"]:
            self._kpis[name].config(text="0" if name in ["Episode", "Step", "Memory Nodes"] else "0.00")
        self._kpis["Success Rate"].config(text="0.0%")
        self._kpis["Avg Comps"].config(text="0.0")

        self._tree_episodes.delete(*self._tree_episodes.get_children())
        self._tree_memory.delete(*self._tree_memory.get_children())

        self._txt_trace.config(state=tk.NORMAL)
        self._txt_trace.delete("1.0", tk.END)
        self._txt_trace.insert(tk.END, "WAITING FOR FIRST STEP\n\nSimulation is ready.")
        self._txt_trace.config(state=tk.DISABLED)

    def _update_debug_banner(self) -> None:
        """Update verification line directly from the live self.env object."""
        if self.env is None:
            return
        self._lbl_debug_env.config(
            text=(
                f"Current Environment: {self.active_preset.name}   |   "
                f"Grid: {self.env.width} × {self.env.height}   |   "
                f"Start: {self.env.start_pos}   |   "
                f"Goal: {self.env.goal_pos}   |   "
                f"Obstacles: {len(self.env.obstacles)}"
            )
        )

    def _set_status(self, text: str, is_error: bool = False) -> None:
        color = COLOR_DANGER if is_error else (
            COLOR_SUCCESS if "RUNNING" in text or "READY" in text else (
                COLOR_WARNING if "PAUSED" in text or "STEPPED" in text else COLOR_PRIMARY
            )
        )
        self._lbl_status.config(text=f"● {text}", foreground=color)
        self._kpis["Status"].config(text=text.split("—")[0].strip())

    def _on_close(self) -> None:
        if self.controller is not None:
            self.controller.stop_worker()
        self.root.destroy()


def run_gui() -> None:
    """Entry point to start the LINKMEM application."""
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()
