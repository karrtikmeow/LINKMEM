"""CustomEnvEditor: Visual editor for placing obstacles and setting start/goal."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional, Set, Tuple

from linkmem.gui.environments import EnvironmentPreset, validate_preset
from linkmem.gui.theme import (
    COLOR_BG_CARD,
    COLOR_BG_MAIN,
    COLOR_AGENT,
    COLOR_DANGER,
    COLOR_GOAL,
    COLOR_GOAL_BORDER,
    COLOR_GRID_BG,
    COLOR_GRID_LINE,
    COLOR_OBSTACLE,
    COLOR_PRIMARY,
    COLOR_START,
    COLOR_START_BORDER,
    COLOR_SUCCESS,
    FONT_BODY_BOLD,
    FONT_SMALL,
    FONT_SUBTITLE,
)

_CELL = 44          # Canvas cell size in pixels
_MARGIN = 12        # Canvas border margin


class CustomEnvEditor(tk.Toplevel):
    """Modal grid editor: click cells to toggle obstacles, S/G to set start/goal.

    Args:
        parent: Parent Tk widget.
        initial_preset: Starting preset to load into the editor.
        on_apply: Callback(EnvironmentPreset) invoked on Apply.
    """

    def __init__(
        self,
        parent: tk.Widget,
        initial_preset: EnvironmentPreset,
        on_apply: Callable[[EnvironmentPreset], None],
    ):
        super().__init__(parent)
        self.title("Custom Environment Editor")
        self.resizable(False, False)
        self.configure(background=COLOR_BG_MAIN)
        self.grab_set()

        self._on_apply = on_apply

        # Working copy of the preset
        self._width = initial_preset.width
        self._height = initial_preset.height
        self._start: Tuple[int, int] = initial_preset.start_pos
        self._goal: Tuple[int, int] = initial_preset.goal_pos
        self._obstacles: Set[Tuple[int, int]] = set(initial_preset.obstacles)

        # Editing mode: "obstacle" | "start" | "goal"
        self._edit_mode = tk.StringVar(value="obstacle")

        self._build_ui()
        self._draw()

        # Center over parent
        self.transient(parent)
        self.update_idletasks()
        px = parent.winfo_rootx() + (parent.winfo_width() - self.winfo_width()) // 2
        py = parent.winfo_rooty() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{px}+{py}")

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        ttk.Label(self, text="Custom Environment Editor", font=FONT_SUBTITLE).pack(
            padx=16, pady=(12, 4), anchor=tk.W
        )

        # Tool selector
        tool_frame = ttk.LabelFrame(self, text="Edit Tool", padding=6)
        tool_frame.pack(padx=12, pady=4, fill=tk.X)

        for mode, label, color in [
            ("obstacle", "Toggle Obstacle", COLOR_OBSTACLE),
            ("start",    "Set Start (S)",   "#0f5132"),
            ("goal",     "Set Goal (G)",    "#664d03"),
        ]:
            rb = ttk.Radiobutton(
                tool_frame,
                text=label,
                value=mode,
                variable=self._edit_mode,
            )
            rb.pack(side=tk.LEFT, padx=8)

        ttk.Label(
            tool_frame,
            text="Click grid cells to edit",
            font=FONT_SMALL,
            foreground="#6c757d",
        ).pack(side=tk.RIGHT, padx=8)

        # Canvas
        canvas_w = self._width * _CELL + 2 * _MARGIN
        canvas_h = self._height * _CELL + 2 * _MARGIN

        self._canvas = tk.Canvas(
            self,
            width=canvas_w,
            height=canvas_h,
            background=COLOR_BG_CARD,
            highlightthickness=1,
            highlightbackground="#ced4da",
        )
        self._canvas.pack(padx=12, pady=8)
        self._canvas.bind("<Button-1>", self._on_click)
        self._canvas.bind("<B1-Motion>", self._on_drag)

        # Feedback
        self._lbl_feedback = ttk.Label(
            self, text="", font=FONT_SMALL, foreground=COLOR_DANGER
        )
        self._lbl_feedback.pack(padx=16)

        # Clear / Apply buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(padx=12, pady=(4, 14), fill=tk.X)

        ttk.Button(btn_frame, text="Clear All Obstacles", command=self._clear_obstacles).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(
            side=tk.RIGHT, padx=4
        )
        ttk.Button(
            btn_frame,
            text="Apply Environment",
            style="Primary.TButton",
            command=self._handle_apply,
        ).pack(side=tk.RIGHT, padx=4)

    # ------------------------------------------------------------------
    # Canvas Rendering
    # ------------------------------------------------------------------

    def _draw(self) -> None:
        self._canvas.delete("all")

        for x in range(self._width):
            for y in range(self._height):
                x1 = _MARGIN + x * _CELL
                y1 = _MARGIN + y * _CELL
                x2 = x1 + _CELL
                y2 = y1 + _CELL
                pos = (x, y)

                if pos in self._obstacles:
                    fill = COLOR_OBSTACLE
                    text = ""
                    text_color = "#ffffff"
                elif pos == self._start:
                    fill = COLOR_START
                    text = "S"
                    text_color = "#0f5132"
                elif pos == self._goal:
                    fill = COLOR_GOAL
                    text = "G"
                    text_color = "#664d03"
                else:
                    fill = COLOR_GRID_BG
                    text = ""
                    text_color = ""

                self._canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=fill, outline=COLOR_GRID_LINE, width=1,
                )
                if text:
                    self._canvas.create_text(
                        (x1 + x2) / 2, (y1 + y2) / 2,
                        text=text,
                        font=FONT_BODY_BOLD,
                        fill=text_color,
                    )

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def _cell_at(self, event_x: int, event_y: int) -> Optional[Tuple[int, int]]:
        x = (event_x - _MARGIN) // _CELL
        y = (event_y - _MARGIN) // _CELL
        if 0 <= x < self._width and 0 <= y < self._height:
            return (x, y)
        return None

    def _on_click(self, event) -> None:
        cell = self._cell_at(event.x, event.y)
        if cell is None:
            return
        self._apply_tool(cell)
        self._draw()

    def _on_drag(self, event) -> None:
        if self._edit_mode.get() != "obstacle":
            return
        cell = self._cell_at(event.x, event.y)
        if cell is not None:
            self._apply_tool(cell)
            self._draw()

    def _apply_tool(self, pos: Tuple[int, int]) -> None:
        mode = self._edit_mode.get()
        if mode == "obstacle":
            if pos == self._start or pos == self._goal:
                return  # cannot block start/goal
            if pos in self._obstacles:
                self._obstacles.discard(pos)
            else:
                self._obstacles.add(pos)
        elif mode == "start":
            if pos in self._obstacles or pos == self._goal:
                return
            self._start = pos
        elif mode == "goal":
            if pos in self._obstacles or pos == self._start:
                return
            self._goal = pos

    def _clear_obstacles(self) -> None:
        self._obstacles.clear()
        self._lbl_feedback.config(text="")
        self._draw()

    def _handle_apply(self) -> None:
        preset = EnvironmentPreset(
            name="Custom Environment",
            description="User-defined custom environment.",
            width=self._width,
            height=self._height,
            start_pos=self._start,
            goal_pos=self._goal,
            obstacles=set(self._obstacles),
        )
        ok, msg = validate_preset(preset)
        if not ok:
            self._lbl_feedback.config(text=f"⚠ {msg}", foreground=COLOR_DANGER)
            return

        self._lbl_feedback.config(text="✔ Valid — applying.", foreground=COLOR_SUCCESS)
        self._on_apply(preset)
        self.after(300, self.destroy)
