"""GridView: Clean 2D GridWorld canvas visualizer for LINKMEM."""

import tkinter as tk
from tkinter import ttk
from typing import Optional, Tuple
from linkmem.core.types import Action
from linkmem.core.environment import GridWorld
from linkmem.gui.theme import (
    COLOR_BG_CARD,
    COLOR_GRID_BG,
    COLOR_GRID_LINE,
    COLOR_OBSTACLE,
    COLOR_START,
    COLOR_START_BORDER,
    COLOR_GOAL,
    COLOR_GOAL_BORDER,
    COLOR_AGENT,
    COLOR_AGENT_PUPIL,
    FONT_BODY_BOLD,
)


class GridView(ttk.Frame):
    """Renders the 2D GridWorld grid, obstacles, start/goal markers, and agent on Canvas."""

    def __init__(self, parent: tk.Widget, env: GridWorld, **kwargs):
        super().__init__(parent, **kwargs)
        self.env = env
        self.last_action: Optional[Action] = None

        # Container Header
        header = ttk.Frame(self)
        header.pack(fill=tk.X, padx=6, pady=(2, 4))
        self.lbl_title = ttk.Label(header, text="Grid World", style="Subtitle.TLabel")
        self.lbl_title.pack(side=tk.LEFT)
        self.lbl_info = ttk.Label(header, text=f"{env.width}×{env.height}", style="Muted.TLabel")
        self.lbl_info.pack(side=tk.RIGHT)

        # Drawing Canvas
        self.canvas = tk.Canvas(
            self,
            background=COLOR_BG_CARD,
            highlightthickness=1,
            highlightbackground="#ced4da",
            width=380,
            height=380,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.canvas.bind("<Configure>", lambda e: self.redraw())

    def set_environment(self, env: GridWorld) -> None:
        """Update reference to GridWorld and refresh view."""
        self.env = env
        self.last_action = None
        self.lbl_info.config(text=f"{env.width}×{env.height}")
        self.redraw()

    def update_agent_position(self, pos: Tuple[int, int], action: Optional[Action] = None) -> None:
        """Update agent's position and optional heading direction."""
        self.last_action = action
        self.redraw()

    def redraw(self) -> None:
        """Redraw the entire grid, obstacles, landmarks, and agent."""
        self.canvas.delete("all")

        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()

        if width < 20 or height < 20:
            return

        margin = 12
        grid_w = self.env.width
        grid_h = self.env.height

        avail_w = width - 2 * margin
        avail_h = height - 2 * margin

        cell_size = max(10, min(avail_w // grid_w, avail_h // grid_h))

        offset_x = margin + (avail_w - (cell_size * grid_w)) // 2
        offset_y = margin + (avail_h - (cell_size * grid_h)) // 2

        # Draw Cells
        for x in range(grid_w):
            for y in range(grid_h):
                x1 = offset_x + x * cell_size
                y1 = offset_y + y * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size

                pos = (x, y)
                if pos in self.env.obstacles:
                    # Obstacle
                    self.canvas.create_rectangle(
                        x1, y1, x2, y2,
                        fill=COLOR_OBSTACLE,
                        outline="#212529",
                        width=1,
                    )
                    mid_y = (y1 + y2) / 2
                    self.canvas.create_line(x1, mid_y, x2, mid_y, fill="#495057", width=1)
                elif pos == self.env.start_pos:
                    # Start Cell
                    self.canvas.create_rectangle(
                        x1, y1, x2, y2,
                        fill=COLOR_START,
                        outline=COLOR_START_BORDER,
                        width=1,
                    )
                    self.canvas.create_text(
                        (x1 + x2) / 2, (y1 + y2) / 2,
                        text="S",
                        font=FONT_BODY_BOLD,
                        fill="#0f5132",
                    )
                elif pos == self.env.goal_pos:
                    # Goal Cell
                    self.canvas.create_rectangle(
                        x1, y1, x2, y2,
                        fill=COLOR_GOAL,
                        outline=COLOR_GOAL_BORDER,
                        width=1,
                    )
                    self.canvas.create_text(
                        (x1 + x2) / 2, (y1 + y2) / 2,
                        text="★ G",
                        font=FONT_BODY_BOLD,
                        fill="#664d03",
                    )
                else:
                    # Empty Cell
                    self.canvas.create_rectangle(
                        x1, y1, x2, y2,
                        fill=COLOR_GRID_BG,
                        outline=COLOR_GRID_LINE,
                        width=1,
                    )

        # Draw Agent
        ax, ay = self.env.agent_pos
        ax1 = offset_x + ax * cell_size + cell_size * 0.15
        ay1 = offset_y + ay * cell_size + cell_size * 0.15
        ax2 = offset_x + (ax + 1) * cell_size - cell_size * 0.15
        ay2 = offset_y + (ay + 1) * cell_size - cell_size * 0.15

        # Agent Body
        self.canvas.create_oval(
            ax1, ay1, ax2, ay2,
            fill=COLOR_AGENT,
            outline="#0a58ca",
            width=2,
        )

        # Heading eye / direction pointer
        cx = (ax1 + ax2) / 2
        cy = (ay1 + ay2) / 2
        r = (ax2 - ax1) / 2

        dx, dy = 0.0, 0.0
        if self.last_action == Action.UP:
            dy = -r * 0.45
        elif self.last_action == Action.DOWN:
            dy = r * 0.45
        elif self.last_action == Action.LEFT:
            dx = -r * 0.45
        elif self.last_action == Action.RIGHT:
            dx = r * 0.45

        eye_r = max(2.0, r * 0.25)
        self.canvas.create_oval(
            cx + dx - eye_r, cy + dy - eye_r,
            cx + dx + eye_r, cy + dy + eye_r,
            fill=COLOR_AGENT_PUPIL,
            outline="#0a58ca",
            width=1,
        )
