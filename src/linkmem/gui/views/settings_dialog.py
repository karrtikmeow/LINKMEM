"""SettingsDialog: Modal dialog for editing LINKMEM simulation parameters."""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional, Tuple

from linkmem.gui.theme import (
    COLOR_BG_MAIN,
    COLOR_DANGER,
    COLOR_PRIMARY,
    COLOR_SUCCESS,
    COLOR_TEXT_SECONDARY,
    FONT_BODY,
    FONT_BODY_BOLD,
    FONT_SMALL,
    FONT_SUBTITLE,
)


class SettingsDialog(tk.Toplevel):
    """A modal settings dialog for theta, N_max, speed, max steps, and seed.

    Args:
        parent: The parent Tk window.
        current_theta: Current theta threshold value.
        current_n_max: Current maximum memory size.
        current_speed: Current simulation speed in steps/second.
        current_max_steps: Current max steps per episode.
        current_seed: Current random seed.
        on_apply: Callback(theta, n_max, speed, max_steps, seed) invoked on Apply.
    """

    def __init__(
        self,
        parent: tk.Widget,
        current_theta: float,
        current_n_max: int,
        current_speed: float,
        current_max_steps: int,
        current_seed: Optional[int],
        on_apply: Callable[[float, int, float, int, Optional[int]], None],
    ):
        super().__init__(parent)
        self.title("Simulation Settings")
        self.resizable(False, False)
        self.configure(background=COLOR_BG_MAIN)
        self.grab_set()  # Modal

        self._on_apply = on_apply
        self._result_ok = False

        self._build_ui(
            current_theta,
            current_n_max,
            current_speed,
            current_max_steps,
            current_seed,
        )

        # Center over parent
        self.transient(parent)
        self.update_idletasks()
        px = parent.winfo_rootx() + (parent.winfo_width() - self.winfo_width()) // 2
        py = parent.winfo_rooty() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{px}+{py}")

    def _build_ui(
        self,
        theta: float,
        n_max: int,
        speed: float,
        max_steps: int,
        seed: Optional[int],
    ) -> None:
        pad = dict(padx=12, pady=6)

        title_lbl = ttk.Label(self, text="Simulation Settings", font=FONT_SUBTITLE)
        title_lbl.pack(padx=16, pady=(14, 4), anchor=tk.W)

        separator = ttk.Separator(self, orient=tk.HORIZONTAL)
        separator.pack(fill=tk.X, padx=12, pady=4)

        form = ttk.Frame(self)
        form.pack(padx=16, pady=4, fill=tk.X)

        def _row(label: str, row: int) -> ttk.Entry:
            ttk.Label(form, text=label, font=FONT_BODY_BOLD, width=24, anchor=tk.W).grid(
                row=row, column=0, sticky=tk.W, padx=4, pady=5
            )
            e = ttk.Entry(form, width=14)
            e.grid(row=row, column=1, sticky=tk.W, padx=4, pady=5)
            return e

        self._theta_entry = _row("Theta (θ)  [0.0 – 5.0]", 0)
        self._theta_entry.insert(0, f"{theta:.4f}")

        self._nmax_entry = _row("Max Memory (N_max)  [2 – 5000]", 1)
        self._nmax_entry.insert(0, str(n_max))

        self._speed_entry = _row("Simulation Speed (Hz)  [0.5 – 100]", 2)
        self._speed_entry.insert(0, f"{speed:.1f}")

        self._steps_entry = _row("Max Steps / Episode  [10 – 5000]", 3)
        self._steps_entry.insert(0, str(max_steps))

        self._seed_entry = _row("Random Seed  (blank = none)", 4)
        self._seed_entry.insert(0, str(seed) if seed is not None else "")

        # Feedback
        self._lbl_feedback = ttk.Label(
            self, text="", font=FONT_SMALL, foreground=COLOR_DANGER
        )
        self._lbl_feedback.pack(padx=16, pady=(0, 2))

        sep2 = ttk.Separator(self, orient=tk.HORIZONTAL)
        sep2.pack(fill=tk.X, padx=12, pady=4)

        btn_row = ttk.Frame(self)
        btn_row.pack(padx=16, pady=(4, 14), anchor=tk.E)

        ttk.Button(btn_row, text="Cancel", command=self.destroy).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(
            btn_row,
            text="Apply",
            style="Primary.TButton",
            command=self._handle_apply,
        ).pack(side=tk.LEFT, padx=4)

    def _handle_apply(self) -> None:
        """Validate inputs and invoke callback."""
        errors = []

        try:
            theta = float(self._theta_entry.get().strip())
            if not (0.0 <= theta <= 5.0):
                errors.append("Theta must be in [0.0, 5.0].")
        except ValueError:
            errors.append("Theta must be a decimal number.")
            theta = 0.35  # placeholder

        try:
            n_max = int(self._nmax_entry.get().strip())
            if not (2 <= n_max <= 5000):
                errors.append("N_max must be in [2, 5000].")
        except ValueError:
            errors.append("N_max must be an integer.")
            n_max = 100

        try:
            speed = float(self._speed_entry.get().strip())
            if not (0.5 <= speed <= 100.0):
                errors.append("Speed must be in [0.5, 100].")
        except ValueError:
            errors.append("Speed must be a decimal number.")
            speed = 12.0

        try:
            max_steps = int(self._steps_entry.get().strip())
            if not (10 <= max_steps <= 5000):
                errors.append("Max steps must be in [10, 5000].")
        except ValueError:
            errors.append("Max steps must be an integer.")
            max_steps = 100

        seed_str = self._seed_entry.get().strip()
        if seed_str == "":
            seed = None
        else:
            try:
                seed = int(seed_str)
            except ValueError:
                errors.append("Seed must be an integer or left blank.")
                seed = None

        if errors:
            self._lbl_feedback.config(text=" | ".join(errors), foreground=COLOR_DANGER)
            return

        self._lbl_feedback.config(
            text="✔ Settings applied.", foreground=COLOR_SUCCESS
        )
        self._on_apply(theta, n_max, speed, max_steps, seed)
        self.after(400, self.destroy)
