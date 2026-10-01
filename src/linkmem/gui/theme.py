"""Visual theme and styling constants for LINKMEM academic GUI."""

import tkinter as tk
from tkinter import ttk

# Color Palette (Clean Academic Theme)
COLOR_BG_MAIN = "#f4f6f8"
COLOR_BG_CARD = "#ffffff"
COLOR_BORDER = "#dcdfe3"
COLOR_TEXT_PRIMARY = "#212529"
COLOR_TEXT_SECONDARY = "#6c757d"
COLOR_TEXT_MUTED = "#adb5bd"

COLOR_PRIMARY = "#0d6efd"
COLOR_SUCCESS = "#198754"
COLOR_DANGER = "#dc3545"
COLOR_WARNING = "#ffc107"
COLOR_INFO = "#0dcaf0"

# Grid World Colors
COLOR_GRID_BG = "#ffffff"
COLOR_GRID_LINE = "#e9ecef"
COLOR_OBSTACLE = "#343a40"
COLOR_START = "#d1e7dd"
COLOR_START_BORDER = "#a3cfbb"
COLOR_GOAL = "#fff3cd"
COLOR_GOAL_BORDER = "#ffe69c"
COLOR_AGENT = "#0d6efd"
COLOR_AGENT_PUPIL = "#ffffff"

# Memory Node Colors
COLOR_NODE_BG = "#ffffff"
COLOR_NODE_BORDER = "#ced4da"
COLOR_NODE_HEADER = "#e9ecef"
COLOR_NODE_SELECTED = "#cfe2ff"
COLOR_NODE_SELECTED_BORDER = "#0d6efd"
COLOR_ARROW = "#495057"

# Typography
FONT_FAMILY = "DejaVu Sans"
FONT_MONO = "DejaVu Sans Mono"

FONT_TITLE = (FONT_FAMILY, 14, "bold")
FONT_SUBTITLE = (FONT_FAMILY, 11, "bold")
FONT_BODY = (FONT_FAMILY, 9)
FONT_BODY_BOLD = (FONT_FAMILY, 9, "bold")
FONT_SMALL = (FONT_FAMILY, 8)
FONT_CODE = (FONT_MONO, 8)
FONT_CODE_BOLD = (FONT_MONO, 8, "bold")


def apply_theme(root: tk.Tk) -> None:
    """Configure modern ttk styles for the application."""
    style = ttk.Style(root)
    
    # Try clam theme as baseline if available
    available_themes = style.theme_names()
    if "clam" in available_themes:
        style.theme_use("clam")

    # Frame styles
    style.configure("TFrame", background=COLOR_BG_MAIN)
    style.configure("Card.TFrame", background=COLOR_BG_CARD, relief="solid", borderwidth=1)
    
    # Label styles
    style.configure("TLabel", background=COLOR_BG_MAIN, foreground=COLOR_TEXT_PRIMARY, font=FONT_BODY)
    style.configure("Card.TLabel", background=COLOR_BG_CARD, foreground=COLOR_TEXT_PRIMARY, font=FONT_BODY)
    style.configure("Title.TLabel", font=FONT_TITLE, foreground=COLOR_TEXT_PRIMARY)
    style.configure("Subtitle.TLabel", font=FONT_SUBTITLE, foreground=COLOR_TEXT_PRIMARY)
    style.configure("Muted.TLabel", foreground=COLOR_TEXT_SECONDARY, font=FONT_SMALL)
    style.configure("MetricVal.TLabel", font=(FONT_FAMILY, 12, "bold"), foreground=COLOR_PRIMARY)

    # Button styles
    style.configure("TButton", font=FONT_BODY_BOLD, padding=5)
    style.configure("Primary.TButton", background=COLOR_PRIMARY, foreground="#ffffff", font=FONT_BODY_BOLD)
    style.map("Primary.TButton", background=[("active", "#0b5ed7")])

    # Radio and Checkbutton
    style.configure("TRadiobutton", background=COLOR_BG_MAIN, font=FONT_BODY)
    
    # LabelFrame styles
    style.configure("TLabelframe", background=COLOR_BG_MAIN, foreground=COLOR_TEXT_PRIMARY)
    style.configure("TLabelframe.Label", background=COLOR_BG_MAIN, foreground=COLOR_TEXT_PRIMARY, font=FONT_SUBTITLE)
