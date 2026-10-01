"""EnvironmentPreset: Predefined and user-configurable GridWorld environments for LINKMEM."""

from __future__ import annotations

import collections
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EnvironmentPreset:
    """Complete description of a GridWorld configuration."""

    name: str
    description: str

    width: int
    height: int
    start_pos: Tuple[int, int]
    goal_pos: Tuple[int, int]
    obstacles: Set[Tuple[int, int]] = field(default_factory=set)

    # Only used for the "Random Obstacles" preset
    obstacle_density: float = 0.0
    random_seed: Optional[int] = None

    def is_custom(self) -> bool:
        return self.name == "Custom Environment"

    def is_random(self) -> bool:
        return self.name == "Random Obstacles"


@dataclass
class RandomObstaclesConfig:
    """Configuration knobs for the random-obstacle environment."""
    width: int = 8
    height: int = 8
    obstacle_count: int = 6
    seed: Optional[int] = 0

    def obstacle_density(self) -> float:
        total = self.width * self.height - 2  # exclude start/goal
        return self.obstacle_count / max(1, total)


# ---------------------------------------------------------------------------
# Built-in preset definitions
# ---------------------------------------------------------------------------

def _empty_grid() -> EnvironmentPreset:
    return EnvironmentPreset(
        name="Empty Grid",
        description="8×8 open grid with no obstacles. The agent explores freely.",
        width=8, height=8,
        start_pos=(0, 0),
        goal_pos=(7, 7),
        obstacles=set(),
    )


def _simple_maze() -> EnvironmentPreset:
    """A small deterministic maze with a clear solution path."""
    obstacles: Set[Tuple[int, int]] = {
        (2, 0), (2, 1), (2, 2), (2, 3),
        (4, 4), (4, 5), (4, 6),
        (1, 5), (1, 6), (1, 7),
    }
    return EnvironmentPreset(
        name="Simple Maze",
        description="Small deterministic maze. Clear single-corridor solution path.",
        width=8, height=8,
        start_pos=(0, 0),
        goal_pos=(7, 7),
        obstacles=obstacles,
    )


def _complex_maze() -> EnvironmentPreset:
    """A larger, more challenging deterministic maze (verified reachable)."""
    obstacles: Set[Tuple[int, int]] = {
        # Horizontal barrier row=2, cols 1-4
        (1, 2), (2, 2), (3, 2), (4, 2),
        # Vertical barrier col=5, rows 0-3
        (5, 0), (5, 1), (5, 2), (5, 3),
        # Horizontal barrier row=5, cols 2-6
        (2, 5), (3, 5), (4, 5), (5, 5), (6, 5),
        # Vertical barrier col=1, rows 4-6
        (1, 4), (1, 5), (1, 6),
        # Interior block
        (3, 3),
    }
    return EnvironmentPreset(
        name="Complex Maze",
        description="Larger challenging maze requiring multi-corridor navigation.",
        width=8, height=8,
        start_pos=(0, 0),
        goal_pos=(7, 7),
        obstacles=obstacles,
    )



def _random_obstacles(cfg: Optional[RandomObstaclesConfig] = None) -> EnvironmentPreset:
    cfg = cfg or RandomObstaclesConfig()
    rng = random.Random(cfg.seed)
    start = (0, 0)
    goal = (cfg.width - 1, cfg.height - 1)

    candidates = [
        (x, y)
        for x in range(cfg.width)
        for y in range(cfg.height)
        if (x, y) != start and (x, y) != goal
    ]
    count = min(cfg.obstacle_count, len(candidates))
    obstacles: Set[Tuple[int, int]] = set(rng.sample(candidates, count))

    return EnvironmentPreset(
        name="Random Obstacles",
        description=f"{cfg.width}×{cfg.height} grid, {count} randomly placed obstacles (seed={cfg.seed}).",
        width=cfg.width, height=cfg.height,
        start_pos=start,
        goal_pos=goal,
        obstacles=obstacles,
        random_seed=cfg.seed,
    )


def _custom_environment() -> EnvironmentPreset:
    return EnvironmentPreset(
        name="Custom Environment",
        description="User-defined environment. Use the visual editor to place obstacles.",
        width=8, height=8,
        start_pos=(0, 0),
        goal_pos=(7, 7),
        obstacles=set(),
    )


# ---------------------------------------------------------------------------
# Preset registry
# ---------------------------------------------------------------------------

PRESET_NAMES: List[str] = [
    "Empty Grid",
    "Simple Maze",
    "Complex Maze",
    "Random Obstacles",
    "Custom Environment",
]


def build_preset(name: str, random_cfg: Optional[RandomObstaclesConfig] = None) -> EnvironmentPreset:
    """Return a fresh EnvironmentPreset by name."""
    if name == "Empty Grid":
        return _empty_grid()
    elif name == "Simple Maze":
        return _simple_maze()
    elif name == "Complex Maze":
        return _complex_maze()
    elif name == "Random Obstacles":
        return _random_obstacles(random_cfg)
    elif name == "Custom Environment":
        return _custom_environment()
    else:
        raise ValueError(f"Unknown preset: {name!r}")


# ---------------------------------------------------------------------------
# Path reachability check (BFS)
# ---------------------------------------------------------------------------

def is_path_reachable(
    width: int,
    height: int,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    obstacles: Set[Tuple[int, int]],
) -> bool:
    """Return True if a path exists from start to goal using BFS.

    Uses the same 4-directional movement as GridWorld (up/down/left/right).
    """
    if start == goal:
        return True
    if start in obstacles or goal in obstacles:
        return False

    visited: Set[Tuple[int, int]] = set()
    queue: collections.deque[Tuple[int, int]] = collections.deque([start])
    visited.add(start)

    while queue:
        x, y = queue.popleft()
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) == goal:
                return True
            if (
                0 <= nx < width
                and 0 <= ny < height
                and (nx, ny) not in obstacles
                and (nx, ny) not in visited
            ):
                visited.add((nx, ny))
                queue.append((nx, ny))
    return False


def validate_preset(preset: EnvironmentPreset) -> Tuple[bool, str]:
    """Validate that a preset's start, goal, and obstacle configuration is legal.

    Returns:
        (is_valid, error_message). error_message is empty string if valid.
    """
    w, h = preset.width, preset.height
    sx, sy = preset.start_pos
    gx, gy = preset.goal_pos

    if not (0 <= sx < w and 0 <= sy < h):
        return False, f"Start position {preset.start_pos} is out of bounds."
    if not (0 <= gx < w and 0 <= gy < h):
        return False, f"Goal position {preset.goal_pos} is out of bounds."
    if preset.start_pos in preset.obstacles:
        return False, "Start position is blocked by an obstacle."
    if preset.goal_pos in preset.obstacles:
        return False, "Goal position is blocked by an obstacle."
    if preset.start_pos == preset.goal_pos:
        return False, "Start and goal positions are identical."

    if not is_path_reachable(w, h, preset.start_pos, preset.goal_pos, preset.obstacles):
        return False, "No valid path exists between start and goal."

    return True, ""


# ---------------------------------------------------------------------------
# EnvironmentManager
# ---------------------------------------------------------------------------

class EnvironmentManager:
    """Manages active environment preset and exposes GridWorld factory."""

    def __init__(self) -> None:
        self._active_preset: EnvironmentPreset = _simple_maze()
        self._random_cfg: RandomObstaclesConfig = RandomObstaclesConfig()
        self._custom_preset: EnvironmentPreset = _custom_environment()

    @property
    def active_preset(self) -> EnvironmentPreset:
        return self._active_preset

    @property
    def preset_name(self) -> str:
        return self._active_preset.name

    @property
    def random_config(self) -> RandomObstaclesConfig:
        return self._random_cfg

    def select_preset(self, name: str) -> Tuple[bool, str]:
        """Switch to a named preset.

        For "Random Obstacles" uses the current RandomObstaclesConfig.
        Returns (success, error_message).
        """
        preset = build_preset(
            name,
            random_cfg=self._random_cfg if name == "Random Obstacles" else None,
        )
        if name == "Custom Environment":
            # Keep existing custom layout
            preset = self._custom_preset
        ok, msg = validate_preset(preset)
        if not ok:
            return False, msg
        self._active_preset = preset
        return True, ""

    def apply_random_config(self, cfg: RandomObstaclesConfig) -> Tuple[bool, str]:
        """Update random obstacle config and regenerate the preset.

        Returns (success, error_message).
        """
        self._random_cfg = cfg
        preset = _random_obstacles(cfg)
        ok, msg = validate_preset(preset)
        if not ok:
            return False, msg
        self._random_cfg = cfg
        if self._active_preset.name == "Random Obstacles":
            self._active_preset = preset
        return True, ""

    def set_custom_preset(self, preset: EnvironmentPreset) -> Tuple[bool, str]:
        """Store and activate a custom-edited preset.

        Returns (success, error_message).
        """
        ok, msg = validate_preset(preset)
        if not ok:
            return False, msg
        self._custom_preset = preset
        self._active_preset = preset
        return True, ""

    def build_gridworld(self, max_steps: int = 100, **reward_kwargs) -> "GridWorld":  # type: ignore[name-defined]
        """Construct and return a fresh GridWorld from the active preset."""
        from linkmem.core.environment import GridWorld

        p = self._active_preset
        return GridWorld(
            width=p.width,
            height=p.height,
            start_pos=p.start_pos,
            goal_pos=p.goal_pos,
            obstacles=set(p.obstacles),
            max_steps=max_steps,
            **reward_kwargs,
        )
