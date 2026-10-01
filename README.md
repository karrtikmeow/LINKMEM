# LINKMEM — Interactive Non-Iterative Learning using Linked-List Experience Memory

> **Academic Software Project** | Python 3.14 | Tkinter GUI | Pytest

LINKMEM demonstrates a **non-iterative learning algorithm** in which an agent stores experiences in a singly linked list and retrieves the most relevant past experience via O(n) linear scan—without gradient descent, backpropagation, or training epochs.

---

## Core Learning Rule

The only Q-value update in the system is the **sample-average update**:

```
Q_new = Q_old + (reward - Q_old) / n
```

where `n` is the visit count after increment. No Bellman equation, no gamma, no max Q(s',a'), no TD learning.

---

## Project Structure

```
LINKMEM/
├── src/linkmem/
│   ├── core/              # Phase 1 — Core data structures
│   │   ├── types.py       # Enums: Action, CellType, PruningStrategy, LearningMode
│   │   ├── node.py        # ExperienceNode with sample-average Q update
│   │   ├── memory.py      # ExperienceMemory — singly linked list (O(1) insert, O(n) search)
│   │   ├── bucketed_memory.py  # BucketedExperienceMemory — spatial hash baseline
│   │   ├── encoder.py     # StateEncoder — 6D state vector
│   │   ├── distance.py    # DistanceCalculator (Euclidean / Manhattan)
│   │   ├── environment.py # GridWorld — 2D discrete environment
│   │   ├── agent.py       # Agent with HEURISTIC_TOWARDS_GOAL / RANDOM_EXPLORATION
│   │   └── pruning.py     # PruningPolicy (LOWEST_Q primary, LFU secondary)
│   ├── engine/            # Phase 2 — Headless learning engine
│   │   ├── events.py      # StepEvent — 22-field immutable step telemetry dataclass
│   │   ├── metrics.py     # EpisodeMetrics — 8-field episode measurement dataclass
│   │   └── learning_engine.py  # LearningEngine: orchestrates full step pipeline
│   ├── gui/               # Phase 3+4 — Tkinter GUI
│   │   ├── app.py         # MainWindow — tabbed notebook application
│   │   ├── controller.py  # SimulationController — worker thread + queue
│   │   ├── environments.py # EnvironmentPreset system (5 presets + BFS validation)
│   │   ├── theme.py       # Color palette, fonts, ttk style configuration
│   │   └── views/
│   │       ├── overview_tab.py    # Overview: GridWorld + Experiment Summary
│   │       ├── trace_tab.py       # Algorithm Trace: step-by-step decision inspector
│   │       ├── memory_tab.py      # Memory: scrollable linked-list visualizer
│   │       ├── analysis_tab.py    # Analysis: 3-chart Matplotlib + episode stats table
│   │       ├── controls_bar.py    # Bottom playback controls bar
│   │       ├── settings_dialog.py # Modal settings dialog (θ, N_max, speed, steps, seed)
│   │       ├── custom_env_editor.py  # Visual grid editor for Custom Environment
│   │       ├── environment_view.py   # GridWorld canvas renderer
│   │       ├── memory_view.py        # Linked-list canvas with click-to-inspect
│   │       └── algorithm_view.py     # Text-based algorithm step inspector
│   └── utils/
│       └── config.py      # AppConfig dataclass
├── tests/                 # 85 tests (84 sandboxed + 1 display integration)
└── pyproject.toml
```

---

## 6D State Vector

```python
state = [
    normalized_dx_goal,   # (goal_x - agent_x) / (width - 1)
    normalized_dy_goal,   # (goal_y - agent_y) / (height - 1)
    wall_up,              # 1 if blocked above, else 0
    wall_down,
    wall_left,
    wall_right,
]
```

---

## Launch

```bash
# Activate virtual environment
source .venv/bin/activate

# Launch interactive GUI
python -m linkmem

# Headless demonstration (no display required)
python -m linkmem --headless
```

---

## GUI Overview (Phase 4)

The application is a **4-tab notebook** with a persistent environment selector and controls bar.

### Top Bar
- **Environment dropdown**: `Empty Grid | Simple Maze | Complex Maze | Random Obstacles | Custom Environment`
- **⚙ Settings**: Opens modal dialog for θ, N_max, speed, max steps, and random seed
- Random Obstacles: exposes Count + Seed + Generate controls inline

### Tabs

| Tab | Contents |
|-----|---------|
| **Overview** | GridWorld canvas (left) + Experiment Summary cards + Episode Log (right) |
| **Algorithm Trace** | Step-by-step decision trace: State → Memory Search → Match/Novel → Action → Reward → Q Update |
| **Memory** | Scrollable singly linked-list visualizer with click-to-inspect node detail panel |
| **Analysis** | 3 Matplotlib charts (Reward, Memory Size, Cumulative Success Rate) + episode statistics table |

### Controls Bar (bottom)
```
Mode: [Autonomous] [Step By Step] [Manual] | ▶ Start | ⏸ Pause | ⏭ Step | ↺ Reset | ● STATUS
```

### Keyboard Shortcuts (Manual Mode)
| Key | Action |
|-----|--------|
| `W` / `↑` | Move Up |
| `S` / `↓` | Move Down |
| `A` / `←` | Move Left |
| `D` / `→` | Move Right |

---

## Environment Presets

| Preset | Description |
|--------|-------------|
| Empty Grid | 8×8 open grid, no obstacles |
| Simple Maze | Small deterministic maze, single corridor |
| Complex Maze | Larger challenging maze, multi-corridor |
| Random Obstacles | Configurable count + seed; BFS validates reachability |
| Custom Environment | Visual editor: click to toggle obstacles, set start/goal |

All environments are validated with BFS before loading — impossible configurations are rejected.

---

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| θ (theta) | 0.35 | Distance threshold for nearest-node match. Configurable; effect on node creation, lookup comparisons, success rate, and memory size will be evaluated empirically. |
| N_max | 100 | Maximum experience memory capacity |
| Speed | 12 Hz | Simulation steps per second in Autonomous mode |
| Max Steps | 100 | Episode timeout limit |
| Pruning | LOWEST_Q | Primary: evict lowest Q-value node; Secondary benchmark: LFU |

---

## Tests

```bash
# Run non-display tests (sandboxed)
.venv/bin/pytest -q --ignore=tests/test_gui_app_integration.py

# Run full suite including GUI integration test (requires X11 display)
DISPLAY=:0 .venv/bin/pytest -q
```

**Test count: 85 total** (84 sandboxed + 1 display integration test)

| Module | Tests |
|--------|-------|
| Phase 1 — Core structures | 35 |
| Phase 2 — Learning Engine | 14 |
| Phase 3 — GUI logic | 9 |
| Phase 4 — Environment presets | 26 |
| Phase 4 — GUI integration | 1 |

---

## Algorithm Pipeline

```
GridWorld.get_state()
    ↓ 6D StateVector
StateEncoder.encode()
    ↓
ExperienceMemory.find_nearest()    ← O(n) linear scan
    ↓ (best_node, distance, comparisons, trace)
Agent.select_action()
    ↓ if distance ≤ θ → reuse node's action
    ↓ if distance > θ → heuristic or random (novel state)
GridWorld.step(action)
    ↓ (next_state, reward, done, info)
ExperienceNode.update_q(reward)    ← Q ← Q + (r - Q) / n
    ↓
PruningPolicy.prune()              ← if memory.size > N_max
    ↓
StepEvent (22 fields) → SimulationController.queue → GUI tabs
```

---

## Academic Notes

- **No iterative training**: The system never replays experiences or runs multiple gradient descent epochs.
- **Theta (θ) is configurable**: The default of 0.35 is a starting value. Its effect on node creation rate, lookup comparison count, success rate, and memory saturation will be evaluated empirically.
- **Bellman-style updates are out of scope**: The core update rule is strictly the sample average. Q-learning with Bellman targets is noted as future scope only.
