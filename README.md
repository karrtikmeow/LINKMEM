# LINKMEM — Interactive Non-Iterative Learning using Linked-List Experience Memory

> **Academic Software Project** | Python 3.14 | Tkinter GUI | Pytest

LINKMEM demonstrates a **non-iterative learning algorithm** in which an agent stores experiences in an explicit singly linked list and retrieves the most relevant past experience via $O(n)$ linear scan—without gradient descent, backpropagation, neural networks, or iterative training epochs.

---

## Current Status

- **Status**: Production-Ready / Fully Verified
- **Test Suite**: **102 / 102 tests passing** (100% passing across unit, engine, GUI logic, environment switching, and navigation regression suites)
- **Architecture**: Single-source-of-truth GUI with atomic environment switching and episode-local cycle prevention

---

## Core Learning Rule

The only Q-value update in the system is the **sample-average update**:

$$\mathcal{Q}_{\text{new}} = \mathcal{Q}_{\text{old}} + \frac{r - \mathcal{Q}_{\text{old}}}{n}$$

where:
- $\mathcal{Q}_{\text{old}}$ is the existing node Q-value.
- $r$ is the scalar reward from the environment step.
- $n$ is the visit count of the node after increment ($n \leftarrow n + 1$).

There is **no Bellman equation**, no discount factor ($\gamma$), no $\max_{a'} \mathcal{Q}(s', a')$, no temporal-difference (TD) error, and no policy iteration.

---

## 6D State Vector

The discrete 2D environment state is encoded as a normalized 6-dimensional continuous feature vector:

$$\mathbf{s} = \begin{bmatrix} \Delta x_{\text{goal}} \\ \Delta y_{\text{goal}} \\ w_{\text{up}} \\ w_{\text{down}} \\ w_{\text{left}} \\ w_{\text{right}} \end{bmatrix} = \begin{bmatrix} (x_{\text{goal}} - x_{\text{agent}}) / (W - 1) \\ (y_{\text{goal}} - y_{\text{agent}}) / (H - 1) \\ \mathbb{I}(\text{wall above}) \\ \mathbb{I}(\text{wall below}) \\ \mathbb{I}(\text{wall left}) \\ \mathbb{I}(\text{wall right}) \end{bmatrix}$$

---

## Algorithm & Execution Pipeline

```
GridWorld.get_state()
    ↓ 6D StateVector
StateEncoder.encode()
    ↓
ExperienceMemory.find_nearest()       ← O(n) linear scan through singly linked list
    ↓ (nearest_node, min_distance, comparisons, trace)
Agent.select_action()
    ├─ Filter legal (unblocked) actions
    ├─ Multi-factor scoring: goal alignment + memory signal - cycle/reversal/visit penalties
    ├─ If min_dist ≤ θ: match reuse (or update action if resolving loop)
    └─ If min_dist > θ: novel state exemplar prepended at head in O(1)
GridWorld.step(action)
    ↓ (next_state, reward, is_terminal, info)
ExperienceNode.update_q(reward)       ← Q_new = Q_old + (r - Q_old) / n
    ↓
PruningPolicy.prune()                 ← if memory.size > N_max (LOWEST_Q or LFU)
    ↓
StepEvent → SimulationController.queue → Tkinter GUI Main Loop
```

### Loop & Cycle Prevention
To eliminate infinite 2-state oscillations ($A \leftrightarrow B$) or local dead-end loops in maze corridors:
- **Episode-Local Tracking**: Tracks `visited_states`, `state_action_visits`, `recent_states` (bounded queue), and `consecutive_cycles`.
- **Visit & Reversal Penalties**: Penalizes taking previously repeated actions and immediate 180° reversals when unvisited legal alternatives exist.
- **Memory Adaptation on Loop Escape**: When an alternative action is selected to escape an active cycle, the memory exemplar's action is updated to the escape action, allowing subsequent episodes to navigate directly without looping.
- **Episode Reset**: Episode-local counters reset between episodes, while the persistent linked-list `ExperienceMemory` is preserved.

---

## Project Structure

```
LINKMEM/
├── src/linkmem/
│   ├── core/                          # Core data structures & algorithm primitives
│   │   ├── types.py                   # Enums: Action, CellType, PruningStrategy, LearningMode, NoveltyStrategy
│   │   ├── node.py                    # ExperienceNode: singly linked list node with sample-average Q update
│   │   ├── memory.py                  # ExperienceMemory: explicit singly linked list (O(1) insert, O(n) search)
│   │   ├── bucketed_memory.py         # BucketedExperienceMemory: spatial hash grid benchmark baseline
│   │   ├── encoder.py                 # StateEncoder: encodes (pos, goal, obstacles) into 6D vector
│   │   ├── distance.py                # DistanceCalculator: Euclidean and Manhattan distance metrics
│   │   ├── environment.py             # GridWorld: 2D discrete environment with bounce-back collisions
│   │   ├── agent.py                   # Agent: memory lookup, cycle prevention, and adaptive action selection
│   │   └── pruning.py                 # PruningPolicy: LOWEST_Q primary and LFU secondary eviction
│   ├── engine/                        # Headless execution engine
│   │   ├── events.py                  # StepEvent: immutable telemetry dataclass emitted after every step
│   │   ├── metrics.py                 # EpisodeMetrics: per-episode measurement and statistics dataclass
│   │   └── learning_engine.py         # LearningEngine: headless loop orchestrating steps and batch episodes
│   ├── gui/                           # Interactive desktop GUI
│   │   ├── app.py                     # MainWindow: clean single-source-of-truth Tkinter application
│   │   ├── controller.py              # SimulationController: thread-safe worker and queue coordinator
│   │   ├── environments.py            # EnvironmentPreset system, BFS reachability, and EnvironmentManager
│   │   ├── theme.py                   # Color palette, fonts, and ttk styling
│   │   └── views/                     # GUI visual components
│   │       ├── grid_view.py           # Canvas visualizer for GridWorld grid, obstacles, and agent
│   │       ├── custom_env_editor.py   # Modal interactive grid editor for Custom Environment
│   │       ├── settings_dialog.py     # Modal dialog for tuning θ, N_max, speed, max steps, and seed
│   │       └── environment_view.py    # Canvas base renderer
│   ├── utils/                         # Utilities and configuration
│   │   └── config.py                  # AppConfig dataclass with system defaults
│   └── __main__.py                    # CLI entry point: interactive GUI or headless demonstration
├── tests/                             # Comprehensive test suite (102 tests)
│   ├── test_agent.py                  # Agent novel state creation, match reuse, random novelty (3 tests)
│   ├── test_bucketed_memory.py        # Bucketed spatial memory operations (3 tests)
│   ├── test_config.py                 # Default and custom AppConfig validation (2 tests)
│   ├── test_distance.py               # Euclidean, Manhattan, dimension checks (3 tests)
│   ├── test_encoder.py                # 6D normalization and wall sensor detection (3 tests)
│   ├── test_environment.py            # GridWorld boundaries, collisions, goal attainment (6 tests)
│   ├── test_environment_switching.py  # Atomic component replacement & switching invariants (12 tests)
│   ├── test_environments.py           # Builtin presets, random configs, BFS reachability (27 tests)
│   ├── test_gui_app_integration.py    # End-to-end full GUI capabilities integration test (1 test)
│   ├── test_gui_logic.py              # SimulationController, modes, worker thread lifecycle (8 tests)
│   ├── test_learning_engine.py        # Single steps, batch runs, Q-updates, determinism (14 tests)
│   ├── test_memory.py                 # Singly linked list order, lookup, node removal, clearing (8 tests)
│   ├── test_navigation_regression.py  # Loop escape, dead-ends, multi-episode improvement (5 tests)
│   ├── test_node.py                   # ExperienceNode sample-average math & serialization (4 tests)
│   └── test_pruning.py                # Lowest-Q and LFU eviction policies (3 tests)
├── pyproject.toml                     # Package configuration and dependencies
└── README.md
```

---

## Interactive Desktop GUI

The GUI provides an academic interface designed around a **single source of truth**:

```
┌────────────────────────────────────────────────────────────────────────┐
│ LINKMEM — Non-Iterative Learning using Experience Memory  [⚙ Settings] │
├────────────────────────────────────────────────────────────────────────┤
│ Environment: [ Simple Maze ▼ ] [Apply]   [Count: 6 Seed: 0 [Generate]] │
├────────────────────────────────────────────────────────────────────────┤
│ Current Environment: Simple Maze | Grid: 8×8 | Start: (0,0) | Goal: (7,7)│
├───────────────────────────────┬────────────────────────────────────────┤
│                               │  LIVE TELEMETRY                        │
│          GRID WORLD           │  Episode: 1        Step: 14            │
│                               │  Last Reward: +1.0 Ep Reward: +0.87    │
│          8 × 8 GRID           │  Memory: 10 nodes  Success Rate: 100%  │
│                               │  Avg Comps: 4.6    Status: RUNNING     │
│          S → → →              ├────────────────────────────────────────┤
│          █ █                  │  [Recent Episodes] [Memory] [Trace]    │
│          █       G            │  Ep 1 | 14 steps | +0.87 | SUCCESS     │
│                               │  Ep 2 | 14 steps | +0.87 | SUCCESS     │
│                               │  Ep 3 | 14 steps | +0.87 | SUCCESS     │
├───────────────────────────────┴────────────────────────────────────────┤
│ Mode: (●) Autonomous  ( ) Step-by-Step  ( ) Manual                     │
│ [ ▶ Start ] [ ⏸ Pause ] [ ⏭ Step ] [ ↺ Reset ]   Speed: [12] Hz ● READY│
└────────────────────────────────────────────────────────────────────────┘
```

### Key UI Features

- **Environment Selector**: Instantly switch between built-in presets or generate custom grids.
- **Atomic Environment Switching**: Completely stops existing worker threads and instantiates fresh `GridWorld`, `ExperienceMemory`, `Agent`, `LearningEngine`, and `SimulationController` instances.
- **Live Debug Banner**: Displays active dimensions, coordinates, and obstacle counts directly from the live `GridWorld` object.
- **Details Notebook**:
  - **Recent Episodes**: Treeview table recording completed episodes with color-coded outcomes (`SUCCESS` / `TIMEOUT`).
  - **Memory Nodes**: Live inspection table displaying stored experience nodes (`Node ID`, `Action`, `Q-Value`, `Visits`, `State Vector`).
  - **Algorithm Trace**: Text inspector showing the full 6-step breakdown for the latest decision with exact arithmetic verification.
- **Controls & Modes**:
  - **Autonomous**: Background thread executes continuous learning loop at configured speed (1–100 Hz).
  - **Step-by-Step**: Clicking **⏭ Step** executes exactly one perception-action-update cycle.
  - **Manual**: Direct keyboard control (`W`/`A`/`S`/`D` or Arrow keys).
  - **Reset**: Resets agent to start, clears memory and episode telemetry, but **preserves** the selected environment.

---

## Environment Presets

| Preset | Dimensions | Obstacles | Description |
|--------|:----------:|:---------:|-------------|
| **Empty Grid** | 8 × 8 | 0 | Open space for unobstructed exploration. |
| **Simple Maze** | 8 × 8 | 10 | Deterministic maze with a single corridor solution. |
| **Complex Maze** | 8 × 8 | 17 | Multi-corridor layout requiring backtracking and loop escape. |
| **Random Obstacles** | 8 × 8 | Configurable | Seed-deterministic obstacle distribution with BFS validation. |
| **Custom Environment** | 8 × 8 | User-defined | Interactive modal grid editor to toggle obstacles and set start/goal. |

Every environment change is validated via **Breadth-First Search (BFS)**; impossible configurations without a legal path between start and goal are rejected.

---

## Hyperparameters

| Parameter | Default | Description |
|-----------|:-------:|-------------|
| $\theta$ (theta) | `0.35` | Distance threshold for nearest-node match. Configurable starting value evaluated empirically across environments. |
| $N_{\text{max}}$ | `100` | Maximum capacity of the singly linked list memory. |
| Speed | `12 Hz` | Simulation steps per second in Autonomous mode (adjustable 0.5–100 Hz). |
| Max Steps | `100` | Maximum allowed steps per episode before timeout. |
| Pruning Strategy | `LOWEST_Q` | Primary strategy: evicts node with minimum Q-value; LFU available as comparison. |

---

## Launch Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Launch interactive desktop GUI
python -m linkmem

# Run headless demonstration (no display required)
python -m linkmem --headless
```

---

## Test Suite & Verification

The project includes **102 automated tests** verifying every layer of the architecture:

```bash
# Run sandboxed headless tests
.venv/bin/pytest -q --ignore=tests/test_gui_app_integration.py

# Run full test suite including GUI integration (requires X11 display)
DISPLAY=:0 .venv/bin/pytest -v
```

### Test Breakdown (102 Tests Total)

| Test Module | Tests | Description |
|-------------|:-----:|-------------|
| `test_agent.py` | 3 | Novel state creation, match reuse, and random novelty strategy |
| `test_bucketed_memory.py` | 3 | Bucketed spatial hash baseline operations |
| `test_config.py` | 2 | Configuration dataclass validation |
| `test_distance.py` | 3 | Euclidean and Manhattan distance metrics |
| `test_encoder.py` | 3 | 6D state vector normalization and obstacle boundary sensing |
| `test_environment.py` | 6 | GridWorld dynamics, collisions, bounce-back, and goal attainment |
| `test_environment_switching.py` | 12 | Atomic simulation component replacement and switching invariants |
| `test_environments.py` | 27 | Built-in presets, random generator determinism, and BFS validation |
| `test_gui_app_integration.py` | 1 | Full end-to-end GUI startup, stepping, memory table, and controls |
| `test_gui_logic.py` | 8 | Thread-safe SimulationController, queue dispatch, and mode switching |
| `test_learning_engine.py` | 14 | Non-iterative Q-updates, batch runs, and deterministic replay |
| `test_memory.py` | 8 | Singly linked list insertion, O(n) search, deletion, and telemetry |
| `test_navigation_regression.py` | 5 | Loop escape, cul-de-sac backtracking, and multi-episode memory reuse |
| `test_node.py` | 4 | ExperienceNode creation, sample-average math, and serialization |
| `test_pruning.py` | 3 | Lowest-Q and LFU memory eviction algorithms |
| **Total** | **102** | **All tests passing** |
