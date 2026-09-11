# Go Playing Program — Design Document (Scala 3)

An educational Go-playing engine written in modern, idiomatic Scala 3.
Inspired by the author's 1979 UCSD Pascal Go program, reimagined with
Monte Carlo Tree Search, neural-network evaluation, and reinforcement
learning self-play — all running on a powerful Apple Silicon Mac desktop.

---

## 1. Key Technology Stack

### 1.1 Language & Build

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Language | **Scala 3.6+** | Algebraic data types, opaque types for board coordinates, immutable collections for functional game trees, inline for hot-path performance. |
| Build | **scala-cli** | Zero-config builds consistent with the other book projects. |
| JVM | **GraalVM 21+ (Apple Silicon)** | Tier-1 aarch64 support, C2 JIT with vector intrinsics for matrix ops, Virtual Threads (Project Loom) for parallel search. |
| GPU (optional) | **Apple Metal via ONNX Runtime** | Run trained ONNX policy/value networks on the Mac's Metal GPU. The `onnxruntime` Java binding supports Apple's CoreML / Metal execution provider natively on Apple Silicon. |

### 1.2 Hardware Assumptions — Powerful Mac Desktop

A Mac Studio or Mac Pro with Apple Silicon (M2 Ultra / M4 Ultra class):

- **24–76 CPU cores** — saturate with parallel MCTS playouts using JVM
  Virtual Threads.
- **64–192 GB unified memory** — keep large transposition tables and
  MCTS trees resident; no disk I/O during search.
- **Integrated GPU (up to 76-core)** — run neural network inference
  via ONNX Runtime + CoreML/Metal provider. Avoids the need for a
  discrete GPU or CUDA.
- **High single-core IPC** — benefits alpha-beta and policy-network
  pruned search.

### 1.3 Dependencies (Minimal)

| Library | Purpose |
|---------|---------|
| `onnxruntime` (Java) | Neural network inference (policy + value heads) on Metal GPU |
| *None for core engine* | Pure Scala 3, zero runtime deps for board logic & search |

---

## 2. Advanced Search Techniques

### 2.1 Monte Carlo Tree Search (MCTS) with UCT

MCTS is the foundation of all modern Go engines. Each iteration consists
of four phases:

```
Selection → Expansion → Simulation → Backpropagation
```

**Upper Confidence bound for Trees (UCT)** balances exploitation
(choosing the move with the best win-rate so far) and exploration
(trying under-visited moves):

```
UCT(child) = Q(child)/N(child) + C_puct * P(child) * sqrt(N(parent)) / (1 + N(child))
```

Where:
- `Q` = total value accumulated
- `N` = visit count
- `P` = prior probability from the policy network
- `C_puct` = exploration constant (typically 1.0–2.5, tuned via self-play)

### 2.2 Policy-Network-Guided Search (AlphaGo / KataGo Style)

Replace random rollouts with a **neural network value head**:

1. **Selection**: Walk the tree using PUCT (Predictor + UCT), where the
   prior `P` comes from the **policy head** of the neural network.
2. **Expansion**: When a leaf is reached, run the neural network to
   obtain `(policy_vector, value)`.
3. **Value backup**: Backpropagate the value estimate `v` up the tree
   — no random rollout needed.

This is the approach used by AlphaGo Zero and KataGo. It is
dramatically stronger than random rollouts and converges with far
fewer playouts.

### 2.3 RAVE (Rapid Action Value Estimation)

RAVE (also called AMAF — All Moves As First) accelerates early tree
convergence by sharing statistics across sibling nodes:

- If move `m` was played *anywhere* in a playout starting from node
  `n`, credit `m` as if it were played *first* from `n`.
- Blend RAVE values with MCTS values using a mixing parameter `β` that
  decays toward zero as visit counts grow.

```
Q_combined = (1 - β) * Q_mcts + β * Q_rave
β = sqrt(k / (3*N + k))     // k ≈ 3000, tuned empirically
```

RAVE is most valuable in the early thousands of playouts before the
neural network prior dominates.

### 2.4 Progressive Widening

In positions with high branching factor (Go can have 200+ legal moves),
progressively widen the set of children considered:

```
num_children_to_expand = C * N(parent)^α    // α ≈ 0.5, C ≈ 1.0
```

Only expand the top-`k` moves ranked by the policy network prior.
This prevents the search tree from growing too wide too fast.

### 2.5 Parallel Search (Virtual Threads)

Exploit the Mac's many cores with **root parallelism** and **virtual
loss**:

- Launch `N` virtual threads, each performing independent MCTS
  iterations on the shared tree.
- Apply a **virtual loss** (temporarily decrement a node's value when
  a thread selects it) to encourage threads to explore different
  branches.
- Backpropagate atomically using `AtomicInteger` / `AtomicDouble`
  accumulators on tree nodes.

JVM Virtual Threads (Project Loom) are ideal here: lightweight, no
thread-pool sizing headaches, and the JVM schedules them across all
available carrier threads automatically.

### 2.6 Alpha-Beta as a Tactical Supplement (Optional)

For capturing-race (semeai) and life-and-death (tsumego) sub-problems,
a focused alpha-beta search with Go-specific null-move and threat
extensions can read out tactical sequences more efficiently than MCTS.
This is similar to KataGo's approach of blending search strategies.

---

## 3. Evaluation via Reinforcement Learning

### 3.1 Network Architecture

A dual-headed convolutional neural network:

```
Input: 19×19 board planes (stone colors, liberties, ko, history, etc.)
   │
   ▼
┌──────────────────────┐
│  Residual Tower       │   6–20 residual blocks (start small, scale up)
│  (128–256 filters)    │   Each block: Conv 3×3 → BatchNorm → ReLU → Conv 3×3 → skip
└──────────┬───────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐ ┌─────────┐
│ Policy  │ │  Value  │
│  Head   │ │  Head   │
│ 19×19+1 │ │ scalar  │
│ softmax │ │  tanh   │
└─────────┘ └─────────┘
```

- **Policy head**: probability distribution over all 19×19 intersections
  + pass. Guides MCTS move selection.
- **Value head**: single scalar in `[-1, +1]` estimating the game outcome
  from the current player's perspective.

### 3.2 Input Feature Planes

| Plane(s) | Description |
|----------|-------------|
| 1–2 | Current player's stones / opponent's stones |
| 3–4 | Liberties count (1, 2, 3, ≥4) for each side |
| 5 | Ko point (illegal recapture) |
| 6–13 | Last 4 move history (2 planes each: stone placed, side to move) |
| 14 | Side to move (all 1s for Black, all 0s for White) |
| 15 | Ladder-escape indicators (optional, helps with tactical patterns) |

### 3.3 Self-Play Training Loop

The reinforcement learning pipeline follows AlphaGo Zero:

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Self-Play  │────▶│  Training    │────▶│  Evaluation  │
│  Games      │     │  Pipeline    │     │  (Gating)    │
└─────────────┘     └──────────────┘     └──────────────┘
       ▲                                        │
       └────────────────────────────────────────┘
              update network if improved
```

1. **Self-Play**: The current best network plays games against itself
   using MCTS (800+ playouts per move). Record
   `(board_state, mcts_policy, game_outcome)` tuples.

2. **Training**: Train the network to minimize the combined loss:
   ```
   L = (z - v)² - π^T · log(p) + c·‖θ‖²
   ```
   Where `z` = actual game outcome, `v` = value prediction, `π` = MCTS
   policy target, `p` = network policy output, `c` = L2 regularization.

3. **Evaluation (Gating)**: Pit the newly trained network against the
   current champion in a match of 100+ games. Accept the new network
   only if it wins ≥ 55% (prevents regression).

4. **Iterate**: Replace the champion, generate new self-play games,
   repeat.

### 3.4 Training Infrastructure (Mac Desktop)

- **Self-play game generation**: Runs entirely on the Mac using MCTS
  + ONNX Runtime on Metal GPU.  A Mac Studio M4 Ultra can generate
  thousands of self-play games per day at modest playout counts.
- **Network training**: Export self-play data to PyTorch or JAX
  (Python). Train on the Mac's GPU via `mps` (Metal Performance
  Shaders) device backend. Export the trained model to ONNX format.
- **Bootstrapping**: Begin with a randomly initialized network. The
  first few generations learn basic patterns (captures, eyes,
  territory) purely from self-play.

### 3.5 Practical Bootstrapping — Progressive Board Sizes

To accelerate early training:

1. Start on **9×9** boards (fast games, quick convergence).
2. Transfer learned features to **13×13**.
3. Finally, scale to **19×19**.

Each board size reuses the convolutional weights (padding as needed)
and fine-tunes with additional self-play.

---

## 4. Application Structure

### 4.1 Source Files

Following the book's convention of flat scala-cli projects:

```
go-playing-program/
├── DESIGN.md           ← This document
├── Makefile            ← Build/run targets (scala-cli)
├── Board.scala         ← Board representation and game rules
├── Search.scala        ← MCTS engine with UCT, RAVE, parallel search
├── Evaluate.scala      ← Neural network evaluation (ONNX Runtime)
├── SelfPlay.scala      ← Self-play game generation for RL training
├── GTP.scala           ← Go Text Protocol interface (for GoGUI, etc.)
├── CLI.scala           ← Interactive terminal UI with ANSI rendering
├── SGF.scala           ← SGF (Smart Game Format) reader/writer
└── README.md           ← Usage instructions and book chapter reference
```

### 4.2 Module Descriptions

#### `Board.scala` — Core Game Logic

```scala
package go

// Immutable board state using opaque types for type-safe coordinates
opaque type Point = Int          // 0..360 for 19×19, or -1 for pass
opaque type Color = Byte         // Black=1, White=2, Empty=0

case class Group(color: Color, stones: Set[Point], liberties: Set[Point])

case class BoardState(
  size: Int,                      // 9, 13, or 19
  grid: Array[Byte],             // flat array, size×size
  groups: Map[Point, Group],     // union-find or direct map
  koPoint: Option[Point],        // illegal recapture point
  captures: (Int, Int),          // (black captured, white captured)
  moveHistory: Vector[Point],    // for superko detection
  zobristHash: Long              // incremental Zobrist hash
)
```

Responsibilities:
- Legal move generation (liberty counting, self-capture prohibition,
  ko and superko rules).
- Stone capture and group merging via union-find.
- Scoring (Chinese rules: area scoring; Japanese rules: territory
  scoring).
- Zobrist hashing for transposition detection and superko.

#### `Search.scala` — MCTS Engine

```scala
package go

class MCTSNode(
  val move: Point,
  val parent: MCTSNode | Null,
  var children: Array[MCTSNode],
  val prior: Float,               // P(move) from policy network
  visits: AtomicInteger,          // N — thread-safe
  totalValue: AtomicDouble,       // Q — thread-safe
  raveVisits: AtomicInteger,      // RAVE N
  raveTotalValue: AtomicDouble    // RAVE Q
)

object Search:
  def bestMove(
    state: BoardState,
    network: PolicyValueNetwork,
    playouts: Int = 1600,
    threads: Int = Runtime.getRuntime.availableProcessors,
    timeLimit: Option[Duration] = None
  ): (Point, Map[Point, Float])  // best move + policy distribution
```

Responsibilities:
- MCTS iteration loop: select → expand → evaluate → backpropagate.
- PUCT selection with policy-network priors.
- RAVE blending for early convergence.
- Progressive widening to manage branching factor.
- Virtual-thread parallel MCTS with virtual loss.
- Time management (sudden death, byo-yomi, per-move budgets).
- Pondering (search on opponent's time).

#### `Evaluate.scala` — Neural Network Interface

```scala
package go

trait PolicyValueNetwork:
  def evaluate(state: BoardState): (Array[Float], Float)
  //                                 policy[361]  value[-1,+1]

class OnnxNetwork(modelPath: String) extends PolicyValueNetwork:
  // ONNX Runtime session with Metal execution provider
  // Batch evaluation support for parallel MCTS

class RandomNetwork extends PolicyValueNetwork:
  // Uniform policy + random value — used before any training
```

Responsibilities:
- Load ONNX model and configure Metal GPU acceleration.
- Convert `BoardState` to input tensor (feature planes).
- Run inference and decode policy + value outputs.
- Support batched evaluation for throughput.
- Fallback `RandomNetwork` for play without a trained model.

#### `SelfPlay.scala` — Reinforcement Learning Data Generation

```scala
package go

case class TrainingExample(
  featurePlanes: Array[Array[Float]],  // input to network
  mctsPolicy: Array[Float],            // π — search policy target
  outcome: Float                        // z — game result {-1, +1}
)

object SelfPlay:
  def generateGame(
    network: PolicyValueNetwork,
    boardSize: Int = 19,
    playouts: Int = 800,
    temperature: Float = 1.0f,    // exploration temperature
    tempDropMove: Int = 30         // drop temp to near-zero after move 30
  ): Vector[TrainingExample]
```

Responsibilities:
- Play full games using MCTS, recording training examples.
- Temperature-based move selection (high early for diversity,
  low later for strength).
- Dirichlet noise at root for exploration.
- Export training data in a format consumable by PyTorch/JAX.
- Support resignation threshold (resign if value head says < -0.9).

#### `GTP.scala` — Go Text Protocol

```scala
package go

object GTP:
  def main(args: Array[String]): Unit =
    // Read GTP commands from stdin, write responses to stdout
    // Supports: play, genmove, boardsize, komi, clear_board, showboard,
    //           time_settings, time_left, final_score, quit, etc.
```

Responsibilities:
- Implement the [GTP v2 specification](https://www.lysator.liu.se/~gunnar/gtp/).
- Enable the engine to connect to GUIs like **GoGUI**, **Sabaki**,
  or **Lizzie**.
- Enable engine-vs-engine matches via `gogui-twogtp`.

#### `CLI.scala` — Terminal User Interface

```scala
package go

object CLI:
  def main(args: Array[String]): Unit =
    // ANSI-rendered board with colored stones
    // Interactive play: human vs engine, engine vs engine
    // Commands: play, pass, resign, undo, score, save, load, quit
```

Responsibilities:
- Render the board with ANSI colors (tan background, ● ○ stones,
  star points, coordinates A–T).
- Sidebar: captured stones, komi, move number, evaluation bar,
  engine thinking info (playouts/sec, principal variation).
- Command parsing for interactive play.

#### `SGF.scala` — Smart Game Format

```scala
package go

object SGF:
  def write(moves: Vector[(Color, Point)], result: String, komi: Float,
            boardSize: Int): String
  def read(sgf: String): (Vector[(Color, Point)], Map[String, String])
```

Responsibilities:
- Save and load games in standard SGF format.
- Preserve metadata (player names, date, result, komi).
- Enable game review in external tools (Sabaki, GoGUI).

### 4.3 Build & Run Targets

```makefile
.PHONY: clean check run gtp selfplay

check:
	scala-cli compile .

run:
	scala-cli run . --main-class go.CLI

gtp:
	scala-cli run . --main-class go.GTP

selfplay:
	scala-cli run . --main-class go.SelfPlay -- --games 100 --size 9

clean:
	scala-cli clean . 2>/dev/null || true
	find . -name ".scala-build" -type d -exec rm -rf {} +
	find . -name ".bsp" -type d -exec rm -rf {} +
```

---

## 5. Development Roadmap

### Phase 1 — Board & Rules (Foundation)
- [ ] `Board.scala`: Full 19×19 board with legal move generation,
      capture, ko, superko, scoring.
- [ ] `CLI.scala`: Terminal rendering and human-vs-human play.
- [ ] `SGF.scala`: Save/load games.

### Phase 2 — Search (Playable Engine)
- [ ] `Search.scala`: Basic MCTS with random rollouts (no neural
      network). This alone produces a reasonable 9×9 player.
- [ ] `GTP.scala`: Connect to GoGUI for visualization.
- [ ] RAVE integration for improved early search quality.

### Phase 3 — Neural Network Evaluation
- [ ] `Evaluate.scala`: ONNX Runtime integration with Metal GPU.
- [ ] Replace random rollouts with policy + value network.
- [ ] Progressive widening guided by policy priors.

### Phase 4 — Reinforcement Learning
- [ ] `SelfPlay.scala`: Self-play game generation.
- [ ] Python training pipeline (PyTorch on Metal `mps` backend).
- [ ] Training loop: self-play → train → evaluate → iterate.
- [ ] Start on 9×9, transfer to 13×13, then 19×19.

### Phase 5 — Polish & Strength
- [ ] Virtual-thread parallel MCTS.
- [ ] Time management and pondering.
- [ ] Opening book from professional games (optional).
- [ ] Benchmark against GnuGo and other engines.

---

## 6. References

- Silver, D. et al. *Mastering the Game of Go without Human Knowledge*
  (AlphaGo Zero), Nature 2017.
- Wu, D. *Accelerating Self-Play Learning in Go* (KataGo), 2019.
- Gelly, S. & Silver, D. *Combining Online and Offline Knowledge in
  UCT* (RAVE/AMAF), ICML 2007.
- Coulom, R. *Efficient Selectivity and Backup Operators in Monte-Carlo
  Tree Search*, 2006.
- GTP v2 Specification: https://www.lysator.liu.se/~gunnar/gtp/
- ONNX Runtime Java API: https://onnxruntime.ai/docs/get-started/with-java.html
