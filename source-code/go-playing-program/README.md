# go-playing-program

A Go engine in Scala 3, written for *Scala AI Book*.  It plays with Monte Carlo
tree search guided by a neural network, generates its own training data by
playing itself, and speaks GTP so it can be driven by any Go GUI.

The whole thing is about 6,210 lines of Scala — 1,371 of them tests — and
runs with **no external dependencies** — just a JDK and `scala-cli`.  A trained neural network is
optional and opt-in.

```
$ make check        # compile and run every test suite
$ make run          # play against it in the terminal
$ make demo-all     # a guided tour of every subsystem
```

## Contents

- [Quick start](#quick-start)
- [Requirements](#requirements)
- [The code, subsystem by subsystem](#the-code-subsystem-by-subsystem)
- [Working on one part at a time](#working-on-one-part-at-a-time)
- [Every make target](#every-make-target)
- [How the engine works](#how-the-engine-works)
- [Conventions you need to know](#conventions-you-need-to-know)
- [The reinforcement learning loop](#the-reinforcement-learning-loop)
- [Using a trained neural network](#using-a-trained-neural-network)
- [Testing](#testing)
- [Performance](#performance)
- [Deviations from DESIGN.md](#deviations-from-designmd)
- [What is not implemented](#what-is-not-implemented)

## Quick start

```bash
cd source-code/go-playing-program

make check                     # compile, then run all five test suites
make run                       # play Black against the engine on a 9x9 board
make run SIZE=19 PLAYOUTS=8000 # a serious game on a full board
make gtp                       # speak GTP on stdin/stdout
make selfplay GAMES=20         # generate reinforcement-learning training data
make demo-all                  # run every subsystem demo
make help                      # list everything
```

Inside `make run`, type moves like `D4`, `Q16` or `J9` — GTP coordinates, with
the letter `I` skipped — plus `hint`, `undo`, `score`, `settle`, `auto 20`,
`save game.sgf`, `load game.sgf` and `help`.

## Requirements

- **JDK 21 or newer** (`java -version`).  The parallel search uses virtual
  threads, which are final in Java 21.
- **scala-cli 1.5 or newer**.  The project pins Scala 3.6.4 itself.
- Nothing else.  No build tool, no dependency download, no test framework.

```bash
brew install openjdk@21 scala-cli     # macOS
```

`make version` prints both versions.

### A note on Bloop

`scala-cli` normally keeps a compile server (Bloop) running for fast incremental
builds.  In restricted environments — containers, some sandboxes, CI images —
Bloop cannot create its socket or write its cache, and every command fails with
a confusing Java exception.  The Makefile therefore **tests whether the Bloop
cache is writable and passes `--server=false` when it is not**, so `make check`
works either way.  Force it with `make check SERVER=0` or `SERVER=1`.

If you are calling `scala-cli` directly rather than through `make` and see a
socket error, add `--server=false`.

## The code, subsystem by subsystem

Eleven source files and six test suites, deliberately flat at the top level (see
[DESIGN.md §4.1](DESIGN.md)).  Each file carries its own
`//> using scala 3.6.4` directive, so it can be read, compiled and reasoned
about on its own.

| File | Lines | What it does | Test | Demo |
|------|------:|--------------|------|------|
| `project.scala` | 20 | Build settings and the default main class | — | — |
| `Board.scala` | 862 | Board state, legal moves, captures, ko/superko, Zobrist hashing, scoring, ASCII rendering | `make test-board` | `make demo-board` |
| `Search.scala` | 904 | MCTS: PUCT selection, RAVE/AMAF, progressive widening, virtual loss, virtual-thread parallelism, time control | `make test-search` | `make demo-search` |
| `Evaluate.scala` | 731 | Feature planes, `HeuristicNetwork`, `RandomNetwork`, influence heuristics, ONNX Runtime bridge | — | `make demo-eval` |
| `SelfPlay.scala` | 456 | Reinforcement-learning data generation: Dirichlet noise, temperature schedule, resignation, binary/JSON/SGF output | `make test-selfplay` | `make demo-selfplay` |
| `GTP.scala` | 502 | Go Text Protocol v2 session, handicap, undo, time control, SGF loading | `make test-gtp` | `make demo-gtp` |
| `CLI.scala` | 471 | Interactive terminal board with ANSI colours, an evaluation bar, and a layout that is taller than it is wide | `make test-cli` | `make run` |
| `SGF.scala` | 364 | SGF reader and writer, for interoperability and reviewing games | `make test-sgf` | `make demo-sgf` |
| `Args.scala` | 87 | The shared `--name value` command-line parser | — | — |
| `Demo.scala` | 444 | One runnable demonstration per subsystem | — | — |
| `*Test.scala` | 1,371 | The six test suites, in the book's `@main def …Test()` style | `make test` | — |
| `python/load_selfplay.py` | 324 | Reads the training data, stdlib-only for summaries, NumPy for training | — | — |

The three files that are not in DESIGN.md's list — `project.scala`,
`Args.scala`, `Demo.scala` — exist to serve "run everything easily"; see
[Deviations](#deviations-from-designmd).

## Working on one part at a time

This is the point of the layout.  Pick the subsystem you are changing, and use
its own test and demo:

```bash
make test-board     && make demo-board      # changed the rules?
make test-search    && make demo-search     # changed the search?
make test-sgf       && make demo-sgf        # changed the file format?
make test-selfplay  && make demo-selfplay   # changed data generation?
make test-gtp       && make demo-gtp        # changed the protocol?
make test-cli                                # changed the board layout?
```

Each demo prints the numbers that subsystem is judged on.  `make demo-search`,
for example, shows the visit distribution, the principal variation, the win
rate and the playouts per second; `make demo-eval` shows all fifteen feature
planes, the policy and an influence map of the position.

Three things make this work:

1. **Self-contained files.**  Every file repeats its `//> using` directives, so
   you can compile a single file in isolation while working on it:
   `scala-cli compile Board.scala --server=false`.
2. **One test suite per subsystem**, each a plain `@main` with no framework, so
   `make test-board` is one fast command that either passes or prints exactly
   which check failed.
3. **No hidden coupling.**  `Board.scala` knows nothing about search;
   `Search.scala` reaches the rules only through `BoardState`; the networks
   implement one small `PolicyValueNetwork` trait; `GTP.scala`, `CLI.scala` and
   `SelfPlay.scala` are three separate front ends over the same engine.

The dependency direction is:

```
                        Board.scala
                             |
        +--------------------+--------------------+
        |                    |                    |
   Search.scala        Evaluate.scala         SGF.scala
        |                    |                    |
        +----------+---------+--------------------+
                   |
   +---------------+----------------+
   |               |                |
SelfPlay.scala  GTP.scala       CLI.scala        Args.scala
                   \               /
                    \             /
                      Demo.scala
```

## Every make target

| Target | What it does |
|--------|--------------|
| `make check` | Compile, then run every test suite (the one to run before committing) |
| `make test` | Run every test suite |
| `make run` | Interactive board (`TIME=5s` for a time budget, `COMPACT=1` for a one-line-per-row board) |
| `make gtp` | GTP v2 on stdin/stdout |
| `make selfplay` | Generate training data (`GAMES=`, `OUTPUT=`) |
| `make clean` | Remove build output and generated games |
| `make help` | The full target list with current option values |
| `make watch` | Recompile on every source change (needs a writable Bloop server) |
| `make fmt`, `make fmt-check` | Format with scalafmt, if a `.scalafmt.conf` exists |
| `make version` | Print the `scala-cli` and Java versions |

Tests and demos, one per subsystem:

| Test | Demo | Subsystem |
|------|------|-----------|
| `make test-board` | `make demo-board` | Rules, captures, ko, scoring |
| `make test-search` | `make demo-search` | Monte Carlo tree search |
| `make test-sgf` | `make demo-sgf` | SGF reading and writing |
| `make test-selfplay` | `make demo-selfplay` | Reinforcement-learning data |
| `make test-gtp` | `make demo-gtp` | Go Text Protocol |
| `make test-cli` | — | Board layout: letter alignment and proportions |
| `make test-all` | `make demo-all` | Everything |

Options, all overridable on the command line:

```bash
make run      SIZE=19 PLAYOUTS=8000 THREADS=8
make run      SIZE=19 TIME=5s               # a time budget instead of playouts
make run      SIZE=19 COMPACT=1             # one line per row, for a short terminal
make selfplay GAMES=100 SIZE=9 OUTPUT=runs/run-01
make gtp      SIZE=19 PLAYOUTS=4000
make check    SERVER=0                      # force the Bloop server off
make onnx-run MODEL=models/policy-9x9.onnx  # needs the ONNX Runtime jar
```

## How the engine works

### Board.scala — the rules

Positions are immutable: every move returns a new `BoardState`, which is what
makes the parallel search safe.  A state carries the stones, whose turn it is,
komi, the rule set, prisoner counts, the move history, and a Zobrist hash.

- Stones are stored as one `Array[Byte]` addressed by `y * size + x`.  The
  `Point` type is an `opaque type Point = Int`, so a coordinate cannot be
  confused with a count.
- Legality is checked by playing the move on a copy: suicide, ko and positional
  superko are all detected.  Superko uses the Zobrist hash, so it also catches
  triple ko.
- Scoring supports both **area** (`Chinese`) and **territory** (`Japanese`)
  rules.  `territory` labels an empty region with a colour only when it borders
  exactly one colour; everything else is dame.
- `isOwnEye` is what stops the engine filling its own eyes, which is the detail
  that makes self-play games actually terminate.

### Search.scala — Monte Carlo tree search

- **Selection** is PUCT: `Q + c_puct * P * sqrt(parent visits) / (1 + visits)`,
  balanced against a small tie-breaking noise so that equally-valued moves do
  not always resolve the same way.
- **RAVE/AMAF** blends the node's own value with the value of moves that did
  well elsewhere in the rollout, using the schedule from DESIGN.md §2.3:
  `beta = sqrt(k / (3N + k))` with `k = 3000`.  Only the first
  `raveHorizon = 30` plies of the path are credited, which bounds the cost.
- **Progressive widening** starts a node with a single child and admits more as
  it is visited: `max(1, ceil(2.0 * sqrt(N)))`, candidates taken in prior order.
- **Virtual loss** lets threads descend the same path without all choosing the
  same child; it is added on the way down and removed on the way back, so the
  final statistics are exactly one visit and one value per playout.
- **Parallelism** uses virtual threads (`Thread.ofVirtual()`), one
  `java.util.Random` per worker, a shared atomic playout budget and an optional
  nanosecond deadline.
- **Leaf evaluation** is either the network's value head (`ValueHead`) or a
  random rollout to the end of the game (`Rollout`).  `Auto`, the default,
  picks rollouts for a network whose `valueIsMeaningful` is false — which is how
  `RandomNetwork` gets used as a pure-rollout engine.

### Evaluate.scala — what the network sees

Fifteen feature planes per position, in the style of AlphaGo Zero, all from the
point of view of the player to move:

| Plane | Meaning |
|------:|---------|
| 0, 1 | Stones of the player to move / of the opponent |
| 2, 3 | Liberties of each side, bucketed as `min(libs, 4) / 4` |
| 4 | The ko point |
| 5–12 | The last four moves: where the stone was played, then a whole-board plane saying whether that move was Black's |
| 13 | Whose turn it is (whole board, 0 or 1) |
| 14 | Stones in atari |

Three networks implement the same trait:

- `HeuristicNetwork` (the default) — no model file needed.  It builds a policy
  from capture and atari bonuses, liberty terms, eyes, self-atari, edge
  distance and star points, and derives a value from an influence heuristic.
  This is what makes the engine play a sensible game out of the box.
- `RandomNetwork` — uniform policy, `valueIsMeaningful = false`, so the search
  falls back to pure rollouts.  Useful as a baseline.
- `OnnxNetwork` — a trained model, loaded **reflectively** so the ONNX Runtime
  jar stays optional.  On Apple Silicon it selects the CoreML (Metal) execution
  provider via `SessionOptions.addCoreML()`.

### SelfPlay.scala — learning from itself

Plays the engine against itself and records, for each position, the triple that
AlphaGo Zero trains on: the feature planes, the MCTS visit distribution, and the
game result.  Root Dirichlet noise keeps openings varied, temperature is 1 for
the first `tempDropMove` moves and 0 after that, and resignation is available
but conservative by default.  Output is one binary file plus one SGF per game,
so every generated game can be reviewed by hand.

### GTP.scala, CLI.scala, SGF.scala — the front ends

`GtpEngine` implements protocol version 2 with the commands GUIs actually use:
`boardsize`, `clear_board`, `komi`, `play`, `genmove`, `undo`, `showboard`,
`final_score`, `final_status_list`, `fixed_handicap`, `place_free_handicap`,
`set_free_handicap`, `time_settings`, `time_left`, `loadsgf` and the
identification commands.  Command ids are echoed, failures use `?`, and an
unknown command is a protocol error rather than a crash.

`CLI.scala` draws the board with a wood-coloured grid, ● and ○ stones, an
evaluation bar and a panel showing captures, komi and the last search.  Two
layout details are deliberate, and `CLITest.scala` guards them:

- **The column letters sit over the points.**  A row label is three characters
  wide and each column is two (" `glyph`"), so the letters and the points share
  an offset.  Getting this wrong by one character still "works", which is why
  it is asserted rather than eyeballed.
- **The board is taller than it is wide.**  Each column is three characters and
  each row is followed by a blank band of wood.  That spare line is what buys
  the height, because a terminal character is about twice as tall as it is
  wide.  Stones are drawn once, never repeated.  On 9x9 that comes to 33
  characters wide by 20 lines.  A 19x19 board would be 40 lines, so
  `make run COMPACT=1` (or `--compact`) drops the bands and fits one line per
  row when the terminal is short.

`SGF.scala` reads and writes the `FF[4]` format, including handicap setup
stones, variations (the main line is followed) and escaped values, so games can
be exchanged with Sabaki, GoGUI or Lizzie.

## Conventions you need to know

These two conventions are used consistently everywhere and are the usual source
of confusion, so they are stated once, here:

**1. Values are always from the point of view of the player to move.**

```scala
val (policy, value) = network.evaluate(state)
// value is in [-1, +1] for state.toMove: +1 means "the player to move wins"
// policy has state.area + 1 entries, with the pass move at index state.area
```

`MCTSNode` stores its value from the point of view of the player who *made* the
move into that node, so backpropagation negates at each level.  The training
target `z` in `SelfPlay.scala` is written from the same perspective, so a model
trained on this data evaluates positions exactly the way the search expects.

**2. Coordinates.**

- Internally: `Point.at(x, y, size) = y * size + x`, with `y = 0` at the **top**
  row — so `(1, 1)` on a 9x9 board is GTP `B8`, not `B2`.
- GTP: column letters `A`–`T` **skipping `I`**, rows numbered from the bottom,
  and `pass` as a vertex.
- SGF: `a`–`s` letters, with `aa` being the top-left corner.

## The reinforcement learning loop

`make selfplay` closes the loop: the engine plays itself, the data trains a
network, and the network makes the engine stronger, which produces better data.

```bash
make selfplay GAMES=100 SIZE=9 OUTPUT=runs/run-01
python3 python/load_selfplay.py runs/run-01/training.bin
```

The output directory gets `training.bin` and one SGF per game:

```
runs/run-01/
  training.bin
  game-0001.sgf
  game-0002.sgf
  ...
```

The binary format (documented in full in both `SelfPlay.scala` and
`python/load_selfplay.py`) is little-endian so NumPy can consume it directly:

```
magic "GOPP" | version int32 | exampleCount | planeCount | boardSize | policySize
then per example: float32[planeCount * boardSize * boardSize] planes
                  float32[policySize]                         MCTS policy
                  float32                                     outcome z
                  float32                                     root search value
```

Training in PyTorch is then the standard two-headed loss:

```python
import torch
from load_selfplay import read_training_data

data = read_training_data("runs/run-01/training.bin")
planes = torch.from_numpy(data["planes"])            # (N, 15, size, size)
target_policy = torch.from_numpy(data["policies"])   # (N, size*size+1)
target_value = torch.from_numpy(data["outcomes"])    # (N,) in {-1, 0, +1}

log_policy, value = model(planes)
loss = -(target_policy * log_policy).sum(dim=1).mean() \
       + ((value - target_value) ** 2).mean()
```

`python3 python/load_selfplay.py file.bin --example 40` draws a single position
with its search target, which is the quickest way to see whether the data looks
sane.

## Using a trained neural network

The ONNX Runtime jar is **not** a dependency of the default build.  The engine
talks to it reflectively, so everything above works without it and the network
is entirely opt-in:

```bash
make onnx-check MODEL=models/policy-9x9.onnx               # does it load and evaluate?
make onnx-run   MODEL=models/policy-9x9.onnx               # interactive board
make onnx-gtp   MODEL=models/policy-9x9.onnx               # GTP with the network
make onnx-selfplay MODEL=models/policy-9x9.onnx GAMES=50   # self-play with it
```

Start with `make onnx-check`: it reports whether ONNX Runtime is on the
classpath and evaluates the opening position once, which catches a mismatch in
the model's input or output names before a game begins.

These targets add `--dep com.microsoft.onnxruntime:onnxruntime:1.20.0`
(override with `ONNX_VERSION=`).  To export a model from PyTorch:

```python
torch.onnx.export(
    model, example_input, "policy-9x9.onnx",
    input_names=["input"], output_names=["policy", "value"],
    dynamic_axes={"input": {0: "batch"}},
)
```

The input is NCHW `(batch, 15, size, size)`; the outputs are the policy logits
(or probabilities — both are handled) of length `size * size + 1` and a value in
`[-1, +1]` from the mover's perspective.  On Apple Silicon the CoreML provider
uses the GPU; pass `useCoreML = false` to `OnnxNetwork` to stay on the CPU.

If the artifact cannot be resolved because the dependency cache is read-only —
containers and sandboxes often mount it that way — point Coursier somewhere
writable:

```bash
COURSIER_CACHE=/tmp/coursier make onnx-check MODEL=models/policy-9x9.onnx
```

> **Note:** the ONNX path is the one part of this project that has not been
> executed end to end, because the environment it was developed in could not
> download the jar (its dependency cache is read-only, as above).  The code
> compiles, the bridge reports exactly what is missing, and the native objects
> are all released on every call — but treat the first run as a smoke test and
> start with `make onnx-check`.

## Testing

Six suites, 307 checks, no test framework:

```
boardTest:      56 checks passed     rules, captures, ko, superko, scoring
searchTest:     50 checks passed     tactics found, eyes never filled, budgets
sgfTest:        38 checks passed     round trips, variations, handicap, errors
selfPlayTest:   41 checks passed     shapes, labels, binary format, SGF
gtpTest:        99 checks passed     every command, including its failure path
cliTest:        23 checks passed     column letters over the points, board shape
```

They are ordinary `@main` methods, so each one is also runnable directly:

```bash
scala-cli run . --server=false --main-class go.boardTest
```

Each suite prints `"<name>Test: N checks passed"` and exits non-zero with a list
of failures otherwise, so `make check` is a real gate rather than a log to read.

Two deliberate properties of these tests:

- **They assert behaviour, not exact numbers.**  The search is parallel and
  stochastic: work is handed out from a shared playout budget, so the split
  between threads — and therefore the visit counts — varies with timing.  The
  tests check what must always hold: a forced capture is found, an eye is never
  filled, a budget is respected, the game terminates.  They never assert a
  specific move ordering, and where a result must be reproducible the search is
  pinned to one thread and a fixed seed.
- **They are fast.**  Small boards and small playout budgets keep the five
  suites to about half a minute, because a test suite you avoid running is
  worthless.  Compiling from scratch takes longer than testing.

## Performance

The parallel search is where the time goes.  Indicative numbers on a 10-core
Apple Silicon laptop, 9x9:

| Configuration | Speed |
|---------------|-------|
| `HeuristicNetwork`, value head, 4 threads | ~3,000 playouts/s |
| `RandomNetwork`, rollouts, 8 threads | ~520 playouts/s |

Feature-plane encoding is allocation-light but not free; `maxBatchSize` on
`SearchConfig` lets `BatchingEvaluator` group leaf evaluations for a GPU-backed
network, which is where batching pays off.

For a real game, raise the budget:

```bash
make run SIZE=19 PLAYOUTS=8000 THREADS=10
make gtp SIZE=19 PLAYOUTS=4000
```

## Deviations from DESIGN.md

The design document is the specification; this implementation follows it
closely.  Where it differs, the change is deliberate:

| DESIGN.md | Here | Why |
|-----------|------|-----|
| `opaque type Color = Byte` | `enum Color { Empty, Black, White }` | An enum is safer and more idiomatic Scala 3, and gives exhaustive matching and `ordinal` for free. `Point` remains an `opaque type` as specified. |
| `AtomicDouble` for node values | `java.util.concurrent.atomic.DoubleAdder` | `AtomicDouble` is not in the JDK. `DoubleAdder` is lock-free and made for exactly this accumulate-from-many-threads pattern. |
| Ko tracked by a ko point | Ko point **and** positional superko via `positionHistory: Set[Long]` | The hash-based rule catches triple ko and is no harder to check; the ko point is still recorded for feature planes and display. |
| The file list in §4.1 | Plus `project.scala`, `Args.scala`, `Demo.scala`, five test files, `python/load_selfplay.py` | `project.scala` holds shared build settings, `Args.scala` is the one argument parser all three entry points share, and the tests and demos are what make each subsystem independently runnable. |
| A Python training pipeline in Phase 4 | `python/load_selfplay.py` only | The reader and the format are here and verified; the model architecture and training loop belong with whichever framework you choose. |
| ONNX Runtime as a normal dependency | Reflective, opt-in, added by the `onnx-*` make targets | Keeps the default build dependency-free and fast to compile, and the engine still runs without a model. |
| §2.6 alpha-beta comparison | Not implemented | Deliberately out of scope for the MCTS engine; there is a marked placeholder in `Search.scala` explaining what would go there. |

Two smaller implementation notes:

- `HeuristicNetwork` reserves its pass probability *inside* the distribution, so
  the policy always sums to exactly 1.  This matters because the root-noise
  renormalisation and the training targets both assume it.
- Scoring does not remove dead stones.  Both players are assumed to have played
  out the game; `isOwnEye` is what keeps the engine from filling its own eyes
  and lets games finish naturally.

## What is not implemented

The short version is below.  `TODOs.md` is the working list: it adds the items
that are gaps rather than omissions, says where each piece belongs, and gives
each one a "done when" so a task can be picked up cold.

- **Alpha-beta search** (DESIGN.md §2.6).  The MCTS engine needs no such
  baseline, so this is left as a documented gap rather than a half-built one.
- **Proper life-and-death for `final_status_list`.**  GTP's `final_status_list
  alive` reports only groups with two or more genuine eyes — a conservative
  answer, because a wrong list is worse than a short one.
- **Distributed self-play.**  Games are generated one at a time on one machine;
  the parallelism is inside each search.
- **Pondering** (thinking on the opponent's time).  `Clock.timeForMove` and the
  search's deadline support it; nothing yet starts a search early.
- **An opening book**, and a benchmark harness against GnuGo or KataGo.  Both
  are Phase 5 items in DESIGN.md that want a stable strong engine first.
- **A trained model.**  The scaffolding to train and load one is here; the model
  itself is the reader's exercise.

## License

Copyright 2026 Mark Watson.  All rights reserved.
