# TODOs

What is still to be built, in the order I would build it.

`README.md` has a short "What is not implemented" section for a reader of the
scaffold.  This file is the working list: it says *where* each piece goes and
*how to tell it is done*, so a task can be picked up cold.

## Where things stand

| | |
|---|---|
| Suites | `make check` — 307 checks across 6 suites, all passing |
| Entry points | `make run`, `make gtp`, `make selfplay`, `make demo-all` |
| Verified here | board rules, MCTS, SGF, self-play, GTP, CLI layout |
| **Not verified here** | the ONNX path — this environment cannot resolve the jar (see item 1) |

Sizes, so you can gauge the work: `Board.scala` 862 lines, `Search.scala` 904,
`Evaluate.scala` 731, `GTP.scala` 502, `CLI.scala` 471, `SelfPlay.scala` 456,
`Demo.scala` 444, `SGF.scala` 364, plus `Args.scala` and six test suites.

---

## 1. Verification gaps — start here

These are cheap and they protect work you have already done.  Item 1.1 is the
most valuable task in this file.

- [ ] **1.1 Verify the ONNX path on a machine that can fetch the dependency.**
  The bridge is written, compiles, and is defensive, but it has never executed a
  real model: the development sandbox has a read-only Coursier cache, so
  `com.microsoft.onnxruntime:onnxruntime` cannot be resolved.
  *Where:* `Evaluate.scala` (`OnnxBridge`, `OnnxNetwork`), `make onnx-check`.
  *Workaround for a restricted cache:* `COURSIER_CACHE=/tmp/coursier make onnx-check`.
  *Done when:* `make onnx-check MODEL=...` prints a policy that sums to 1.0 and a
  value in [-1, 1] for a real export, on both a 9x9 and a 19x19 model, and
  `make onnx-gtp` plays a legal game.
  *Likely to surface:* output ordering assumptions (`policyOutput=0`,
  `valueOutput=1`), input name (`input`), and whether CoreML registration
  succeeds on Apple Silicon.

- [ ] **1.2 A test suite for `Evaluate.scala`.**  It is the largest untested file.
  `FeaturePlanes.encode` is exactly the kind of code that fails silently — an
  off-by-one in a plane or a wrong "was Black" plane poisons every training
  example and every search.  A wrong encoding cannot be seen by reading output.
  *Where:* new `EvaluateTest.scala` + `make test-eval`; add to `SUITES` in the
  `Makefile`.
  *Done when:* plane count/shape, plane 0 is the side-to-move stones, the
  history planes, the `toMove` perspective of the value, komi handling, and
  `toNchw`'s flat layout are all asserted; `HeuristicNetwork`'s policy sums to
  1.0 and moves a forced capture; `HeuristicEval.influenceAt` is non-zero
  adjacent to a stone and zero far away.

- [ ] **1.3 A test for `Args.scala`.**  It is shared by all three entry points and
  has fiddly parsing (`--name value`, `--name=value`, `-name value`, bare
  flags, `5s`/`250ms`/`2m`/`1h`/bare-seconds durations).
  *Where:* new `ArgsTest.scala`, or fold it into `CLITest.scala`.
  *Done when:* every form parses, `ms` is not misread as seconds, and an
  unknown flag is reported rather than ignored.

- [ ] **1.4 A test for the training-data reader.**  `python/load_selfplay.py`
  currently has no automated check, so a format change could break the training
  path without anything noticing.
  *Where:* `python/` — a small `unittest`, runnable with `python3 -m unittest`.
  *Done when:* a generated file round-trips (header, example count, policy sums),
  and a deliberately truncated file fails loudly instead of silently.

- [ ] **1.5 Continuous integration.**  Nothing runs the suites on a push, and the
  zero-dependency design means it is a one-line job: install `scala-cli`, run
  `make check`.
  *Where:* a new `.github/workflows/check.yml`.
  *Done when:* a push runs `make check` on Linux, where `SERVER=0` is forced.

## 2. Rules and protocol completeness

- [ ] **2.1 `final_status_list` only answers `alive`.**  Every other status —
  `dead`, `seki`, `black`, `white` — returns an empty response, which is a wrong
  answer rather than a missing one.
  *Where:* `GTP.scala`, `finalStatusList`/`aliveGroups`.
  *Done when:* `dead` lists captured-or-doomed groups and `seki` lists
  double-live groups; `GTPTest.scala` covers each status.
  *Note:* this is the same life-and-death problem as 2.2 — solve it once.

- [ ] **2.2 Life-and-death for scoring.**  Dead stones are not removed before
  scoring, and `final_score` can therefore disagree with a human count.  The
  engine plays legally and terminates, but its score is optimistic.
  *Where:* `Board.scala` scoring, `GTP.scala` `final_score`/`final_status_list`.
  *Done when:* a position with a clearly dead group scores as if those stones
  were captured, and `BoardTest.scala` asserts it.
  *Approach:* a small Benson-style or search-based life-and-death pass is
  enough; full Japanese-rules resolution is a research project.

- [ ] **2.3 Expose the rule set.**  `RuleSet` (Area/Territory) exists and is
  written to and read from SGF, but no entry point can set it.
  *Where:* `CLI.scala` and `GTP.scala` argument parsing (`--rules area|territory`),
  as `--komi` already does.
  *Done when:* `make run RULES=territory` scores with prisoners and komi, and
  `make gtp RULES=territory` does the same.

- [ ] **2.4 Human-versus-human play.**  DESIGN.md Phase 1 asks for it.  Today
  `humanColor = None` means the *engine* plays both colours, so there is no way
  to have two humans at one terminal.
  *Where:* `CLI.scala` — `humanColor: Option[Color]` conflates "no human" with
  "both humans".  Replace it with an explicit `HumanSides` setting.
  *Done when:* `--both` is engine-vs-engine and a new `--hotseat` prompts for
  both colours; `CLITest.scala` checks which side is prompted.

- [ ] **2.5 SGF variations are read but not written.**  Loading a game with
  variations follows the main line; saving always writes a single line, so
  annotations and alternative branches are lost on a round trip.
  *Where:* `SGF.scala`, `write(record)`.
  *Done when:* a record with variations writes `(...)` branches and re-reads to
  the same tree; `SGFTest.scala` covers a two-branch record.

- [ ] **2.6 Handicap in the CLI.**  GTP has `fixed_handicap`,
  `place_free_handicap` and `set_free_handicap`; `StarPoints.handicapPoints`
  exists.  The interactive CLI has no equivalent.
  *Where:* `CLI.scala` — a `--handicap N` option reusing `StarPoints`.
  *Done when:* `make run HANDICAP=4` starts with four stones placed and White to
  play, matching GTP's placement.

## 3. Engine strength

Ordered by value per unit of effort.

- [ ] **3.1 Reuse the search tree between moves.**  Every move currently rebuilds
  the tree from scratch, throwing away everything learned about the position.
  This is the largest strength-per-line change available.
  *Where:* `Search.scala` — `SearchSession` needs a `advanceTo(point)` that
  re-roots on the child matching the played move and keeps its `visits` and
  `totalValue`; `CLI.scala`/`GTP.scala`/`SelfPlay.scala` call it after each move.
  *Watch out:* superko history, virtual-loss bookkeeping, and the parent links
  that RAVE walks.
  *Done when:* a benchmark shows fewer playouts give the same move quality, and
  `SearchTest.scala` asserts re-rooting preserves the best move on a fixed
  position.

- [ ] **3.2 Pondering.**  Think on the opponent's time.  The clock side is
  already built: `Clock.timeForMove` and the search's nanosecond deadline take a
  budget.
  *Where:* `GTP.scala` (start a search after `genmove`, keep it if the opponent
  plays the predicted move, discard it otherwise), `Search.scala` for
  cancellation.
  *Done when:* `time_settings` plus a ponder cycle does not corrupt state and
  the reply to the predicted move is measurably faster.

- [ ] **3.3 Tune the search constants.**  `raveEquivalence = 3000.0`,
  `raveHorizon = 30`, `wideningC = 2.0`, `wideningAlpha = 0.5`, and
  `virtualLoss = 1.0` are reasoned guesses, not measured values.
  *Where:* `SearchConfig` in `Search.scala`.
  *Done when:* a small round-robin (see 5.3) justifies each value, and the
  chosen numbers are recorded with the measurement.

- [ ] **3.4 An opening book (optional, DESIGN.md Phase 5).**  `BoardState` already
  carries `zobristHash: Long`, which is the index such a book needs.
  *Where:* new `Book.scala` + loading in `GTP.scala`/`Search.scala`.
  *Done when:* a book built from a directory of SGF files answers the first few
  moves and falls through to search otherwise.

- [ ] **3.5 Alpha-beta tactical supplement (DESIGN.md §2.6) — deliberately
  deferred.**  There is a marker comment in `Search.scala` explaining why: a
  trustworthy Go alpha-beta needs its own capture-race move generation, threat
  extensions and a static exchange evaluator, and a half-built one is worse than
  none.
  *Where:* a new `Tactics.scala`, opted into from `SearchConfig`, aimed at
  semeai and tsumego subproblems rather than the whole game.
  *Done when:* it finds forced captures that MCTS needs thousands of playouts
  for, and never returns a losing line as winning.

## 4. Learning pipeline (DESIGN.md Phase 4)

The generation half exists; the training half does not.  Nothing in this section
can be finished without a machine with a GPU and PyTorch.

- [ ] **4.1 The Python training script.**  `python/load_selfplay.py` reads the
  data and documents the loss; nothing trains.
  *Where:* new `python/train.py` (PyTorch, `mps` on Apple Silicon, `cuda`
  otherwise), reusing the existing reader.
  *Contract to honour:* input `(N, 15, size, size)` float32; policy target
  length `area + 1` with pass at index `area`; value target in [-1, 1] from
  `state.toMove`'s perspective; loss = policy cross-entropy + value MSE.
  *Done when:* it trains on a generated file, checkpoints, and reports policy
  and value loss on a held-out set.

- [ ] **4.2 The ONNX export script.**  The engine expects specific names and
  order and nothing produces a file that matches them.
  *Where:* new `python/export_onnx.py` (or a documented `torch.onnx.export`
  snippet in `README.md`).
  *Contract:* input named `input`; output 0 the policy `(N, area + 1)`; output 1
  the value `(N, 1)`; 15 input planes.  These are the `OnnxNetwork` defaults.
  *Done when:* `make onnx-check MODEL=<exported file>` passes (this is 1.1).

- [ ] **4.3 Data augmentation by symmetry.**  Go positions are invariant under the
  eight dihedral transforms, so the same file can train eight times the data.
  *Where:* the reader or the training loop (transform both the planes and the
  policy index).
  *Done when:* one example expands to eight, and the pass index is left alone.

- [ ] **4.4 A promotion gate: network versus network.**  AlphaZero-style: a new
  model replaces the old only after winning a match.
  *Where:* new `Match.scala` or a `SelfPlay` mode that plays two networks
  against each other over N games with alternating colours.
  *Done when:* it reports wins/losses and refuses to promote on a tie or a
  losing record.

- [ ] **4.5 Iterate the loop, 9x9 then 13x13 then 19x19.**  DESIGN.md Phase 4
  describes self-play → train → evaluate → repeat, transferring up in size.
  *Where:* a driver script (`Makefile` targets or a shell script) and README
  notes on the hyperparameters used.
  *Done when:* there is a 9x9 model that beats `HeuristicNetwork` in a match.

- [ ] **4.6 Parallel or distributed self-play.**  Games are generated one at a
  time; the parallelism is inside each search, so the CPU is idle between
  moves and across games.
  *Where:* `SelfPlay.scala` — a worker pool over games, writing to one output
  file under a lock or one file per worker merged at the end.
  *Done when:* `GAMES=100` scales with cores and the output is identical in
  format to the serial path (`SelfPlayTest.scala` guards the format).

## 5. Tooling, packaging and docs

- [ ] **5.1 Packaging.**  There is no way to ship a single artifact.
  *Where:* `project.scala` plus Makefile targets — `scala-cli package --assembly`
  or a native image, and a `--jvm` / native-image note.
  *Done when:* `make package` produces something runnable without `scala-cli`.

- [ ] **5.2 A scalafmt config.**  `make fmt` and `make fmt-check` currently do
  nothing, because there is no `.scalafmt.conf`.
  *Where:* new `.scalafmt.conf` (Scala 3 dialect) — then run `make fmt` once.
  *Done when:* `make fmt-check` passes on the committed tree.

- [ ] **5.3 A benchmark and gauntlet harness (DESIGN.md Phase 5).**  Strength
  claims are currently anecdotal.
  *Where:* new `Benchmark.scala` — fixed positions with known best moves, plus
  engine-vs-engine matches over N games with alternating colours, reporting
  win rate and Elo-like differences.
  *Done when:* it can compare two `SearchConfig`s or two networks and say which
  is stronger, and `README.md`'s performance table comes from it.

- [ ] **5.4 Profiling and performance notes.**  The README quotes playouts per
  second from the demos; there is no profile of where the time goes.
  *Where:* a `make profile` target using a JFR or async-profiler run.
  *Done when:* the hot spots are known and either fixed or written down.

- [ ] **5.5 Map the scaffold to the book chapters.**  `NOT_YET_IN_BOOK.md` marks
  this code as not yet in the book.  A reader would benefit from knowing which
  file corresponds to which chapter.
  *Where:* `README.md`.
  *Done when:* each `*.scala` names its chapter, or a table maps them.

## 6. Deliberate non-goals

Not oversights; recorded so they are not rediscovered as "bugs".

- **Situational superko.**  This engine uses positional superko
  (`positionHistory: Set[Long]`), which is what most rulesets use.  A
  situational option would be a rule variant, not a fix.
- **Japanese-rules suicide.**  Suicide is illegal here, as it is in most
  rulesets.  This is correct for play, even where territory scoring is selected.
- **A trained model in the repository.**  Design and build the pipeline; the
  model belongs to the reader's run, not to the source tree.
- **Exact SGF preservation of unknown properties.**  Unknown properties are
  ignored rather than round-tripped; preserving them would need a full property
  tree in `SgfNode`.

---

## Suggested next three

1. **1.1** — verify ONNX on an unrestricted machine.  It is the only part of the
   scaffold with no execution evidence at all, and it is the thing a reader will
   try first.
2. **1.2** — an `EvaluateTest.scala`.  It is the biggest untested surface, and
   feature-plane bugs are invisible.
3. **3.1** — tree reuse between moves.  The largest strength gain available, and
   it is contained inside `SearchSession`.
