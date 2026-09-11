# TODOs

What is still to be built, in the order I would build it.

`README.md` has a short "What is not implemented" section for a reader of the
scaffold.  This file is the working list: it says *where* each piece goes and
*how to tell it is done*, so a task can be picked up cold.

## Where things stand

| | |
|---|---|
| Suites | `make check` — 465 checks across 8 suites, plus `make test-python` (13 tests) |
| Entry points | `make run`, `make gtp`, `make selfplay`, `make demo-all` |
| Verified here | board rules, MCTS, SGF, self-play, GTP, CLI layout, eval planes, tactics, life-and-death, tree reuse, python reader |
| **Not verified here** | the ONNX path and a full `make check` run — this environment cannot execute `scala-cli` (see item 1.1) |

New files since the scaffold: `Tactics.scala`, `LifeDeath.scala`,
`EvaluateTest.scala`, `TacticsTest.scala`, `python/test_load_selfplay.py`.

---

## 1. Verification gaps — start here

These are cheap and they protect work you have already done.  Item 1.1 is the
most valuable task in this file.

- [ ] **1.1 Verify the ONNX path on a machine that can fetch the dependency.**
  The bridge is written, compiles, and is defensive, but it has never executed a
  real model.  Attempted 2026-09-11 on an M2 Mac: `scala-cli` itself cannot
  start inside the sandbox (its launcher dies on a signal call the sandbox
  forbids, and escalation is unavailable in this session), so even the
  dependency fetch could not be attempted here — this needs a normal terminal.
  *Where:* `Evaluate.scala` (`OnnxBridge`, `OnnxNetwork`), `make onnx-check`.
  *Workaround for a restricted cache:* `COURSIER_CACHE=/tmp/coursier make onnx-check`.
  *Try, in order:* `make onnx-check` (no model: reports jar presence);
  `COURSIER_CACHE=/tmp/coursier make onnx-check` if the cache is read-only;
  then with a real export for 9x9 and 19x19.
  *Done when:* `make onnx-check MODEL=...` prints a policy that sums to 1.0 and a
  value in [-1, 1] for a real export, on both a 9x9 and a 19x19 model, and
  `make onnx-gtp` plays a legal game.
  *Likely to surface:* output ordering assumptions (`policyOutput=0`,
  `valueOutput=1`), input name (`input`), and whether CoreML registration
  succeeds on Apple Silicon.

- [x] **1.2 A test suite for `Evaluate.scala`.**  Done 2026-09-11:
  `EvaluateTest.scala` (67 checks) + `make test-eval`, in `SUITES`.
  Asserts plane count/shape, plane 0 as side-to-move stones, history slots
  (including passes), `toMove` value symmetry, komi-aware value
  (`HeuristicEval.influenceValue` now subtracts komi — behaviour change, was
  komi-blind), `toNchw` layout, policy sums + forced-capture top move, and
  `influenceAt` near/far/off-board.  Not yet executed here (`scala-cli`
  cannot run in this sandbox); needs one `make check` on an unrestricted
  machine.

- [ ] **1.3 A test for `Args.scala`.**  It is shared by all three entry points and
  has fiddly parsing (`--name value`, `--name=value`, `-name value`, bare
  flags, `5s`/`250ms`/`2m`/`1h`/bare-seconds durations).
  *Where:* new `ArgsTest.scala`, or fold it into `CLITest.scala`.
  *Done when:* every form parses, `ms` is not misread as seconds, and an
  unknown flag is reported rather than ignored.

- [x] **1.4 A test for the training-data reader.**  Done 2026-09-11:
  `python/test_load_selfplay.py` (13 tests, `make test-python`), executed
  here: 12 pass, 1 skipped (NumPy absent).  Round-trips header/count/policy
  sums/outcomes/rendering; truncated header/body, wrong magic, wrong policy
  size, and missing file all fail loudly.  The test caught one real wart:
  a truncated body raised bare `EOFError` from `array.fromfile`; the reader
  now wraps it in `SelfPlayFormatError`.

- [ ] **1.5 Continuous integration.**  Nothing runs the suites on a push, and the
  zero-dependency design means it is a one-line job: install `scala-cli`, run
  `make check`.
  *Where:* a new `.github/workflows/check.yml`.
  *Done when:* a push runs `make check` on Linux, where `SERVER=0` is forced.

## 2. Rules and protocol completeness

- [x] **2.1 `final_status_list` answers every status.**  Done 2026-09-11:
  `alive`/`dead`/`seki` via `LifeDeath`, `black`/`white` list their stones,
  unknown statuses are refused (were silently empty).  `GTPTest.scala` covers
  each status on a trapped-stones position and a solver-verified corner seki.
  Solved once with 2.2, as the note suggested.

- [x] **2.2 Life-and-death for scoring.**  Done 2026-09-11: new
  `LifeDeath.scala` (Benson unconditional life, territory-control rule,
  tactical proofs via `Tactics`), `BoardState.score` removes dead stones and
  counts them as prisoners, plus `deadStones`/`removeDeadStones`.  `BoardTest`
  asserts a walled-in group scores exactly as if captured by hand (B+81 area,
  B+67 territory on the fixture).  Deliberate limits, documented in the file:
  the per-rollout path proves tactically only for enclosed ≤2-liberty groups
  and counts open fights as they stand; ko-active positions decline proofs.
  Classifications were cross-checked with an independent Python port and an
  exhaustive seki solver.  Not yet executed here; needs `make check`.

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

- [x] **3.1 Reuse the search tree between moves.**  Done 2026-09-11:
  `SearchSession.advanceTo(point)` re-roots on the played child, keeping
  visits, values and RAVE stats (superko history travels inside the child's
  state; virtual loss always cancels per iteration so none is in flight;
  stale parent links are harmless — nothing walks them).  Wired into
  `CLI.scala`, `GTP.scala` and `SelfPlay.scala` (advance on every move, fresh
  session when the move was unexplored, reset on undo/load/handicap/komi).
  `SearchTest` asserts state/visits/history preservation and continued search
  legality on a fixed position.  No benchmark yet: fewer playouts for the
  same quality is expected but unmeasured — see 5.3.  The batcher now closes
  per session (`close()`), not per `runPlayouts`, so reused sessions stay
  batched.  Not yet executed here; needs `make check`.

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

- [x] **3.5 Alpha-beta tactical supplement (DESIGN.md §2.6).**  Done
  2026-09-11: new `Tactics.scala` — depth-bounded alpha-beta over the local
  region around one victim group, opted into with
  `SearchConfig(tactics = true, tacticsDepth = 8)`, which boosts a proven
  capture's prior before the search (the search still decides).  `Some(move)`
  is a proof (every in-region defender reply still loses the group); `None`
  is "not proven".  Open fights, big regions and active ko are declined, and
  every call has a node budget (100k).  `TacticsTest.scala` (20 checks):
  hanging-group proof, walled-in proof with a tightening first move, `None`
  on empty/open/ko/seki positions (seki shape found by an exhaustive solver),
  prior-boost and boosted-search integration.  One re-read bug fixed
  (`groupAt(original.head)` misfired on partial captures).  The old
  "deliberately not implemented" marker in `Search.scala` now points here.
  Not yet executed here; needs `make check`.

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

1. **Run `make check` on an unrestricted machine.**  Everything in this batch
   (1.2, 2.1, 2.2, 3.1, 3.5) is written and cross-checked but not yet
   executed: `scala-cli` cannot start in the sandbox this was built in.
   `make test-python` already passes (12 pass, 1 skipped without NumPy).
2. **1.1** — verify ONNX on an unrestricted machine.  Still the only part of
   the project with no execution evidence at all, and the thing a reader will
   try first.  Needs a real exported model (see 4.2).
3. **5.3** — the benchmark harness.  It would turn 3.1's expected win ("fewer
   playouts, same quality") and 3.5's ("captures in dozens, not thousands")
   from reasoning into measurements.
