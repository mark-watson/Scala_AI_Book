//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package go

import scala.collection.mutable
import scala.concurrent.duration.FiniteDuration

/**
 * Offline checks for the MCTS engine in Search.scala.
 *
 * Run with: `make test-search`.
 *
 * Search is stochastic and parallel, so these tests are written to be robust
 * rather than byte-exact: they assert the properties a correct engine must
 * have (legal moves, forced captures found, eyes never filled, time limits
 * respected) rather than exact visit counts.
 */
@main def searchTest(): Unit =
  val failures = mutable.ArrayBuffer.empty[String]
  var checks = 0

  def check(label: String)(cond: Boolean): Unit =
    checks += 1
    if !cond then failures += label

  def eq[A](label: String)(actual: A, expected: A): Unit =
    checks += 1
    if actual != expected then failures += s"$label: expected $expected, got $actual"

  val heuristic = HeuristicNetwork()
  val random = RandomNetwork()

  // --------------------------------------------------------------------------
  // Basic: a legal move comes back, and the policy is a distribution
  // --------------------------------------------------------------------------
  val empty9 = BoardState.initial(9, komi = 6.5)

  for (label, network) <- List("heuristic" -> heuristic, "random" -> random) do
    val result = Search.search(
      empty9,
      network,
      SearchConfig(playouts = 120, threads = 4, seed = 1L)
    )
    check(s"$label returns a legal move") {
      result.move.isPass || empty9.isLegal(result.move, Color.Black)
    }
    check(s"$label reports playouts") { result.playouts > 0 }
    check(s"$label policy sums to about 1") {
      val sum = result.policy.values.sum
      math.abs(sum - 1.0) < 1e-3
    }
    check(s"$label principal variation is not longer than the cap") {
      result.principalVariation.length <= 20
    }
    check(s"$label created search nodes") { result.nodesCreated > 1 }

  // A network with no meaningful value must fall back to random rollouts.
  check("random network falls back to rollouts") {
    val session = Search.newSession(empty9, random, SearchConfig(playouts = 1))
    session.usesRollouts
  }
  check("heuristic network uses its value head") {
    val session = Search.newSession(empty9, heuristic, SearchConfig(playouts = 1))
    !session.usesRollouts
  }

  // --------------------------------------------------------------------------
  // Tactics: a two-stone capture is forced, and must be found
  // --------------------------------------------------------------------------
  // White's two stones at (4,4)-(4,5) have a single liberty at (4,6).  Taking
  // it wins a whole group, so any sane engine plays there.
  val atari = BoardState.empty(9, komi = 6.5).setupStones(
    Seq(
      Point.at(3, 4, 9) -> Color.Black,
      Point.at(3, 5, 9) -> Color.Black,
      Point.at(5, 4, 9) -> Color.Black,
      Point.at(5, 5, 9) -> Color.Black,
      Point.at(4, 3, 9) -> Color.Black,
      Point.at(4, 4, 9) -> Color.White,
      Point.at(4, 5, 9) -> Color.White
    )
  )
  eq("the white group is in atari")(
    atari.groupAt(Point.at(4, 4, 9)).map(_.liberties.size),
    Some(1)
  )
  val captureMove = Point.at(4, 6, 9)
  check("the capture is legal") { atari.isLegal(captureMove, Color.Black) }

  val tactical = Search.search(
    atari,
    heuristic,
    SearchConfig(playouts = 1000, threads = 4, seed = 7L)
  )
  eq("search finds the forced two-stone capture")(tactical.move, captureMove)
  check("the capture also leads the visit distribution") {
    tactical.visits.getOrElse(captureMove, 0) == tactical.visits.values.max
  }
  eq("the captured group is really gone after the move")(
    atari.playOrThrow(captureMove).stoneAt(Point.at(4, 4, 9)),
    Color.Empty
  )

  // Rollouts, with no value network at all.
  //
  // This is deliberately NOT an assertion that random rollouts find the
  // capture.  Measured across seeds, a pure-rollout search on this position
  // ranks the capture anywhere from first to about fiftieth, because random
  // playouts spread their visits thin: that weakness is the whole reason the
  // design document trains a value network.  Asserting a top-three finish here
  // would be asserting a coin flip.  What can be asserted, and is asserted
  // below, is that the rollout search is legal, is reproducible when it runs on
  // one thread with a fixed seed, and does put some visits on the capture.
  val rolloutConfig = SearchConfig(playouts = 2000, threads = 1, seed = 2L)
  val tacticalRollout = Search.search(atari, random, rolloutConfig)
  val repeatRollout = Search.search(atari, random, rolloutConfig)
  check("rollout search returns a legal move") {
    tacticalRollout.move.isPass || atari.isLegal(tacticalRollout.move, Color.Black)
  }
  check("rollout search spreads its visits over the candidates") {
    tacticalRollout.visits.size > 10
  }
  check("rollout search reaches the forced capture") {
    tacticalRollout.visits.getOrElse(captureMove, 0) > 0
  }
  // A single thread with a fixed seed must reproduce exactly; this is what
  // makes the search testable at all, and it holds because each worker owns one
  // Random seeded from config.seed.
  eq("a single-threaded search with a fixed seed is reproducible")(
    repeatRollout.visits.toVector.sortBy(_._1.index),
    tacticalRollout.visits.toVector.sortBy(_._1.index)
  )
  eq("the reproducible search picks the same move")(repeatRollout.move, tacticalRollout.move)

  // --------------------------------------------------------------------------
  // A real eye must never be filled
  // --------------------------------------------------------------------------
  // Black at B9 and A8 makes A9 a genuine one-point corner eye.  Filling it
  // would be self-destructive, and it must not even be a candidate winner.
  val eyeShape = BoardState.empty(9, komi = 6.5).setupStones(
    Seq(
      Point.at(1, 0, 9) -> Color.Black,
      Point.at(0, 1, 9) -> Color.Black,
      Point.at(4, 4, 9) -> Color.White,
      Point.at(5, 5, 9) -> Color.White
    )
  )
  val eyePoint = Point.at(0, 0, 9)
  check("the eye point is legal to play, but bad") { eyeShape.isLegal(eyePoint, Color.Black) }
  val eyeResult = Search.search(
    eyeShape,
    heuristic,
    SearchConfig(playouts = 800, threads = 4, seed = 3L)
  )
  check("search does not fill its own eye")(eyeResult.move != eyePoint)
  check("the eye is not among the most visited moves") {
    val ranked = eyeResult.visits.toVector.sortBy(-_._2).take(3).map(_._1)
    !ranked.contains(eyePoint)
  }

  // --------------------------------------------------------------------------
  // Parallelism and budgets
  // --------------------------------------------------------------------------
  val parallel = Search.search(
    empty9,
    heuristic,
    SearchConfig(playouts = 400, threads = 16, seed = 5L)
  )
  check("16 virtual threads complete without losing the budget") {
    parallel.playouts >= 300 && parallel.playouts <= 400
  }
  check("parallel search still returns a legal move") {
    parallel.move.isPass || empty9.isLegal(parallel.move, Color.Black)
  }

  val single = Search.search(
    empty9,
    heuristic,
    SearchConfig(playouts = 400, threads = 1, seed = 5L)
  )
  check("single-threaded search completes its budget")(single.playouts >= 300)

  // A very short time limit must stop the search early but still return a move.
  val timed = Search.search(
    empty9,
    heuristic,
    SearchConfig(
      playouts = 100_000_000,
      threads = 4,
      timeLimit = Some(FiniteDuration(250, "ms")),
      seed = 2L
    )
  )
  check("the time limit stops the search") { timed.playouts < 100_000_000 }
  check("a stopped search still returns a legal move") {
    timed.move.isPass || empty9.isLegal(timed.move, Color.Black)
  }
  check("the time limit is respected to within a wide margin") {
    timed.elapsed.toMillis < 4000
  }

  // --------------------------------------------------------------------------
  // RAVE on and off
  // --------------------------------------------------------------------------
  val withRave = Search.search(
    atari,
    heuristic,
    SearchConfig(playouts = 400, threads = 2, raveEnabled = true, seed = 13L)
  )
  val withoutRave = Search.search(
    atari,
    heuristic,
    SearchConfig(playouts = 400, threads = 2, raveEnabled = false, seed = 13L)
  )
  check("RAVE enabled still finds the capture")(withRave.move == captureMove)
  check("RAVE disabled still finds the capture")(withoutRave.move == captureMove)

  // --------------------------------------------------------------------------
  // Terminal values
  // --------------------------------------------------------------------------
  check("a resignation is worth -1 to the resigner") {
    val resigned = empty9.resign(Color.Black).withToMove(Color.Black)
    Search.terminalValue(resigned) == -1f
  }
  check("a resignation is worth +1 to the opponent") {
    val resigned = empty9.resign(Color.Black)
    Search.terminalValueFrom(resigned, Color.White) == 1f
  }
  check("a won position scores positively for the winner") {
    // Black's wall at C leaves it the whole left strip; White's wall at H
    // leaves it only the last column, so Black is ahead by nine points.
    val won = BoardState
      .empty(9, komi = 0.0)
      .setupStones((0 until 9).flatMap(y =>
        Seq(Point.at(2, y, 9) -> Color.Black, Point.at(7, y, 9) -> Color.White)
      ))
      .playPass
      .playPass
      .withToMove(Color.Black)
    Search.terminalValue(won) > 0f
  }

  // --------------------------------------------------------------------------
  // Self-play smoke test: a complete game at low playout counts
  // --------------------------------------------------------------------------
  var state = BoardState.initial(9, komi = 6.5)
  var moves = 0
  val moveCap = 250
  while !state.isTerminal && moves < moveCap do
    val r = Search.search(
      state,
      heuristic,
      SearchConfig(playouts = 40, threads = 2, seed = moves.toLong + 100L)
    )
    state = state.place(r.move, state.toMove).fold(_ => state.playPass, identity)
    moves += 1
  check("a self-play game terminates within the move cap") { state.isTerminal }
  check("the engine eventually passes instead of filling its own eyes") {
    state.moveHistory.count(_.isPass) >= 2
  }

  // --------------------------------------------------------------------------
  // Dirichlet noise used for self-play exploration
  // --------------------------------------------------------------------------
  val rng = new java.util.Random(42L)
  val noise = Search.dirichlet(20, 0.3, rng)
  eq("Dirichlet sample has the right length")(noise.length, 20)
  check("Dirichlet sample sums to 1") { math.abs(noise.sum - 1.0) < 1e-9 }
  check("Dirichlet sample is non-negative") { noise.forall(_ >= 0.0) }
  check("Dirichlet sample is not uniform") { noise.max - noise.min > 1e-6 }

  val noisySession = Search.newSession(
    empty9,
    heuristic,
    SearchConfig(playouts = 1, dirichletWeight = 0.25, dirichletAlpha = 0.3)
  )
  noisySession.applyRootNoise(rng)
  check("root noise leaves the priors normalised") {
    val priors = noisySession.root.candidatePriors
    math.abs(priors.sum - 1.0) < 1e-3
  }

  // --------------------------------------------------------------------------
  // Time control and clocks
  // --------------------------------------------------------------------------
  val sudden = Clock(TimeControl.suddenDeath(FiniteDuration(60, "s")))
  check("sudden death is not byo-yomi") { !sudden.inByoYomi }
  check("sudden death budgets a slice of the remaining time") {
    val budget = sudden.timeForMove(Some(10))
    budget.toMillis > 0 && budget.toMillis < 60_000
  }
  val byo = Clock(TimeControl.byoYomi(FiniteDuration(2, "s"), FiniteDuration(30, "s"), 3))
  check("byo-yomi starts with periods available")(byo.periodsRemaining == 3)
  check("byo-yomi is not active while main time remains") { !byo.inByoYomi }
  byo.recordMove(FiniteDuration(3, "s")) // burns all the main time
  check("running out of main time enters byo-yomi") { byo.inByoYomi }
  val byoBudget = byo.timeForMove()
  check("byo-yomi budget is within the period") {
    byoBudget.toMillis > 0 && byoBudget.toMillis <= 30_000
  }
  check("byo-yomi clock describes itself") { byo.toString.contains("byo-yomi") }

  // --------------------------------------------------------------------------
  // Report
  // --------------------------------------------------------------------------
  if failures.isEmpty then println(s"searchTest: $checks checks passed")
  else
    println(s"searchTest: ${failures.size} of $checks checks FAILED")
    failures.foreach(f => println(s"  - $f"))
    System.exit(1)
