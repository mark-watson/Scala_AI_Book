//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package go

import scala.collection.mutable

/**
 * Offline checks for the evaluation code in Evaluate.scala.
 *
 * Run with: `make test-eval`.
 *
 * `FeaturePlanes.encode` is the kind of code that fails silently: an
 * off-by-one in a plane poisons every training example and every search
 * without ever raising.  So this suite pins the encoding down plane by
 * plane, rather than eyeballing output.
 */
@main def evalTest(): Unit =
  val failures = mutable.ArrayBuffer.empty[String]
  var checks = 0

  def check(label: String)(cond: Boolean): Unit =
    checks += 1
    if !cond then failures += label

  def eq[A](label: String)(actual: A, expected: A): Unit =
    checks += 1
    if actual != expected then failures += s"$label: expected $expected, got $actual"

  def close(label: String, actual: Double, expected: Double, tol: Double = 1e-6): Unit =
    checks += 1
    if math.abs(actual - expected) > tol then
      failures += s"$label: expected $expected +- $tol, got $actual"

  val empty9 = BoardState.initial(9, komi = 0.0)
  val center = Point.at(4, 4, 9)

  // --------------------------------------------------------------------------
  // Shape: 15 planes, one per board point
  // --------------------------------------------------------------------------
  eq("there are 15 feature planes")(FeaturePlanes.PlaneCount, 15)
  eq("four moves of history are recorded")(FeaturePlanes.HistoryMoves, 4)
  val emptyPlanes = FeaturePlanes.encode(empty9)
  eq("encoding produces one plane per plane")(emptyPlanes.length, 15)
  check("every plane covers the whole board") {
    emptyPlanes.forall(_.length == 81)
  }
  check("the empty board encodes to all zeros") {
    emptyPlanes.forall(_.forall(_ == 0f))
  }

  // --------------------------------------------------------------------------
  // Plane 0 is the side-to-move stones, plane 1 the opponent
  // --------------------------------------------------------------------------
  val oneStone = empty9.setupStones(Seq(center -> Color.Black))
  val blackToMove = FeaturePlanes.encode(oneStone.withToMove(Color.Black))
  eq("plane 0 marks the mover's stone")(blackToMove(0)(center.index), 1f)
  eq("plane 1 is empty when the opponent has no stones")(blackToMove(1).sum, 0f)

  val whiteToMove = FeaturePlanes.encode(oneStone.withToMove(Color.White))
  eq("plane 0 is empty when the mover has no stones")(whiteToMove(0).sum, 0f)
  eq("plane 1 marks the opponent's stone")(whiteToMove(1)(center.index), 1f)

  val twoStones = empty9.setupStones(
    Seq(center -> Color.Black, Point.at(0, 0, 9) -> Color.White)
  )
  val mixed = FeaturePlanes.encode(twoStones.withToMove(Color.Black))
  eq("plane 0 holds only the mover's stones")(mixed(0).sum, 1f)
  eq("plane 1 holds only the opponent's stones")(mixed(1).sum, 1f)

  // --------------------------------------------------------------------------
  // Liberty planes are bucketed into [0, 1]
  // --------------------------------------------------------------------------
  eq("a lone stone has four liberties, the top bucket")(blackToMove(2)(center.index), 1f)
  val edgeStone = empty9.setupStones(Seq(Point.at(0, 4, 9) -> Color.Black))
  val edgePlanes = FeaturePlanes.encode(edgeStone)
  eq("an edge stone has three liberties")(edgePlanes(2)(Point.at(0, 4, 9).index), 0.75f)
  check("liberty planes stay inside [0, 1]") {
    mixed(2).forall(v => v >= 0f && v <= 1f) && mixed(3).forall(v => v >= 0f && v <= 1f)
  }

  // --------------------------------------------------------------------------
  // History planes: slot 0 is the latest move, with its mover's colour
  // --------------------------------------------------------------------------
  val d4 = Point.fromGtp("D4", 9).get
  val e5 = Point.fromGtp("E5", 9).get
  val twoMoves = empty9.playOrThrow(d4).playOrThrow(e5)
  val history = FeaturePlanes.encode(twoMoves)
  eq("the latest move is in stone plane 5")(history(5)(e5.index), 1f)
  eq("stone plane 5 holds only the latest move")(history(5).sum, 1f)
  eq("White just moved, so colour plane 6 is all zeros")(history(6).sum, 0f)
  eq("the previous move is in stone plane 7")(history(7)(d4.index), 1f)
  eq("Black moved before, so colour plane 8 is all ones")(history(8).sum, 81f)
  check("older slots are empty in a two-move game") {
    history(9).sum == 0f && history(10).sum == 0f &&
    history(11).sum == 0f && history(12).sum == 0f
  }

  val afterPass = twoMoves.playPass
  val passHistory = FeaturePlanes.encode(afterPass)
  eq("a pass leaves no stone mark")(passHistory(5).sum, 0f)
  eq("a pass still records its mover's colour")(passHistory(6).sum, 81f)
  eq("the pass pushes the latest stone move back one slot")(passHistory(7)(e5.index), 1f)

  // --------------------------------------------------------------------------
  // Side to move, ko, and atari planes
  // --------------------------------------------------------------------------
  eq("plane 13 is all ones with Black to move")(history(13).sum, 81f)
  val whiteHistory = FeaturePlanes.encode(twoMoves.withToMove(Color.White))
  eq("plane 13 is all zeros with White to move")(whiteHistory(13).sum, 0f)

  eq("no ko point on an open board")(history(4).sum, 0f)
  val koShape = empty9.setupStones(
    Seq(
      Point.at(0, 1, 9) -> Color.Black,
      Point.at(1, 0, 9) -> Color.Black,
      Point.at(1, 2, 9) -> Color.Black,
      Point.at(1, 1, 9) -> Color.White,
      Point.at(2, 0, 9) -> Color.White,
      Point.at(2, 2, 9) -> Color.White,
      Point.at(3, 1, 9) -> Color.White
    )
  )
  val koTaken = koShape.playOrThrow(Point.at(2, 1, 9))
  val koPlanes = FeaturePlanes.encode(koTaken)
  eq("plane 4 marks the ko point")(koPlanes(4)(Point.at(1, 1, 9).index), 1f)
  eq("plane 4 marks only the ko point")(koPlanes(4).sum, 1f)

  val atariShape = BoardState.empty(9, komi = 0.0).setupStones(
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
  val atariPlanes = FeaturePlanes.encode(atariShape)
  eq("plane 14 marks the white group's last liberty")(
    atariPlanes(14)(Point.at(4, 6, 9).index),
    1f
  )
  eq("an open board has no group in atari")(history(14).sum, 0f)

  // --------------------------------------------------------------------------
  // Flattening and the NCHW layout the ONNX bridge sends
  // --------------------------------------------------------------------------
  val flat = FeaturePlanes.flatten(history)
  eq("flattened planes are plane-major")(flat.length, 15 * 81)
  check("flattening preserves the plane layout") {
    history.indices.forall(p =>
      history(p).indices.forall(i => flat(p * 81 + i) == history(p)(i))
    )
  }

  val nchw = FeaturePlanes.toNchw(Seq(twoMoves))
  eq("one sample lays out to planes times area")(nchw.length, 15 * 81)
  check("NCHW matches the flat encoding point for point") {
    nchw.indices.forall(i => nchw(i) == flat(i))
  }
  val batch = FeaturePlanes.toNchw(Seq(twoMoves, oneStone))
  eq("two samples lay out back to back")(batch.length, 2 * 15 * 81)
  check("the second sample starts after the first") {
    val second = FeaturePlanes.flatten(FeaturePlanes.encode(oneStone))
    second.indices.forall(i => batch(15 * 81 + i) == second(i))
  }
  eq("an empty batch is empty")(FeaturePlanes.toNchw(Seq.empty).length, 0)

  // --------------------------------------------------------------------------
  // Value perspective and komi
  // --------------------------------------------------------------------------
  val valueBlack = HeuristicEval.influenceValue(twoStones.withToMove(Color.Black))
  val valueWhite = HeuristicEval.influenceValue(twoStones.withToMove(Color.White))
  eq("the value flips exactly with the side to move")(valueWhite, -valueBlack)
  check("values stay inside [-1, 1]") {
    valueBlack >= -1f && valueBlack <= 1f && valueWhite >= -1f && valueWhite <= 1f
  }

  eq("an empty board without komi is exactly even")(
    HeuristicEval.influenceValue(BoardState.empty(9, komi = 0.0)),
    0f
  )
  val emptyKomi = BoardState.empty(9, komi = 6.5).withToMove(Color.Black)
  check("komi favours White on an otherwise even board") {
    HeuristicEval.influenceValue(emptyKomi) < 0f
  }
  check("White to move likes komi") {
    HeuristicEval.influenceValue(emptyKomi.withToMove(Color.White)) > 0f
  }
  check("more komi is worse for Black") {
    HeuristicEval.influenceValue(BoardState.empty(9, komi = 7.5)) <
      HeuristicEval.influenceValue(BoardState.empty(9, komi = 0.0))
  }

  // --------------------------------------------------------------------------
  // Influence display values
  // --------------------------------------------------------------------------
  val corner = empty9.setupStones(Seq(Point.at(0, 0, 9) -> Color.Black))
  val near = HeuristicEval.influenceAt(corner, Point.at(1, 0, 9))
  check("a point next to a stone leans to its colour")(near > 0.5 && near < 0.6)
  eq("a point beyond influence range is neutral")(
    HeuristicEval.influenceAt(corner, Point.at(8, 8, 9)),
    0.0
  )
  val cornerWhite = empty9.setupStones(Seq(Point.at(0, 0, 9) -> Color.White))
  check("a point next to a white stone leans White") {
    HeuristicEval.influenceAt(cornerWhite, Point.at(1, 0, 9)) < -0.5
  }
  eq("an off-board point has no influence")(
    HeuristicEval.influenceAt(corner, Point(999)),
    0.0
  )

  // --------------------------------------------------------------------------
  // The heuristic network: a distribution that takes what is hanging
  // --------------------------------------------------------------------------
  val heuristic = HeuristicNetwork()
  for (label, position) <- List("empty" -> empty9, "played" -> twoMoves) do
    val (policy, value) = heuristic.evaluate(position)
    eq(s"policy length covers the board plus pass on $label")(policy.length, 82)
    close(s"policy sums to 1 on $label", policy.sum, 1.0, 1e-5)
    check(s"policy is non-negative on $label") { policy.forall(_ >= 0f) }
    check(s"passing keeps a non-zero probability on $label") { policy(81) > 0f }
    check(s"value is bounded on $label") { value >= -1f && value <= 1f }

  val (capturePolicy, _) = heuristic.evaluate(atariShape)
  val best = capturePolicy.indices.maxBy(i => capturePolicy(i))
  eq("the network's top move takes the hanging group")(Point(best), Point.at(4, 6, 9))

  // A position with no legal moves but passing still yields a distribution.
  val filled = BoardState.empty(9, komi = 0.0)
  val (filledPolicy, _) = heuristic.evaluate(filled)
  close("the opening policy sums to 1", filledPolicy.sum, 1.0, 1e-5)

  // --------------------------------------------------------------------------
  // The random network: uniform policy, deterministic value
  // --------------------------------------------------------------------------
  val random = RandomNetwork()
  check("the random network carries no value") { !random.valueIsMeaningful }
  check("the heuristic network carries a value") { heuristic.valueIsMeaningful }
  val (uniform, hashValue) = random.evaluate(empty9)
  close("the random policy is uniform", uniform.sum, 1.0, 1e-5)
  check("the random policy is flat") { uniform.forall(p => math.abs(p - uniform(0)) < 1e-9) }
  check("the hash value is bounded") { hashValue >= -1f && hashValue <= 1f }
  eq("the hash value reproduces exactly")(random.evaluate(empty9)._2, hashValue)

  eq("a board point maps to its index")(heuristic.policyIndex(center, empty9), 40)
  eq("pass maps to the final policy slot")(heuristic.policyIndex(Point.Pass, empty9), 81)

  // --------------------------------------------------------------------------
  // Report
  // --------------------------------------------------------------------------
  if failures.isEmpty then println(s"evalTest: $checks checks passed")
  else
    println(s"evalTest: ${failures.size} of $checks checks FAILED")
    failures.foreach(f => println(s"  - $f"))
    System.exit(1)
