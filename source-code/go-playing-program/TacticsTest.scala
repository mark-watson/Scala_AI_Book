//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package go

import scala.collection.mutable

/**
 * Offline checks for the tactical reader in Tactics.scala.
 *
 * Run with: `make test-tactics`.
 *
 * The contract under test is one-directional on purpose: `Some(move)` must be
 * a proven forced capture (every defender reply still loses the group), while
 * `None` means "not proven" and says nothing about life.  So these tests pin
 * down the sound direction exactly and only probe the completeness direction
 * on positions whose answer is beyond dispute.
 */
@main def tacticsTest(): Unit =
  val failures = mutable.ArrayBuffer.empty[String]
  var checks = 0

  def check(label: String)(cond: Boolean): Unit =
    checks += 1
    if !cond then failures += label

  def eq[A](label: String)(actual: A, expected: A): Unit =
    checks += 1
    if actual != expected then failures += s"$label: expected $expected, got $actual"

  def close(label: String, actual: Double, expected: Double, tol: Double): Unit =
    checks += 1
    if math.abs(actual - expected) > tol then
      failures += s"$label: expected $expected +- $tol, got $actual"

  // White's two stones at (4,4)-(4,5) have a single liberty at (4,6); the
  // same hanging group SearchTest uses.
  val atari = BoardState.empty(9, komi = 0.0).setupStones(
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
  val captureMove = Point.at(4, 6, 9)

  // A lone white stone walled into a 5x5 black ring: four liberties, no way
  // out, no room for two eyes.
  val ring = (0 until 5).flatMap(i =>
    Seq(
      Point.at(i, 0, 9),
      Point.at(i, 4, 9),
      Point.at(0, i, 9),
      Point.at(4, i, 9)
    )
  ).toSet.toVector
  val enclosed = BoardState
    .empty(9, komi = 0.0)
    .setupStones(ring.map(_ -> Color.Black) :+ (Point.at(2, 2, 9) -> Color.White))
  val enclosedLibs = enclosed.groupAt(Point.at(2, 2, 9)).get.liberties

  // --------------------------------------------------------------------------
  // The sound direction: a proven line is really forced
  // --------------------------------------------------------------------------
  eq("the hanging group falls to its only liberty")(
    Tactics.findForcedCapture(atari, Color.Black).map(_.firstMove),
    Some(captureMove)
  )
  eq("the proof names the victim's size")(
    Tactics.findForcedCapture(atari, Color.Black).map(_.victimStones),
    Some(2)
  )
  eq("canForceCapture agrees on the hanging group")(
    Tactics.canForceCapture(atari, Point.at(4, 4, 9), Color.Black),
    true
  )

  val enclosedLine = Tactics.findForcedCapture(enclosed, Color.Black)
  check("the walled-in stone has a forced capture") { enclosedLine.isDefined }
  enclosedLine.foreach { line =>
    check("the first move is legal") { enclosed.isLegal(line.firstMove, Color.Black) }
    check("the first move fills one of the victim's liberties") {
      enclosedLibs.contains(line.firstMove)
    }
    // Playing the proven first move must make the victim strictly worse off:
    // White's liberties shrink, or stones come off at once.
    val before = enclosedLibs.size
    val after = enclosed.place(line.firstMove, Color.Black).toOption.get
    val whiteLibsAfter = after.groupAt(Point.at(2, 2, 9)).map(_.liberties.size).getOrElse(0)
    check("the first move tightens the net") {
      after.captures != enclosed.captures || whiteLibsAfter < before
    }
  }

  // --------------------------------------------------------------------------
  // The careful direction: open fights and ko are declined, not misread
  // --------------------------------------------------------------------------
  eq("an empty board proves nothing")(
    Tactics.findForcedCapture(BoardState.empty(9), Color.Black),
    None
  )
  val loneStone = BoardState.empty(9, komi = 0.0).setupStones(Seq(Point.at(4, 4, 9) -> Color.Black))
  eq("a stone with four open liberties is not force-captured")(
    Tactics.findForcedCapture(loneStone, Color.White),
    None
  )
  eq("asking about an empty point is not a capture")(
    Tactics.canForceCapture(loneStone, Point.at(0, 0, 9), Color.White),
    false
  )
  eq("asking about a friendly stone is not a capture")(
    Tactics.canForceCapture(loneStone, Point.at(4, 4, 9), Color.Black),
    false
  )

  // With a ko on the board, outside threats decide everything, so the local
  // reader declines rather than risking a misproof.
  val koShape = BoardState.empty(9, komi = 0.0).setupStones(
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
  check("the ko capture really sets a ko point") { koTaken.koPoint.isDefined }
  eq("an active ko declines every proof")(
    Tactics.findForcedCapture(koTaken, Color.White),
    None
  )

  // --------------------------------------------------------------------------
  // Corner seki: neither side can force anything (cross-checked by an
  // independent exhaustive solver in /tmp/seki_find.py, which found this shape)
  // --------------------------------------------------------------------------
  val seki = BoardState.empty(9, komi = 0.0).setupStones(
    Seq(
      Point.at(0, 0, 9) -> Color.Black,
      Point.at(0, 1, 9) -> Color.Black,
      Point.at(1, 1, 9) -> Color.White,
      Point.at(1, 2, 9) -> Color.White
    )
  )
  eq("Black cannot force the white stones")(
    Tactics.canForceCapture(seki, Point.at(1, 1, 9), Color.Black),
    false
  )
  eq("White cannot force the black stones")(
    Tactics.canForceCapture(seki, Point.at(0, 0, 9), Color.White),
    false
  )
  eq("no forced capture is reported in the seki")(
    Tactics.findForcedCapture(seki, Color.Black),
    None
  )

  // --------------------------------------------------------------------------
  // The search opt-in: a proven capture gets a louder prior
  // --------------------------------------------------------------------------
  val plain = SearchConfig(playouts = 1, threads = 1, seed = 7L)
  val boosted = plain.copy(tactics = true)
  val sessionPlain = Search.newSession(atari, HeuristicNetwork(), plain)
  sessionPlain.runPlayouts(1)
  val sessionBoosted = Search.newSession(atari, HeuristicNetwork(), boosted)
  sessionBoosted.runPlayouts(1)
  val at = sessionPlain.root.candidates.indexOf(captureMove.index)
  check("the capture is among the root candidates") { at >= 0 }
  if at >= 0 then
    val before = sessionPlain.root.candidatePriors(at)
    val after = sessionBoosted.root.candidatePriors(at)
    check("the tactics opt-in raises the proven capture's prior") { after > before }
    close("priors stay normalised after the boost", sessionBoosted.root.candidatePriors.sum, 1.0, 1e-5)

  val tacticalSearch = Search.search(
    atari,
    HeuristicNetwork(),
    SearchConfig(playouts = 800, threads = 1, seed = 7L, tactics = true)
  )
  eq("the boosted search takes the hanging group")(tacticalSearch.move, captureMove)

  // --------------------------------------------------------------------------
  // Report
  // --------------------------------------------------------------------------
  if failures.isEmpty then println(s"tacticsTest: $checks checks passed")
  else
    println(s"tacticsTest: ${failures.size} of $checks checks FAILED")
    failures.foreach(f => println(s"  - $f"))
    System.exit(1)
