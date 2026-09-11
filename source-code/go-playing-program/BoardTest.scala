//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package go

import scala.collection.mutable

/**
 * Offline checks for the rules engine in Board.scala.
 *
 * Run with: `make test-board`  (or `scala-cli run . --main-class go.boardTest`).
 *
 * Like the other book projects, this uses a plain `@main` method and `assert`
 * style checks rather than a test framework, so the project keeps its zero
 * runtime dependencies.  Every check is collected and reported at the end, and
 * a non-zero exit code is returned if anything failed, so `make test` is a
 * reliable gate.
 */
@main def boardTest(): Unit =
  val failures = mutable.ArrayBuffer.empty[String]
  var checks = 0

  def check(label: String)(cond: Boolean): Unit =
    checks += 1
    if !cond then failures += label

  def eq[A](label: String)(actual: A, expected: A): Unit =
    checks += 1
    if actual != expected then failures += s"$label: expected $expected, got $actual"

  // --------------------------------------------------------------------------
  // Coordinates
  // --------------------------------------------------------------------------
  eq("GTP A1 is bottom-left")(Point.fromGtp("A1", 9), Some(Point.at(0, 8, 9)))
  eq("GTP A9 is top-left")(Point.fromGtp("A9", 9), Some(Point.at(0, 0, 9)))
  eq("GTP skips the letter I")(Point.fromGtp("J9", 9), Some(Point.at(8, 0, 9)))
  eq("GTP rejects K on a 9x9 board")(Point.fromGtp("K9", 9), None)
  eq("GTP rejects row 0")(Point.fromGtp("A0", 9), None)
  eq("GTP parses pass")(Point.fromGtp("pass", 9), Some(Point.Pass))
  eq("SGF lower left is a9")(Point.fromSgf("ai", 9), Some(Point.at(0, 8, 9)))
  eq("SGF empty string is pass")(Point.fromSgf("", 9), Some(Point.Pass))

  val allPoints = (0 until 9 * 9).map(Point(_))
  check("GTP round-trips every point") {
    allPoints.forall(p => Point.fromGtp(p.toGtp(9), 9).contains(p))
  }
  check("SGF round-trips every point") {
    allPoints.forall(p => Point.fromSgf(p.toSgf(9), 9).contains(p))
  }

  // --------------------------------------------------------------------------
  // Legal move generation
  // --------------------------------------------------------------------------
  val empty9 = BoardState.initial(9, komi = 0.0)
  eq("empty 9x9 has 81 legal moves")(empty9.legalMoves(Color.Black).size, 81)
  check("empty 9x9 includes pass only when asked") {
    empty9.legalMoves(Color.Black).forall(!_.isPass) &&
    empty9.legalMoves(Color.Black, includePass = true).exists(_.isPass)
  }
  check("star points are present on 9x9") {
    val stars = empty9.starPoints
    stars.contains(Point.at(2, 2, 9)) && stars.contains(Point.at(4, 4, 9)) && stars.size == 5
  }

  // --------------------------------------------------------------------------
  // Capture
  // --------------------------------------------------------------------------
  val surround = empty9.setupStones(
    Seq(
      Point.at(3, 4, 9) -> Color.Black,
      Point.at(5, 4, 9) -> Color.Black,
      Point.at(4, 3, 9) -> Color.Black,
      Point.at(4, 4, 9) -> Color.White
    )
  )
  eq("surrounded stone is in atari")(surround.groupAt(Point.at(4, 4, 9)).map(_.inAtari), Some(true))
  val captured = surround.play(Point.at(4, 5, 9)).toOption.get
  eq("capture removes the stone")(captured.stoneAt(Point.at(4, 4, 9)), Color.Empty)
  eq("capture is recorded for Black")(captured.captures, (1, 0))
  eq("capturing stone survives")(captured.stoneAt(Point.at(4, 5, 9)), Color.Black)

  // A two-stone group is captured together.
  val groupCapture = empty9
    .setupStones(
      Seq(
        Point.at(3, 4, 9) -> Color.White,
        Point.at(4, 4, 9) -> Color.White,
        Point.at(3, 3, 9) -> Color.Black,
        Point.at(4, 3, 9) -> Color.Black,
        Point.at(3, 5, 9) -> Color.Black,
        Point.at(4, 5, 9) -> Color.Black,
        Point.at(2, 4, 9) -> Color.Black
      )
    )
    .play(Point.at(5, 4, 9))
    .toOption
    .get
  eq("both stones of the group are captured")(groupCapture.captures, (2, 0))
  eq("the second captured stone is gone")(groupCapture.stoneAt(Point.at(4, 4, 9)), Color.Empty)

  // --------------------------------------------------------------------------
  // Suicide
  // --------------------------------------------------------------------------
  // Black at B9 and A8 seals the corner; White filling A9 would have no
  // liberties and captures nothing, so the move is illegal.
  val suicideShape = empty9.setupStones(
    Seq(Point.at(1, 0, 9) -> Color.Black, Point.at(0, 1, 9) -> Color.Black)
  )
  check("suicide is illegal")(suicideShape.place(Point.at(0, 0, 9), Color.White).isLeft)
  check("suicide is reported as such") {
    suicideShape.place(Point.at(0, 0, 9), Color.White).left.exists(_.contains("suicide"))
  }
  check("filling a liberty is legal for the enclosing colour") {
    suicideShape.place(Point.at(0, 0, 9), Color.Black).isRight
  }
  check("playing on an occupied point is illegal") {
    surround.play(Point.at(4, 4, 9)).isLeft
  }
  check("playing off the board is illegal") {
    empty9.isLegal(Point(999), Color.Black) == false
  }
  check("pass is always legal")(empty9.isLegal(Point.Pass, Color.White))

  // --------------------------------------------------------------------------
  // Ko
  // --------------------------------------------------------------------------
  // The textbook ko shape, drawn at the top of the board (y=0 is the top row,
  // so (1,1) is the GTP point B8).  White's stone at B8 has a single liberty at
  // C8; Black takes it, and the capturing stone is itself in atari.
  //     . X O .
  //     X O . O     <- Black to play at the marked point
  //     . X O .
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
  check("the ko stone starts in atari") {
    koShape.groupAt(Point.at(1, 1, 9)).map(_.inAtari).contains(true)
  }
  val koTaken = koShape.playOrThrow(Point.at(2, 1, 9))
  eq("taking the ko captures one stone")(koTaken.captures, (1, 0))
  eq("the ko point is recorded")(koTaken.koPoint, Some(Point.at(1, 1, 9)))
  check("immediate recapture is illegal") {
    koTaken.play(Point.at(1, 1, 9)).isLeft
  }

  // After a ko threat and the answer, the recapture becomes legal.
  val afterThreat = koTaken
    .playOrThrow(Point.at(7, 7, 9)) // White plays a threat elsewhere
    .playOrThrow(Point.at(8, 8, 9)) // Black answers
  val koRetaken = afterThreat.play(Point.at(1, 1, 9))
  check("recapture is legal once the position has moved on")(koRetaken.isRight)
  eq("recapture removes the capturing stone")(
    koRetaken.toOption.map(_.stoneAt(Point.at(2, 1, 9))),
    Some(Color.Empty)
  )

  // --------------------------------------------------------------------------
  // Zobrist hashing
  // --------------------------------------------------------------------------
  def recomputeHash(s: BoardState): Long =
    var h = 0L
    var i = 0
    while i < s.area do
      val c = Color.fromByte(s.grid(i))
      if c.isStone then h ^= Zobrist.stone(c, Point(i))
      i += 1
    h

  eq("empty board hash is zero")(empty9.zobristHash, 0L)
  check("incremental hash matches a full recomputation after a capture") {
    recomputeHash(captured) == captured.zobristHash
  }
  check("incremental hash tracks groups of stones") {
    recomputeHash(groupCapture) == groupCapture.zobristHash
  }
  check("setupStones maintains the hash") {
    recomputeHash(koShape) == koShape.zobristHash
  }
  check("position history records distinct positions") {
    koTaken.positionHistory.size > koShape.positionHistory.size
  }

  // --------------------------------------------------------------------------
  // Passing and game end
  // --------------------------------------------------------------------------
  val afterPass = empty9.playPass
  check("one pass does not end the game")(!afterPass.isTerminal)
  check("two passes end the game")(afterPass.playPass.isTerminal)
  eq("white moves after black passes")(afterPass.toMove, Color.White)
  eq("a stone move resets the pass counter")(
    empty9.playPass.playOrThrow(Point.at(4, 4, 9)).passes,
    0
  )

  // --------------------------------------------------------------------------
  // Scoring
  // --------------------------------------------------------------------------
  // Two facing walls.  The left strip is bordered only by Black, the right
  // strip only by White, and the middle strip touches both colours, so it is
  // dame and belongs to neither side.
  //
  //   x: 0 1 | 2 | 3 4 5 | 6 | 7 8
  //          | B | dame  | W |
  val walls = BoardState.empty(9, komi = 0.0).setupStones(
    (0 until 9).flatMap(y =>
      Seq(Point.at(2, y, 9) -> Color.Black, Point.at(6, y, 9) -> Color.White)
    )
  )
  val wallScore = walls.score
  // 9 stones + 18 points of territory on each side, with no komi: a draw.
  eq("area scoring counts stones plus territory")(wallScore.black, 27.0)
  eq("area scoring is symmetric here")(wallScore.white, 27.0)
  eq("a balanced position without komi is a draw")(wallScore.winner, Color.Empty)
  eq("black owns the left strip")(walls.territory(Point.at(0, 0, 9).index), 1.toByte)
  eq("white owns the right strip")(walls.territory(Point.at(8, 8, 9).index), 2.toByte)
  eq("the middle strip is dame")(walls.territory(Point.at(4, 4, 9).index), 0.toByte)

  val komiWall = walls.copyWithKomi(6.5)
  eq("komi decides a balanced position")(komiWall.score.winner, Color.White)
  eq("result string reflects the margin")(komiWall.resultString, "W+6.5")

  // Capturing a stone in the dame strip leaves the strip's ownership alone but
  // creates a one-point eye: the captured point is surrounded entirely by the
  // four new Black stones, so it becomes Black territory.  Chinese and
  // Japanese rules therefore disagree only by the prisoners, as expected:
  //   area      = 13 stones + 19 territory        = 32 vs 27  -> +5
  //   territory = 19 territory + 1 prisoner       = 20 vs 18  -> +2
  val withPrisoner = BoardState
    .empty(9, komi = 0.0)
    .setupStones(
      (0 until 9).flatMap(y =>
        Seq(Point.at(2, y, 9) -> Color.Black, Point.at(6, y, 9) -> Color.White)
      ) :+ (Point.at(4, 4, 9) -> Color.White)
    )
  val prisonersGame = Seq(
    Point.at(3, 4, 9),
    Point.at(4, 3, 9),
    Point.at(4, 5, 9),
    Point.at(5, 4, 9)
  ).foldLeft(withPrisoner)((s, p) => s.placeOrThrow(p, Color.Black))
  eq("the surrounded white stone is captured")(prisonersGame.captures, (1, 0))
  eq("the captured stone becomes an eye")(prisonersGame.territory(Point.at(4, 4, 9).index), 1.toByte)
  eq("area scoring counts the four new stones and the eye")(prisonersGame.score.diff, 5.0)
  eq("territory scoring counts the eye and the prisoner")(
    prisonersGame.copyWithRules(RuleSet.Territory).score.diff,
    2.0
  )

  // Resignation short-circuits the count.
  val resign = empty9.resign(Color.White)
  eq("the opponent of the resigner wins")(resign.winner, Some(Color.Black))
  eq("a resignation is reported clearly")(resign.resultString, "B+Resign")

  // --------------------------------------------------------------------------
  // Life and death: dead stones do not own points
  // --------------------------------------------------------------------------
  // A solid 5x5 block with two separate one-point eyes is unconditionally
  // alive: nothing the opponent does can touch it.
  val twoEyed = BoardState
    .empty(9, komi = 0.0)
    .setupStones(
      (for
        x <- 2 to 6
        y <- 2 to 6
        if (x, y) != (3, 3) && (x, y) != (5, 5)
      yield Point.at(x, y, 9) -> Color.Black).toVector
    )
  check("the two-eyed block is alive") {
    LifeDeath.aliveGroups(twoEyed).exists(_.stones.contains(Point.at(2, 2, 9)))
  }
  eq("nothing is dead around two eyes")(twoEyed.deadStones, Set.empty[Point])
  check("removing nothing returns the same position") { twoEyed.removeDeadStones eq twoEyed }

  // Two white stones walled into a 5x5 black ring: five liberties, no way
  // out, no room for two eyes.  They are dead, and the score must say so.
  val ringPoints = (0 until 5)
    .flatMap(i => Seq(Point.at(i, 0, 9), Point.at(i, 4, 9), Point.at(0, i, 9), Point.at(4, i, 9)))
    .toSet
  val whiteA = Point.at(2, 2, 9)
  val whiteB = Point.at(2, 3, 9)
  val trapped = BoardState
    .empty(9, komi = 0.0)
    .setupStones((ringPoints.map(_ -> Color.Black) + (whiteA -> Color.White) + (whiteB -> Color.White)).toVector)
  eq("the trapped stones share five liberties")(
    trapped.groupAt(whiteA).map(_.liberties.size),
    Some(5)
  )
  eq("both trapped stones are dead")(
    trapped.deadStones,
    Set(whiteA, whiteB)
  )
  check("none of the wall is dead") {
    trapped.deadStones.forall(p => trapped.stoneAt(p) == Color.White)
  }
  val cleared = trapped.removeDeadStones
  eq("removal empties the trapped points")(cleared.stoneAt(whiteA), Color.Empty)
  eq("removal empties the second trapped point")(cleared.stoneAt(whiteB), Color.Empty)
  eq("removed stones count as prisoners")(cleared.captures, (2, 0))
  // Fill the five liberties by hand: Black captures, and the count after a
  // real capture is what the life-and-death score must equal.
  val filled = trapped
    .groupAt(whiteA)
    .get
    .liberties
    .toVector
    .foldLeft(trapped)((s, p) => s.place(p, Color.Black).fold(_ => s, identity))
  eq("filling every liberty captures the group")(filled.captures, (2, 0))
  eq("the score counts the trapped stones as captured")(trapped.score, filled.score)
  eq("removal scores the same as capturing")(cleared.score, filled.score)
  // Concrete counts: 16 wall stones plus 65 points of owned emptiness (9
  // inside plus 56 outside the 5x5 box) under area rules; under territory
  // rules the 65 points plus 2 prisoners.
  // (Filling the liberties by hand instead would fill 5 points of Black's own
  // territory, so territory scoring rightly differs there: dame is not free.)
  eq("area counts the wall and the whole corner")(trapped.score, Score(81.0, 0.0))
  eq("territory counts the corner plus 2 prisoners")(
    trapped.copyWithRules(RuleSet.Territory).score,
    Score(67.0, 0.0)
  )

  // --------------------------------------------------------------------------
  // Rendering
  // --------------------------------------------------------------------------
  val rendered = koShape.toAscii()
  check("rendering shows both colours")(rendered.contains("X") && rendered.contains("O"))
  eq("rendering has a line per board row plus headers")(
    rendered.linesIterator.size,
    11
  )

  // --------------------------------------------------------------------------
  // Report
  // --------------------------------------------------------------------------
  if failures.isEmpty then println(s"boardTest: $checks checks passed")
  else
    println(s"boardTest: ${failures.size} of $checks checks FAILED")
    failures.foreach(f => println(s"  - $f"))
    System.exit(1)
