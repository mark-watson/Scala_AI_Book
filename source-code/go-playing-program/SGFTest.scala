//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package go

import scala.collection.mutable

/**
 * Offline checks for the SGF reader and writer.
 *
 * Run with: `make test-sgf`.
 *
 * The interesting cases are not the happy path but the things other tools do:
 * property values containing brackets, variations that must not derail the main
 * line, setup stones, and passes written as an empty value.
 */
@main def sgfTest(): Unit =
  val failures = mutable.ArrayBuffer.empty[String]
  var checks = 0

  def check(label: String)(cond: Boolean): Unit =
    checks += 1
    if !cond then failures += label

  def eq[A](label: String)(actual: A, expected: A): Unit =
    checks += 1
    if actual != expected then failures += s"$label: expected $expected, got $actual"

  // --------------------------------------------------------------------------
  // Round trip of a complete record
  // --------------------------------------------------------------------------
  val moves: Vector[(Color, Point)] = Vector(
    (Color.Black, Point.at(2, 2, 9)),
    (Color.White, Point.at(6, 6, 9)),
    (Color.Black, Point.at(2, 6, 9)),
    (Color.White, Point.Pass),
    (Color.Black, Point.at(6, 2, 9))
  )
  val original = GameRecord(
    boardSize = 9,
    komi = 7.5,
    moves = moves,
    result = "B+2.5",
    blackName = "Alice",
    whiteName = "Bob",
    date = "2026-01-15",
    ruleSet = RuleSet.Area,
    comment = "a short test game"
  )
  val text = SGF.write(original)
  check("written SGF starts a collection") { text.startsWith("(;") }
  check("written SGF is balanced") {
    text.count(_ == '(') == text.count(_ == ')')
  }
  check("written SGF carries the komi") { text.contains("KM[7.5]") }
  check("written SGF carries the result") { text.contains("RE[B+2.5]") }

  val reread = SGF.readRecord(text)
  check("written SGF parses back") { reread.isRight }
  val record = reread.toOption.get
  eq("board size survives the round trip")(record.boardSize, 9)
  eq("komi survives the round trip")(record.komi, 7.5)
  eq("result survives the round trip")(record.result, "B+2.5")
  eq("player names survive the round trip")(record.blackName, "Alice")
  eq("the opponent name survives too")(record.whiteName, "Bob")
  eq("date survives the round trip")(record.date, "2026-01-15")
  eq("the comment survives the round trip")(record.comment, "a short test game")
  eq("the move count survives")(record.moves.length, 5)
  eq("move colours survive")(record.moves.map(_._1).toVector, moves.map(_._1))
  eq("move points survive")(record.moves.map(_._2).toVector, moves.map(_._2))
  eq("a pass is written as an empty value")(record.moves(3)._2.isPass, true)

  // Replaying the record must reproduce the same position.
  eq("replaying gives the same position")(
    record.toBoard.samePosition(BoardState.fromMoves(moves, 9, 7.5)),
    true
  )

  // --------------------------------------------------------------------------
  // The simpler read() signature from the design document
  // --------------------------------------------------------------------------
  val (parsedMoves, props) = SGF.read(text)
  eq("read() returns the moves")(parsedMoves.length, 5)
  eq("read() exposes the root properties")(props.get("SZ"), Some("9"))
  eq("read() exposes the komi as text")(props.get("KM"), Some("7.5"))

  // --------------------------------------------------------------------------
  // Escaping and tolerance
  // --------------------------------------------------------------------------
  val tricky = GameRecord(
    boardSize = 9,
    moves = Vector((Color.Black, Point.at(0, 0, 9))),
    comment = "a ] bracket and a \\ backslash"
  )
  val trickyText = SGF.write(tricky)
  check("brackets are escaped on write") { trickyText.contains("\\]") }
  eq("escaped comments survive a round trip")(
    SGF.readRecord(trickyText).toOption.map(_.comment),
    Some("a ] bracket and a \\ backslash")
  )

  // Whitespace, line breaks and a trailing game must not upset the parser.
  val messy = "  (\n ;GM[1]SZ[9]KM[6.5]\n;B[cc]\n;W[dd]\n)\n"
  eq("whitespace is tolerated")(
    SGF.readRecord(messy).toOption.map(_.moves.length),
    Some(2)
  )

  // The main line is followed; variations are skipped rather than parsed.
  val withVariations = "(;SZ[9];B[cc](;W[dd])(;W[ee];B[ff]))"
  eq("the main line follows the first variation")(
    SGF.readRecord(withVariations).toOption.map(_.moves.map(_._2).toVector),
    Some(Vector(Point.at(2, 2, 9), Point.at(3, 3, 9)))
  )

  // --------------------------------------------------------------------------
  // Handicap and setup stones
  // --------------------------------------------------------------------------
  val handicapText =
    "(;GM[1]SZ[9]KM[0.5]HA[2]AB[cc][gg];W[ee];B[ff])"
  val handicap = SGF.readRecord(handicapText).toOption.get
  eq("handicap is read")(handicap.handicap, 2)
  eq("setup stones are read")(handicap.setupStones.length, 2)
  eq("the setup stones are Black")(handicap.setupStones.map(_._2).toSet, Set(Color.Black))
  eq("handicap games start with White")(handicap.firstColor, Color.White)
  eq("the handicap board has the stones placed")(
    handicap.toBoard.stoneAt(Point.at(2, 2, 9)),
    Color.Black
  )
  eq("the handicap board is White to move")(handicap.toBoard.toMove, Color.White)
  eq("setup stones survive a write/read round trip")(
    SGF.readRecord(SGF.write(handicap)).toOption.map(_.setupStones.length),
    Some(2)
  )

  // --------------------------------------------------------------------------
  // Collections and malformed input
  // --------------------------------------------------------------------------
  val collection = "(;SZ[9];B[cc])(;SZ[9];B[dd];W[ee])"
  eq("a collection is read in full")(
    SGF.readCollection(collection).toOption.map(_.length),
    Some(2)
  )
  eq("the first game of a collection is used by readRecord")(
    SGF.readRecord(collection).toOption.map(_.moves.length),
    Some(1)
  )
  check("malformed input is reported, not thrown") {
    SGF.readRecord("not an sgf at all").isLeft
  }
  check("an unterminated value is reported") {
    SGF.readRecord("(;SZ[9];B[cc)").isLeft
  }
  check("an empty collection is reported") { SGF.readRecord("").isLeft }

  // --------------------------------------------------------------------------
  // Japanese rules naming
  // --------------------------------------------------------------------------
  val japanese = SGF.write(GameRecord(boardSize = 9, ruleSet = RuleSet.Territory, moves = Vector.empty))
  check("Japanese rules are named in the SGF") { japanese.contains("RU[Japanese]") }
  eq("Japanese rules are read back")(
    SGF.readRecord(japanese).toOption.map(_.ruleSet),
    Some(RuleSet.Territory)
  )

  // --------------------------------------------------------------------------
  // Report
  // --------------------------------------------------------------------------
  if failures.isEmpty then println(s"sgfTest: $checks checks passed")
  else
    println(s"sgfTest: ${failures.size} of $checks checks FAILED")
    failures.foreach(f => println(s"  - $f"))
    System.exit(1)
