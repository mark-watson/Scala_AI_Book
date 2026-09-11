//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package go

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import scala.collection.mutable

/**
 * Offline checks for the GTP session in GTP.scala.
 *
 * Run with: `make test-gtp`.
 *
 * These drive [[GtpEngine]] directly, so no stdin or stdout is involved and
 * every command can be checked including its failure path.
 */
@main def gtpTest(): Unit =
  val failures = mutable.ArrayBuffer.empty[String]
  var checks = 0

  def check(label: String)(cond: Boolean): Unit =
    checks += 1
    if !cond then failures += label

  def eq[A](label: String)(actual: A, expected: A): Unit =
    checks += 1
    if actual != expected then failures += s"$label: expected $expected, got $actual"

  def ok(label: String, response: Either[String, String]): String =
    checks += 1
    response match
      case Right(body) => body
      case Left(error) =>
        failures += s"$label failed: $error"
        ""

  def refuses(label: String, response: Either[String, String]): Unit =
    checks += 1
    if response.isRight then failures += s"$label should have been refused, got ${response}"

  val engine = new GtpEngine(config = SearchConfig(playouts = 60, threads = 2, seed = 7L))

  // --------------------------------------------------------------------------
  // Identification and protocol commands
  // --------------------------------------------------------------------------
  eq("protocol_version is 2")(ok("protocol_version", engine.handle("protocol_version")), "2")
  eq("name is reported")(ok("name", engine.handle("name")), "GoPlayingProgram")
  check("version is reported") { ok("version", engine.handle("version")).nonEmpty }
  eq("known_command knows genmove")(ok("known", engine.handle("known_command genmove")), "true")
  eq("known_command rejects nonsense")(ok("known", engine.handle("known_command frobnicate")), "false")
  check("list_commands advertises genmove") { ok("list", engine.handle("list_commands")).contains("genmove") }
  check("list_commands advertises boardsize") { engine.handle("list_commands").toOption.get.contains("boardsize") }
  // A leading command id is stripped by the engine and echoed by the front end.
  eq("a numeric command id is accepted")(ok("id", engine.handle("1 protocol_version")), "2")
  eq("kgs-game_over is a no-op")(ok("kgs", engine.handle("kgs-game_over")), "")
  refuses("an unknown command is refused", engine.handle("frobnicate"))

  // --------------------------------------------------------------------------
  // Board setup
  // --------------------------------------------------------------------------
  eq("boardsize 9 is accepted")(ok("boardsize", engine.handle("boardsize 9")), "")
  eq("the board is 9x9")(engine.currentState.size, 9)
  refuses("boardsize 0 is refused", engine.handle("boardsize 0"))
  refuses("boardsize 25 is refused", engine.handle("boardsize 25"))
  refuses("a non-numeric boardsize is refused", engine.handle("boardsize big"))
  eq("komi is accepted")(ok("komi", engine.handle("komi 7.5")), "")
  eq("komi reaches the position")(engine.currentState.komi, 7.5)
  refuses("a non-numeric komi is refused", engine.handle("komi abc"))
  eq("clear_board empties the board")(ok("clear", engine.handle("clear_board")).isEmpty, true)
  eq("the cleared board has no stones")(engine.currentState.moveCount, 0)

  // --------------------------------------------------------------------------
  // play
  // --------------------------------------------------------------------------
  eq("play accepts a move")(ok("play", engine.handle("play black D4")), "")
  eq("the stone is on the board")(engine.currentState.stoneAt(Point.fromGtp("D4", 9).get), Color.Black)
  eq("the turn has passed to White")(engine.currentState.toMove, Color.White)
  refuses("an unknown colour is refused", engine.handle("play green D4"))
  refuses("an invalid vertex is refused", engine.handle("play black ZZ"))
  refuses("playing on a occupied point is refused", engine.handle("play white D4"))
  eq("pass is a legal vertex")(ok("pass", engine.handle("play white pass")), "")
  eq("the turn returns to Black")(engine.currentState.toMove, Color.Black)

  // --------------------------------------------------------------------------
  // genmove
  // --------------------------------------------------------------------------
  val generated = ok("genmove", engine.handle("genmove black"))
  check("genmove returns a vertex or pass") {
    generated == "pass" || Point.fromGtp(generated, 9).isDefined
  }
  check("genmove actually plays the move") {
    Point.fromGtp(generated, 9).exists(p => engine.currentState.stoneAt(p) == Color.Black)
  }
  eq("genmove hands the turn over")(engine.currentState.toMove, Color.White)
  check("debug_search reports the last search") {
    engine.handle("debug_search").toOption.exists(_.contains("playouts="))
  }

  // --------------------------------------------------------------------------
  // undo
  // --------------------------------------------------------------------------
  val beforeUndo = engine.currentState.moveCount
  eq("undo is accepted")(ok("undo", engine.handle("undo")), "")
  eq("undo removes one move")(engine.currentState.moveCount, beforeUndo - 1)
  ok("undo twice", engine.handle("undo"))

  // --------------------------------------------------------------------------
  // showboard and scoring
  // --------------------------------------------------------------------------
  val shown = ok("showboard", engine.handle("showboard"))
  check("showboard includes the board") { shown.contains("9") }
  check("showboard lists the column letters") { shown.contains("A") && shown.contains("J") }
  check("final_score is reported") { ok("final_score", engine.handle("final_score")).nonEmpty }
  check("final_status_list alive answers") {
    engine.handle("final_status_list alive").isRight
  }

  // --------------------------------------------------------------------------
  // Handicap
  // --------------------------------------------------------------------------
  eq("clear the board")(ok("clear", engine.handle("clear_board")), "")
  val handicap = ok("fixed_handicap", engine.handle("fixed_handicap 2"))
  eq("two handicap stones are placed")(handicap.split("\\s+").length, 2)
  eq("the first handicap stone is the upper right")(handicap.split("\\s+")(0), "G7")
  eq("the second handicap stone is the lower left")(handicap.split("\\s+")(1), "C3")
  eq("handicap games are White to move")(engine.currentState.toMove, Color.White)
  eq("the handicap stones are Black")(
    engine.currentState.stones(Color.Black).size,
    2
  )
  refuses("an impossible handicap count is refused", engine.handle("fixed_handicap 10"))
  ok("clear", engine.handle("clear_board"))
  eq("place_free_handicap places three")(ok("pfh", engine.handle("place_free_handicap 3")).split("\\s+").length, 3)
  ok("clear", engine.handle("clear_board"))
  eq("set_free_handicap is accepted")(ok("sfh", engine.handle("set_free_handicap D4 E4")), "")
  eq("set_free_handicap places two stones")(engine.currentState.stones(Color.Black).size, 2)
  refuses("set_free_handicap rejects a bad vertex", engine.handle("set_free_handicap D4 ZZ"))

  // The standard table's quirk: five stones include the centre, six do not.
  eq("five stones include the centre")(
    StarPoints.handicapPoints(9, 5).map(_.last.toGtp(9)),
    Some("E5")
  )
  eq("six stones do not include the centre")(
    StarPoints.handicapPoints(9, 6).map(_.exists(_.toGtp(9) == "E5")),
    Some(false)
  )
  eq("nine stones use the whole star point set")(
    StarPoints.handicapPoints(9, 9).map(_.toSet.size),
    Some(9)
  )
  eq("handicap points are distinct")(
    StarPoints.handicapPoints(19, 9).map(_.toSet.size),
    Some(9)
  )

  // --------------------------------------------------------------------------
  // Life and death: final_status_list answers every status (2.1)
  // --------------------------------------------------------------------------
  // Two white stones walled into a black ring: dead, and the score says so.
  eq("clear the board")(ok("clear", engine.handle("clear_board")), "")
  eq("komi zero for exact counting")(ok("komi", engine.handle("komi 0")), "")
  for vertex <- Vector(
    "A9", "B9", "C9", "D9", "E9",
    "A5", "B5", "C5", "D5", "E5",
    "A8", "A7", "A6", "E8", "E7", "E6"
  ) do
    ok(s"wall $vertex", engine.handle(s"play black $vertex"))
  ok("trap one", engine.handle("play white C7"))
  ok("trap two", engine.handle("play white C6"))
  eq("dead lists the trapped stones")(ok("dead", engine.handle("final_status_list dead")), "C6 C7")
  eq("alive does not list the trapped stones")(ok("alive", engine.handle("final_status_list alive")), "")
  eq("nothing is seki here")(ok("seki", engine.handle("final_status_list seki")), "")
  check("black lists the wall") { ok("black", engine.handle("final_status_list black")).contains("A9") }
  eq("white lists the trapped group")(ok("white", engine.handle("final_status_list white")), "C6 C7")
  refuses("an unknown status is refused", engine.handle("final_status_list bogus"))
  eq("final_score counts the trapped stones as dead")(ok("score", engine.handle("final_score")), "B+81")

  // Corner seki: neither side can force a capture, so both groups are seki
  // and neither is dead.
  eq("clear the board")(ok("clear", engine.handle("clear_board")), "")
  ok("seki B1", engine.handle("play black A9"))
  ok("seki B2", engine.handle("play black A8"))
  ok("seki W1", engine.handle("play white B8"))
  ok("seki W2", engine.handle("play white B7"))
  val sekiBody = ok("seki", engine.handle("final_status_list seki"))
  check("seki lists the shared-life groups") {
    Vector("A9", "A8", "B8", "B7").forall(sekiBody.contains)
  }
  eq("neither seki group is dead")(ok("dead", engine.handle("final_status_list dead")), "")
  eq("neither seki group is alive")(ok("alive", engine.handle("final_status_list alive")), "")

  // --------------------------------------------------------------------------
  // Time control
  // --------------------------------------------------------------------------
  eq("time_settings is accepted")(ok("time_settings", engine.handle("time_settings 60 10 5")), "")
  refuses("time_settings needs three numbers", engine.handle("time_settings 60"))
  eq("time_left is accepted")(ok("time_left", engine.handle("time_left black 30 0")), "")
  refuses("time_left rejects an unknown colour", engine.handle("time_left purple 30"))
  refuses("time_left needs a colour", engine.handle("time_left"))
  // With very little time left, genmove must still return a legal move fast.
  ok("give Black almost no time", engine.handle("time_left black 1 0"))
  val quick = ok("genmove under time pressure", engine.handle("genmove black"))
  check("a move is produced under time pressure") {
    quick == "pass" || Point.fromGtp(quick, 9).isDefined
  }

  // --------------------------------------------------------------------------
  // loadsgf
  // --------------------------------------------------------------------------
  val sgfFile: Path = Files.createTempFile("go-gtp-test", ".sgf")
  try
    val record = BoardState
      .fromMoves(
        Vector(
          (Color.Black, Point.fromGtp("D4", 9).get),
          (Color.White, Point.fromGtp("E5", 9).get),
          (Color.Black, Point.fromGtp("C3", 9).get)
        ),
        9,
        6.5
      )
    Files.writeString(sgfFile, SGF.write(GameRecord.fromBoard(record)), StandardCharsets.UTF_8)
    eq("loadsgf is accepted")(ok("loadsgf", engine.handle(s"loadsgf $sgfFile")), "")
    eq("loadsgf restores the board size")(engine.currentState.size, 9)
    eq("loadsgf restores the moves")(engine.currentState.moveCount, 3)
    eq("loadsgf restores the stones")(
      engine.currentState.stoneAt(Point.fromGtp("D4", 9).get),
      Color.Black
    )
    refuses("loadsgf rejects a missing file", engine.handle("loadsgf /nonexistent/file.sgf"))
    refuses("loadsgf requires a filename", engine.handle("loadsgf"))
  finally Files.deleteIfExists(sgfFile)

  // --------------------------------------------------------------------------
  // A short engine-vs-engine game over GTP, to prove the session is usable
  // --------------------------------------------------------------------------
  ok("new board", engine.handle("boardsize 9"))
  var moveCount = 0
  var passes = 0
  var guard = 0
  // The engine passes only once every remaining point is an own-eye fill or is
  // illegal, so a 9x9 game needs a generous cap before the board fills up.
  while !engine.currentState.isTerminal && guard < 300 do
    // Deliberately not counted as a check: the length of a self-play game
    // varies from run to run, and counting each move would make the suite's
    // total drift.  The per-command behaviour is covered above.
    val vertex = engine.handle("genmove " + (if guard % 2 == 0 then "black" else "white")) match
      case Right(v) => v
      case Left(error) =>
        failures += s"genmove failed mid-game: $error"
        ""
    if vertex == "pass" then passes += 1
    moveCount += 1
    guard += 1
  check("the GTP game terminated") { engine.currentState.isTerminal }
  check("the GTP game passed at least twice") { passes >= 2 }
  check("the GTP game produced moves") { moveCount > 4 }
  check("the GTP game has a result") { engine.currentState.resultString.nonEmpty }

  // --------------------------------------------------------------------------
  // Report
  // --------------------------------------------------------------------------
  if failures.isEmpty then println(s"gtpTest: $checks checks passed")
  else
    println(s"gtpTest: ${failures.size} of $checks checks FAILED")
    failures.foreach(f => println(s"  - $f"))
    System.exit(1)
