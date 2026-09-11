//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package go

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths}
import scala.collection.mutable

/**
 * Offline checks for self-play data generation in SelfPlay.scala.
 *
 * Run with: `make test-selfplay`.
 *
 * These use a tiny playout count so the whole suite stays fast; the shape and
 * labelling of the training data is what matters here, not the strength of the
 * games.
 */
@main def selfPlayTest(): Unit =
  val failures = mutable.ArrayBuffer.empty[String]
  var checks = 0

  def check(label: String)(cond: Boolean): Unit =
    checks += 1
    if !cond then failures += label

  def eq[A](label: String)(actual: A, expected: A): Unit =
    checks += 1
    if actual != expected then failures += s"$label: expected $expected, got $actual"

  val network = HeuristicNetwork()
  val config = SelfPlayConfig(
    boardSize = 9,
    playouts = 20,
    threads = 2,
    komi = 6.5,
    temperature = 1.0,
    tempDropMove = 4,
    dirichletWeight = 0.25,
    resignThreshold = None, // never resign, so we always get a labelled game
    seed = 4242L
  )

  val rng = new java.util.Random(4242L)
  val game = SelfPlay.generateGame(network, config, rng)

  // --------------------------------------------------------------------------
  // The game itself
  // --------------------------------------------------------------------------
  check("a self-play game produces training examples")(game.examples.nonEmpty)
  eq("there is one example per move")(game.examples.length, game.moveCount)
  check("the game ends") { game.finalBoard.isTerminal }
  check("the game has a result string") { game.result.nonEmpty }
  check("the move list and the board history agree") {
    game.finalBoard.moveCount >= game.moveCount
  }
  check("no resignation was requested") { !game.resigned }
  check("colours alternate") {
    game.moves.map(_._1).zipWithIndex.forall((c, i) => c == (if i % 2 == 0 then Color.Black else Color.White))
  }

  // --------------------------------------------------------------------------
  // Shape and normalisation of the training tensors
  // --------------------------------------------------------------------------
  val first = game.examples.head
  eq("feature planes use the documented plane count")(first.featurePlanes.length, FeaturePlanes.PlaneCount)
  eq("each plane covers the whole board")(first.featurePlanes.head.length, 81)
  eq("the policy vector covers the board plus pass")(first.mctsPolicy.length, 82)
  eq("flattened planes have the expected length")(
    first.flattened.length,
    FeaturePlanes.PlaneCount * 81
  )
  check("every policy vector is a distribution") {
    game.examples.forall(e => math.abs(e.mctsPolicy.sum - 1.0) < 1e-3)
  }
  check("every policy vector is non-negative") {
    game.examples.forall(_.mctsPolicy.forall(_ >= 0f))
  }
  check("every example records a legal move that was played") {
    game.examples.forall(e => e.move.isPass || e.move.index < 81)
  }
  check("move numbers increase") {
    game.examples.map(_.moveNumber).toVector == (0 until game.moveCount).toVector
  }

  // --------------------------------------------------------------------------
  // Labels: z must be from the perspective of the player to move
  // --------------------------------------------------------------------------
  check("outcomes are only -1, 0 or +1") {
    game.examples.forall(e => e.outcome == 1f || e.outcome == -1f || e.outcome == 0f)
  }
  game.winner match
    case Some(winner) =>
      check("the winner's positions are labelled +1") {
        game.examples.filter(_.toMove == winner).forall(_.outcome == 1f)
      }
      check("the loser's positions are labelled -1") {
        game.examples.filter(_.toMove == winner.opposite).forall(_.outcome == -1f)
      }
    case None =>
      check("a drawn game is labelled 0") { game.examples.forall(_.outcome == 0f) }

  // --------------------------------------------------------------------------
  // Binary format
  // --------------------------------------------------------------------------
  val dir: Path = Files.createTempDirectory("go-selfplay-test")
  try
    val binPath = dir.resolve("training.bin")
    SelfPlay.Writer.writeBinary(game.examples, binPath)
    check("the binary file exists") { Files.exists(binPath) }

    val planeCount = FeaturePlanes.PlaneCount
    val policySize = 82
    val floatsPerExample = SelfPlay.Writer.floatsPerExample(planeCount, 9, policySize)
    val expectedBytes = 24L + game.examples.length.toLong * floatsPerExample * 4L
    eq("the binary file has exactly the expected size")(Files.size(binPath), expectedBytes)

    // The file is little-endian by design (see SelfPlay.Writer), so it is read
    // back with an explicitly little-endian buffer rather than DataInputStream.
    val bytes = Files.readAllBytes(binPath)
    val buffer = java.nio.ByteBuffer.wrap(bytes).order(java.nio.ByteOrder.LITTLE_ENDIAN)
    eq("the magic number is right")(
      new String(bytes, 0, 4, StandardCharsets.US_ASCII),
      "GOPP"
    )
    eq("the format version is right")(buffer.getInt(4), 1)
    eq("the example count is right")(buffer.getInt(8), game.examples.length)
    eq("the plane count is right")(buffer.getInt(12), planeCount)
    eq("the board size is right")(buffer.getInt(16), 9)
    eq("the policy size is right")(buffer.getInt(20), policySize)
    // Spot-check the first example: its first plane float, and its label, which
    // is written last.  Matching the Java values proves the float encoding.
    eq("the first plane float round-trips")(buffer.getFloat(24), first.flattened(0))
    val labelOffset = 24 + (floatsPerExample - 2) * 4
    eq("the first label round-trips")(buffer.getFloat(labelOffset), first.outcome)
    eq("the first search value round-trips")(buffer.getFloat(labelOffset + 4), first.searchValue)
    // A little-endian reader must disagree with a big-endian one, which is the
    // bug this layout deliberately avoids.
    val bigEndian = java.nio.ByteBuffer.wrap(bytes).order(java.nio.ByteOrder.BIG_ENDIAN)
    check("the format is genuinely little-endian") { bigEndian.getInt(4) != 1 }

    // JSON Lines, for eyeballing and for tools that dislike binary.
    val jsonPath = dir.resolve("training.jsonl")
    SelfPlay.Writer.writeJsonLines(game.examples, jsonPath)
    val lines = Files.readAllLines(jsonPath, StandardCharsets.UTF_8)
    eq("there is one JSON line per example")(lines.size(), game.examples.length)
    check("JSON lines carry the planes") { lines.get(0).contains("\"planes\":[") }
    check("JSON lines carry the policy") { lines.get(0).contains("\"policy\":[") }
    check("JSON lines carry the outcome") { lines.get(0).contains("\"outcome\":") }
    check("JSON lines carry the played move") { lines.get(0).contains("\"played\":\"") }
  finally
    // Tidy up whatever the checks created.
    try
      Files.walk(dir).sorted(java.util.Comparator.reverseOrder()).forEach(p => Files.deleteIfExists(p))
    catch case _: Throwable => ()

  // --------------------------------------------------------------------------
  // Self-play games are reviewable: SGF round trip
  // --------------------------------------------------------------------------
  val record = game.toRecord
  eq("the record has the right board size")(record.boardSize, 9)
  val reread = SGF.readRecord(SGF.write(record))
  check("a self-play game survives an SGF round trip") { reread.isRight }
  eq("the replayed game reaches the same position")(
    reread.toOption.map(_.toBoard.samePosition(game.finalBoard)),
    Some(true)
  )

  // --------------------------------------------------------------------------
  // Greedy play at zero temperature, and the safety move cap
  // --------------------------------------------------------------------------
  val greedy = SelfPlayConfig(boardSize = 9, playouts = 20, threads = 2, temperature = 0.0, seed = 9L)
  val greedyGame = SelfPlay.generateGame(network, greedy, new java.util.Random(9L))
  check("a greedy game still produces examples")(greedyGame.examples.nonEmpty)

  eq("the default move cap is twice the board area")(
    SelfPlayConfig(boardSize = 9).effectiveMaxMoves,
    162
  )
  eq("an explicit move cap is respected")(
    SelfPlayConfig(boardSize = 9, maxMoves = 12).effectiveMaxMoves,
    12
  )

  // --------------------------------------------------------------------------
  // Report
  // --------------------------------------------------------------------------
  if failures.isEmpty then println(s"selfPlayTest: $checks checks passed")
  else
    println(s"selfPlayTest: ${failures.size} of $checks checks FAILED")
    failures.foreach(f => println(s"  - $f"))
    System.exit(1)
