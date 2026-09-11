//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package go

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

// ============================================================================
// Demo.scala -- one runnable demonstration per subsystem.
//
// The point of this file is that each part of the engine can be exercised on
// its own, without starting a game or a protocol session:
//
//   make demo-board      rules: captures, ko, scoring, rendering
//   make demo-search     MCTS: visits, principal variation, timing
//   make demo-eval       networks: feature planes, policy, influence map
//   make demo-sgf        file format: write, read, replay, round trip
//   make demo-gtp        protocol: a scripted GTP session
//   make demo-selfplay   reinforcement learning: one game's training data
//   make demo-all        all of the above
//
// Each one is deliberately chatty: when you are working on one subsystem, the
// printed numbers are usually what you want to see.
// ============================================================================

/** Small helpers so the demos read as a narrative rather than as printf spam. */
private object Demo:
  def title(text: String): Unit =
    println()
    println("=" * 72)
    println(text)
    println("=" * 72)

  def step(text: String): Unit =
    println()
    println(s"-- $text")
    println()

  def indent(text: String, prefix: String = "   "): Unit =
    text.linesIterator.foreach(line => println(prefix + line))

  def say(text: String): Unit = println(s"   $text")

// ----------------------------------------------------------------------------
// Board: rules and rendering
// ----------------------------------------------------------------------------
@main def demoBoard(): Unit =
  Demo.title("Board.scala -- board state, captures, ko, scoring")

  val empty = BoardState.initial(9, komi = 6.5)
  Demo.step("An empty 9x9 board")
  Demo.indent(empty.toAscii())
  Demo.say(s"legal moves for Black: ${empty.legalMoves(Color.Black).size}")
  Demo.say(s"area: ${empty.area} points, komi ${empty.komi}")

  // A capture.  White plays the centre point E5, Black fills its four
  // liberties one at a time, and the last Black stone takes it off the board.
  Demo.step("A capture, move by move")
  var position = BoardState.initial(9, komi = 6.5)
  val sequence = Vector(
    Color.Black -> Point.fromGtp("D5", 9).get,
    Color.White -> Point.fromGtp("E5", 9).get,
    Color.Black -> Point.fromGtp("F5", 9).get,
    Color.White -> Point.fromGtp("A1", 9).get, // White plays elsewhere
    Color.Black -> Point.fromGtp("E4", 9).get,
    Color.White -> Point.fromGtp("B1", 9).get, // White plays elsewhere again
    Color.Black -> Point.fromGtp("E6", 9).get  // the capturing move
  )
  for (colour, move) <- sequence do
    position.play(move) match
      case Right(next) =>
        position = next
        val atari = if position.groupAt(move).exists(_.inAtari) then "  (in atari)" else ""
        Demo.say(
          f"${colour.toString}%-5s ${move.toGtp(9)}%-4s captures now " +
            s"${position.captures._1} B / ${position.captures._2} W$atari"
        )
      case Left(reason) => Demo.say(s"${colour} ${move.toGtp(9)} rejected: $reason")
  Demo.say("")
  Demo.indent(position.toAscii())
  Demo.say(s"the stone at E5 is now ${position.stoneAt(Point.fromGtp("E5", 9).get)}")
  Demo.say(s"Black has captured ${position.captures._1} stone(s) in total")
  val group = position.groupAt(Point.fromGtp("D5", 9).get)
  Demo.say(s"Black's D5 group: ${group.map(_.stones.size).getOrElse(0)} stones, " +
    s"liberties ${group.map(_.liberties.size).getOrElse(0)}")

  // The textbook ko.
  Demo.step("Ko: taking, then the illegal immediate recapture")
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
  Demo.indent(koShape.toAscii())
  val koTaken = koShape.playOrThrow(Point.at(2, 1, 9))
  Demo.say(s"Black takes the ko at C8; captures ${koTaken.captures._1}, ko point ${koTaken.koPoint.map(_.toGtp(9))}")
  Demo.say(s"White's immediate recapture at B8: ${koTaken.play(Point.at(1, 1, 9)).left.getOrElse("allowed")}")
  val afterThreat = koTaken
    .playOrThrow(Point.at(7, 7, 9))
    .playOrThrow(Point.at(8, 8, 9))
  Demo.say(s"after a ko threat and the answer, the recapture is ${afterThreat.play(Point.at(1, 1, 9)).map(_ => "legal").getOrElse("illegal")}")

  // Scoring two facing walls, which is the clearest way to show area scoring.
  Demo.step("Scoring: two facing walls on a 9x9")
  val walls = BoardState.empty(9, komi = 0.0).setupStones(
    // Black wall on column C, White wall on column G.
    (0 until 9).map(y => Point.at(2, y, 9) -> Color.Black) ++
      (0 until 9).map(y => Point.at(6, y, 9) -> Color.White)
  )
  Demo.indent(walls.toAscii())
  val score = walls.score
  Demo.say(f"area score: B ${score.black}%.1f  W ${score.white}%.1f  ->  ${walls.resultString}")
  Demo.say(f"with komi 6.5: ${walls.copyWithKomi(6.5).resultString}")

  // Ending a game.
  Demo.step("Ending a game: two passes, then resignation")
  val ended = position.playPass.playPass
  Demo.say(s"two passes: terminal=${ended.isTerminal}, result ${ended.resultString}")
  Demo.say(s"resignation: ${position.resign(Color.White).resultString}")

// ----------------------------------------------------------------------------
// Search: Monte Carlo tree search
// ----------------------------------------------------------------------------
@main def demoSearch(): Unit =
  Demo.title("Search.scala -- Monte Carlo tree search")

  // White's two stones are in atari; the capture at (4,6) is forced.
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
  Demo.step("A forced two-stone capture")
  Demo.indent(atari.toAscii())
  Demo.say("White's two stones have one liberty; Black to play.")

  def report(label: String, result: SearchResult): Unit =
    Demo.step(label)
    Demo.say(f"best move: ${result.move.toGtp(9)}   playouts: ${result.playouts}   " +
      f"nodes: ${result.nodesCreated}   time: ${result.elapsed.toMillis} ms")
    val nps = if result.elapsed.toMillis > 0 then result.playouts * 1000L / result.elapsed.toMillis else 0L
    Demo.say(f"speed: $nps%,d playouts/s")
    Demo.say(f"win rate for Black: ${result.winRate}%.4f")
    Demo.say(f"principal variation: ${result.principalVariation.take(8).map(_.toGtp(9)).mkString(" ")}")
    Demo.say("")
    Demo.say("  move   visits   share   prior")
    val total = result.visits.values.sum
    for (p, n) <- result.visits.toVector.sortBy(-_._2).take(8) do
      val share = if total > 0 then n.toDouble / total * 100 else 0.0
      val prior = result.policy.getOrElse(p, 0f) * 100
      Demo.say(f"  ${p.toGtp(9)}%-5s $n%6d  $share%5.1f%%  $prior%5.1f%%")

  report(
    "Value-head search (heuristic network, 800 playouts)",
    Search.search(atari, HeuristicNetwork(), SearchConfig(playouts = 800, threads = 4, seed = 1L))
  )
  // Pure random playouts are far weaker than a value head, so this needs many
  // more playouts before the tactic emerges.  The contrast is worth seeing.
  val capture = Point.fromGtp("E3", 9).get
  val rollout = Search.search(atari, RandomNetwork(), SearchConfig(playouts = 4000, threads = 8, seed = 2L))
  report("Rollout search (random network, 4000 playouts)", rollout)
  val ranked = rollout.visits.toVector.sortBy(-_._2).map(_._1)
  Demo.say("")
  Demo.say(s"the forced capture ${capture.toGtp(9)} ranks #${ranked.indexOf(capture) + 1} of ${ranked.length} candidates")
  Demo.say("with no value head the visits are spread thin, which is exactly why the")
  Demo.say("design document trains a neural network to guide the search.")

  Demo.step("Time-limited search")
  val timed = Search.search(
    atari,
    HeuristicNetwork(),
    SearchConfig(playouts = 1_000_000, threads = 4, timeLimit = Some(scala.concurrent.duration.FiniteDuration(250, "ms")))
  )
  Demo.say(f"asked for a 250 ms budget: ${timed.playouts} playouts in ${timed.elapsed.toMillis} ms")
  Demo.say("the playout cap is not reached because the clock stops the search")

// ----------------------------------------------------------------------------
// Evaluation: feature planes and networks
// ----------------------------------------------------------------------------
@main def demoEval(): Unit =
  Demo.title("Evaluate.scala -- feature planes, policy networks, influence")

  val position = BoardState
    .fromMoves(
      Vector(
        (Color.Black, Point.fromGtp("D4", 9).get),
        (Color.White, Point.fromGtp("E5", 9).get),
        (Color.Black, Point.fromGtp("E4", 9).get),
        (Color.White, Point.fromGtp("F4", 9).get)
      ),
      9,
      6.5
    )

  Demo.step("The position")
  Demo.indent(position.toAscii())

  Demo.step("Feature planes: what the neural network sees")
  val planes = FeaturePlanes.encode(position)
  Demo.say(f"${FeaturePlanes.PlaneCount} planes of ${position.size}x${position.size} floats")
  // Planes 5-12 are four history moves, each contributing two planes: where the
  // stone was played, and a whole-board plane saying whether that move was
  // Black's.  A sum of 81 therefore means "yes, Black played it".
  val names = Vector(
    "stones of the player to move",
    "stones of the opponent",
    "liberties / 4 (player to move)",
    "liberties / 4 (opponent)",
    "ko point",
    "stone played on the last move",
    "the last move was Black (whole board)",
    "stone played two moves ago",
    "two moves ago was Black",
    "stone played three moves ago",
    "three moves ago was Black",
    "stone played four moves ago",
    "four moves ago was Black",
    "side to move (whole board, 0 or 81)",
    "liberties == 1 (in atari)"
  )
  names.zipWithIndex.foreach { (name, i) =>
    val sum = planes(i).sum
    Demo.say(f"  plane $i%2d  sum=$sum%6.1f  $name")
  }
  Demo.say(f"  flattened length: ${FeaturePlanes.flatten(planes).length}")
  val batch = FeaturePlanes.toNchw(Vector(position, position.copyWithKomi(7.5)))
  Demo.say(f"  as one flat NCHW array: ${batch.length} floats = 2 x ${FeaturePlanes.PlaneCount} x ${position.size} x ${position.size}")

  Demo.step("Policy and value")
  val heuristic = HeuristicNetwork()
  val (policy, value) = heuristic.evaluate(position)
  Demo.say(f"heuristic network: ${policy.length} policy values, value ${value}%.4f from ${position.toMove}'s point of view")
  Demo.say("top policy choices:")
  val ranked = position.legalMoves(position.toMove).filterNot(position.isOwnEye(_, position.toMove))
  for
    (p, prob) <- ranked.map(p => p -> policy(heuristic.policyIndex(p, position))).sortBy(-_._2).take(6)
  do Demo.say(f"  ${p.toGtp(9)}%-5s ${prob * 100}%5.1f%%")
  Demo.say(f"  pass   ${policy(heuristic.policyIndex(Point.Pass, position)) * 100}%5.1f%%")
  Demo.say(f"  total: ${policy.sum * 100}%.1f%%")

  val random = RandomNetwork()
  val (rpolicy, rvalue) = random.evaluate(position)
  Demo.say(f"random network: value $rvalue%.4f, valueIsMeaningful=${random.valueIsMeaningful}, " +
    f"policy sum ${rpolicy.sum}%.4f")

  Demo.step("Influence map (HeuristicEval)")
  Demo.say("positive is Black territory, negative is White; digits are 1-9, '.' is neutral")
  val size = 9
  val rows = (0 until size).map { y =>
    (0 until size).map { x =>
      val p = Point.at(x, y, size)
      position.stoneAt(p) match
        case Color.Black => "X"
        case Color.White => "O"
        case Color.Empty =>
          val v = HeuristicEval.influenceAt(position, p)
          if math.abs(v) < 0.05 then "."
          // Influence at the edge of its radius is small but still real, so
          // anything visible is shown as at least 1 rather than rounding to 0.
          else if v > 0 then math.max(1, math.min(9, math.round(v).toInt)).toString
          else math.max(1, math.min(9, math.round(-v).toInt)).toString
    }.mkString(" ")
  }
  Demo.indent(rows.mkString("\n"))
  Demo.say(f"balance for Black: ${HeuristicEval.balanceFor(position, Color.Black)}%.2f")

// ----------------------------------------------------------------------------
// SGF: reading and writing game records
// ----------------------------------------------------------------------------
@main def demoSgf(): Unit =
  Demo.title("SGF.scala -- reading and writing game records")

  val moves = Vector(
    Color.Black -> Point.fromGtp("D4", 9).get,
    Color.White -> Point.fromGtp("E5", 9).get,
    Color.Black -> Point.fromGtp("E4", 9).get,
    Color.White -> Point.Pass,
    Color.Black -> Point.fromGtp("F4", 9).get
  )

  Demo.step("Writing a game record")
  val record = GameRecord(
    boardSize = 9,
    komi = 6.5,
    moves = moves,
    result = "B+1.5",
    blackName = "Alice",
    whiteName = "Bob",
    date = "2026-01-15",
    comment = "an SGF demo -- with a bracket ] and a backslash \\ in the comment"
  )
  val text = SGF.write(record)
  Demo.indent(text)
  Demo.say(s"${text.length} characters, ${text.linesIterator.size} lines")

  Demo.step("Reading it back")
  SGF.readRecord(text) match
    case Left(error) => Demo.say(s"parse error: $error")
    case Right(parsed) =>
      Demo.say(s"board size ${parsed.boardSize}, komi ${parsed.komi}, result ${parsed.result}")
      Demo.say(s"players: ${parsed.blackName} (B) vs ${parsed.whiteName} (W), ${parsed.date}")
      Demo.say(s"comment: ${parsed.comment}")
      Demo.say(s"moves: ${parsed.moves.map((c, p) => s"${if c == Color.Black then "B" else "W"}${p.toGtp(9)}").mkString(" ")}")
      Demo.say("")
      Demo.say("Replaying the moves reproduces this position:")
      Demo.indent(parsed.toBoard.toAscii())

  Demo.step("The simple read() API")
  val (simpleMoves, properties) = SGF.read(text)
  Demo.say(s"${simpleMoves.length} moves, root properties: ${properties.toVector.sortBy(_._1).mkString(", ")}")

  Demo.step("What other tools produce: handicap and a variation")
  val foreign = "(;GM[1]FF[4]SZ[9]KM[0.5]HA[2]AB[cc][gg]C[A handicap game](;W[ee];B[ff])(;W[dd]))"
  Demo.say(foreign)
  SGF.readRecord(foreign) match
    case Left(error) => Demo.say(s"parse error: $error")
    case Right(parsed) =>
      Demo.say(s"handicap ${parsed.handicap}, ${parsed.setupStones.size} setup stones, " +
        s"${parsed.firstColor} to move, comment '${parsed.comment}'")
      Demo.say(s"main line: ${parsed.moves.map((_, p) => p.toGtp(9)).mkString(" ")}")
      Demo.indent(parsed.toBoard.toAscii())

  Demo.step("Round trip to a file")
  val path = Paths.get("demo-game.sgf")
  Files.writeString(path, text, StandardCharsets.UTF_8)
  val reread = SGF.readRecord(new String(Files.readAllBytes(path), StandardCharsets.UTF_8))
  Demo.say(s"wrote $path (${Files.size(path)} bytes)")
  Demo.say(s"read it back: ${reread.map(r => s"${r.moves.length} moves, result ${r.result}").getOrElse("failed")}")
  Demo.say(s"same position: ${reread.toOption.exists(_.toBoard.samePosition(record.toBoard))}")

// ----------------------------------------------------------------------------
// GTP: a scripted protocol session
// ----------------------------------------------------------------------------
@main def demoGtp(): Unit =
  Demo.title("GTP.scala -- a scripted Go Text Protocol session")

  val engine = GtpEngine(config = SearchConfig(playouts = 300, threads = 4, seed = 5L))
  val script = Vector(
    "protocol_version",
    "name",
    "version",
    "boardsize 9",
    "komi 6.5",
    "play black D4",
    "play white E5",
    "genmove black",
    "showboard",
    "final_score",
    "undo",
    "genmove white",
    "final_status_list alive",
    "known_command genmove",
    "frobnicate"
  )
  for command <- script do
    println(s"> $command")
    engine.handle(command) match
      case Right("")        => println("=")
      case Right(response)  => Demo.indent(response, "= ")
      case Left(error)      => println(s"? $error")
    println()

  Demo.say("Every command gets exactly one response, and failures use '?'.")

// ----------------------------------------------------------------------------
// Self-play: one game's worth of training data
// ----------------------------------------------------------------------------
@main def demoSelfPlay(): Unit =
  Demo.title("SelfPlay.scala -- reinforcement learning data generation")

  val config = SelfPlayConfig(
    boardSize = 9,
    playouts = 40,
    threads = 4,
    temperature = 1.0,
    tempDropMove = 6,
    dirichletWeight = 0.25,
    resignThreshold = None,
    seed = 99L
  )
  Demo.step(s"Playing one game: 9x9, ${config.playouts} playouts per move, Dirichlet noise ${config.dirichletWeight}")
  val game = SelfPlay.generateGame(HeuristicNetwork(), config, new java.util.Random(99L))

  Demo.say(s"result: ${game.result}")
  Demo.say(s"moves: ${game.moveCount}, examples: ${game.examples.size}, time: ${game.elapsed.toSeconds}s")
  Demo.say(s"moves played: ${game.moves.map((c, p) => s"${if c == Color.Black then "B" else "W"}${p.toGtp(9)}").mkString(" ")}")
  Demo.say("")
  Demo.indent(game.finalBoard.toAscii())

  Demo.step("Each move contributed one training example")
  Demo.say("  move  plane dims   policy dims  z    search value  move played")
  for example <- game.examples.take(10) do
    Demo.say(
      f"  ${example.moveNumber}%4d  ${example.featurePlanes.length}%2d x ${example.boardSize * example.boardSize}%3d" +
        f"  ${example.mctsPolicy.length}%3d        ${example.outcome}%+.0f   ${example.searchValue}%+.3f      ${example.move.toGtp(9)}"
    )
  Demo.say("  ...")

  Demo.step("Labels are from the point of view of the player to move")
  val blackExamples = game.examples.count(_.toMove == Color.Black)
  val whiteExamples = game.examples.count(_.toMove == Color.White)
  val positive = game.examples.count(_.outcome > 0)
  Demo.say(s"$blackExamples examples with Black to move, $whiteExamples with White to move")
  Demo.say(s"$positive examples labelled +1, ${game.examples.count(_.outcome < 0)} labelled -1")
  Demo.say(s"the policy target is the MCTS visit distribution, so it sums to ${game.examples.head.mctsPolicy.sum}")

  Demo.step("Writing the data out")
  val outDir = Paths.get("demo-selfplay")
  Files.createDirectories(outDir)
  val binary = outDir.resolve("training.bin")
  SelfPlay.Writer.writeBinary(game.examples, binary)
  Demo.say(f"$binary: ${Files.size(binary)} bytes " +
    f"(${SelfPlay.Writer.floatsPerExample(FeaturePlanes.PlaneCount, 9, 82)} floats per example)")
  Demo.say("Read it in Python with python/load_selfplay.py.")
  val sgf = outDir.resolve("game-0001.sgf")
  Files.writeString(sgf, SGF.write(game.toRecord), StandardCharsets.UTF_8)
  Demo.say(s"$sgf: the game, reviewable in Sabaki or GoGUI")

// ----------------------------------------------------------------------------
// Everything
// ----------------------------------------------------------------------------
@main def demoAll(): Unit =
  demoBoard()
  demoEval()
  demoSgf()
  demoSearch()
  demoGtp()
  demoSelfPlay()
  Demo.title("All demos finished")
  println("Run one at a time with make demo-board, demo-search, demo-eval,")
  println("demo-sgf, demo-gtp or demo-selfplay.")
