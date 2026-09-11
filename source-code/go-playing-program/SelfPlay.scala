//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package go

import java.io.{BufferedOutputStream, FileOutputStream}
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths}
import scala.collection.mutable
import scala.concurrent.duration.FiniteDuration

// ============================================================================
// SelfPlay.scala -- reinforcement-learning data generation (design doc section 3.3).
//
// Plays games between the engine and itself, recording the tuple that
// AlphaGo Zero learns from:
//
//     (position planes, MCTS policy, game outcome)
//
// The MCTS policy -- the visit distribution, not the raw network output -- is
// the whole point: the search is a stronger player than the network, so
// training the network to imitate the search is what makes it improve.
//
// Two details from the design document that matter for data quality:
//   * temperature: the first moves are sampled at T=1 for opening diversity,
//     then temperature drops to near zero so the rest of the game is strong.
//   * Dirichlet noise at the root, which stops every game from being identical.
//
// Output is a compact binary format (documented below and read by
// python/load_selfplay.py) plus an optional JSON Lines form for eyeballing.
// ============================================================================

/**
 * One training example: the three tensors AlphaGo Zero trains on.
 *
 * `outcome` is from the perspective of [[toMove]], so it is `+1` when the
 * player to move in this position went on to win, `-1` when they lost, and `0`
 * for a draw.  It is filled in after the game ends, which is why the whole
 * game has to be kept in memory before it can be written.
 */
final case class TrainingExample(
    featurePlanes: Array[Array[Float]],
    /** The MCTS visit distribution -- the search policy target, pi. */
    mctsPolicy: Array[Float],
    /** The game result from the perspective of the player to move, z. */
    outcome: Float,
    /** The move actually played, for inspection and debugging. */
    move: Point = Point.Pass,
    moveNumber: Int = 0,
    toMove: Color = Color.Black,
    boardSize: Int = 19,
    /** Root win rate the search reported, from `toMove`'s perspective. */
    searchValue: Float = 0f
):
  /** Flat plane data, `planeCount * area` long. */
  def flattened: Array[Float] = FeaturePlanes.flatten(featurePlanes)

  /** A compact single-line JSON encoding, for `--format json` output. */
  def toJson: String =
    val planes = flattened
    val sb = new StringBuilder
    sb.append("{\"size\":").append(boardSize)
    sb.append(",\"move\":").append(moveNumber)
    sb.append(",\"toMove\":\"").append(if toMove == Color.Black then "B" else "W").append('"')
    sb.append(",\"played\":\"").append(move.toGtp(boardSize)).append('"')
    sb.append(",\"outcome\":").append(outcome)
    sb.append(",\"searchValue\":").append(searchValue)
    sb.append(",\"policy\":[")
    sb.append(mctsPolicy.mkString(","))
    sb.append("],\"planes\":[")
    sb.append(planes.mkString(","))
    sb.append("]}")
    sb.result()

/** Everything produced by one self-play game. */
final case class SelfPlayGame(
    examples: Vector[TrainingExample],
    moves: Vector[(Color, Point)],
    result: String,
    winner: Option[Color],
    finalBoard: BoardState,
    elapsed: FiniteDuration,
    resigned: Boolean = false
):
  def moveCount: Int = moves.length

  /** The game as an SGF record, so self-play games can be reviewed. */
  def toRecord: GameRecord =
    GameRecord(
      boardSize = finalBoard.size,
      komi = finalBoard.komi,
      moves = moves,
      result = result,
      blackName = "selfplay",
      whiteName = "selfplay",
      ruleSet = finalBoard.ruleSet
    )

/** How self-play games are produced and what data is kept. */
final case class SelfPlayConfig(
    boardSize: Int = 19,
    /** Playouts per move.  The design document suggests 800 or more. */
    playouts: Int = 800,
    threads: Int = math.max(1, Runtime.getRuntime.availableProcessors()),
    komi: Double = 6.5,
    ruleSet: RuleSet = RuleSet.Area,
    /** Exploration temperature for the opening. */
    temperature: Double = 1.0,
    /** After this many moves the temperature is dropped, making play greedy. */
    tempDropMove: Int = 30,
    /** Temperature used after [[tempDropMove]]. */
    lateTemperature: Double = 0.0,
    dirichletAlpha: Double = 0.3,
    dirichletWeight: Double = 0.25,
    /**
     * Resign when the search is this pessimistic, from the mover's point of
     * view, and the game is already long enough for the estimate to be
     * trustworthy.  Set to `None` to disable resignation entirely.
     */
    resignThreshold: Option[Double] = Some(-0.97),
    resignMinMoves: Int = 40,
    /** Safety cap, so a pathological game cannot run forever. */
    maxMoves: Int = 0,
    seed: Long = 0x5eedL
):
  require(boardSize >= 2 && boardSize <= 19, "boardSize must be between 2 and 19")
  require(playouts > 0, "playouts must be positive")
  require(threads > 0, "threads must be positive")
  require(temperature >= 0, "temperature must not be negative")
  require(maxMoves >= 0, "maxMoves must not be negative")

  /** The effective move cap: twice the board area is the usual safety net. */
  def effectiveMaxMoves: Int = if maxMoves > 0 then maxMoves else boardSize * boardSize * 2

  /** The search configuration implied by these settings. */
  def searchConfig: SearchConfig = SearchConfig(
    playouts = playouts,
    threads = threads,
    dirichletAlpha = dirichletAlpha,
    dirichletWeight = dirichletWeight,
    seed = seed
  )

/**
 * Self-play game generation.
 *
 * `main` implements the `make selfplay` target from the design document:
 * {{{
 *   scala-cli run . --main-class go.SelfPlay -- --games 100 --size 9
 * }}}
 */
object SelfPlay:

  /**
   * Plays one complete game and returns its training data.
   *
   * This is the signature given in the design document; it delegates to the
   * config-driven [[generateGame]] below.
   */
  def generateGame(
      network: PolicyValueNetwork,
      boardSize: Int = 19,
      playouts: Int = 800,
      temperature: Float = 1.0f,
      tempDropMove: Int = 30
  ): Vector[TrainingExample] =
    generateGame(
      network,
      SelfPlayConfig(
        boardSize = boardSize,
        playouts = playouts,
        temperature = temperature.toDouble,
        tempDropMove = tempDropMove
      ),
      new java.util.Random(boardSize * 1000L + playouts)
    ).examples

  /** Plays one complete game with full control over the process. */
  def generateGame(
      network: PolicyValueNetwork,
      config: SelfPlayConfig,
      rng: java.util.Random
  ): SelfPlayGame =
    val started = System.nanoTime()
    var state = BoardState.empty(config.boardSize, config.komi, config.ruleSet)
    val recorded = mutable.ArrayBuffer.empty[TrainingExample]
    val played = mutable.ArrayBuffer.empty[(Color, Point)]
    var resigned = false
    var moveNumber = 0

    while !state.isTerminal && moveNumber < config.effectiveMaxMoves && !resigned do
      val mover = state.toMove
      val session = Search.newSession(state, network, config.searchConfig)

      // Root noise is what stops a generation of games from collapsing onto a
      // single opening, which would starve the training set of variety.
      session.applyRootNoise(rng)
      session.runPlayouts(config.playouts)

      val temperature =
        if moveNumber < config.tempDropMove then config.temperature else config.lateTemperature
      val move = session.sampleMove(math.max(temperature, 0.0), rng)
      val winRate = session.winRate

      recorded += TrainingExample(
        featurePlanes = FeaturePlanes.encode(state),
        mctsPolicy = policyVector(session, state),
        outcome = 0f, // filled in once the game is over
        move = move,
        moveNumber = moveNumber,
        toMove = mover,
        boardSize = config.boardSize,
        searchValue = winRate.toFloat
      )

      state.play(move) match
        case Right(next) => state = next
        case Left(_)     => state = state.playPass

      played += ((mover, move))
      moveNumber += 1

      // Resign only once the estimate can be trusted; a premature resignation
      // would poison the training data with a wrong label.
      config.resignThreshold match
        case Some(threshold) if moveNumber >= config.resignMinMoves && winRate < threshold =>
          resigned = true
          state = state.resign(mover)
        case _ => ()

    val winner = state.winner
    // z is from the perspective of whoever was to move in that position.
    val labelled = recorded.toVector.map { example =>
      val z = winner match
        case Some(w) => if w == example.toMove then 1f else -1f
        case None    => 0f
      example.copy(outcome = z)
    }

    SelfPlayGame(
      examples = labelled,
      moves = played.toVector,
      result = state.resultString,
      winner = winner,
      finalBoard = state,
      elapsed = FiniteDuration(System.nanoTime() - started, java.util.concurrent.TimeUnit.NANOSECONDS),
      resigned = resigned
    )

  /** The search's visit distribution as a full-length policy vector. */
  private def policyVector(session: SearchSession, state: BoardState): Array[Float] =
    val out = new Array[Float](state.area + 1)
    for (p, prob) <- session.visitDistribution do
      // Hoisted: `out[a.b.c(...)]` is ambiguous to the parser, which reads the
      // path as a type argument rather than as an index expression.
      val index = session.network.policyIndex(p, state)
      out(index) = prob
    out

  // --------------------------------------------------------------------------
  // Writing training data
  // --------------------------------------------------------------------------

  /**
   * Binary format, version 1.  Little-endian, which is what NumPy and the
   * PyTorch ecosystem expect; note that this is *not* what Java's
   * `DataOutputStream` would produce, so the bytes are assembled with an
   * explicitly little-endian `ByteBuffer`:
   *
   * {{{
   *   magic         4 bytes, ASCII "GOPP"
   *   version       int32    = 1
   *   exampleCount  int32
   *   planeCount    int32
   *   boardSize     int32
   *   policySize    int32    = boardSize * boardSize + 1
   *   then, for each example, tightly packed float32:
   *     planeCount * boardSize * boardSize   feature planes
   *     policySize                           MCTS policy
   *     1                                    outcome
   *     1                                    search value
   * }}}
   *
   * Every record is the same size, so a reader can simply `np.fromfile` once
   * and reshape.  See `python/load_selfplay.py`.
   */
  object Writer:
    val Magic = "GOPP"
    val Version = 1

    /** Per-example float count, useful for sizing a reader. */
    def floatsPerExample(planeCount: Int, boardSize: Int, policySize: Int): Int =
      planeCount * boardSize * boardSize + policySize + 2

    def writeBinary(examples: Seq[TrainingExample], path: Path): Unit =
      Option(path.toAbsolutePath.getParent).foreach(Files.createDirectories(_))
      val out = BufferedOutputStream(FileOutputStream(path.toFile))
      val little = java.nio.ByteOrder.LITTLE_ENDIAN
      try
        out.write(header(examples).array())
        // One reusable buffer per record: every example has the same shape.
        val first = examples.headOption
        val perExample = first
          .map(e => floatsPerExample(e.featurePlanes.length, e.boardSize, e.mctsPolicy.length))
          .getOrElse(0)
        val record = java.nio.ByteBuffer.allocate(perExample * 4).order(little)
        for example <- examples do
          record.clear()
          putFloats(record, example.flattened)
          putFloats(record, example.mctsPolicy)
          record.putFloat(example.outcome)
          record.putFloat(example.searchValue)
          out.write(record.array(), 0, record.position())
      finally out.close()

    private def header(examples: Seq[TrainingExample]): java.nio.ByteBuffer =
      val size = examples.headOption.map(_.boardSize).getOrElse(19)
      val planeCount = examples.headOption.map(_.featurePlanes.length).getOrElse(FeaturePlanes.PlaneCount)
      val buf = java.nio.ByteBuffer.allocate(24).order(java.nio.ByteOrder.LITTLE_ENDIAN)
      buf.put(Magic.getBytes(StandardCharsets.US_ASCII))
      buf.putInt(Version)
      buf.putInt(examples.size)
      buf.putInt(planeCount)
      buf.putInt(size)
      buf.putInt(size * size + 1)
      buf.flip()
      buf

    private def putFloats(buffer: java.nio.ByteBuffer, values: Array[Float]): Unit =
      var i = 0
      while i < values.length do
        buffer.putFloat(values(i))
        i += 1

    /** One JSON object per line; larger, but readable with `head`/`jq`. */
    def writeJsonLines(examples: Seq[TrainingExample], path: Path): Unit =
      Option(path.toAbsolutePath.getParent).foreach(Files.createDirectories(_))
      val writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)
      try examples.foreach(e => writer.write(e.toJson + "\n"))
      finally writer.close()

  // --------------------------------------------------------------------------
  // Command line entry point: make selfplay
  // --------------------------------------------------------------------------

  def main(args: Array[String]): Unit =
    val parsed = Args.parse(args)
    if parsed.has("help") then
      println(usage)
      return

    val games = parsed.int("games", 10)
    val size = parsed.int("size", 9)
    val playouts = parsed.int("playouts", if size <= 9 then 400 else 800)
    val threads = parsed.int("threads", math.max(1, Runtime.getRuntime.availableProcessors()))
    val temperature = parsed.double("temperature", 1.0)
    val tempDrop = parsed.int("temp-drop", 30)
    val komi = parsed.double("komi", 6.5)
    val noiseWeight = parsed.double("noise", 0.25)
    val resignThreshold = if parsed.bool("no-resign", false) then None else Some(parsed.double("resign", -0.97))
    val output = parsed.string("output", s"selfplay-${size}x$size")
    val format = parsed.string("format", "bin")
    val modelPath = parsed.get("model")
    val maxMoves = parsed.int("max-moves", 0)
    val seed = parsed.get("seed").flatMap(_.toLongOption).getOrElse(System.nanoTime())

    val network: PolicyValueNetwork = modelPath match
      case Some(path) =>
        println(s"loading network from $path")
        OnnxNetwork(path)
      case None =>
        println("no --model given, using the built-in heuristic network")
        HeuristicNetwork()

    val config = SelfPlayConfig(
      boardSize = size,
      playouts = playouts,
      threads = threads,
      komi = komi,
      temperature = temperature,
      tempDropMove = tempDrop,
      dirichletWeight = noiseWeight,
      resignThreshold = resignThreshold,
      maxMoves = maxMoves,
      seed = seed
    )

    println(
      f"self-play: $games games on ${size}x$size, $playouts playouts/move, " +
        f"$threads threads, network=${network.name}"
    )
    val rng = new java.util.Random(seed)
    val outDir = Paths.get(output)
    var allExamples = Vector.empty[TrainingExample]
    var blackWins = 0
    var whiteWins = 0
    var draws = 0
    var resigns = 0
    var gameIndex = 0

    while gameIndex < games do
      val game = generateGame(network, config, rng)
      game.winner match
        case Some(Color.Black) => blackWins += 1
        case Some(Color.White) => whiteWins += 1
        case _                 => draws += 1
      if game.resigned then resigns += 1
      allExamples ++= game.examples

      val record = game.toRecord
      val sgf = SGF.write(record.copy(comment = s"self-play game ${gameIndex + 1}, result ${game.result}"))
      val sgfPath = outDir.resolve(f"game-${gameIndex + 1}%04d.sgf")
      Option(sgfPath.toAbsolutePath.getParent).foreach(Files.createDirectories(_))
      Files.writeString(sgfPath, sgf, StandardCharsets.UTF_8)

      println(
        f"game ${gameIndex + 1}%3d/${games}: ${game.result}%-10s " +
          f"in ${game.moveCount}%3d moves, ${game.elapsed.toSeconds}%4ds, " +
          f"${game.examples.size}%4d examples"
      )
      gameIndex += 1

    val dataPath =
      if format == "json" then outDir.resolve("training.jsonl")
      else outDir.resolve("training.bin")
    if format == "json" then Writer.writeJsonLines(allExamples, dataPath)
    else Writer.writeBinary(allExamples, dataPath)

    println()
    println(s"games:    $games (black $blackWins, white $whiteWins, draw $draws, resignations $resigns)")
    println(s"examples: ${allExamples.size}")
    println(s"data:     $dataPath")
    println(s"sgf:      $outDir")

  val usage: String =
    """Usage: scala-cli run . --main-class go.SelfPlay -- [options]
      |
      |  --games N         number of self-play games to play          (default 10)
      |  --size N          board size, 9, 13 or 19                    (default 9)
      |  --playouts N      MCTS playouts per move                     (default 400 on 9x9)
      |  --threads N       search threads                             (default all cores)
      |  --temperature F   opening exploration temperature            (default 1.0)
      |  --temp-drop N     move after which temperature drops         (default 30)
      |  --noise F         Dirichlet root noise weight                (default 0.25)
      |  --resign F        resign below this win rate                 (default -0.97)
      |  --no-resign       never resign (better labels, slower games)
      |  --komi F          komi                                       (default 6.5)
      |  --max-moves N     safety cap on game length                  (default 2 * area)
      |  --model PATH      ONNX model to use (needs the onnxruntime jar)
      |  --output DIR      output directory                           (default selfplay-NxN)
      |  --format bin|json training data format                       (default bin)
      |  --seed N          random seed
      |  --help            this message
      |
      |Outputs training data plus one SGF per game, so the games can be reviewed
      |in Sabaki or GoGUI.  The binary format is read by python/load_selfplay.py.""".stripMargin
