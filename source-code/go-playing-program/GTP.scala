//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package go

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}
import scala.collection.mutable
import scala.concurrent.duration.FiniteDuration

// ============================================================================
// GTP.scala -- Go Text Protocol (https://www.lysator.liu.se/~gunnar/gtp/)
//
// The engine speaks GTP v2 on stdin/stdout, which is what lets it be driven by
// GoGUI, Sabaki, Lizzie, and `gogui-twogtp` for engine-vs-engine matches.
//
// Protocol requirements that are easy to get wrong and are handled here:
//   * every command gets exactly one response, terminated by a blank line;
//   * a failure is reported with '?' rather than silence;
//   * a numeric command id is echoed back (`=1 ...`);
//   * an unknown command is a failure, not a crash;
//   * coordinates omit the letter I and `pass` is a valid vertex.
// ============================================================================

/** Engine identification reported over GTP. */
final case class GtpInfo(
    name: String = "GoPlayingProgram",
    version: String = "1.0",
    author: String = "Mark Watson"
)

/**
 * A GTP session: the board plus the engine that plays it.
 *
 * State lives here rather than in globals so that a session can be driven
 * directly from a test without touching stdin or stdout.
 */
final class GtpEngine(
    val network: PolicyValueNetwork = HeuristicNetwork(),
    var config: SearchConfig = SearchConfig(threads = 1),
    val info: GtpInfo = GtpInfo(),
    /** Return `resign` from `genmove` when the search is hopeless. */
    val resignAllowed: Boolean = false,
    val resignThreshold: Double = -0.97
):
  private var boardSize: Int = 19
  private var state: BoardState = BoardState.initial(boardSize, 6.5)
  /** Undo stack, oldest first; the current position is the last entry. */
  private var history: Vector[BoardState] = Vector(state)
  private var komi: Double = 6.5
  private val clocks = mutable.Map.empty[Color, Clock]
  /** Search statistics from the last genmove, for `debug_search`. */
  private var lastSearch: Option[SearchResult] = None
  /**
   * The persistent search tree, re-rooted on every played move.  Any command
   * that rewrites the position out from under it (undo, loadsgf, handicap,
   * ...) resets it; the next genmove then starts a fresh tree.
   */
  private var searchSession: SearchSession | Null = null

  /** Releases the search tree, if any. */
  def close(): Unit = resetSession()

  private def sessionFor(position: BoardState, effective: SearchConfig): SearchSession =
    val current = searchSession
    if current != null && current.root.state.samePosition(position) then current
    else
      if current != null then current.close()
      val fresh = Search.newSession(position, network, effective)
      searchSession = fresh
      fresh

  private def advanceOrReset(move: Point): Unit =
    val current = searchSession
    if current != null && !current.advanceTo(move) then
      current.close()
      searchSession = null

  private def resetSession(): Unit =
    if searchSession != null then searchSession.close()
    searchSession = null

  def currentState: BoardState = state

  /** The commands this engine implements, in the order advertised. */
  val supportedCommands: Vector[String] = Vector(
    "protocol_version", "name", "version", "known_command", "list_commands",
    "quit", "boardsize", "clear_board", "komi", "play", "genmove", "undo",
    "showboard", "final_score", "final_status_list", "fixed_handicap",
    "place_free_handicap", "set_free_handicap", "time_settings", "time_left",
    "loadsgf", "kgs-game_over", "debug_search"
  )

  // --------------------------------------------------------------------------
  // Command dispatch
  // --------------------------------------------------------------------------

  /** Handles one raw command line, returning the response body or an error. */
  def handle(line: String): Either[String, String] =
    val trimmed = line.trim
    if trimmed.isEmpty then Right("")
    else
      var tokens = trimmed.split("\\s+").toVector
      // An optional leading integer is the command id, echoed in the reply.
      if tokens.headOption.exists(_.forall(_.isDigit)) then tokens = tokens.tail
      if tokens.isEmpty then Right("")
      else
        val command = tokens.head.toLowerCase
        val args = tokens.tail
        try dispatch(command, args)
        catch
          case e: IllegalArgumentException => Left(e.getMessage)
          case e: Throwable               => Left(s"internal error: ${e.getClass.getSimpleName}: ${e.getMessage}")

  private def dispatch(command: String, args: Vector[String]): Either[String, String] =
    command match
      case "protocol_version" => Right("2")
      case "name"             => Right(info.name)
      case "version"          => Right(info.version)
      case "known_command" =>
        val name = args.headOption.getOrElse("").toLowerCase
        Right(if supportedCommands.contains(name) then "true" else "false")
      case "list_commands" => Right(supportedCommands.mkString("\n"))
      case "quit"          => Right("")
      case "boardsize"     => boardsize(args)
      case "clear_board"   => clearBoard()
      case "komi"          => setKomi(args)
      case "play"          => play(args)
      case "genmove"       => genmove(args)
      case "undo"          => undo()
      case "showboard"     => Right("\n" + state.toAscii())
      case "final_score"   => Right(finalScore)
      case "final_status_list" => finalStatusList(args)
      case "fixed_handicap"    => fixedHandicap(args)
      case "place_free_handicap" => placeFreeHandicap(args)
      case "set_free_handicap"   => setFreeHandicap(args)
      case "time_settings"       => timeSettingsCmd(args)
      case "time_left"           => timeLeft(args)
      case "loadsgf"             => loadSgf(args)
      case "kgs-game_over"       => Right("")
      case "debug_search"        => Right(debugSearch)
      case other                 => Left(s"unknown command: $other")

  // --------------------------------------------------------------------------
  // Board and rules commands
  // --------------------------------------------------------------------------

  private def boardsize(args: Vector[String]): Either[String, String] =
    args.headOption.flatMap(_.toIntOption) match
      case None => Left("boardsize requires a number")
      case Some(n) if n < 2 || n > 19 => Left(s"unsupported board size $n")
      case Some(n) =>
        boardSize = n
        state = BoardState.initial(n, komi)
        history = Vector(state)
        clocks.clear()
        resetSession()
        Right("")

  private def clearBoard(): Either[String, String] =
    state = BoardState.initial(boardSize, komi)
    history = Vector(state)
    lastSearch = None
    resetSession()
    Right("")

  private def setKomi(args: Vector[String]): Either[String, String] =
    args.headOption.flatMap(_.toDoubleOption) match
      case None      => Left("komi requires a number")
      case Some(k) =>
        komi = k
        state = state.copyWithKomi(k)
        history = history.updated(history.length - 1, state)
        resetSession()
        Right("")

  private def play(args: Vector[String]): Either[String, String] =
    if args.length < 2 then Left("play requires a colour and a vertex")
    else
      val color = Color.parse(args(0)).toRight(s"unknown colour: ${args(0)}") match
        case Left(msg)  => return Left(msg)
        case Right(c)   => c
      val vertex = Point.fromGtp(args(1), boardSize).toRight(s"invalid vertex: ${args(1)}") match
        case Left(msg) => return Left(msg)
        case Right(p)  => p
      // `play` sets up a position literally, so it may override whose turn it is.
      state.withToMove(color).place(vertex, color) match
        case Right(next) =>
          state = next
          history = history :+ state
          advanceOrReset(vertex)
          Right("")
        case Left(reason) => Left(s"illegal move: $reason")

  private def genmove(args: Vector[String]): Either[String, String] =
    val color = args.headOption.flatMap(Color.parse).getOrElse(state.toMove)
    // genmove must play for the requested colour even if the board disagrees,
    // which happens when a GUI replays a game move by move.
    val position = state.withToMove(color)
    val budget = budgetFor(color)
    val effective = budget.map(b => config.copy(timeLimit = Some(b))).getOrElse(config)
    val session = sessionFor(position, effective)
    val started = System.nanoTime()
    session.runPlayouts(
      effective.playouts,
      effective.timeLimit.map(t => started + t.toNanos)
    )
    val result = session.result(
      FiniteDuration(System.nanoTime() - started, java.util.concurrent.TimeUnit.NANOSECONDS)
    )
    lastSearch = Some(result)
    recordTime(color, result.elapsed)

    val move =
      if resignAllowed && result.winRate < resignThreshold && !result.move.isPass then "resign"
      else result.move.toGtp(boardSize)

    if move == "resign" then
      state = position.resign(color)
      resetSession()
      Right("resign")
    else
      position.place(result.move, color) match
        case Right(next) =>
          state = next
          history = history :+ state
          advanceOrReset(result.move)
          Right(move)
        case Left(reason) => Left(s"search produced an illegal move: $reason")

  private def undo(): Either[String, String] =
    if history.length <= 1 then Left("cannot undo: no moves have been played")
    else
      history = history.dropRight(1)
      state = history.last
      resetSession()
      Right("")

  private def finalScore: String = state.resultString

  /**
   * Life-and-death status, decided by [[LifeDeath]]: Benson unconditional
   * life for `alive`, forced-capture proofs and territory control for `dead`,
   * and mutually-unsettled neighbours for `seki`.  `black` and `white` list
   * every group of that colour.  One group per line, as GTP expects.
   */
  private def finalStatusList(args: Vector[String]): Either[String, String] =
    val groups = args.headOption.getOrElse("").toLowerCase match
      case "alive" => LifeDeath.aliveGroups(state)
      case "dead"  => LifeDeath.deadGroups(state)
      case "seki"  => LifeDeath.sekiGroups(state)
      case "black" => state.groups.values.toVector.distinct.filter(_.color == Color.Black)
      case "white" => state.groups.values.toVector.distinct.filter(_.color == Color.White)
      case other   => return Left(s"unknown status: $other (try alive, dead or seki)")
    Right(
      groups
        .map(group => group.stones.map(_.toGtp(boardSize)).toVector.sorted.mkString(" "))
        .filter(_.nonEmpty)
        .mkString("\n")
    )

  // --------------------------------------------------------------------------
  // Handicap
  // --------------------------------------------------------------------------

  private def fixedHandicap(args: Vector[String]): Either[String, String] =
    val count = args.headOption.flatMap(_.toIntOption).getOrElse(0)
    StarPoints.handicapPoints(boardSize, count) match
      case None => Left(s"cannot place $count handicap stones on a $boardSize board")
      case Some(points) =>
        state = state.setupStones(points.map(p => (p, Color.Black))).withToMove(Color.White)
        history = history :+ state
        resetSession()
        Right(points.map(_.toGtp(boardSize)).mkString(" "))

  private def placeFreeHandicap(args: Vector[String]): Either[String, String] =
    val count = args.headOption.flatMap(_.toIntOption).getOrElse(0)
    fixedHandicap(Vector(count.toString))

  private def setFreeHandicap(args: Vector[String]): Either[String, String] =
    val parsed = args.map(a => Point.fromGtp(a, boardSize))
    if parsed.exists(_.isEmpty) then Left("invalid handicap vertex")
    else
      val points = parsed.flatten
      state = state.setupStones(points.map(p => (p, Color.Black))).withToMove(Color.White)
      history = history :+ state
      resetSession()
      Right("")

  // --------------------------------------------------------------------------
  // Time
  // --------------------------------------------------------------------------

  private def timeSettingsCmd(args: Vector[String]): Either[String, String] =
    if args.length < 3 then Left("time_settings requires main, byo-yomi and stones")
    else
      val parsed =
        for
          main <- args(0).toIntOption
          byo <- args(1).toIntOption
          stones <- args(2).toIntOption
        yield (math.max(0, main), math.max(0, byo), math.max(0, stones))
      parsed match
        case None => Left("time_settings requires integer arguments")
        case Some((mainSeconds, byoSeconds, stones)) =>
          val mainTime = FiniteDuration(mainSeconds.toLong, "s")
          val control =
            if byoSeconds > 0 && stones > 0 then
              TimeControl(mainTime, Some(ByoYomi(FiniteDuration(byoSeconds.toLong, "s"), stones)))
            else TimeControl.suddenDeath(mainTime)
          clocks.clear()
          clocks(Color.Black) = Clock(control)
          clocks(Color.White) = Clock(control)
          Right("")

  private def timeLeft(args: Vector[String]): Either[String, String] =
    if args.length < 2 then Left("time_left requires a colour and a time")
    else
      val color = Color.parse(args(0)).toRight(s"unknown colour: ${args(0)}") match
        case Left(msg) => return Left(msg)
        case Right(c)  => c
      val seconds = args(1).toDoubleOption.toRight("time_left requires a number of seconds") match
        case Left(msg) => return Left(msg)
        case Right(s)  => s
      val stones = args.lift(2).flatMap(_.toIntOption).getOrElse(0)
      // Rebuild the clock with exactly the remaining time the GUI reported.
      val period = if stones > 0 then FiniteDuration(math.max(0L, seconds.round), "s") else FiniteDuration(0, "s")
      val control =
        if stones > 0 then
          TimeControl(FiniteDuration(math.max(0L, seconds.round), "s"), Some(ByoYomi(period, math.max(1, stones))))
        else TimeControl.suddenDeath(FiniteDuration(math.max(0L, seconds.round), "s"))
      clocks(color) = Clock(control)
      Right("")

  /** The time budget for this move, if the GUI has told us about time limits. */
  private def budgetFor(color: Color): Option[FiniteDuration] =
    clocks.get(color).map(_.timeForMove()) match
      case Some(d) => Some(d)
      case None    => config.timeLimit

  private def recordTime(color: Color, elapsed: FiniteDuration): Unit =
    clocks.get(color).foreach(_.recordMove(elapsed))

  // --------------------------------------------------------------------------
  // SGF
  // --------------------------------------------------------------------------

  private def loadSgf(args: Vector[String]): Either[String, String] =
    val file = args.headOption.toRight("loadsgf requires a filename") match
      case Left(msg) => return Left(msg)
      case Right(f)  => f
    if !Files.exists(Paths.get(file)) then return Left(s"no such file: $file")
    val text = new String(Files.readAllBytes(Paths.get(file)), StandardCharsets.UTF_8)
    SGF.readRecord(text) match
      case Left(msg) => Left(s"cannot read SGF: $msg")
      case Right(record) =>
        boardSize = record.boardSize
        komi = record.komi
        state = record.toBoard
        // `loadsgf <file> <move-number>` may ask to stop part-way through.
        args.lift(1).flatMap(_.toIntOption) match
          case Some(limit) if limit > 0 && limit < record.moves.length =>
            state = record.copy(moves = record.moves.take(limit)).toBoard
          case _ => ()
        history = Vector(state)
        resetSession()
        Right("")

  // --------------------------------------------------------------------------
  // Diagnostics
  // --------------------------------------------------------------------------

  private def debugSearch: String =
    lastSearch match
      case None => "no search has been run yet"
      case Some(r) =>
        val top = r.visits.toVector.sortBy(-_._2).take(5)
        val lines = top.map((p, n) => f"  ${p.toGtp(boardSize)}%-5s visits=$n%5d")
        (Vector(
          f"playouts=${r.playouts} elapsed=${r.elapsed.toMillis}ms nodes=${r.nodesCreated}",
          f"winRate=${r.winRate}%.4f",
          s"pv=${r.principalVariation.map(_.toGtp(boardSize)).mkString(" ")}"
        ) ++ lines).mkString("\n")

object GtpEngine:
  /** Convenience factory for tests: a 9x9 session with a fixed seed. */
  def forTesting(size: Int = 9, playouts: Int = 50): GtpEngine =
    val engine = new GtpEngine(config = SearchConfig(playouts = playouts, threads = 2, seed = 1L))
    engine.handle(s"boardsize $size")
    engine

object StarPoints:
  /**
   * The traditional handicap placements for `count` stones, or `None` when
   * that count is not defined for this board size.
   *
   * The order is the conventional one, and it matters: `fixed_handicap`
   * returns the vertices in this sequence and GUIs display them in it.  Note
   * that five stones includes the centre point while six does not -- that
   * quirk is the standard table, not a mistake.
   */
  def handicapPoints(size: Int, count: Int): Option[Vector[Point]] =
    if count <= 0 then Some(Vector.empty)
    else
      val low = if size >= 13 then 3 else 2
      val high = size - 1 - low
      val mid = size / 2
      val upperRight = Point.at(high, low, size)
      val lowerLeft = Point.at(low, high, size)
      val upperLeft = Point.at(low, low, size)
      val lowerRight = Point.at(high, high, size)
      val centre = Point.at(mid, mid, size)
      val leftMiddle = Point.at(low, mid, size)
      val rightMiddle = Point.at(high, mid, size)
      val topMiddle = Point.at(mid, low, size)
      val bottomMiddle = Point.at(mid, high, size)
      val table: Map[Int, Vector[Point]] = Map(
        1 -> Vector(upperRight),
        2 -> Vector(upperRight, lowerLeft),
        3 -> Vector(upperRight, lowerLeft, upperLeft),
        4 -> Vector(upperRight, lowerLeft, upperLeft, lowerRight),
        5 -> Vector(upperRight, lowerLeft, upperLeft, lowerRight, centre),
        6 -> Vector(upperRight, lowerLeft, upperLeft, lowerRight, leftMiddle, rightMiddle),
        7 -> Vector(upperRight, lowerLeft, upperLeft, lowerRight, leftMiddle, rightMiddle, topMiddle),
        8 -> Vector(
          upperRight,
          lowerLeft,
          upperLeft,
          lowerRight,
          leftMiddle,
          rightMiddle,
          topMiddle,
          bottomMiddle
        ),
        9 -> Vector(
          upperRight,
          lowerLeft,
          upperLeft,
          lowerRight,
          leftMiddle,
          rightMiddle,
          topMiddle,
          bottomMiddle,
          centre
        )
      )
      table.get(count)

/**
 * GTP over stdin/stdout.
 *
 * `main` is the entry point used by `make gtp` and by any GUI that launches the
 * engine as a subprocess.
 */
object GTP:
  def main(args: Array[String]): Unit =
    val parsed = Args.parse(args)
    val size = parsed.int("size", 19)
    val playouts = parsed.int("playouts", if size <= 9 then 2000 else 4000)
    val threads = parsed.int("threads", math.max(1, Runtime.getRuntime.availableProcessors()))
    val komi = parsed.double("komi", 6.5)
    val modelPath = parsed.get("model")
    val resign = parsed.bool("resign", false)
    val log = parsed.bool("log", false)

    if parsed.has("help") then
      println(usage)
      return

    val network: PolicyValueNetwork = modelPath match
      case Some(path) => OnnxNetwork(path)
      case None       => HeuristicNetwork()

    val engine = new GtpEngine(
      network = network,
      config = SearchConfig(
        playouts = playouts,
        threads = threads,
        // Used only until the GUI sends time_settings or time_left.
        timeLimit = parsed.duration("time"),
        seed = System.nanoTime()
      ),
      resignAllowed = resign
    )
    engine.handle(s"boardsize $size")
    engine.handle(s"komi $komi")

    if log then System.err.println(s"${engine.info.name} ready: ${network.name}, $playouts playouts")

    val reader = scala.io.Source.stdin.bufferedReader()
    var running = true
    while running do
      val line = reader.readLine()
      if line == null then running = false
      else
        val trimmed = line.trim
        if log then System.err.println(s"> $trimmed")
        // Echo the command id, if the GUI supplied one.
        val id = trimmed.split("\\s+").headOption.filter(_.forall(_.isDigit)).getOrElse("")
        engine.handle(trimmed) match
          case Right(body) =>
            val tag = s"=$id"
            val out = if body.isEmpty then tag else s"$tag $body"
            println(out)
            println()
            System.out.flush()
            if log then System.err.println(s"< $out")
            if trimmed.toLowerCase.split("\\s+").contains("quit") then running = false
          case Left(error) =>
            println(s"?$id $error")
            println()
            System.out.flush()
            if log then System.err.println(s"< ?$error")
    engine.close()

  val usage: String =
    """Usage: scala-cli run . --main-class go.GTP -- [options]
      |
      |Speaks GTP v2 on stdin/stdout.  Point a GUI at this command, for example
      |in Sabaki or GoGUI, or run engine-vs-engine matches with gogui-twogtp.
      |
      |  --size N        initial board size                (default 19)
      |  --playouts N    MCTS playouts per move            (default 2000 on 9x9)
      |  --time T        time budget per move until the GUI sends time_left
      |  --threads N     search threads                    (default all cores)
      |  --komi F        komi                              (default 6.5)
      |  --resign        allow genmove to return "resign"
      |  --model PATH    ONNX model (needs the onnxruntime jar)
      |  --log           trace commands on stderr
      |  --help          this message""".stripMargin
