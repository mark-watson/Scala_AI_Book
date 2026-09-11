//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package go

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}
import scala.collection.mutable
import scala.concurrent.duration.{FiniteDuration, TimeUnit}

// ============================================================================
// CLI.scala -- the interactive Go board (design doc section 4.2).
//
// A terminal board with ANSI colours, an evaluation bar, and a small command
// language for playing, asking the engine for a move, reviewing self-play
// games, and saving to SGF.
//
// The board is drawn as a wood-coloured grid with ● and ○ stones; set
// --no-color or NO_COLOR for a plain X/O board that works in any terminal.
// ============================================================================

/** ANSI escape helpers, kept in one place so colour can be switched off. */
object Ansi:
  val Reset = "\u001b[0m"
  val Wood = "\u001b[48;5;180m"
  val BlackStone = "\u001b[38;5;232m"
  val WhiteStone = "\u001b[38;5;255m"
  val LastMoveMark = "\u001b[38;5;160m"
  val StarPoint = "\u001b[38;5;94m"
  val Label = "\u001b[38;5;245m"
  val Heading = "\u001b[1;36m"
  val Dim = "\u001b[2m"
  val Warn = "\u001b[33m"
  val Bad = "\u001b[31m"
  val Good = "\u001b[32m"

/**
 * The interactive application.
 *
 * State is kept in the instance rather than in globals, so a session can be
 * driven programmatically (see Demo.scala) as well as from a terminal.
 */
final class CliApp(
    val network: PolicyValueNetwork = HeuristicNetwork(),
    var config: SearchConfig = SearchConfig.interactive(9),
    var humanColor: Option[Color] = Some(Color.Black),
    var komi: Double = 6.5,
    var boardSize: Int = 9,
    val color: Boolean = true,
    val unicode: Boolean = true,
    /**
     * Draw each board row on two lines, so the stones are round and the board
     * comes out taller than it is wide.  Turn it off for 19x19 in a short
     * terminal, where 40 lines will not fit.
     */
    val tall: Boolean = true
):
  private var state: BoardState = BoardState.initial(boardSize, komi)
  private var history: Vector[BoardState] = Vector(state)
  private var lastSearch: Option[SearchResult] = None
  private var running = true
  private val engineName: String = network.name
  /**
   * The persistent search tree, re-rooted on every move played.  Anything
   * that rewrites the position (undo, new, load, komi) resets it instead.
   */
  private var searchSession: SearchSession | Null = null

  def currentState: BoardState = state

  private def sessionFor(position: BoardState): SearchSession =
    val current = searchSession
    if current != null && current.root.state.samePosition(position) then current
    else
      if current != null then current.close()
      val fresh = Search.newSession(position, network, config)
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

  // --------------------------------------------------------------------------
  // Rendering
  // --------------------------------------------------------------------------

  private def paint(code: String, text: String): String =
    if color then s"$code$text${Ansi.Reset}" else text

  /** A point's glyph, its colour code, and whether it holds a stone. */
  private def pointAt(x: Int, y: Int, lastMove: Option[Point]): (String, String, Boolean) =
    val p = Point.at(x, y, boardSize)
    state.stoneAt(p) match
      case Color.Black => (if unicode then "●" else "X", Ansi.BlackStone, true)
      case Color.White => (if unicode then "○" else "O", Ansi.WhiteStone, true)
      case Color.Empty =>
        val (glyph, code) =
          if lastMove.contains(p) then (if unicode then "•" else "*", Ansi.LastMoveMark)
          else if state.starPoints.contains(p) then ("+", Ansi.StarPoint)
          else (if unicode then "·" else ".", Ansi.StarPoint)
        (glyph, code, false)

  /** Three characters on the grid line: the point, centred in its column. */
  private def gridCell(x: Int, y: Int, lastMove: Option[Point]): String =
    val (glyph, fg, _) = pointAt(x, y, lastMove)
    // The wood background is painted even on empty points, so the board reads
    // as a solid block of colour rather than as loose characters.
    val padded = s" $glyph "
    if color then s"${Ansi.Wood}$fg$padded${Ansi.Reset}" else padded

  /** The evaluation bar, from Black's point of view. */
  private def evaluationBar(width: Int = 22): String =
    val balance = HeuristicEval.balanceFor(state, Color.Black)
    val share = 1.0 / (1.0 + math.exp(-balance / (0.35 * state.area)))
    val filled = math.max(0, math.min(width, math.round(share * width).toInt))
    val bar = "█" * filled + "░" * (width - filled)
    if color then bar else bar

  private def evaluationText: String =
    val balance = HeuristicEval.balanceFor(state, Color.Black)
    val label =
      if math.abs(balance) < 1.0 then "even"
      else if balance > 0 then f"B +${balance / 3.0}%.1f"
      else f"W +${-balance / 3.0}%.1f"
    // The empty argument list matters: `$evaluationBar` alone would be
    // eta-expanded into a function value and printed as a lambda.
    f"${evaluationBar()}  $label%-8s"

  /** The information panel shown to the right of the board. */
  private def sidebar: Vector[String] =
    val captures = state.captures
    val lines = mutable.ArrayBuffer.empty[String]
    lines += paint(Ansi.Heading, "Go Playing Program")
    lines += paint(Ansi.Dim, s"engine   $engineName")
    lines += paint(Ansi.Dim, s"board    ${boardSize}x$boardSize, komi ${state.komi}")
    lines += paint(Ansi.Dim, s"rules    ${state.ruleSet}")
    lines += ""
    lines += s"move     ${state.moveCount + 1}"
    val who = state.toMove match
      case Color.Black => if unicode then "● Black" else "X Black"
      case _           => if unicode then "○ White" else "O White"
    lines += s"to play  $who"
    lines += s"captures B ${captures._1}  W ${captures._2}"
    lines += ""
    lines += "estimate"
    lines += evaluationText
    lines += ""
    lastSearch.foreach { r =>
      lines += paint(Ansi.Dim, "last search")
      lines += paint(Ansi.Dim, f"playouts ${r.playouts}")
      val nps = if r.elapsed.toMillis > 0 then r.playouts * 1000L / r.elapsed.toMillis else 0L
      lines += paint(Ansi.Dim, f"speed    ${nps}%,d n/s")
      lines += paint(Ansi.Dim, f"winrate  ${r.winRate}%.3f")
      val pv = r.principalVariation.take(5).map(_.toGtp(boardSize)).mkString(" ")
      lines += paint(Ansi.Dim, s"pv       $pv")
      lines += ""
    }
    lines += paint(Ansi.Dim, "type help for commands")
    lines.toVector

  /** The board plus the information panel, side by side. */
  def render(): String =
    val sb = new StringBuilder
    val last = state.lastMove
    val panel = sidebar
    // Three characters, matching the width of the row labels, so each column
    // letter sits directly over the points in its column.  The old layout used
    // a four-character indent with three-character cells, which put every
    // letter one column to the right of the point it named.
    val indent = "   "
    val header = indent + (0 until boardSize)
      .map(x => s" ${Point.columnLetter(x)} ")
      .mkString
    sb.append(paint(Ansi.Label, header)).append('\n')
    var y = 0
    while y < boardSize do
      val points = (0 until boardSize).map(x => gridCell(x, y, last)).mkString
      val side = panel.lift(y).map(s => s"   $s").getOrElse("")
      sb.append(paint(Ansi.Label, f"${boardSize - y}%2d ")).append(points)
      sb.append(paint(Ansi.Label, f" ${boardSize - y}%2d")).append(side).append('\n')
      if tall then
        // A blank band of wood between rows.  It gives the board its vertical
        // proportions -- taller than wide -- without drawing the stones twice.
        val blank = " " * (3 * boardSize)
        sb.append("   ")
        sb.append(if color then s"${Ansi.Wood}$blank${Ansi.Reset}" else blank)
        sb.append("   ").append('\n')
      y += 1
    sb.append(paint(Ansi.Label, header)).append('\n')
    // Anything the panel still has to say goes below the board.
    panel.drop(boardSize).foreach(line => sb.append("   ").append(line).append('\n'))
    sb.result()

  // --------------------------------------------------------------------------
  // Game actions
  // --------------------------------------------------------------------------

  private def push(next: BoardState): Unit =
    state = next
    history = history :+ state

  private def play(point: Point): Either[String, Unit] =
    state.play(point) match
      case Right(next) =>
        push(next)
        advanceOrReset(point)
        Right(())
      case Left(reason) => Left(reason)

  private def engineMove(verbose: Boolean = true): Either[String, Point] =
    if state.isTerminal then Left("the game is over")
    else
      val position = state
      val session = sessionFor(position)
      val started = System.nanoTime()
      session.runPlayouts(
        config.playouts,
        config.timeLimit.map(t => started + t.toNanos)
      )
      val result = session.result(
        FiniteDuration(System.nanoTime() - started, TimeUnit.NANOSECONDS)
      )
      lastSearch = Some(result)
      position.play(result.move) match
        case Right(next) =>
          push(next)
          session.advanceTo(result.move)
          if verbose then
            val nps =
              if result.elapsed.toMillis > 0 then result.playouts * 1000L / result.elapsed.toMillis else 0L
            println(
              paint(
                Ansi.Dim,
                f"engine plays ${result.move.toGtp(boardSize)} " +
                  f"(${result.playouts} playouts, ${nps}%,d n/s, winrate ${result.winRate}%.3f)"
              )
            )
          Right(result.move)
        case Left(reason) => Left(reason)

  private def reset(size: Int, newKomi: Double): Unit =
    boardSize = size
    komi = newKomi
    state = BoardState.initial(size, newKomi)
    history = Vector(state)
    lastSearch = None
    resetSession()

  // --------------------------------------------------------------------------
  // Command handling
  // --------------------------------------------------------------------------

  /** Runs one command; returns false when the session should end. */
  def command(line: String): Boolean =
    val tokens = line.trim.split("\\s+").toVector.filter(_.nonEmpty)
    if tokens.isEmpty then return true
    val name = tokens.head.toLowerCase
    val args = tokens.tail
    try
      name match
        case "quit" | "exit" | "q" => running = false
        case "help" | "h" | "?"    => println(helpText)
        case "board" | "show" | "b" => () // the board is redrawn every turn
        case "pass" =>
          push(state.playPass)
          advanceOrReset(Point.Pass)
          println(paint(Ansi.Dim, s"${state.toMove.opposite} passes"))
        case "resign" =>
          val loser = state.toMove
          push(state.resign(loser))
          println(paint(Ansi.Warn, s"${loser} resigns: ${state.resultString}"))
        case "undo" | "u" =>
          if history.length <= 1 then println(paint(Ansi.Warn, "nothing to undo"))
          else
            history = history.dropRight(1)
            state = history.last
            lastSearch = None
            resetSession()
        case "genmove" | "g" | "move" => engineMove()
        case "new" =>
          val size = args.headOption.flatMap(_.toIntOption).getOrElse(boardSize)
          val newKomi = args.lift(1).flatMap(_.toDoubleOption).getOrElse(6.5)
          reset(size, newKomi)
          println(paint(Ansi.Dim, s"new ${size}x$size game, komi $newKomi"))
        case "komi" =>
          args.headOption.flatMap(_.toDoubleOption) match
            case Some(k) =>
              komi = k
              state = state.copyWithKomi(k)
              history = history.updated(history.length - 1, state)
              resetSession()
            case None => println(paint(Ansi.Warn, "usage: komi <number>"))
        case "score" | "final" =>
          val s = state.score
          println(f"score: B ${s.black}%.1f - W ${s.white}%.1f   (${state.resultString})")
        case "hint" =>
          val count = args.headOption.flatMap(_.toIntOption).getOrElse(5)
          showHint(count)
        case "settle" | "finish" =>
          // Play out the rest of the game with both sides passing as soon as
          // the engine judges there is nothing left to gain.
          var guard = 0
          while !state.isTerminal && guard < boardSize * boardSize * 2 do
            engineMove(verbose = false)
            guard += 1
          println(paint(Ansi.Good, s"game finished: ${state.resultString}"))
          println(state.toAscii())
        case "auto" =>
          val moves = args.headOption.flatMap(_.toIntOption).getOrElse(10)
          var i = 0
          while i < moves && !state.isTerminal do
            engineMove(verbose = false)
            i += 1
          println(render())
        case "eval" =>
          println(evaluationText)
        case "save" =>
          args.headOption match
            case Some(file) => saveSgf(file)
            case None       => println(paint(Ansi.Warn, "usage: save <file.sgf>"))
        case "load" =>
          args.headOption match
            case Some(file) => loadSgf(file)
            case None       => println(paint(Ansi.Warn, "usage: load <file.sgf>"))
        case "color" =>
          args.headOption.flatMap(Color.parse) match
            case Some(c) =>
              humanColor = Some(c)
              println(paint(Ansi.Dim, s"you play $c"))
            case None => println(paint(Ansi.Warn, "usage: color black|white"))
        case other =>
          // Anything else that looks like a vertex is treated as a move.
          Point.fromGtp(other, boardSize) match
            case Some(p) =>
              play(p) match
                case Right(_)   => ()
                case Left(reason) => println(paint(Ansi.Bad, s"illegal: $reason"))
            case None => println(paint(Ansi.Warn, s"unknown command: $name (try help)"))
    catch
      case e: Throwable => println(paint(Ansi.Bad, s"error: ${e.getMessage}"))
    running

  private def showHint(count: Int): Unit =
    val result = Search.search(state, network, config)
    lastSearch = Some(result)
    val ranked = result.visits.toVector.sortBy(-_._2).take(count)
    val total = result.visits.values.sum
    println(paint(Ansi.Heading, s"top $count moves:"))
    ranked.foreach { (p, n) =>
      val share = if total > 0 then n.toDouble / total else 0.0
      val prior = result.policy.getOrElse(p, 0f)
      println(f"  ${p.toGtp(boardSize)}%-5s visits=$n%6d  ${share * 100}%5.1f%% of playouts  prior=${prior * 100}%5.1f%%")
    }
    println(paint(Ansi.Dim, f"winrate ${result.winRate}%.3f over ${result.playouts} playouts"))

  private def saveSgf(file: String): Unit =
    val record = GameRecord
      .fromBoard(state, firstColor = Color.Black, blackName = "human", whiteName = engineName)
      .copy(result = if state.isTerminal then state.resultString else "")
    Files.writeString(Paths.get(file), SGF.write(record), StandardCharsets.UTF_8)
    println(paint(Ansi.Good, s"saved ${state.moveCount} moves to $file"))

  private def loadSgf(file: String): Unit =
    if !Files.exists(Paths.get(file)) then println(paint(Ansi.Bad, s"no such file: $file"))
    else
      val text = new String(Files.readAllBytes(Paths.get(file)), StandardCharsets.UTF_8)
      SGF.readRecord(text) match
        case Left(msg) => println(paint(Ansi.Bad, s"cannot read $file: $msg"))
        case Right(record) =>
          reset(record.boardSize, record.komi)
          var current = state
          record.moves.foreach { (c, p) => current = current.place(p, c).fold(_ => current, identity) }
          state = current
          history = Vector(state)
          println(paint(Ansi.Good, s"loaded ${record.moves.length} moves from $file"))

  // --------------------------------------------------------------------------
  // Main loop
  // --------------------------------------------------------------------------

  /** True while the session should keep going. */
  def isRunning: Boolean = running && !state.isTerminal

  /** Plays one engine move if it is not the human's turn. */
  def stepEngineIfNeeded(): Unit =
    humanColor match
      case Some(human) if state.toMove != human && !state.isTerminal => engineMove()
      case None if !state.isTerminal                                => engineMove()
      case _                                                        => ()

  /** The interactive read-eval-print loop. */
  def run(): Unit =
    println(banner)
    while running && !state.isTerminal do
      println(render())
      if humanColor.exists(_ == state.toMove) then
        print(paint(Ansi.Heading, "> "))
        Console.flush()
        val line = scala.io.StdIn.readLine()
        if line == null then running = false
        else command(line)
      else engineMove()
    if state.isTerminal then
      println(render())
      println(paint(Ansi.Good, s"game over: ${state.resultString}"))
    println(paint(Ansi.Dim, "bye"))

  private def banner: String =
    paint(
      Ansi.Heading,
      s"""Go Playing Program -- Scala 3 MCTS engine
         |engine: $engineName, ${config.playouts} playouts, ${config.threads} threads
         |""".stripMargin
    )

  val helpText: String =
    """Commands
      |  D4 / play D4     play a move (GTP coordinates, column letters skip I)
      |  pass             pass
      |  resign           resign the game
      |  genmove, g       let the engine move now
      |  hint [n]         show the engine's top n candidate moves
      |  undo, u          take back a move
      |  score            score the current position
      |  settle           play the game out to a result
      |  auto [n]         let the engine play n moves against itself
      |  new [size] [komi] start a new game
      |  komi <f>         change komi
      |  color black|white  choose your colour
      |  save <file.sgf>  save the game
      |  load <file.sgf>  load a game
      |  board            redraw
      |  quit             exit
      |
      |Coordinates look like D4, Q16 or J9.  Columns run A..T with I omitted,
      |rows are numbered from 1 at the bottom.""".stripMargin

object CLI:
  def main(args: Array[String]): Unit =
    val parsed = Args.parse(args)
    if parsed.has("help") then
      println(usage)
      return

    val size = parsed.int("size", 9)
    val playouts = parsed.int("playouts", if size <= 9 then 2000 else 4000)
    val threads = parsed.int("threads", math.max(1, Runtime.getRuntime.availableProcessors()))
    val komi = parsed.double("komi", 6.5)
    val modelPath = parsed.get("model")

    val noColor = parsed.bool("no-color", false) ||
      sys.env.contains("NO_COLOR") ||
      sys.env.get("TERM").contains("dumb")
    val ascii = parsed.bool("ascii", false)
    val compact = parsed.bool("compact", false)

    val humanColor: Option[Color] =
      if parsed.bool("auto", false) || parsed.bool("both", false) then None
      else parsed.get("color").flatMap(Color.parse).orElse(Some(Color.Black))

    val network: PolicyValueNetwork = modelPath match
      case Some(path) => OnnxNetwork(path)
      case None       => HeuristicNetwork()

    // A per-move time limit, as an alternative to a playout count.
    val timeLimit = parsed.duration("time")

    val app = new CliApp(
      network = network,
      config = SearchConfig(
        playouts = playouts,
        threads = threads,
        timeLimit = timeLimit,
        seed = System.nanoTime()
      ),
      humanColor = humanColor,
      komi = komi,
      boardSize = size,
      color = !noColor,
      unicode = !ascii,
      tall = !compact
    )
    app.run()

  val usage: String =
    """Usage: scala-cli run . -- [options]
      |
      |Runs the interactive board.  By default you play Black and the engine
      |plays White; commands are listed in-game with `help`.
      |
      |  --size N        board size                      (default 9)
      |  --playouts N    MCTS playouts per move          (default 2000 on 9x9)
      |  --time T        time budget per move, e.g. 5s or 250ms
      |  --threads N     search threads                  (default all cores)
      |  --komi F        komi                            (default 6.5)
      |  --color C       your colour, black or white     (default black)
      |  --auto          engine plays both sides
      |  --model PATH    ONNX model (needs the onnxruntime jar)
      |  --no-color      plain output, no ANSI colour
      |  --ascii         X/O stones instead of ●/○
      |  --compact       one line per row, for 19x19 in a short terminal
      |  --help          this message""".stripMargin
