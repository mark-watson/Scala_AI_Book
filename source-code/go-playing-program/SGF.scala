//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package go

/**
 * A complete, tool-portable record of a Go game.
 *
 * This is the in-memory counterpart of an SGF file: everything needed to
 * reconstruct the position and to describe the game to a human or another
 * program.  `SGF.write` turns it into text and `SGF.readRecord` parses it back.
 */
final case class GameRecord(
    boardSize: Int = 19,
    komi: Double = 6.5,
    moves: Vector[(Color, Point)] = Vector.empty,
    /** Stones placed before play starts, e.g. handicap stones. */
    setupStones: Vector[(Point, Color)] = Vector.empty,
    /** Who moves first.  White, in a handicap game. */
    firstColor: Color = Color.Black,
    result: String = "",
    blackName: String = "Black",
    whiteName: String = "White",
    date: String = "",
    handicap: Int = 0,
    ruleSet: RuleSet = RuleSet.Area,
    comment: String = ""
):
  /** Replays the record onto a fresh board, ignoring any move that is illegal. */
  def toBoard: BoardState =
    var s = BoardState.empty(boardSize, komi, ruleSet)
    if setupStones.nonEmpty then s = s.setupStones(setupStones)
    s = s.withToMove(firstColor).copyWithRules(ruleSet, komi)
    moves.foreach { case (color, p) =>
      s = s.place(p, color).fold(_ => s, identity)
    }
    s

  def moveCount: Int = moves.length

object GameRecord:
  /**
   * Captures a played board as a record.  Move colours are recovered from the
   * usual alternation, which holds for every game this engine plays.
   */
  def fromBoard(
      state: BoardState,
      firstColor: Color = Color.Black,
      blackName: String = "Black",
      whiteName: String = "White",
      date: String = "",
      comment: String = ""
  ): GameRecord =
    val moves = state.moveHistory.zipWithIndex.map { (p, i) =>
      val color = if i % 2 == 0 then firstColor else firstColor.opposite
      (color, p)
    }
    GameRecord(
      boardSize = state.size,
      komi = state.komi,
      moves = moves,
      firstColor = firstColor,
      result = if state.isTerminal then state.resultString else "",
      blackName = blackName,
      whiteName = whiteName,
      date = date,
      ruleSet = state.ruleSet,
      comment = comment
    )

  /** Builds a record from a parsed game tree. */
  private[go] def fromTree(tree: SGF.SgfTree): GameRecord =
    val nodes = tree.mainLine
    val root = nodes.headOption.getOrElse(SGF.SgfNode(Vector.empty))

    val size = root.first("SZ").flatMap(parseSize).getOrElse(19)
    val komi = root.first("KM").flatMap(_.trim.toDoubleOption).getOrElse(6.5)
    val handicap = root.first("HA").flatMap(_.trim.toIntOption).getOrElse(0)
    val ruleSet = root.first("RU").map(parseRules).getOrElse(RuleSet.Area)

    // Walk every node's properties in document order.  Setup stones may appear
    // anywhere before play, and move order is positional, so the order matters.
    val setup = Vector.newBuilder[(Point, Color)]
    val moves = Vector.newBuilder[(Color, Point)]
    for node <- nodes do
      for (ident, values) <- node.entries do
        ident match
          case "AB" =>
            values.foreach(v => Point.fromSgf(v, size).foreach(p => setup += ((p, Color.Black))))
          case "AW" =>
            values.foreach(v => Point.fromSgf(v, size).foreach(p => setup += ((p, Color.White))))
          case "B" =>
            values.foreach(v => Point.fromSgf(v, size).foreach(p => moves += ((Color.Black, p))))
          case "W" =>
            values.foreach(v => Point.fromSgf(v, size).foreach(p => moves += ((Color.White, p))))
          case _ => ()

    val setupStones = setup.result()
    GameRecord(
      boardSize = size,
      komi = komi,
      moves = moves.result(),
      setupStones = setupStones,
      // With handicap or setup stones on the board it is White who moves first.
      firstColor = if handicap > 0 || setupStones.nonEmpty then Color.White else Color.Black,
      result = root.first("RE").map(_.trim).filter(_.nonEmpty).getOrElse(""),
      blackName = root.first("PB").getOrElse("Black"),
      whiteName = root.first("PW").getOrElse("White"),
      date = root.first("DT").getOrElse(""),
      handicap = handicap,
      ruleSet = ruleSet,
      comment = root.first("C").getOrElse("")
    )

  private def parseSize(value: String): Option[Int] =
    value.trim.split(':').headOption.flatMap(_.toIntOption)

  private def parseRules(value: String): RuleSet =
    val v = value.trim.toLowerCase
    if v.startsWith("jap") || v.startsWith("terr") then RuleSet.Territory else RuleSet.Area

/**
 * SGF (Smart Game Format) reader and writer.
 *
 * SGF is the interchange format understood by essentially every Go tool:
 * Sabaki, GoGUI, Lizzie, OGS.  The subset implemented here covers what this
 * engine produces and needs:
 *
 *   - the root properties `GM`, `FF`, `SZ`, `KM`, `RE`, `PB`, `PW`, `DT`,
 *     `HA`, `RU` and `C`,
 *   - `AB`/`AW` setup stones,
 *   - `B`/`W` moves, including passes,
 *   - variations and comments, which are parsed and skipped (the main line is
 *     followed) so that files written by other tools do not fail to load.
 */
object SGF:
  /** Writes a full record, including metadata, using FF[4]. */
  def write(record: GameRecord): String =
    val sb = new StringBuilder
    sb.append("(;GM[1]FF[4]CA[UTF-8]AP[go-playing-program:1.0]")
    sb.append("SZ[").append(record.boardSize).append(']')
    sb.append("KM[").append(record.komi).append(']')
    sb.append("RU[").append(ruleName(record.ruleSet)).append(']')
    if record.handicap > 0 then sb.append("HA[").append(record.handicap).append(']')
    if record.result.nonEmpty then sb.append("RE[").append(escape(record.result)).append(']')
    if record.blackName.nonEmpty then sb.append("PB[").append(escape(record.blackName)).append(']')
    if record.whiteName.nonEmpty then sb.append("PW[").append(escape(record.whiteName)).append(']')
    if record.date.nonEmpty then sb.append("DT[").append(escape(record.date)).append(']')
    if record.comment.nonEmpty then sb.append("C[").append(escape(record.comment)).append(']')
    sb.append('\n')

    // Setup stones, grouped by colour the way SGF expects.
    for color <- List(Color.Black, Color.White) do
      val stones = record.setupStones.collect { case (p, c) if c == color => p }
      if stones.nonEmpty then
        val tag = if color == Color.Black then "AB" else "AW"
        sb.append(';').append(tag)
        stones.foreach(p => sb.append('[').append(p.toSgf(record.boardSize)).append(']'))
        sb.append('\n')

    // Moves.  SGF requires exactly one move property per node, so each move
    // gets its own `;B[..]` node; they are wrapped every ten for readability.
    record.moves.grouped(10).foreach { group =>
      group.foreach { (color, p) =>
        sb.append(';')
        sb.append(if color == Color.Black then "B[" else "W[")
        if !p.isPass then sb.append(p.toSgf(record.boardSize))
        sb.append(']')
      }
      sb.append('\n')
    }
    sb.append(")\n")
    sb.result()

  /** Convenience overload matching the signature in the design document. */
  def write(moves: Vector[(Color, Point)], result: String, komi: Float, boardSize: Int): String =
    write(
      GameRecord(boardSize = boardSize, komi = komi.toDouble, moves = moves, result = result)
    )

  /**
   * Parses the first game in `sgf`.
   *
   * Returns the move list and the root properties, matching the design
   * document's signature.  Use [[readRecord]] for the richer form.
   */
  def read(sgf: String): (Vector[(Color, Point)], Map[String, String]) =
    val record = readRecord(sgf).fold(msg => throw new IllegalArgumentException(msg), identity)
    (record.moves, rootProperties(sgf))

  /** Parses the first game in `sgf`, or explains why it could not be read. */
  def readRecord(sgf: String): Either[String, GameRecord] =
    try
      val trees = SgfParser(sgf).parseCollection()
      if trees.isEmpty then Left("no SGF game tree found (expected a '(' ... ')' collection)")
      else Right(GameRecord.fromTree(trees.head))
    catch case e: IllegalArgumentException => Left(e.getMessage)

  /** Parses every game in an SGF collection. */
  def readCollection(sgf: String): Either[String, Vector[GameRecord]] =
    try
      val trees = SgfParser(sgf).parseCollection()
      if trees.isEmpty then Left("no SGF game tree found") else Right(trees.map(GameRecord.fromTree))
    catch case e: IllegalArgumentException => Left(e.getMessage)

  /** Only the root properties, as a `String -> String` map. */
  def rootProperties(sgf: String): Map[String, String] =
    try
      SgfParser(sgf)
        .parseCollection()
        .headOption
        .flatMap(_.nodes.headOption)
        .map(_.props.map((k, v) => k -> v.mkString(",")))
        .getOrElse(Map.empty)
    catch case _: IllegalArgumentException => Map.empty

  private def ruleName(ruleSet: RuleSet): String = ruleSet match
    case RuleSet.Area      => "Chinese"
    case RuleSet.Territory => "Japanese"

  /** Escapes the characters that would otherwise end an SGF property value. */
  private def escape(value: String): String =
    value.replace("\\", "\\\\").replace("]", "\\]").replace("\n", " ")

  /**
   * One SGF node: properties in the order they appeared, each with one or more
   * values.
   *
   * The order is preserved rather than collapsed into a `Map` because SGF
   * encodes move order positionally: in a well-formed file each node holds a
   * single `B` or `W`, but keeping the order means a file that packs several
   * moves into one node (as some hand-written files do) still replays in the
   * right sequence.
   */
  private[go] final case class SgfNode(entries: Vector[(String, Vector[String])]):
    /** A lookup view, for callers that only need a property by name. */
    lazy val props: Map[String, Vector[String]] = entries.toMap

    def first(key: String): Option[String] = props.get(key).flatMap(_.headOption)

    def all(key: String): Vector[String] = props.getOrElse(key, Vector.empty)

  /** One SGF game tree: a linear run of nodes followed by variations. */
  private[go] final case class SgfTree(nodes: Vector[SgfNode], children: Vector[SgfTree]):
    /**
     * Flattens the main line: the nodes of this tree, then those of its first
     * child, and so on.  Variations are deliberately ignored.
     */
    def mainLine: Vector[SgfNode] =
      val out = Vector.newBuilder[SgfNode]
      var current = this
      var running = true
      while running do
        out ++= current.nodes
        current.children.headOption match
          case Some(next) => current = next
          case None       => running = false
      out.result()

/**
 * A small recursive-descent SGF parser.
 *
 * The grammar is tiny:
 * {{{
 *   Collection = GameTree { GameTree }
 *   GameTree   = "(" Sequence { GameTree } ")"
 *   Sequence   = Node { Node }
 *   Node       = ";" { Property }
 *   Property   = PropIdent PropValue { PropValue }
 *   PropValue  = "[" text "]"
 * }}}
 * It is hand-written rather than generated so that the project keeps its
 * zero-dependency promise, and because SGF's escaping rules are easier to get
 * right directly than to express in a grammar.
 */
private[go] final class SgfParser(text: String):
  import SGF.{SgfNode, SgfTree}

  private var pos = 0

  private def peek: Char = if pos < text.length then text.charAt(pos) else '\u0000'

  private def skipWhitespace(): Unit =
    while pos < text.length && text.charAt(pos).isWhitespace do pos += 1

  private def fail(message: String): Nothing =
    throw new IllegalArgumentException(s"SGF parse error at offset $pos: $message")

  private def expect(c: Char): Unit =
    skipWhitespace()
    if peek != c then
      val found = if peek == '\u0000' then "end of input" else s"'$peek'"
      fail(s"expected '$c' but found $found")
    pos += 1

  def parseCollection(): Vector[SgfTree] =
    val out = Vector.newBuilder[SgfTree]
    skipWhitespace()
    while peek == '(' do
      out += parseTree()
      skipWhitespace()
    out.result()

  private def parseTree(): SgfTree =
    expect('(')
    skipWhitespace()
    val nodes = Vector.newBuilder[SgfNode]
    while peek == ';' do
      nodes += parseNode()
      skipWhitespace()
    val children = Vector.newBuilder[SgfTree]
    while peek == '(' do
      children += parseTree()
      skipWhitespace()
    expect(')')
    SgfTree(nodes.result(), children.result())

  private def parseNode(): SgfNode =
    expect(';')
    // Entries are kept in encounter order; repeating a property appends to it.
    val entries = scala.collection.mutable.ArrayBuffer.empty[(String, Vector[String])]
    var indexOf = Map.empty[String, Int]
    skipWhitespace()
    while peek.isLetter && peek.isUpper do
      val ident = parseIdent()
      skipWhitespace()
      val values = Vector.newBuilder[String]
      while peek == '[' do
        values += parseValue()
        skipWhitespace()
      val vs = values.result()
      if vs.nonEmpty then
        indexOf.get(ident) match
          case Some(i) =>
            val (key, existing) = entries(i)
            entries(i) = (key, existing ++ vs)
          case None =>
            indexOf = indexOf.updated(ident, entries.length)
            entries += ((ident, vs))
    SgfNode(entries.toVector)

  private def parseIdent(): String =
    val start = pos
    while peek.isLetter && peek.isUpper do pos += 1
    text.substring(start, pos)

  private def parseValue(): String =
    expect('[')
    val sb = new StringBuilder
    var closed = false
    while !closed do
      if pos >= text.length then fail("unterminated property value")
      val c = text.charAt(pos)
      pos += 1
      if c == '\\' then
        if pos >= text.length then fail("dangling escape in property value")
        val escaped = text.charAt(pos)
        pos += 1
        // A backslash before a newline is a line continuation and vanishes.
        if escaped != '\n' && escaped != '\r' then sb.append(escaped)
      else if c == ']' then closed = true
      else if c == '\n' || c == '\r' || c == '\t' then sb.append(' ')
      else sb.append(c)
    sb.result()
