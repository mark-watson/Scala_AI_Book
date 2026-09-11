//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package go

import scala.collection.mutable

// ============================================================================
// Board.scala -- Core game logic: board representation, Go rules, scoring.
//
// Everything here is immutable and free of runtime dependencies.  A move never
// mutates a BoardState; it returns a brand new one.  That property is what
// makes the parallel MCTS in Search.scala safe: many virtual threads can read
// the same position at once without locks.
//
// Performance notes for the hot paths (legalMoves / play, called millions of
// times per self-play game):
//   * cells is a flat Array[Byte]; no per-point objects.
//   * neighbour tables are precomputed once per board size.
//   * group traversal is an iterative flood fill over primitive arrays; no
//     boxing, no recursion (a 19x19 group can be 361 stones deep).
//   * Zobrist hashes are updated incrementally, never recomputed.
// ============================================================================

/**
 * A board intersection, encoded as a flat index into the `size * size` grid.
 *
 * This is an `opaque type`, Scala 3's zero-cost newtype.  At runtime a Point
 * *is* an `Int`, so arrays of them stay primitive and cache friendly, but the
 * compiler refuses to let you mix up a point index with, say, a board size.
 *
 * Index layout is row-major with the origin at the top-left corner, which is
 * how the grid is printed:
 *
 * {{{
 *   index = y * size + x        x: 0 = left column ("A")
 *                               y: 0 = top row
 * }}}
 *
 * Note that GTP numbers rows from the *bottom*, so `Point.at(0, 18, 19)` is
 * `A1` in GTP but row 1 in the internal array.  The conversions below hide
 * that difference.
 */
opaque type Point = Int

object Point:
  /** Builds a point from a flat board index. */
  inline def apply(index: Int): Point = index

  /** Builds a point from a zero-based column `x` and row `y` (top-left origin). */
  inline def at(x: Int, y: Int, size: Int): Point = y * size + x

  /** The pass move.  It has no board location, so it is encoded as `-1`. */
  val Pass: Point = -1

  /** Column letters used by Go, which by long tradition omits the letter `I`. */
  private val GtpColumns = "ABCDEFGHJKLMNOPQRSTUVWXYZ"

  extension (p: Point)
    /** The raw flat index, for indexing into arrays. */
    inline def index: Int = p

    inline def isPass: Boolean = p < 0

    /** True when this point lies inside a board of the given size. */
    inline def onBoard(size: Int): Boolean = p >= 0 && p < size * size

    /** Column, `0` at the left edge. */
    inline def x(size: Int): Int = p % size

    /** Row, `0` at the top edge. */
    inline def y(size: Int): Int = p / size

    /** `"D4"`-style coordinate, the notation used by GTP and terminal output. */
    def toGtp(size: Int): String =
      if p.isPass then "pass" else s"${GtpColumns.charAt(p.x(size))}${size - p.y(size)}"

    /** `"dd"`-style coordinate, the notation used inside SGF files. */
    def toSgf(size: Int): String =
      if p.isPass then "" else s"${('a' + p.x(size)).toChar}${('a' + p.y(size)).toChar}"

    /** Alias of [[toGtp]], for readable call sites in UI code. */
    def label(size: Int): String = p.toGtp(size)

  /** The letter for column `x`, skipping `I` as Go convention requires. */
  def columnLetter(x: Int): Char = GtpColumns.charAt(x)

  /**
   * Parses a GTP coordinate such as `"Q16"`, or `"pass"`.
   * Returns `None` for anything off the board or malformed.
   */
  def fromGtp(text: String, size: Int): Option[Point] =
    val t = text.trim.toUpperCase
    if t.isEmpty || t == "PASS" then Some(Pass)
    else if t.length < 2 then None
    else
      val col = GtpColumns.indexOf(t.charAt(0))
      for
        row <- t.substring(1).toIntOption
        if col >= 0 && col < size && row >= 1 && row <= size
      yield at(col, size - row, size)

  /**
   * Parses an SGF coordinate such as `"dd"`.  Both the empty string and the
   * legacy `"tt"` pass encoding map to [[Pass]].
   */
  def fromSgf(text: String, size: Int): Option[Point] =
    val t = text.trim.toLowerCase
    if t.isEmpty || t == "tt" then Some(Pass)
    else if t.length != 2 then None
    else
      val x = t.charAt(0) - 'a'
      val y = t.charAt(1) - 'a'
      if x >= 0 && x < size && y >= 0 && y < size then Some(at(x, y, size)) else None

/** Stone colours.  `ordinal` doubles as the byte stored in the board grid. */
enum Color:
  case Empty, Black, White

  def opposite: Color = this match
    case Black => White
    case White => Black
    case Empty => Empty

  def isStone: Boolean = this != Empty

  /** Stone glyph for ASCII rendering. */
  def glyph: Char = this match
    case Black => 'X'
    case White => 'O'
    case Empty => '.'

object Color:
  /** Decodes the byte stored in [[BoardState.grid]]. */
  def fromByte(b: Byte): Color = b match
    case 1 => Black
    case 2 => White
    case _ => Empty

  /** Parses `"b"`, `"black"`, `"w"`, `"white"`. */
  def parse(text: String): Option[Color] = text.trim.toLowerCase match
    case "b" | "black" => Some(Black)
    case "w" | "white" => Some(White)
    case _             => None

/**
 * A connected set of same-coloured stones, together with the empty points it
 * could grow into.  A group with zero liberties is captured and removed.
 */
final case class Group(color: Color, stones: Set[Point], liberties: Set[Point]):
  def size: Int = stones.size

  /** True when the group is one move away from being captured. */
  def inAtari: Boolean = liberties.size == 1

/** Which counting rule to use when the game ends. */
enum RuleSet:
  /** Chinese rules: score = your stones on the board + territory you surround. */
  case Area
  /** Japanese rules: score = territory surrounded + prisoners you captured. */
  case Territory

/**
 * Zobrist hashing: a random 64-bit value per (colour, point) pair.  XOR-ing the
 * values of all stones on the board yields a position fingerprint that can be
 * updated incrementally as stones are added and removed.
 *
 * The engine uses it for two things:
 *   - superko detection (see [[BoardState.place]]), and
 *   - a cheap position key for the search and for self-play data de-duplication.
 */
object Zobrist:
  private val rng = new java.util.Random(0x5eed5eedL)

  /** `table(colour.ordinal)(point.index)`, sized for the largest board. */
  private val table: Array[Array[Long]] =
    Array.fill(3)(Array.fill(19 * 19)(rng.nextLong()))

  def stone(color: Color, p: Point): Long = table(color.ordinal)(p.index)

/** Precomputed adjacency tables, cached per board size. */
private[go] object Geometry:
  private val cache = mutable.Map.empty[Int, Array[Array[Int]]]

  /** `neighbors(size)(p.index)` gives the orthogonal neighbours of a point. */
  def neighbors(size: Int): Array[Array[Int]] = cache.synchronized {
    cache.getOrElseUpdate(size, buildNeighbors(size))
  }

  private def buildNeighbors(size: Int): Array[Array[Int]] =
    Array.tabulate(size * size) { idx =>
      val x = idx % size
      val y = idx / size
      val buf = mutable.ArrayBuffer.empty[Int]
      if x > 0 then buf += idx - 1
      if x < size - 1 then buf += idx + 1
      if y > 0 then buf += idx - size
      if y < size - 1 then buf += idx + size
      buf.toArray
    }

  /** The traditional star points (hoshi) for a given board size. */
  def starPoints(size: Int): Set[Point] = size match
    case 19 =>
      val e = 3
      val m = 9
      val f = 15
      (for
        x <- List(e, m, f)
        y <- List(e, m, f)
      yield Point.at(x, y, size)).toSet
    case 13 =>
      Set(3, 9).flatMap(x => Set(3, 9).map(y => Point.at(x, y, size))) + Point.at(6, 6, size)
    case 9 =>
      Set(2, 6).flatMap(x => Set(2, 6).map(y => Point.at(x, y, size))) + Point.at(4, 4, size)
    case _ =>
      Set(Point.at(size / 2, size / 2, size))

/** The outcome of a finished game, from Black's point of view. */
final case class Score(black: Double, white: Double):
  /** Black's margin; negative means White wins. */
  def diff: Double = black - white

  def winner: Color =
    if black > white then Color.Black else if white > black then Color.White else Color.Empty

  /** The margin as seen by `color` (positive = that colour is ahead). */
  def fromPerspective(color: Color): Double =
    if color == Color.White then -diff else diff

  override def toString: String = f"B ${black}%.1f - W ${white}%.1f"

/**
 * A complete, immutable Go position.
 *
 * A move returns a new `BoardState` rather than mutating this one, so any
 * position can be shared freely between threads and kept in a search tree.
 */
final class BoardState private[go] (
    /** Board width: normally 9, 13 or 19. */
    val size: Int,
    /** Flat `size * size` grid holding a [[Color]] ordinal per point. */
    private[go] val cells: Array[Byte],
    /** The single point where an immediate recapture is forbidden, if any. */
    val koPoint: Option[Point],
    /**
     * Prisoners taken, as `(capturedByBlack, capturedByWhite)`.  Japanese
     * scoring adds these to the capturer's score.
     */
    val captures: (Int, Int),
    /** Every move played so far, including passes, oldest first. */
    val moveHistory: Vector[Point],
    /** Whose turn it is. */
    val toMove: Color,
    /** Compensation points awarded to White. */
    val komi: Double,
    val ruleSet: RuleSet,
    /** Incremental Zobrist fingerprint of the stones (side-to-move excluded). */
    val zobristHash: Long,
    /** Every stone-position fingerprint seen so far; drives superko detection. */
    val positionHistory: Set[Long],
    /** Consecutive passes.  Two in a row ends the game. */
    val passes: Int,
    /** Set when a player has resigned; the remaining moves are meaningless. */
    val resigned: Option[Color]
):

  // --------------------------------------------------------------------------
  // Accessors
  // --------------------------------------------------------------------------

  /**
   * A defensive copy of the raw grid, kept for compatibility with the design
   * document.  Prefer [[stoneAt]]: it avoids copying 361 bytes per call.
   */
  def grid: Array[Byte] = cells.clone()

  def stoneAt(p: Point): Color =
    if p.onBoard(size) then Color.fromByte(cells(p.index)) else Color.Empty

  def isEmptyAt(p: Point): Boolean = p.onBoard(size) && cells(p.index) == 0

  /** Number of moves played, passes included. */
  def moveCount: Int = moveHistory.length

  def lastMove: Option[Point] = moveHistory.lastOption

  /** All points holding a stone of `color`. */
  def stones(color: Color): Vector[Point] =
    val buf = Vector.newBuilder[Point]
    var i = 0
    while i < size * size do
      if cells(i) == color.ordinal.toByte then buf += Point(i)
      i += 1
    buf.result()

  /** The board area, `size * size`. */
  def area: Int = size * size

  // --------------------------------------------------------------------------
  // Groups and liberties
  // --------------------------------------------------------------------------

  /**
   * Flood fills the group containing `start`.
   *
   * Two hooks let callers test a hypothetical move without building a whole
   * new board, which is how [[place]] checks captures and suicide in one pass
   * over an unmodified grid:
   *
   *   - `extraPoint`/`extraColor` treat one point as if it already held that
   *     colour (the stone being placed);
   *   - `removed` marks points that have been captured by that stone and so
   *     count as empty.  Without it a ko recapture looks like suicide, because
   *     the captured stone would still be blocking the new stone's liberty.
   *
   * Returns `(stones, liberties)` as primitive arrays, no boxing.
   */
  private[go] def flood(
      start: Int,
      extraPoint: Int = -1,
      extraColor: Byte = 0,
      removed: Array[Boolean] = null
  ): (Array[Int], Array[Int]) =
    inline def cellOf(n: Int): Byte =
      if removed != null && removed(n) then 0
      else if n == extraPoint then extraColor
      else cells(n)

    val color = cellOf(start)
    val nbrs = Geometry.neighbors(size)
    val seen = new Array[Boolean](area)
    val stones = new Array[Int](area)
    val libFlags = new Array[Boolean](area)
    val stack = new Array[Int](area)
    var sp = 0
    var stoneCount = 0
    var libCount = 0
    stack(sp) = start
    sp += 1
    seen(start) = true
    while sp > 0 do
      sp -= 1
      val cur = stack(sp)
      stones(stoneCount) = cur
      stoneCount += 1
      val adj = nbrs(cur)
      var i = 0
      while i < adj.length do
        val n = adj(i)
        val nc = cellOf(n)
        if nc == 0 then
          if !libFlags(n) then
            libFlags(n) = true
            libCount += 1
        else if nc == color && !seen(n) then
          seen(n) = true
          stack(sp) = n
          sp += 1
        i += 1
    var i = 0
    val libs = new Array[Int](libCount)
    var j = 0
    while i < area do
      if libFlags(i) then
        libs(j) = i
        j += 1
      i += 1
    (java.util.Arrays.copyOf(stones, stoneCount), libs)

  /** The group (stones + liberties) containing `p`, or `None` if `p` is empty. */
  def groupAt(p: Point): Option[Group] =
    if !p.onBoard(size) || cells(p.index) == 0 then None
    else
      val color = Color.fromByte(cells(p.index))
      val (stones, libs) = flood(p.index)
      Some(Group(color, stones.map(Point(_)).toSet, libs.map(Point(_)).toSet))

  /**
   * Every group on the board, keyed by each of its stones.  This is what the
   * design document calls `groups`; it is computed on demand because carrying
   * it inside every immutable position would dominate the cost of a move.
   */
  def groups: Map[Point, Group] =
    val out = mutable.Map.empty[Point, Group]
    val visited = new Array[Boolean](area)
    var i = 0
    while i < area do
      if cells(i) != 0 && !visited(i) then
        val (stones, libs) = flood(i)
        val color = Color.fromByte(cells(i))
        val group = Group(color, stones.map(Point(_)).toSet, libs.map(Point(_)).toSet)
        var k = 0
        while k < stones.length do
          visited(stones(k)) = true
          out(Point(stones(k))) = group
          k += 1
      i += 1
    out.toMap

  /** Liberty count of the group at `p`, or `0` for an empty point. */
  def libertiesAt(p: Point): Int =
    if !p.onBoard(size) || cells(p.index) == 0 then 0 else flood(p.index)._2.length

  /** Orthogonal neighbours of an in-bounds point. */
  def neighborsOf(p: Point): Vector[Point] =
    if !p.onBoard(size) then Vector.empty
    else Geometry.neighbors(size)(p.index).toVector.map(Point(_))

  def starPoints: Set[Point] = Geometry.starPoints(size)

  // --------------------------------------------------------------------------
  // Rules: what a move would do
  // --------------------------------------------------------------------------

  /** The captured points, ko point and resulting hash of a hypothetical move. */
  private[go] final case class MoveOutcome(captured: Array[Int], koIndex: Int, hash: Long)

  /**
   * Applies the rules to a single stone placement without allocating a new
   * board, returning either the reason it is illegal or what would happen.
   *
   * Handles, in order: off-board, occupied, capture of adjacent enemy groups,
   * suicide, and positional superko.
   */
  private[go] def computeMove(index: Int, color: Color): Either[String, MoveOutcome] =
    if index < 0 || index >= area then Left("point is off the board")
    else if cells(index) != 0 then Left("point is already occupied")
    else
      val me = color.ordinal.toByte
      val opp = color.opposite.ordinal.toByte
      val nbrs = Geometry.neighbors(size)(index)
      val captured = mutable.ArrayBuffer.empty[Int]
      val capSeen = new Array[Boolean](area)
      var hash = zobristHash ^ Zobrist.stone(color, Point(index))
      var i = 0
      while i < nbrs.length do
        val n = nbrs(i)
        if cells(n) == opp then
          val (stones, libs) = flood(n, index, me)
          if libs.isEmpty then
            var k = 0
            while k < stones.length do
              val s = stones(k)
              // A U-shaped enemy group touches the new stone twice; count it once.
              if !capSeen(s) then
                capSeen(s) = true
                captured += s
                hash ^= Zobrist.stone(Color.fromByte(opp), Point(s))
              k += 1
        i += 1
      // The new stone's own group is assessed *after* the captures, so any
      // points just removed count as liberties (this is what makes a ko
      // recapture legal rather than suicidal).
      val (_, ownLibs) = flood(index, index, me, capSeen)
      if ownLibs.isEmpty then Left("suicide is illegal")
      else if positionHistory.contains(hash) then
        Left("positional superko: this position has already occurred")
      else
        // Simple-ko bookkeeping, used for display and for the ko feature plane.
        // Legality itself is decided by the superko test above, which also
        // catches triple ko and other long cycles.
        val ko =
          if captured.length == 1 && ownLibs.length == 1 then captured(0) else -1
        Right(MoveOutcome(captured.toArray, ko, hash))

  /** True when `color` may legally play at `p` (a pass is always legal). */
  def isLegal(p: Point, color: Color): Boolean =
    p.isPass || computeMove(p.index, color).isRight

  /**
   * True when `p` is a genuine one-point eye of `color`: every orthogonal
   * neighbour is already that colour.  Playing there can only reduce the
   * group's liberties, so search and random playouts both refuse to do it.
   *
   * Points on the edge and in the corner have fewer neighbours and are still
   * recognised, which is exactly what makes corner life work.
   */
  def isOwnEye(p: Point, color: Color): Boolean =
    if !p.onBoard(size) || cells(p.index) != 0 then false
    else
      val nbrs = Geometry.neighbors(size)(p.index)
      nbrs.length > 0 && {
        var i = 0
        var allOwn = true
        while i < nbrs.length && allOwn do
          if cells(nbrs(i)) != color.ordinal.toByte then allOwn = false
          i += 1
        allOwn
      }

  /** Why a move is illegal, or `Right(())` when it is legal. */
  def legality(p: Point, color: Color): Either[String, Unit] =
    if p.isPass then Right(()) else computeMove(p.index, color).map(_ => ())

  /**
   * Every legal non-pass move for `color`.  Pass is appended only when
   * `includePass` is set, because search treats passing as a special case.
   */
  def legalMoves(color: Color, includePass: Boolean = false): Vector[Point] =
    val buf = Vector.newBuilder[Point]
    var i = 0
    while i < area do
      if cells(i) == 0 && computeMove(i, color).isRight then buf += Point(i)
      i += 1
    if includePass then buf += Point.Pass
    buf.result()

  // --------------------------------------------------------------------------
  // Applying moves
  // --------------------------------------------------------------------------

  /** Plays `p` for `color`, returning the resulting position or the rule violation. */
  def place(p: Point, color: Color): Either[String, BoardState] =
    if p.isPass then Right(applyPass(color))
    else
      computeMove(p.index, color).map { out =>
        val next = cells.clone()
        next(p.index) = color.ordinal.toByte
        var i = 0
        while i < out.captured.length do
          next(out.captured(i)) = 0
          i += 1
        val (byBlack, byWhite) = captures
        val newCaptures =
          if color == Color.Black then (byBlack + out.captured.length, byWhite)
          else (byBlack, byWhite + out.captured.length)
        new BoardState(
          size = size,
          cells = next,
          koPoint = if out.koIndex >= 0 then Some(Point(out.koIndex)) else None,
          captures = newCaptures,
          moveHistory = moveHistory :+ p,
          toMove = color.opposite,
          komi = komi,
          ruleSet = ruleSet,
          zobristHash = out.hash,
          positionHistory = positionHistory + out.hash,
          passes = 0,
          resigned = resigned
        )
      }

  /** Plays `p` for whoever is to move. */
  def play(p: Point): Either[String, BoardState] = place(p, toMove)

  /** Plays `p` for `color`, throwing on an illegal move. */
  def placeOrThrow(p: Point, color: Color): BoardState =
    place(p, color).fold(
      err => throw new IllegalArgumentException(s"illegal move ${p.label(size)} for $color: $err"),
      identity
    )

  /** Plays a move for whoever is to move, throwing on an illegal move. */
  def playOrThrow(p: Point): BoardState = placeOrThrow(p, toMove)

  /** Passes for `color`.  Always legal. */
  def passFor(color: Color): BoardState = applyPass(color)

  /** Passes for whoever is to move. */
  def playPass: BoardState = applyPass(toMove)

  private def applyPass(color: Color): BoardState =
    // A pass changes no stones, so the Zobrist hash is unchanged.  Recording it
    // again is harmless: superko compares stone positions, and a duplicate
    // fingerprint simply means "this position is already on the record".
    new BoardState(
      size = size,
      cells = cells,
      koPoint = None,
      captures = captures,
      moveHistory = moveHistory :+ Point.Pass,
      toMove = color.opposite,
      komi = komi,
      ruleSet = ruleSet,
      zobristHash = zobristHash,
      positionHistory = positionHistory + zobristHash,
      passes = passes + 1,
      resigned = resigned
    )

  /** Resigns on behalf of `color`; the game ends immediately. */
  def resign(color: Color): BoardState =
    new BoardState(
      size = size,
      cells = cells,
      koPoint = None,
      captures = captures,
      moveHistory = moveHistory,
      toMove = toMove,
      komi = komi,
      ruleSet = ruleSet,
      zobristHash = zobristHash,
      positionHistory = positionHistory,
      passes = passes,
      resigned = Some(color)
    )

  /**
   * Places stones without applying the rules (no captures, no ko, no superko).
   * Used by SGF setup nodes, handicap games and tests.
   */
  def setupStones(stones: Seq[(Point, Color)]): BoardState =
    val next = cells.clone()
    var hash = zobristHash
    for (p, c) <- stones if p.onBoard(size) do
      if next(p.index) != 0 then hash ^= Zobrist.stone(Color.fromByte(next(p.index)), p)
      next(p.index) = c.ordinal.toByte
      hash ^= Zobrist.stone(c, p)
    new BoardState(
      size = size,
      cells = next,
      koPoint = None,
      captures = captures,
      moveHistory = moveHistory,
      toMove = toMove,
      komi = komi,
      ruleSet = ruleSet,
      zobristHash = hash,
      positionHistory = positionHistory + hash,
      passes = passes,
      resigned = resigned
    )

  /** Same position, ignoring move history.  Used by tests and repetition checks. */
  def samePosition(other: BoardState): Boolean =
    size == other.size && toMove == other.toMove && java.util.Arrays.equals(cells, other.cells)

  /** A copy with a different komi, for komi-aware scoring and match setup. */
  def copyWithKomi(komi: Double): BoardState = copyWith(ruleSet = ruleSet, komi = komi)

  /** A copy scored under different rules, e.g. to compare Chinese vs Japanese. */
  def copyWithRules(ruleSet: RuleSet, komi: Double = komi): BoardState =
    copyWith(ruleSet = ruleSet, komi = komi)

  /** A copy with the other player to move.  Handicap games start with White. */
  def withToMove(color: Color): BoardState =
    new BoardState(
      size = size,
      cells = cells,
      koPoint = koPoint,
      captures = captures,
      moveHistory = moveHistory,
      toMove = color,
      komi = komi,
      ruleSet = ruleSet,
      zobristHash = zobristHash,
      positionHistory = positionHistory,
      passes = passes,
      resigned = resigned
    )

  private def copyWith(ruleSet: RuleSet, komi: Double): BoardState =
    new BoardState(
      size = size,
      cells = cells,
      koPoint = koPoint,
      captures = captures,
      moveHistory = moveHistory,
      toMove = toMove,
      komi = komi,
      ruleSet = ruleSet,
      zobristHash = zobristHash,
      positionHistory = positionHistory,
      passes = passes,
      resigned = resigned
    )

  // --------------------------------------------------------------------------
  // Game end and scoring
  // --------------------------------------------------------------------------

  /** Two consecutive passes, or a resignation, ends the game. */
  def isTerminal: Boolean = passes >= 2 || resigned.isDefined

  /** True when both sides have just passed and play stopped without resignation. */
  def isScoredEnd: Boolean = passes >= 2 && resigned.isEmpty

  /**
   * Ownership of each empty point after a flood fill of the empty regions:
   * `1`/`2` for a region touching only Black/White, `0` for neutral (dame).
   */
  def territory: Array[Byte] = BoardState.territoryOf(cells, size)

  /** Every stone that is dead in this position (see [[LifeDeath]]). */
  def deadStones: Set[Point] = LifeDeath.deadStones(this)

  /**
   * The position with dead stones removed and counted as prisoners.
   * Returns `this` when nothing is dead, so the common case allocates nothing.
   */
  def removeDeadStones: BoardState =
    val dead = LifeDeath.deadStones(this)
    if dead.isEmpty then this
    else
      val next = cells.clone()
      var hash = zobristHash
      var (byBlack, byWhite) = captures
      for p <- dead do
        val color = Color.fromByte(next(p.index))
        if color.isStone then
          hash ^= Zobrist.stone(color, p)
          next(p.index) = 0
          if color == Color.Black then byWhite += 1 else byBlack += 1
      new BoardState(
        size = size,
        cells = next,
        koPoint = None,
        captures = (byBlack, byWhite),
        moveHistory = moveHistory,
        toMove = toMove,
        komi = komi,
        ruleSet = ruleSet,
        zobristHash = hash,
        positionHistory = positionHistory + hash,
        passes = passes,
        resigned = resigned
      )

  /** Counts the position under the configured [[RuleSet]]. */
  def score: Score =
    if resigned.isDefined then
      // A resignation is scored as the largest possible loss for the resigner;
      // callers normally use `winner` instead of the numbers.
      resigned.get match
        case Color.Black => Score(0.0, Double.MaxValue)
        case _           => Score(Double.MaxValue, 0.0)
    else
      // Dead stones are removed before counting, so a group that cannot live
      // does not own points it merely stands on.  The analysis is gated on
      // short liberties (see LifeDeath.deadStonesForScore); settled positions
      // score exactly as before and pay nothing extra.
      val dead = LifeDeath.deadStonesForScore(this)
      if dead.isEmpty then count(cells, captures)
      else
        val cleared = cells.clone()
        var (byBlack, byWhite) = captures
        for p <- dead do
          if Color.fromByte(cleared(p.index)) == Color.Black then byWhite += 1
          else byBlack += 1
          cleared(p.index) = 0
        count(cleared, (byBlack, byWhite))

  private def count(grid: Array[Byte], prisoners: (Int, Int)): Score =
    val terr = BoardState.territoryOf(grid, size)
    var blackStones = 0
    var whiteStones = 0
    var blackTerr = 0
    var whiteTerr = 0
    var i = 0
    while i < area do
      grid(i) match
        case 1 => blackStones += 1
        case 2 => whiteStones += 1
        case _ =>
          terr(i) match
            case 1 => blackTerr += 1
            case 2 => whiteTerr += 1
            case _ => ()
      i += 1
    val (byBlack, byWhite) = prisoners
    ruleSet match
      case RuleSet.Area => Score(blackStones + blackTerr, whiteStones + whiteTerr + komi)
      case RuleSet.Territory =>
        Score(blackTerr + byBlack, whiteTerr + byWhite + komi)

  /**
   * The winner, or `None` for a draw.  Resignation short-circuits the count.
   */
  def winner: Option[Color] =
    resigned match
      case Some(loser) => Some(loser.opposite)
      case None =>
        score.winner match
          case Color.Empty => None
          case c           => Some(c)

  /** Human-readable result in the usual notation, e.g. `"B+3.5"` or `"W+Resign"`. */
  def resultString: String =
    val code = (c: Color) => if c == Color.Black then "B" else "W"
    resigned match
      case Some(loser) => s"${code(loser.opposite)}+Resign"
      case None =>
        val s = score
        s.winner match
          case Color.Empty => "0"
          case c =>
            val margin = math.abs(s.diff)
            val shown = if margin == margin.floor then margin.toInt.toString else f"$margin%.1f"
            s"${code(c)}+$shown"

  // --------------------------------------------------------------------------
  // Rendering
  // --------------------------------------------------------------------------

  /**
   * Renders the board as text.  `unicode` switches to the conventional
   * `X`/`O` glyphs for the plain form or heavy dots when a terminal supports
   * UTF-8; ANSI colouring lives in CLI.scala.
   */
  def toAscii(unicode: Boolean = false, showCoordinates: Boolean = true): String =
    val sb = new StringBuilder
    if showCoordinates then
      sb.append("   ")
      var x = 0
      while x < size do
        sb.append(Point.columnLetter(x)).append(' ')
        x += 1
      sb.append('\n')
    val stars = starPoints
    var y = 0
    while y < size do
      sb.append(f"${size - y}%2d ")
      var x = 0
      while x < size do
        val ch = cells(y * size + x) match
          case 1 => 'X'
          case 2 => 'O'
          case _ => if stars.contains(Point.at(x, y, size)) then '+' else '.'
        sb.append(ch).append(' ')
        x += 1
      sb.append(f"${size - y}%2d")
      sb.append('\n')
      y += 1
    if showCoordinates then
      sb.append("   ")
      var x = 0
      while x < size do
        sb.append(Point.columnLetter(x)).append(' ')
        x += 1
      sb.append('\n')
    sb.result()

  override def toString: String = toAscii()

object BoardState:
  /** A board with no stones and Black to move. */
  def empty(
      size: Int = 19,
      komi: Double = 6.5,
      ruleSet: RuleSet = RuleSet.Area
  ): BoardState =
    require(size >= 2 && size <= 19, s"unsupported board size: $size")
    val hash = 0L
    new BoardState(
      size = size,
      cells = new Array[Byte](size * size),
      koPoint = None,
      captures = (0, 0),
      moveHistory = Vector.empty,
      toMove = Color.Black,
      komi = komi,
      ruleSet = ruleSet,
      zobristHash = hash,
      positionHistory = Set(hash),
      passes = 0,
      resigned = None
    )

  /** The standard starting position. */
  def initial(
      size: Int = 19,
      komi: Double = 6.5,
      ruleSet: RuleSet = RuleSet.Area
  ): BoardState = empty(size, komi, ruleSet)

  /** Replays a move list onto an empty board, stopping at the first illegal move. */
  def fromMoves(
      moves: Seq[(Color, Point)],
      size: Int = 19,
      komi: Double = 6.5,
      ruleSet: RuleSet = RuleSet.Area
  ): BoardState =
    var state = empty(size, komi, ruleSet)
    for (color, p) <- moves do
      state = state.place(p, color).fold(err => state, identity)
    state

  /**
   * Ownership of each empty point in `cells` (`1`/`2` for Black/White-only
   * regions, `0` for dame).  The instance method [[BoardState.territory]]
   * delegates here; life-and-death analysis calls this directly on grids with
   * a candidate group lifted off.
   */
  private[go] def territoryOf(cells: Array[Byte], size: Int): Array[Byte] =
    val area = size * size
    val owner = new Array[Byte](area)
    val seen = new Array[Boolean](area)
    val nbrs = Geometry.neighbors(size)
    var i = 0
    while i < area do
      if cells(i) == 0 && !seen(i) then
        val region = mutable.ArrayBuffer.empty[Int]
        val stack = mutable.ArrayBuffer(i)
        seen(i) = true
        var touchesBlack = false
        var touchesWhite = false
        while stack.nonEmpty do
          val cur = stack.remove(stack.length - 1)
          region += cur
          val adj = nbrs(cur)
          var k = 0
          while k < adj.length do
            val n = adj(k)
            cells(n) match
              case 1 => touchesBlack = true
              case 2 => touchesWhite = true
              case _ =>
                if !seen(n) then
                  seen(n) = true
                  stack += n
            k += 1
        val o: Byte =
          if touchesBlack && !touchesWhite then 1
          else if touchesWhite && !touchesBlack then 2
          else 0
        var r = 0
        while r < region.length do
          owner(region(r)) = o
          r += 1
      i += 1
    owner
