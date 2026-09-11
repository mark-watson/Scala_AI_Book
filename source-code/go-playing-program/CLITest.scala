//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package go

import scala.collection.mutable

/**
 * Offline checks for the interactive board's layout.
 *
 * Run with: `make test-cli`.
 *
 * This suite exists because the layout is easy to break silently.  A board
 * whose column letters sit one character to the right of the points still
 * "works" -- it just looks wrong -- so the alignment and the proportions are
 * asserted here rather than eyeballed.
 *
 * A terminal character cell is assumed to be about twice as tall as it is
 * wide, which is what makes "taller than wide" a statement about
 * `2 * lines` versus `width`.
 */
@main def cliTest(): Unit =
  val failures = mutable.ArrayBuffer.empty[String]
  var checks = 0

  def check(label: String)(cond: Boolean): Unit =
    checks += 1
    if !cond then failures += label

  def eq[A](label: String)(actual: A, expected: A): Unit =
    checks += 1
    if actual != expected then failures += s"$label: expected $expected, got $actual"

  /** A board with two stones on it and no search, so the output is fixed. */
  def app(size: Int, tall: Boolean, unicode: Boolean = true): CliApp =
    val a = CliApp(
      network = HeuristicNetwork(),
      config = SearchConfig(playouts = 20, threads = 1, seed = 1L),
      boardSize = size,
      color = false,
      unicode = unicode,
      tall = tall
    )
    a.command("D4")
    a.command("F6")
    a

  val lines9 = app(9, tall = true).render().split("\n").toVector
  val compact9 = app(9, tall = false).render().split("\n").toVector

  /** The index of every character in `text` matching `pattern`. */
  def positions(text: String, pattern: Char => Boolean): Vector[Int] =
    text.zipWithIndex.collect { case (c, i) if pattern(c) => i }.toVector

  /**
   * Just the board: the top header through the bottom header.
   *
   * The information panel is longer than a 9x9 board, so it continues below
   * the bottom header; counting whole renders would measure the panel too.
   */
  def boardLines(lines: Vector[String]): Vector[String] =
    lines.slice(0, lines.lastIndexWhere(_ == lines.head) + 1)

  val size = 9
  /** Row label, three characters per column, trailing label. */
  val boardWidth = 3 + 3 * size + 3

  /**
   * The board half of each line.
   *
   * The panel to the right also contains stone glyphs -- "to play  ● Black" --
   * so glyph counting has to ignore everything past the board.
   */
  def boardColumns(lines: Vector[String]): Vector[String] = lines.map(_.take(boardWidth))

  def glyphLines(lines: Vector[String], glyph: String): Vector[Int] =
    lines.zipWithIndex.collect { case (l, i) if l.contains(glyph) => i }.toVector

  val isColumnLetter = (c: Char) => c >= 'A' && c <= 'T' && c != 'I'
  val isPointGlyph = (c: Char) => c == '·' || c == '+' || c == '●' || c == '○'

  val board9 = boardLines(lines9)
  val header9 = board9.head
  // Row 9 is the first board row, so the header sits directly above it.
  val row9 = board9(1)

  // --------------------------------------------------------------------------
  // The column letters sit over the points, not beside them
  // --------------------------------------------------------------------------
  eq("the header carries one letter per column")(positions(header9, isColumnLetter).size, 9)
  // Three characters of row label, then each column is " glyph ", so the first
  // letter and the first point both land on index 4.
  eq("the first letter sits over the first point")(
    positions(header9, isColumnLetter).head,
    positions(row9, isPointGlyph).head
  )
  eq("the letters line up with the points in row 9")(
    positions(header9, isColumnLetter),
    positions(row9, isPointGlyph)
  )
  // The old layout used three-character cells with a four-character indent,
  // which put every letter one column to the right of its points.
  check("the letters are not one character late")(
    positions(header9, isColumnLetter) != positions(row9, isPointGlyph).map(_ + 1)
  )

  // --------------------------------------------------------------------------
  // The board is taller than it is wide
  // --------------------------------------------------------------------------
  eq("a tall board is two lines per row plus two headers")(board9.size, 2 * size + 2)
  eq("a compact board is one line per row plus two headers")(
    boardLines(compact9).size,
    size + 2
  )
  check("the tall board is taller than it is wide")(
    2 * board9.size > boardWidth
  )
  check("the compact board trades that away to fit a short terminal")(
    2 * boardLines(compact9).size <= boardWidth + 2
  )
  eq("the header is the row label plus three characters per column")(
    header9.length,
    3 + 3 * size
  )
  eq("the board is 33 characters wide on 9x9")(boardWidth, 33)

  // --------------------------------------------------------------------------
  // Stones are round, not flat discs
  // --------------------------------------------------------------------------
  // Black played D4 and White F6, so there is one stone of each colour.
  val board9Columns = boardColumns(board9)
  val blackLines = glyphLines(board9Columns, "●")
  val whiteLines = glyphLines(board9Columns, "○")
  eq("the tall board draws the black stone once")(blackLines.size, 1)
  eq("the tall board draws the white stone once")(whiteLines.size, 1)
  // The stones were once repeated on the following line to make them look
  // round, which read as two stones stacked.  These two checks stop that
  // coming back: no two adjacent lines carry a stone, and the lines between
  // the rows are blank wood.
  val stoneLines = blackLines ++ whiteLines
  check("no two adjacent lines both carry a stone")(
    !stoneLines.exists(i => stoneLines.contains(i + 1))
  )
  eq("every stone sits on a grid line")(
    stoneLines.count(i => board9Columns(i).exists(c => c == '·' || c == '+')),
    stoneLines.size
  )
  eq("half the board lines are the blank bands between rows")(
    board9.drop(1).dropRight(1).count(_.trim.isEmpty),
    size
  )
  eq("the compact board draws the black stone once")(
    glyphLines(boardColumns(boardLines(compact9)), "●").size,
    1
  )

  // --------------------------------------------------------------------------
  // Board sizes and the ASCII fallback
  // --------------------------------------------------------------------------
  val lines19 = app(19, tall = true).render().split("\n").toVector
  val letters19 = positions(lines19.head, isColumnLetter).size
  eq("a 19x19 board has 19 column letters")(letters19, 19)
  check("the letter I is skipped on 19x19")(!lines19.head.contains("I"))
  val board19 = boardLines(lines19)
  eq("a 19x19 tall board is 40 lines")(board19.size, 40)
  check("the 19x19 letters line up with its points")(
    positions(board19.head, isColumnLetter) ==
      positions(board19(1), isPointGlyph).take(19)
  )

  val ascii = app(9, tall = true, unicode = false).render()
  check("ascii mode uses X and O")(ascii.contains("X") && ascii.contains("O"))
  check("ascii mode does not use the unicode empty point")(!ascii.contains("·"))
  check("ascii mode marks empty points with a full stop")(ascii.contains(" ."))

  // --------------------------------------------------------------------------
  // Report
  // --------------------------------------------------------------------------
  if failures.isEmpty then println(s"cliTest: $checks checks passed")
  else
    println(s"cliTest: ${failures.size} of $checks checks FAILED")
    failures.foreach(f => println(s"  - $f"))
    System.exit(1)
