// Copyright 2025-2026 Mark Watson. All rights reserved.

/** Tic-Tac-Toe with minimax + alpha-beta pruning.
  *
  * Demonstrates game-tree search — the computer plays optimally.
  */

enum Player:
  case X, O
  def opponent: Player = this match
    case X => O
    case O => X
  override def toString: String = this match
    case X => "X"
    case O => "O"

enum Cell:
  case Filled(player: Player)
  case Empty
  override def toString: String = this match
    case Filled(p) => p.toString
    case Empty     => "."

case class Board(cells: Vector[Cell]):

  def get(row: Int, col: Int): Cell = cells(row * 3 + col)

  def set(row: Int, col: Int, player: Player): Board =
    Board(cells.updated(row * 3 + col, Cell.Filled(player)))

  def emptyCells: List[(Int, Int)] =
    (for
      r <- 0 until 3
      c <- 0 until 3
      if cells(r * 3 + c) == Cell.Empty
    yield (r, c)).toList

  def winner: Option[Player] =
    val lines = List(
      List(0,1,2), List(3,4,5), List(6,7,8), // rows
      List(0,3,6), List(1,4,7), List(2,5,8), // cols
      List(0,4,8), List(2,4,6)               // diagonals
    )
    lines.collectFirst:
      case line if line.forall(i => cells(i) == Cell.Filled(Player.X)) => Player.X
      case line if line.forall(i => cells(i) == Cell.Filled(Player.O)) => Player.O

  def isDraw: Boolean = winner.isEmpty && emptyCells.isEmpty

  def isOver: Boolean = winner.isDefined || isDraw

  def display: String =
    (0 until 3).map: r =>
      (0 until 3).map(c => get(r, c)).mkString(" ")
    .mkString("\n")

object Board:
  val empty: Board = Board(Vector.fill(9)(Cell.Empty))

object Minimax:

  /** Evaluate the board from the perspective of `player`.
    * Returns a score: +10 for win, -10 for loss, 0 for draw.
    */
  def score(board: Board, player: Player, depth: Int): Int =
    board.winner match
      case Some(p) if p == player => 10 - depth
      case Some(_)                => depth - 10
      case None                   => 0

  /** Minimax with alpha-beta pruning. Returns (bestScore, bestMove). */
  def bestMove(board: Board, player: Player): Option[(Int, Int)] =
    import scala.util.boundary, boundary.break

    def minimax(b: Board, isMaximizing: Boolean, depth: Int, alpha: Int, beta: Int): Int =
      if b.isOver then score(b, player, depth)
      else if isMaximizing then
        boundary:
          var best = Int.MinValue
          var a = alpha
          for (r, c) <- b.emptyCells do
            val s = minimax(b.set(r, c, player), false, depth + 1, a, beta)
            best = math.max(best, s)
            a = math.max(a, s)
            if beta <= a then break(best)
          best
      else
        boundary:
          var best = Int.MaxValue
          var b2 = beta
          for (r, c) <- b.emptyCells do
            val s = minimax(b.set(r, c, player.opponent), true, depth + 1, alpha, b2)
            best = math.min(best, s)
            b2 = math.min(b2, s)
            if b2 <= alpha then break(best)
          best

    if board.isOver then None
    else
      val moves = board.emptyCells
      val scored = moves.map: (r, c) =>
        val s = minimax(board.set(r, c, player), false, 1, Int.MinValue, Int.MaxValue)
        (s, (r, c))
      Some(scored.maxBy(_._1)._2)
