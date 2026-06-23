package chess

import Piece._

object Perft {
  def perft(board: Board, depth: Int): Long = {
    if (depth == 0) return 1L
    var nodes = 0L
    val legal = board.getLegalMoves()
    for (move <- legal) {
      val state = board.makeMove(move)
      
      // Verify Zobrist hash consistency
      val recomputed = board.computeZobristHash()
      if (recomputed != board.zobristHash) {
        System.err.println(s"Hash mismatch after ${move.uci}: stored=${board.zobristHash} recomputed=$recomputed")
        System.exit(1)
      }
      
      nodes += perft(board, depth - 1)
      board.unmakeMove(move, state)
      
      // Verify hash restored
      if (board.zobristHash != state.zobristHash) {
        System.err.println(s"Hash not restored after unmake ${move.uci}")
        System.exit(1)
      }
    }
    nodes
  }

  def main(args: Array[String]): Unit = {
    val board = new Board()
    val expected = Array(20L, 400L, 8902L)

    println("Perft tests from starting position:\n")

    for (depth <- 1 to 3) {
      val start = System.nanoTime()
      val nodes = perft(board, depth)
      val elapsed = (System.nanoTime() - start) / 1e9
      val status = if (nodes == expected(depth - 1)) "PASS" else "FAIL"
      val nps = Math.round(nodes.toDouble / elapsed)
      println(f"Depth $depth: $nodes nodes (expected ${expected(depth - 1)}) [$status] $elapsed%.2fs ($nps%,d nps)")
    }
  }
}
