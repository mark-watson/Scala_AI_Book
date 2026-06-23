package chess

import Piece._
import AI._

object CLI {
  val PIECE_GLYPHS: Map[Int, String] = Map(
    (WHITE | KING) -> "♔", (WHITE | QUEEN) -> "♕", (WHITE | ROOK) -> "♖",
    (WHITE | BISHOP) -> "♗", (WHITE | KNIGHT) -> "♘", (WHITE | PAWN) -> "♙",
    (BLACK | KING) -> "♚", (BLACK | QUEEN) -> "♛", (BLACK | ROOK) -> "♜",
    (BLACK | BISHOP) -> "♝", (BLACK | KNIGHT) -> "♞", (BLACK | PAWN) -> "♟"
  )

  def printBoard(board: Board): Unit = {
    val info = scala.collection.mutable.ArrayBuffer[String]()
    info += s"\u001b[1;37mTurn: ${if (board.turn == WHITE) "White" else "Black"}\u001b[0m"
    info += s"Move: ${board.fullmoveNumber}"
    info += s"50-move: ${board.halfmoveClock}"
    if (board.isInCheck()) {
      info += "\u001b[1;31mIN CHECK!\u001b[0m"
    }
    info += s"EP: ${if (board.enPassantSquare == -1) "-" else Board.SQUARE_NAMES(board.enPassantSquare)}"
    
    var castling = ""
    if ((board.castlingRights & Board.WK) != 0) castling += "K"
    if ((board.castlingRights & Board.WQ) != 0) castling += "Q"
    if ((board.castlingRights & Board.BK) != 0) castling += "k"
    if ((board.castlingRights & Board.BQ) != 0) castling += "q"
    info += s"Castling: ${if (castling.nonEmpty) castling else "-"}"
    info += f"Eval: ${evaluate(board) / 100.0}%.1f"

    println("")
    for (rank <- 7 to 0 by -1) {
      val row = scala.collection.mutable.ArrayBuffer[String](s"\u001b[90m${rank + 1}\u001b[0m ")
      for (file <- 0 until 8) {
        val sq = rank * 8 + file
        val p = board.board(sq).toInt
        val bg = if ((rank + file) % 2 == 0) "\u001b[48;5;237m" else "\u001b[48;5;94m"
        if (p == EMPTY) {
          row += s"$bg  \u001b[0m"
        } else {
          val color = if (pieceColor(p) == WHITE) "\u001b[1;37m" else "\u001b[1;35m"
          row += s"$bg$color${PIECE_GLYPHS.getOrElse(p, " ")} \u001b[0m"
        }
      }
      val extra = if (7 - rank < info.length) info(7 - rank) else ""
      println(row.mkString("") + (if (extra.nonEmpty) s"  $extra" else ""))
    }
    println("\u001b[90m  a b c d e f g h\u001b[0m\n")
  }

  def parseMove(board: Board, input: String): Option[Move] = {
    val s = input.trim.toLowerCase
    if (s.length < 4 || s.length > 5) return None
    val fromOpt = Board.NAME_TO_SQUARE.get(s.substring(0, 2))
    val toOpt = Board.NAME_TO_SQUARE.get(s.substring(2, 4))
    if (fromOpt.isEmpty || toOpt.isEmpty) return None
    val from = fromOpt.get
    val to = toOpt.get

    val promChar = if (s.length == 5) s(4) else ' '
    val promType = promChar match {
      case 'q' => QUEEN
      case 'r' => ROOK
      case 'b' => BISHOP
      case 'n' => KNIGHT
      case _   => EMPTY
    }

    val legal = board.getLegalMoves()
    legal.find { m =>
      m.from == from && m.to == to && (
        (promType != EMPTY && m.promotion == promType) ||
        (promType == EMPTY && (m.promotion == EMPTY || m.promotion == QUEEN))
      )
    }
  }

  def main(args: Array[String]): Unit = {
    playGame()
  }

  def playGame(): Unit = {
    def ask(prompt: String): String = {
      print(prompt)
      System.out.flush()
      scala.io.StdIn.readLine()
    }

    println("\n\u001b[1;36m=== Chess Game ===\u001b[0m\n")
    println("1. Play as White")
    println("2. Play as Black")
    println("3. Program vs Program")

    val modeStr = ask("\nChoose mode (1-3): ")
    val mode = try { modeStr.toInt } catch { case _: Exception => 1 }
    val humanColor = if (mode == 1) WHITE else if (mode == 2) BLACK else -1
    
    val depthStr = ask("Program depth (1-6, default 3): ")
    val depthInput = try { depthStr.toInt } catch { case _: Exception => 3 }
    val depth = Math.min(6, Math.max(1, depthInput))

    val board = new Board()
    printBoard(board)

    var playing = true
    while (playing) {
      val legalMoves = board.getLegalMoves()
      if (legalMoves.isEmpty) {
        if (board.isInCheck()) {
          val winner = if (board.turn == WHITE) "Black" else "White"
          println(s"\u001b[1;33mCheckmate! $winner wins.\u001b[0m")
        } else {
          println("\u001b[1;33mStalemate! Draw.\u001b[0m")
        }
        playing = false
      } else if (board.halfmoveClock >= 100) {
        println("\u001b[1;33m50-move rule! Draw.\u001b[0m")
        playing = false
      } else {
        if (humanColor == -1 || board.turn != humanColor) {
          println(s"\u001b[90mProgram is thinking (depth $depth)...\u001b[0m")
          val startTime = System.nanoTime()
          val (move, score) = getBestMove(board, depth)
          val elapsed = ((System.nanoTime() - startTime) / 1e9)
          println(f"Program plays: ${move.uci} | eval: ${score / 100.0}%.1f | nodes: $nodesVisited | time: $elapsed%.1fs")
          board.makeMove(move)
          printBoard(board)
          if (humanColor == -1) {
            Thread.sleep(500)
          }
        } else {
          val input = ask("Your move (or help): ")
          input.trim.toLowerCase match {
            case "exit" | "quit" =>
              println("Goodbye.")
              playing = false
            case "help" =>
              println("Commands: <uci move> (e.g. e2e4), fen, setfen, legal, reset, help, exit")
            case "fen" =>
              println(board.toFen)
            case "setfen" =>
              val fenStr = ask("Enter FEN: ")
              try {
                board.fromFen(fenStr)
                printBoard(board)
              } catch {
                case e: Exception => println(s"Error parsing FEN: ${e.getMessage}")
              }
            case "legal" =>
              println(legalMoves.map(_.uci).mkString(" "))
            case "reset" =>
              board.reset()
              printBoard(board)
            case other =>
              parseMove(board, other) match {
                case Some(move) =>
                  board.makeMove(move)
                  printBoard(board)
                case None =>
                  println("Invalid move. Try UCI format like e2e4, or \"help\" for commands.")
              }
          }
        }
      }
    }
  }
}
