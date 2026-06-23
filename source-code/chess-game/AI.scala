package chess

import Piece._

object AI {
  val PIECE_VALUES: Map[Int, Int] = Map(
    PAWN -> 100, KNIGHT -> 320, BISHOP -> 330, ROOK -> 500, QUEEN -> 900, KING -> 20000
  )

  val PAWN_PST: Array[Int] = Array(
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
  )

  val KNIGHT_PST: Array[Int] = Array(
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
  )

  val BISHOP_PST: Array[Int] = Array(
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
  )

  val ROOK_PST: Array[Int] = Array(
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0
  )

  val QUEEN_PST: Array[Int] = Array(
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
  )

  val KING_MIDDLE_PST: Array[Int] = Array(
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20
  )

  val KING_END_PST: Array[Int] = Array(
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50
  )

  val PST_TABLES: Map[Int, Array[Int]] = Map(
    PAWN -> PAWN_PST, KNIGHT -> KNIGHT_PST, BISHOP -> BISHOP_PST,
    ROOK -> ROOK_PST, QUEEN -> QUEEN_PST
  )

  def isEndgame(board: Board): Boolean = {
    var material = 0
    for (ci <- 0 until 2) {
      for (sq <- board.pieces(ci)) {
        val pt = pieceType(board.board(sq).toInt)
        if (pt != PAWN && pt != KING) {
          material += PIECE_VALUES(pt)
        }
      }
    }
    material <= 3000
  }

  def evaluate(board: Board): Int = {
    var score = 0
    // White pieces (ci = 0)
    for (sq <- board.pieces(0)) {
      val pt = pieceType(board.board(sq).toInt)
      score += PIECE_VALUES(pt)
      if (pt != KING) score += PST_TABLES(pt)(sq)
    }
    // Black pieces (ci = 1)
    for (sq <- board.pieces(1)) {
      val pt = pieceType(board.board(sq).toInt)
      score -= PIECE_VALUES(pt)
      if (pt != KING) score -= PST_TABLES(pt)(sq ^ 56)
    }

    // King PST
    val endgame = isEndgame(board)
    val kingTable = if (endgame) KING_END_PST else KING_MIDDLE_PST
    for (ci <- 0 until 2) {
      val it = board.pieces(ci).iterator
      var foundKing = false
      while (it.hasNext && !foundKing) {
        val sq = it.next()
        if (pieceType(board.board(sq).toInt) == KING) {
          foundKing = true
          val mult = if (ci == 0) 1 else -1
          val tableSq = if (ci == 0) sq else (sq ^ 56)
          score += mult * kingTable(tableSq)
        }
      }
    }

    if (board.turn == WHITE) score else -score
  }

  def moveValue(board: Board, move: Move, ttMove: Move): Int = {
    if (ttMove != null && move == ttMove) 1000000
    else if (move.pieceCaptured != EMPTY) {
      10000 + PIECE_VALUES(pieceType(move.pieceCaptured)) - (PIECE_VALUES(pieceType(move.pieceMoved)) / 100)
    }
    else if (move.promotion != EMPTY) 8000 + PIECE_VALUES(move.promotion)
    else if (move.isCastling) 1000
    else {
      val pt = pieceType(move.pieceMoved)
      if (pt != KING) {
        PST_TABLES(pt)(move.to) - PST_TABLES(pt)(move.from)
      } else {
        0
      }
    }
  }

  // Transposition table flags
  val TT_EXACT = 0
  val TT_ALPHA = 1
  val TT_BETA = 2

  case class TTEntry(
    depth: Int,
    score: Int,
    flag: Int,
    bestMove: Move
  )

  val transpositionTable = scala.collection.mutable.LongMap[TTEntry]()

  var maxDepth: Int = 3
  var nodesVisited: Int = 0

  def search(board: Board, depth: Int, alphaInput: Int, betaInput: Int): Int = {
    nodesVisited += 1

    if (board.halfmoveClock >= 100) return 0

    var alpha = alphaInput
    var beta = betaInput

    val ttEntryOpt = transpositionTable.get(board.zobristHash)
    var ttBestMove: Move = null
    val originalAlpha = alpha

    if (ttEntryOpt.isDefined) {
      val ttEntry = ttEntryOpt.get
      if (ttEntry.depth >= depth) {
        ttBestMove = ttEntry.bestMove
        var score = ttEntry.score
        if (score > 29000) score -= (maxDepth - depth)
        else if (score < -29000) score += (maxDepth - depth)

        if (ttEntry.flag == TT_EXACT) return score
        if (ttEntry.flag == TT_ALPHA && score <= alpha) return score
        if (ttEntry.flag == TT_BETA && score >= beta) return score
        if (ttEntry.flag == TT_ALPHA) alpha = Math.max(alpha, score)
        if (ttEntry.flag == TT_BETA) beta = Math.min(beta, score)
        if (alpha >= beta) return score
      } else {
        ttBestMove = ttEntry.bestMove
      }
    }

    val legalMoves = board.getLegalMoves()
    if (legalMoves.isEmpty) {
      if (board.isInCheck()) return -30000 + (maxDepth - depth) // Checkmate
      return 0 // Stalemate
    }

    if (depth == 0) return quiescenceSearch(board, alpha, beta)

    // Sort moves
    val sortedMoves = legalMoves.sortBy(m => -moveValue(board, m, ttBestMove))

    var bestScore = -2000000000
    var bestMove: Move = null

    val it = sortedMoves.iterator
    var cutOff = false
    while (it.hasNext && !cutOff) {
      val move = it.next()
      val state = board.makeMove(move)
      val score = -search(board, depth - 1, -beta, -alpha)
      board.unmakeMove(move, state)

      if (score > bestScore) {
        bestScore = score
        bestMove = move
      }
      alpha = Math.max(alpha, score)
      if (alpha >= beta) {
        cutOff = true
      }
    }

    // Store in TT
    val flag = if (bestScore <= originalAlpha) {
      TT_BETA
    } else if (bestScore >= beta) {
      TT_ALPHA
    } else {
      TT_EXACT
    }

    var storedScore = bestScore
    if (storedScore > 29000) storedScore += (maxDepth - depth)
    else if (storedScore < -29000) storedScore -= (maxDepth - depth)

    val existingOpt = transpositionTable.get(board.zobristHash)
    if (existingOpt.isEmpty || depth >= existingOpt.get.depth) {
      transpositionTable.put(board.zobristHash, TTEntry(depth, storedScore, flag, bestMove))
    }

    bestScore
  }

  def quiescenceSearch(board: Board, alphaInput: Int, beta: Int): Int = {
    nodesVisited += 1
    val standPat = evaluate(board)
    if (standPat >= beta) return beta
    var alpha = alphaInput
    if (standPat > alpha) alpha = standPat

    val captures = board.getPseudoLegalMoves().filter(m => m.pieceCaptured != EMPTY || m.promotion != EMPTY)
      .filter { m =>
        val state = board.makeMove(m)
        val legal = !board.isSquareAttacked(
          board.kingSquare(if (pieceColor(m.pieceMoved) == WHITE) 0 else 1), board.turn
        )
        board.unmakeMove(m, state)
        legal
      }

    val sortedCaptures = captures.sortBy(m => -moveValue(board, m, null))

    val it = sortedCaptures.iterator
    var cutOff = false
    while (it.hasNext && !cutOff) {
      val move = it.next()
      val state = board.makeMove(move)
      val score = -quiescenceSearch(board, -beta, -alpha)
      board.unmakeMove(move, state)

      if (score >= beta) {
        return beta
      }
      if (score > alpha) {
        alpha = score
      }
    }
    alpha
  }

  def getBestMove(board: Board, depth: Int = 3): (Move, Int) = {
    if (transpositionTable.size > 500000) {
      transpositionTable.clear()
    }

    var bestMove: Move = null
    var bestScore = 0

    // Iterative deepening
    for (d <- 1 to depth) {
      maxDepth = d
      nodesVisited = 0

      val legalMoves = board.getLegalMoves()
      val ttEntryOpt = transpositionTable.get(board.zobristHash)
      val ttBest = if (ttEntryOpt.isDefined) ttEntryOpt.get.bestMove else null

      val sortedMoves = legalMoves.sortBy(m => -moveValue(board, m, ttBest))

      var alpha = -2000000000
      val beta = 2000000000
      var currentBest: Move = null
      var currentScore = -2000000000

      for (move <- sortedMoves) {
        val state = board.makeMove(move)
        val score = -search(board, d - 1, -beta, -alpha)
        board.unmakeMove(move, state)

        if (score > currentScore) {
          currentScore = score
          currentBest = move
        }
        if (score > alpha) {
          alpha = score
        }
      }

      if (currentBest != null) {
        bestMove = currentBest
        bestScore = currentScore
        transpositionTable.put(board.zobristHash, TTEntry(
          depth = d,
          score = currentScore,
          flag = TT_EXACT,
          bestMove = currentBest
        ))
      }
    }

    (bestMove, bestScore)
  }
}
