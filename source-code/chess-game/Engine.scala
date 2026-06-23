package chess

import scala.collection.mutable

object Piece {
  val EMPTY = 0
  val PAWN = 1
  val KNIGHT = 2
  val BISHOP = 3
  val ROOK = 4
  val QUEEN = 5
  val KING = 6
  val TYPE_MASK = 7
  val WHITE = 8
  val BLACK = 16
  val COLOR_MASK = 24

  inline def pieceType(p: Int): Int = p & TYPE_MASK
  inline def pieceColor(p: Int): Int = p & COLOR_MASK
  inline def opponent(c: Int): Int = if (c == WHITE) BLACK else WHITE
  inline def colorIdx(c: Int): Int = if (c == WHITE) 0 else 1
}

case class Move(
  from: Int,
  to: Int,
  pieceMoved: Int,
  pieceCaptured: Int = Piece.EMPTY,
  promotion: Int = Piece.EMPTY,
  isEnPassant: Boolean = false,
  isCastling: Boolean = false,
  isDoublePush: Boolean = false
) {
  def uci: String = {
    val base = Board.SQUARE_NAMES(from) + Board.SQUARE_NAMES(to)
    if (promotion != Piece.EMPTY) {
      val promChar = promotion match {
        case Piece.QUEEN  => "q"
        case Piece.ROOK   => "r"
        case Piece.BISHOP => "b"
        case Piece.KNIGHT => "n"
        case _            => ""
      }
      base + promChar
    } else {
      base
    }
  }

  override def equals(other: Any): Boolean = other match {
    case m: Move => from == m.from && to == m.to && promotion == m.promotion
    case _ => false
  }

  override def hashCode(): Int = {
    from * 31 * 31 + to * 31 + promotion
  }

  override def toString: String = uci
}

case class BoardState(
  enPassantSquare: Int,
  castlingRights: Int,
  halfmoveClock: Int,
  zobristHash: Long
)

object Precomputed {
  val KNIGHT_MOVES: Array[Array[Int]] = new Array(64)
  val KING_MOVES: Array[Array[Int]] = new Array(64)
  val ROOK_RAYS: Array[Array[Array[Int]]] = new Array(64)
  val BISHOP_RAYS: Array[Array[Array[Int]]] = new Array(64)
  val QUEEN_RAYS: Array[Array[Array[Int]]] = new Array(64)

  private def precompute(): Unit = {
    val knightOffsets = Array((-2,-1), (-2,1), (-1,-2), (-1,2), (1,-2), (1,2), (2,-1), (2,1))
    val kingOffsets = Array((-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1))
    val rookOffsets = Array((1,0), (-1,0), (0,1), (0,-1))
    val bishopOffsets = Array((1,1), (1,-1), (-1,1), (-1,-1))

    for (sq <- 0 until 64) {
      val r = sq >> 3
      val f = sq & 7

      // Knights
      val kMoves = mutable.ArrayBuffer[Int]()
      for ((dr, df) <- knightOffsets) {
        val nr = r + dr
        val nf = f + df
        if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) kMoves += (nr * 8 + nf)
      }
      KNIGHT_MOVES(sq) = kMoves.toArray

      // Kings
      val kgMoves = mutable.ArrayBuffer[Int]()
      for ((dr, df) <- kingOffsets) {
        val nr = r + dr
        val nf = f + df
        if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) kgMoves += (nr * 8 + nf)
      }
      KING_MOVES(sq) = kgMoves.toArray

      // Rook Rays
      val rRays = mutable.ArrayBuffer[Array[Int]]()
      for ((dr, df) <- rookOffsets) {
        val ray = mutable.ArrayBuffer[Int]()
        var nr = r + dr
        var nf = f + df
        while (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) {
          ray += (nr * 8 + nf)
          nr += dr
          nf += df
        }
        rRays += ray.toArray
      }
      ROOK_RAYS(sq) = rRays.toArray

      // Bishop Rays
      val bRays = mutable.ArrayBuffer[Array[Int]]()
      for ((dr, df) <- bishopOffsets) {
        val ray = mutable.ArrayBuffer[Int]()
        var nr = r + dr
        var nf = f + df
        while (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) {
          ray += (nr * 8 + nf)
          nr += dr
          nf += df
        }
        bRays += ray.toArray
      }
      BISHOP_RAYS(sq) = bRays.toArray

      QUEEN_RAYS(sq) = ROOK_RAYS(sq) ++ BISHOP_RAYS(sq)
    }
  }

  precompute()
}

class Board(fen: String = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1") {
  import Piece._
  import Precomputed._

  val board: Array[Byte] = new Array[Byte](64)
  val pieces: Array[mutable.Set[Int]] = Array(
    mutable.Set[Int](),
    mutable.Set[Int]()
  )
  val kingSquare: Array[Int] = Array(4, 60)
  var turn: Int = WHITE
  var castlingRights: Int = Board.WK | Board.WQ | Board.BK | Board.BQ
  var enPassantSquare: Int = -1
  var halfmoveClock: Int = 0
  var fullmoveNumber: Int = 1
  var zobristHash: Long = 0L

  fromFen(fen)

  def reset(): Unit = {
    fromFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
  }

  def fromFen(fenString: String): Unit = {
    java.util.Arrays.fill(board, 0.toByte)
    pieces(0).clear()
    pieces(1).clear()

    val parts = fenString.split(' ')
    val placement = parts(0)
    val activeColor = parts(1)
    val castling = parts(2)
    val ep = parts(3)
    val halfmove = parts(4)
    val fullmove = parts(5)

    var rank = 7
    var file = 0
    for (ch <- placement) {
      if (ch == '/') {
        rank -= 1
        file = 0
      } else if (ch >= '1' && ch <= '8') {
        file += (ch - '0')
      } else {
        val color = if (ch.isUpper) WHITE else BLACK
        val typeVal = ch.toLower match {
          case 'p' => PAWN
          case 'n' => KNIGHT
          case 'b' => BISHOP
          case 'r' => ROOK
          case 'q' => QUEEN
          case 'k' => KING
        }
        val piece = color | typeVal
        val sq = rank * 8 + file
        board(sq) = piece.toByte
        pieces(colorIdx(color)).add(sq)
        file += 1
      }
    }

    turn = if (activeColor == "w") WHITE else BLACK

    castlingRights = 0
    if (castling.contains('K')) castlingRights |= Board.WK
    if (castling.contains('Q')) castlingRights |= Board.WQ
    if (castling.contains('k')) castlingRights |= Board.BK
    if (castling.contains('q')) castlingRights |= Board.BQ

    enPassantSquare = if (ep == "-") -1 else Board.NAME_TO_SQUARE(ep)
    halfmoveClock = halfmove.toInt
    fullmoveNumber = fullmove.toInt

    kingSquare(0) = findKing(WHITE)
    kingSquare(1) = findKing(BLACK)

    zobristHash = computeZobristHash()
  }

  private def findKing(color: Int): Int = {
    val kPiece = color | KING
    var found = -1
    val s = pieces(colorIdx(color))
    val it = s.iterator
    while (it.hasNext && found == -1) {
      val sq = it.next()
      if (board(sq) == kPiece) found = sq
    }
    found
  }

  def toFen: String = {
    val ranks = mutable.ArrayBuffer[String]()
    for (rank <- 7 to 0 by -1) {
      var empty = 0
      val row = new java.lang.StringBuilder()
      for (file <- 0 until 8) {
        val p = board(rank * 8 + file)
        if (p == EMPTY) {
          empty += 1
        } else {
          if (empty > 0) {
            row.append(empty)
            empty = 0
          }
          row.append(Board.PIECE_CHARS(p.toInt))
        }
      }
      if (empty > 0) {
        row.append(empty)
      }
      ranks += row.toString
    }

    var castling = ""
    if ((castlingRights & Board.WK) != 0) castling += "K"
    if ((castlingRights & Board.WQ) != 0) castling += "Q"
    if ((castlingRights & Board.BK) != 0) castling += "k"
    if ((castlingRights & Board.BQ) != 0) castling += "q"
    if (castling.isEmpty) castling = "-"

    val ep = if (enPassantSquare == -1) "-" else Board.SQUARE_NAMES(enPassantSquare)

    s"${ranks.mkString("/")} ${if (turn == WHITE) "w" else "b"} $castling $ep $halfmoveClock $fullmoveNumber"
  }

  def computeZobristHash(): Long = {
    var h = 0L
    for (sq <- 0 until 64) {
      val p = board(sq).toInt
      if (p != EMPTY) {
        h ^= Board.ZOBRIST_PIECES(sq)(p)
      }
    }
    if (turn == BLACK) {
      h ^= Board.ZOBRIST_SIDE
    }
    h ^= Board.ZOBRIST_CASTLING(castlingRights)
    if (enPassantSquare != -1) {
      h ^= Board.ZOBRIST_EP(enPassantSquare)
    }
    h
  }

  def isSquareAttacked(sq: Int, attackerColor: Int): Boolean = {
    val r = sq >> 3
    val f = sq & 7

    // Pawn attacks
    val pawnDir = if (attackerColor == WHITE) -8 else 8
    val dfValues = Array(-1, 1)
    var i = 0
    var attacked = false
    while (i < 2 && !attacked) {
      val df = dfValues(i)
      val nf = f + df
      if (nf >= 0 && nf <= 7) {
        val pawnSq = sq + pawnDir + df
        if (pawnSq >= 0 && pawnSq < 64 && board(pawnSq) == (attackerColor | PAWN)) {
          attacked = true
        }
      }
      i += 1
    }
    if (attacked) return true

    // Knight attacks
    val kMoves = KNIGHT_MOVES(sq)
    i = 0
    while (i < kMoves.length && !attacked) {
      val t = kMoves(i)
      if (board(t) == (attackerColor | KNIGHT)) {
        attacked = true
      }
      i += 1
    }
    if (attacked) return true

    // King attacks
    val kgMoves = KING_MOVES(sq)
    i = 0
    while (i < kgMoves.length && !attacked) {
      val t = kgMoves(i)
      if (board(t) == (attackerColor | KING)) {
        attacked = true
      }
      i += 1
    }
    if (attacked) return true

    // Rook/Queen rays
    val rRays = ROOK_RAYS(sq)
    var j = 0
    while (j < rRays.length && !attacked) {
      val ray = rRays(j)
      var k = 0
      var blocked = false
      while (k < ray.length && !blocked && !attacked) {
        val t = ray(k)
        val p = board(t).toInt
        if (p != EMPTY) {
          blocked = true
          if (pieceColor(p) == attackerColor && (pieceType(p) == ROOK || pieceType(p) == QUEEN)) {
            attacked = true
          }
        }
        k += 1
      }
      j += 1
    }
    if (attacked) return true

    // Bishop/Queen rays
    val bRays = BISHOP_RAYS(sq)
    j = 0
    while (j < bRays.length && !attacked) {
      val ray = bRays(j)
      var k = 0
      var blocked = false
      while (k < ray.length && !blocked && !attacked) {
        val t = ray(k)
        val p = board(t).toInt
        if (p != EMPTY) {
          blocked = true
          if (pieceColor(p) == attackerColor && (pieceType(p) == BISHOP || pieceType(p) == QUEEN)) {
            attacked = true
          }
        }
        k += 1
      }
      j += 1
    }

    attacked
  }

  def isInCheck(color: Int = turn): Boolean = {
    isSquareAttacked(kingSquare(colorIdx(color)), opponent(color))
  }

  def getPseudoLegalMoves(): List[Move] = {
    val moves = mutable.ListBuffer[Move]()
    val color = turn
    val opp = opponent(color)
    val forward = if (color == WHITE) 8 else -8
    val startRank = if (color == WHITE) 1 else 6
    val promoRank = if (color == WHITE) 7 else 0
    val promoPieces = Array(QUEEN, ROOK, BISHOP, KNIGHT)

    val s = pieces(colorIdx(color))
    for (sq <- s) {
      val piece = board(sq).toInt
      val ptype = pieceType(piece)
      val r = sq >> 3
      val f = sq & 7

      if (ptype == PAWN) {
        val pushSq = sq + forward
        if (board(pushSq) == EMPTY) {
          if ((pushSq >> 3) == promoRank) {
            for (pp <- promoPieces) {
              moves += Move(sq, pushSq, piece, EMPTY, pp)
            }
          } else {
            moves += Move(sq, pushSq, piece)
            if (r == startRank) {
              val doubleSq = pushSq + forward
              if (board(doubleSq) == EMPTY) {
                moves += Move(sq, doubleSq, piece, EMPTY, EMPTY, isDoublePush = true)
              }
            }
          }
        }

        val dfValues = Array(-1, 1)
        for (df <- dfValues) {
          val nf = f + df
          if (nf >= 0 && nf <= 7) {
            val capSq = sq + forward + df
            val target = board(capSq).toInt
            if (target != EMPTY && pieceColor(target) == opp) {
              if ((capSq >> 3) == promoRank) {
                for (pp <- promoPieces) {
                  moves += Move(sq, capSq, piece, target, pp)
                }
              } else {
                moves += Move(sq, capSq, piece, target)
              }
            } else if (capSq == enPassantSquare) {
              val epCaptured = board(capSq - forward).toInt
              moves += Move(sq, capSq, piece, epCaptured, EMPTY, isEnPassant = true)
            }
          }
        }
      } else if (ptype == KNIGHT) {
        val kMoves = KNIGHT_MOVES(sq)
        for (t <- kMoves) {
          val target = board(t).toInt
          if (target == EMPTY) {
            moves += Move(sq, t, piece)
          } else if (pieceColor(target) == opp) {
            moves += Move(sq, t, piece, target)
          }
        }
      } else if (ptype == KING) {
        val kgMoves = KING_MOVES(sq)
        for (t <- kgMoves) {
          val target = board(t).toInt
          if (target == EMPTY) {
            moves += Move(sq, t, piece)
          } else if (pieceColor(target) == opp) {
            moves += Move(sq, t, piece, target)
          }
        }

        // Castling
        if (color == WHITE && sq == 4) {
          if ((castlingRights & Board.WK) != 0 && board(5) == EMPTY && board(6) == EMPTY) {
            if (!isSquareAttacked(4, opp) && !isSquareAttacked(5, opp) && !isSquareAttacked(6, opp)) {
              moves += Move(4, 6, piece, EMPTY, EMPTY, isCastling = true)
            }
          }
          if ((castlingRights & Board.WQ) != 0 && board(3) == EMPTY && board(2) == EMPTY && board(1) == EMPTY) {
            if (!isSquareAttacked(4, opp) && !isSquareAttacked(3, opp) && !isSquareAttacked(2, opp)) {
              moves += Move(4, 2, piece, EMPTY, EMPTY, isCastling = true)
            }
          }
        } else if (color == BLACK && sq == 60) {
          if ((castlingRights & Board.BK) != 0 && board(61) == EMPTY && board(62) == EMPTY) {
            if (!isSquareAttacked(60, opp) && !isSquareAttacked(61, opp) && !isSquareAttacked(62, opp)) {
              moves += Move(60, 62, piece, EMPTY, EMPTY, isCastling = true)
            }
          }
          if ((castlingRights & Board.BQ) != 0 && board(59) == EMPTY && board(58) == EMPTY && board(57) == EMPTY) {
            if (!isSquareAttacked(60, opp) && !isSquareAttacked(59, opp) && !isSquareAttacked(58, opp)) {
              moves += Move(60, 58, piece, EMPTY, EMPTY, isCastling = true)
            }
          }
        }
      } else {
        // Sliding pieces
        val rays = ptype match {
          case BISHOP => BISHOP_RAYS(sq)
          case ROOK   => ROOK_RAYS(sq)
          case _      => QUEEN_RAYS(sq)
        }
        for (ray <- rays) {
          var blocked = false
          var k = 0
          while (k < ray.length && !blocked) {
            val t = ray(k)
            val target = board(t).toInt
            if (target == EMPTY) {
              moves += Move(sq, t, piece)
            } else {
              if (pieceColor(target) == opp) {
                moves += Move(sq, t, piece, target)
              }
              blocked = true
            }
            k += 1
          }
        }
      }
    }
    moves.toList
  }

  def getLegalMoves(): List[Move] = {
    val legal = mutable.ListBuffer[Move]()
    for (move <- getPseudoLegalMoves()) {
      val state = makeMove(move)
      if (!isSquareAttacked(kingSquare(colorIdx(pieceColor(move.pieceMoved))), turn)) {
        legal += move
      }
      unmakeMove(move, state)
    }
    legal.toList
  }

  def makeMove(move: Move): BoardState = {
    val state = BoardState(enPassantSquare, castlingRights, halfmoveClock, zobristHash)

    val color = pieceColor(move.pieceMoved)
    val opp = opponent(color)
    val ci = colorIdx(color)
    val oi = colorIdx(opp)

    // XOR out old state
    zobristHash ^= Board.ZOBRIST_SIDE
    zobristHash ^= Board.ZOBRIST_CASTLING(castlingRights)
    if (enPassantSquare != -1) {
      zobristHash ^= Board.ZOBRIST_EP(enPassantSquare)
    }
    zobristHash ^= Board.ZOBRIST_PIECES(move.from)(move.pieceMoved)

    // Clear from square
    board(move.from) = EMPTY.toByte
    pieces(ci).remove(move.from)

    // Handle captures
    if (move.isEnPassant) {
      val capturedSq = move.to - (if (color == WHITE) 8 else -8)
      val capturedPiece = board(capturedSq).toInt
      board(capturedSq) = EMPTY.toByte
      pieces(oi).remove(capturedSq)
      zobristHash ^= Board.ZOBRIST_PIECES(capturedSq)(capturedPiece)
    } else if (move.pieceCaptured != EMPTY) {
      board(move.to) = EMPTY.toByte
      pieces(oi).remove(move.to)
      zobristHash ^= Board.ZOBRIST_PIECES(move.to)(move.pieceCaptured)
    }

    // Place piece
    val finalPiece = if (move.promotion != EMPTY) color | move.promotion else move.pieceMoved
    board(move.to) = finalPiece.toByte
    pieces(ci).add(move.to)
    zobristHash ^= Board.ZOBRIST_PIECES(move.to)(finalPiece)

    // Update king square
    if (pieceType(move.pieceMoved) == KING) {
      kingSquare(ci) = move.to
    }

    // Move castling rook
    if (move.isCastling) {
      var rookFrom = 0
      var rookTo = 0
      if (move.to == 6) {
        rookFrom = 7; rookTo = 5
      } else if (move.to == 2) {
        rookFrom = 0; rookTo = 3
      } else if (move.to == 62) {
        rookFrom = 63; rookTo = 61
      } else {
        rookFrom = 56; rookTo = 59
      }
      val rook = color | ROOK
      board(rookFrom) = EMPTY.toByte
      pieces(ci).remove(rookFrom)
      zobristHash ^= Board.ZOBRIST_PIECES(rookFrom)(rook)

      board(rookTo) = rook.toByte
      pieces(ci).add(rookTo)
      zobristHash ^= Board.ZOBRIST_PIECES(rookTo)(rook)
    }

    // Update castling rights
    if (pieceType(move.pieceMoved) == KING) {
      castlingRights &= (if (color == WHITE) ~(Board.WK | Board.WQ) else ~(Board.BK | Board.BQ))
    }
    if (move.from == 7 || move.to == 7) castlingRights &= ~Board.WK
    if (move.from == 0 || move.to == 0) castlingRights &= ~Board.WQ
    if (move.from == 63 || move.to == 63) castlingRights &= ~Board.BK
    if (move.from == 56 || move.to == 56) castlingRights &= ~Board.BQ

    // Update en passant
    enPassantSquare = if (move.isDoublePush) move.to - (if (color == WHITE) 8 else -8) else -1

    // Clocks
    if (pieceType(move.pieceMoved) == PAWN || move.pieceCaptured != EMPTY) {
      halfmoveClock = 0
    } else {
      halfmoveClock += 1
    }

    if (color == BLACK) {
      fullmoveNumber += 1
    }

    turn = opp

    // XOR in new state
    zobristHash ^= Board.ZOBRIST_CASTLING(castlingRights)
    if (enPassantSquare != -1) {
      zobristHash ^= Board.ZOBRIST_EP(enPassantSquare)
    }

    state
  }

  def unmakeMove(move: Move, state: BoardState): Unit = {
    val color = pieceColor(move.pieceMoved)
    val ci = colorIdx(color)
    val oi = colorIdx(opponent(color))

    enPassantSquare = state.enPassantSquare
    castlingRights = state.castlingRights
    halfmoveClock = state.halfmoveClock
    zobristHash = state.zobristHash
    turn = color

    if (color == BLACK) {
      fullmoveNumber -= 1
    }

    board(move.to) = EMPTY.toByte
    pieces(ci).remove(move.to)

    board(move.from) = move.pieceMoved.toByte
    pieces(ci).add(move.from)

    if (move.isEnPassant) {
      val capturedSq = move.to - (if (color == WHITE) 8 else -8)
      board(capturedSq) = move.pieceCaptured.toByte
      pieces(oi).add(capturedSq)
    } else if (move.pieceCaptured != EMPTY) {
      board(move.to) = move.pieceCaptured.toByte
      pieces(oi).add(move.to)
    }

    if (pieceType(move.pieceMoved) == KING) {
      kingSquare(ci) = move.from
    }

    if (move.isCastling) {
      var rookFrom = 0
      var rookTo = 0
      if (move.to == 6) {
        rookFrom = 7; rookTo = 5
      } else if (move.to == 2) {
        rookFrom = 0; rookTo = 3
      } else if (move.to == 62) {
        rookFrom = 63; rookTo = 61
      } else {
        rookFrom = 56; rookTo = 59
      }
      val rook = color | ROOK
      board(rookTo) = EMPTY.toByte
      pieces(ci).remove(rookTo)
      board(rookFrom) = rook.toByte
      pieces(ci).add(rookFrom)
    }
  }

  def printBoard(): Unit = {
    for (rank <- 7 to 0 by -1) {
      val row = mutable.ArrayBuffer[String](s"${rank + 1} ")
      for (file <- 0 until 8) {
        val p = board(rank * 8 + file).toInt
        row += (if (p == EMPTY) "." else Board.PIECE_CHARS(p))
      }
      println(row.mkString(" "))
    }
    println("  a b c d e f g h")
  }
}

object Board {
  val WK = 1
  val WQ = 2
  val BK = 4
  val BQ = 8

  val PIECE_CHARS: Map[Int, String] = Map(
    (Piece.WHITE | Piece.PAWN) -> "P", (Piece.WHITE | Piece.KNIGHT) -> "N", (Piece.WHITE | Piece.BISHOP) -> "B",
    (Piece.WHITE | Piece.ROOK) -> "R", (Piece.WHITE | Piece.QUEEN) -> "Q", (Piece.WHITE | Piece.KING) -> "K",
    (Piece.BLACK | Piece.PAWN) -> "p", (Piece.BLACK | Piece.KNIGHT) -> "n", (Piece.BLACK | Piece.BISHOP) -> "b",
    (Piece.BLACK | Piece.ROOK) -> "r", (Piece.BLACK | Piece.QUEEN) -> "q", (Piece.BLACK | Piece.KING) -> "k"
  )

  val SQUARE_NAMES: Array[String] = {
    val arr = new Array[String](64)
    for (row <- 0 until 8; col <- 0 until 8) {
      arr(row * 8 + col) = s"${"abcdefgh"(col)}${"12345678"(row)}"
    }
    arr
  }

  val NAME_TO_SQUARE: Map[String, Int] = SQUARE_NAMES.zipWithIndex.toMap

  class ZobristRng(seed: Long) {
    private var state: Long = seed | 1L
    def nextLong(): Long = {
      state = state * 6364136223846793005L + 1442695040888963407L
      state
    }
  }

  val rng = new ZobristRng(1337L)
  val ZOBRIST_PIECES: Array[Array[Long]] = {
    val arr = Array.ofDim[Long](64, 32)
    for (sq <- 0 until 64; p <- 0 until 32) {
      arr(sq)(p) = rng.nextLong()
    }
    arr
  }

  val ZOBRIST_SIDE: Long = rng.nextLong()

  val ZOBRIST_CASTLING: Array[Long] = {
    val arr = new Array[Long](16)
    for (i <- 0 until 16) arr(i) = rng.nextLong()
    arr
  }

  val ZOBRIST_EP: Array[Long] = {
    val arr = new Array[Long](64)
    for (i <- 0 until 64) arr(i) = rng.nextLong()
    arr
  }
}
