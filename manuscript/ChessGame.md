# A Chess Engine and AI Bot

Building a high-performance chess engine is a classic milestone in AI development. It tests both the execution speed of the programming language and the elegance of its design. In this chapter, we explore a chess engine and AI bot written in modern, idiomatic Scala 3. 

The project compiles to efficient JVM bytecode, achieving searches of over **550,000+ NPS** (Nodes Per Second) on standard developer laptops without any native dependencies.

All code is located in `source-code/chess-game`.

## Architecture of a Chess Engine

A competitive chess engine is divided into three parts:
1. **Board Representation & Move Generator**: The rules of the game. It models the board and computes all legal moves for a given position.
2. **Evaluation Function**: The heuristics. It assigns a numerical score to a board position, indicating which side is winning.
3. **Search Engine**: The foresight. It explores the tree of possible future moves to select the best one.

### 1. Board Representation

To achieve maximum performance on the JVM, we avoid object allocation during the search. In **chess-game/Engine.scala**, the board is represented as an array of 64 bytes:

```scala
object Piece {
  val EMPTY = 0
  val PAWN = 1
  val KNIGHT = 2
  val BISHOP = 3
  val ROOK = 4
  val QUEEN = 5
  val KING = 6
  
  val WHITE = 8
  val BLACK = 16
  ...
}

class Board {
  val board: Array[Byte] = new Array[Byte](64)
  val pieces: Array[mutable.Set[Int]] = Array(
    mutable.Set[Int](), // White piece squares
    mutable.Set[Int]()  // Black piece squares
  )
  val kingSquare: Array[Int] = Array(4, 60)
  ...
}
```

Using a 64-element `Array[Byte]` makes square lookups `O(1)`. We track active piece coordinates in `mutable.Set[Int]` collection lists for each player, allowing us to iterate over all active pieces in `O(1)` time rather than scanning all 64 squares.

### 2. Move Generation and Precomputed Tables

Calculating how slider pieces (Rooks, Bishops, Queens) and jump pieces (Knights, Kings) move is computationally expensive. We bypass this cost by precomputing move rays and targets at class initialization time inside the `Precomputed` object:

```scala
object Precomputed {
  val KNIGHT_MOVES: Array[Array[Int]] = new Array(64)
  val KING_MOVES: Array[Array[Int]] = new Array(64)
  val ROOK_RAYS: Array[Array[Array[Int]]] = new Array(64)
  val BISHOP_RAYS: Array[Array[Array[Int]]] = new Array(64)
  ...
}
```

During search, `makeMove` modifies the board arrays in-place to avoid copying overhead, but it returns a lightweight `BoardState` snapshot containing the information needed to restore the state:

```scala
case class BoardState(
  enPassantSquare: Int,
  castlingRights: Int,
  halfmoveClock: Int,
  zobristHash: Long
)
```

The matching `unmakeMove` rolls back mutations in `O(1)` time using the snapshot.

### 3. Zobrist Hashing

To identify duplicate positions in the search tree (transpositions), we use **Zobrist Hashing**. Each square-piece combination, active side, castling right, and en passant file is mapped to a unique random 64-bit number. 

When a move is made, we update the board's hash incrementally using bitwise XOR operations, keeping the hash value accurate in `O(1)` time:

```scala
def computeZobristHash(): Long = {
  var h = 0L
  for (sq <- 0 until 64) {
    val p = board(sq).toInt
    if (p != Piece.EMPTY) {
      h ^= ZOBRIST_PIECES(Piece.colorIdx(Piece.pieceColor(p)))(Piece.pieceType(p))(sq)
    }
  }
  if (turn == Piece.BLACK) h ^= ZOBRIST_BLACK_TO_MOVE
  h ^= ZOBRIST_CASTLING(castlingRights)
  if (enPassantSquare != -1) h ^= ZOBRIST_EP(enPassantSquare & 7)
  h
}
```

## Search Engine: Negamax with Alpha-Beta Pruning

The AI search in **chess-game/AI.scala** uses the **Negamax** formulation of minimax. Because chess is a zero-sum game, White's gain is Black's loss. Negamax takes advantage of this by using the relation:

```$
\max(a, b) = -\min(-a, -b)
```

This allows us to write a single search loop instead of duplicating code for maximizing and minimizing players:

```scala
  def search(board: Board, depth: Int, alphaInput: Int, betaInput: Int): Int = {
    nodesVisited += 1
    if (board.halfmoveClock >= 100) return 0

    var alpha = alphaInput
    var beta = betaInput

    val ttEntryOpt = transpositionTable.get(board.zobristHash)
    var ttBestMove: Move = null
    val originalAlpha = alpha

    // 1. Transposition Table Lookup
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

    // 2. Move Ordering
    val sortedMoves = legalMoves.sortBy(m => -moveValue(board, m, ttBestMove))

    var bestScore = -2000000000
    var bestMove: Move = null

    // 3. Negamax Loop
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
        cutOff = true // Alpha-Beta cut-off
      }
    }

    // 4. Save results to Transposition Table
    ...
```

### Search Optimizations

- **Transposition Table**: We cache previously searched positions in a JVM primitive-optimized `mutable.LongMap` to prevent object boxing and skip re-searching known branches.
- **Iterative Deepening**: We search progressively from depth 1 up to the target depth. Shallow searches compile a list of best moves, which are used to order moves in deeper searches, causing significantly more alpha-beta cut-offs.
- **Quiescence Search**: To prevent the "horizon effect" (where a bad move looks good because a capture happens just past the search limit), we extend the search for captures and promotions until the position stabilizes.
- **Smart Move Ordering**: We order moves by looking at Transposition Table moves first, then captures using MVV-LVA (Most Valuable Victim, Least Valuable Attacker), promotion scores, and positional values from Piece-Square Tables (PSTs).

## Positional Evaluation

Our evaluation combines raw material count with positional bonuses defined by Piece-Square Tables (PSTs). The bot evaluates coordinate maps differently depending on the stage of the game. For example, during the middlegame, it rewards keeping the King safe in the corners; during the endgame, it uses a centralization table to pull the King into active play:

```scala
  val KING_MIDDLE_PST: Array[Int] = Array(
    -30,-40,-40,-50,-50,-40,-40,-30,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20
  )

  val KING_END_PST: Array[Int] = Array(
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -50,-30,-30,-30,-30,-30,-30,-50
  )
```

## Running the Chess CLI and Perft Tests

You can play against the bot or watch it play against itself using `CLI.scala`. Launch it with:

```bash
scala-cli run . --main-class chess.CLI
```

The CLI features an interactive ANSI board rendering and a sidebar tracking the game statistics:

```text
Turn: White  Move: 1  50-move: 0  Eval: 0.0  Castling: KQkq

8 ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜  
7 ♟ ♟ ♟ ♟ ♟ ♟ ♟ ♟  
6                  
5                  
4                  
3                  
2 ♙ ♙ ♙ ♙ ♙ ♙ ♙ ♙  
1 ♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖  
  a b c d e f g h

Your move (or help): e2e4
```

To validate the move generator and check the integrity of Zobrist hashing, run the Perft verification suite:

```bash
scala-cli run . --main-class chess.Perft
```

This counts every reachable terminal state at deep levels, ensuring correctness against official chess standards:

```text
Perft tests from starting position:

Depth 1: 20 nodes (expected 20) [PASS] 0.00s (11,447 nps)
Depth 2: 400 nodes (expected 400) [PASS] 0.00s (272,108 nps)
Depth 3: 8902 nodes (expected 8902) [PASS] 0.01s (706,508 nps)
```
