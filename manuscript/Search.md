# Search Algorithms

Search is one of the most fundamental concepts in artificial intelligence. Historically, before the rise of machine learning, AI was almost synonymous with search. In this chapter, we explore how to implement classic search algorithms in Scala 3, taking advantage of its elegant functional features and clean syntax.

We will build three examples:
1. **Graph Search**: Implementing Depth-First Search (DFS) and Breadth-First Search (BFS) over a graph of cities.
2. **Maze Solving**: Generating a random maze using depth-first carving and solving it with BFS to guarantee the shortest path.
3. **Game Tree Search**: Implementing the Minimax algorithm with alpha-beta pruning to play an optimal game of Tic-Tac-Toe.

All code is in the directory `source-code/search`.

## Representing Graphs

To search a graph, we first need to represent it. We define a simple immutable representation in **search/Graph.scala** using Scala 3 case classes. A node contains a name and coordinates, and the graph itself tracks a vector of nodes and undirected edge pairs:

```scala
case class Node(name: String, x: Int, y: Int)

case class Graph(nodes: Vector[Node], edges: Vector[(Int, Int)]):

  /** Return indices of all nodes directly connected to `node`. */
  def neighbors(node: Int): List[Int] =
    edges.collect:
      case (a, b) if a == node => b
      case (a, b) if b == node => a
    .toList

  /** Euclidean distance between two nodes (for A* or display). */
  def distance(a: Int, b: Int): Double =
    val dx = nodes(a).x - nodes(b).x
    val dy = nodes(a).y - nodes(b).y
    math.sqrt(dx * dx + dy * dy)

object Graph:
  /** Build a graph from a list of (name, x, y) and edge pairs (name, name). */
  def apply(nodeDefs: List[(String, Int, Int)], edgeDefs: List[(String, String)]): Graph =
    val nodes = nodeDefs.map((n, x, y) => Node(n, x, y)).toVector
    val nameToIndex = nodes.zipWithIndex.map((n, i) => n.name -> i).toMap
    val edges = edgeDefs.map((a, b) => (nameToIndex(a), nameToIndex(b))).toVector
    Graph(nodes, edges)
```

## Depth-First and Breadth-First Graph Search

We implement DFS and BFS in a purely functional manner in **search/GraphSearch.scala**. Both return an `Option[List[Int]]` representing the path of node indices from start to goal, or `None` if no path is found.

DFS uses recursive backtracking and explores as deep as possible along each branch before backtracking, keeping track of visited nodes to avoid cycles:

```scala
  /** Depth-first search using recursive backtracking. */
  def depthFirst(graph: Graph, start: Int, goal: Int): Option[List[Int]] =
    def search(node: Int, visited: Set[Int]): Option[List[Int]] =
      if node == goal then Some(List(goal))
      else
        val nextNodes = graph.neighbors(node).filterNot(visited.contains)
        nextNodes.view
          .flatMap(n => search(n, visited + n))
          .headOption
          .map(node :: _)

    search(start, Set(start))
```

BFS, on the other hand, explores all neighbors at the current depth before moving deeper. We implement it using a queue of partial paths. Because it processes shallower paths first, BFS is guaranteed to find the shortest path in terms of the number of edges:

```scala
  /** Breadth-first search using a queue of partial paths.
    * Always finds the shortest path (fewest edges).
    */
  def breadthFirst(graph: Graph, start: Int, goal: Int): Option[List[Int]] =
    import scala.collection.immutable.Queue

    @annotation.tailrec
    def search(queue: Queue[(Int, List[Int])], visited: Set[Int]): Option[List[Int]] =
      if queue.isEmpty then None
      else
        val ((node, path), rest) = queue.dequeue
        if node == goal then Some(path.reverse)
        else
          val nextNodes = graph.neighbors(node).filterNot(visited.contains)
          val newVisited = visited ++ nextNodes
          val newEntries = nextNodes.map(n => (n, n :: path))
          search(rest.enqueueAll(newEntries), newVisited)

    search(Queue((start, List(start))), Set(start))
```

## Maze Generation and Solving

Solving mazes is another classic search benchmark. A maze is represented as a 2D grid of booleans where `true` is a wall and `false` is an open passage. We generate a random maze using a DFS-based recursive backtracking algorithm (carving paths through walls) in **search/Maze.scala**:

```scala
case class Location(row: Int, col: Int)

case class Maze(grid: Vector[Vector[Boolean]], rows: Int, cols: Int):
  def isBlocked(loc: Location): Boolean =
    loc.row < 0 || loc.row >= rows ||
    loc.col < 0 || loc.col >= cols ||
    grid(loc.row)(loc.col)

  def openNeighbors(loc: Location): List[Location] =
    List(
      Location(loc.row - 1, loc.col),
      Location(loc.row + 1, loc.col),
      Location(loc.row, loc.col - 1),
      Location(loc.row, loc.col + 1)
    ).filterNot(isBlocked)
```

We solve the maze using either `MazeSearch.depthFirst` or `MazeSearch.breadthFirst`. The BFS solver uses the exact same queue-based strategy to guarantee the shortest path to the exit.

## Game Tree Search: Minimax for Tic-Tac-Toe

Game playing requires looking ahead to predict opponent moves. The Minimax algorithm does this by maximizing the player's score while assuming the opponent will play optimally to minimize it. To make search practical, we use **alpha-beta pruning**, which cuts off branches of the game tree that cannot affect the final decision.

We define the game state in **search/TicTacToe.scala**:

```scala
enum Player:
  case X, O
  def opponent: Player = this match
    case X => O
    case O => X

enum Cell:
  case Filled(player: Player)
  case Empty

case class Board(cells: Vector[Cell]):
  def winner: Option[Player] = ...
  def isDraw: Boolean = winner.isEmpty && emptyCells.isEmpty
  def isOver: Boolean = winner.isDefined || isDraw
```

The minimax search checks game states and recursively computes scores. We use Scala 3's new `scala.util.boundary` and `break` utilities to cleanly implement early pruning:

```scala
object Minimax:

  def score(board: Board, player: Player, depth: Int): Int =
    board.winner match
      case Some(p) if p == player => 10 - depth
      case Some(_)                => depth - 10
      case None                   => 0

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
      ...
```

## Running the Search Demos

The entry point in **search/Main.scala** coordinates all three search demos. First, it searches a small city graph:

```scala
  val graph = Graph(
    nodeDefs = List(
      ("Atlanta",    110, 520),
      ("Baltimore",  520, 360),
      ("Chicago",    310, 200),
      ("Denver",      75, 280),
      ("Erie",       480, 225),
      ("Fresno",      10, 400),
      ("Miami",        20, 800),
      ("Seattle",     600, 100)
    ),
    edgeDefs = List(
      ("Atlanta",   "Baltimore"),
      ("Atlanta",   "Chicago"),
      ("Atlanta",   "Denver"),
      ("Atlanta",   "Miami"),
      ("Baltimore", "Chicago"),
      ("Chicago",   "Erie"),
      ("Chicago",   "Seattle"),
      ("Denver",    "Fresno"),
      ("Denver",    "Chicago")
    )
  )
```

Running the project via `scala-cli run .` produces outputs showing the DFS and BFS paths, the solved maze grid (drawing a path using `.`), and a complete optimal Tic-Tac-Toe game where two computers play each other to a draw:

```text
==================================================
Graph Search Demo
==================================================

Searching from Atlanta to Erie:
  DFS path: Atlanta → Denver → Chicago → Erie
  BFS path: Atlanta → Chicago → Erie

==================================================
Maze Search Demo
==================================================

Maze (15×31):
###############################
# #       #   #   #   #   #   #
# # ### # # # # # # # # # # # #
#   #   #   #   #   #   #   # #
### # ####### # # # # # # # # #
#   # #   #   #   #   #   #   #
# ### # # # ### ### ### ### # #
#   #   # # #   #   #   #   # #
# # ##### # # ### ####### # # #
# #   #   # #   # #   #   # # #
# ### # ### ### # # # # ### # #
#   # #   #   # #   #   #   # #
# # # ##### # ####### ##### # #
# # #       #                 #
###############################

BFS solution (61 steps):
###############################
#.#.......#   #   #   #   #   #
#.#.###.#.# # # # # # # # # # #
#...#   #...#   #   #   #   # #
### # #######.# # # # # # # # #
#   # #   #...#   #   #   #   #
# ### # # #.### ### ### ### # #
#   #   # #.#   #   #   #   # #
# # ##### #.# ### ####### # # #
# #   #   #.#   # #   #   # # #
# #   #   #.# ### # # # # ### # #
#   # #   #.# #   #   #   #   # #
# # # #####.####### ##### # # #
# # #.......#.................#
###############################

==================================================
Tic-Tac-Toe: Computer (X) vs Computer (O)
==================================================

X plays (0, 0):
X . .
. . .
. . .

O plays (1, 1):
X . .
. O .
. . .

...

Draw!
```
