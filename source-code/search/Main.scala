// Copyright 2025-2026 Mark Watson. All rights reserved.

/** Main demo for all search algorithms: graph, maze, and Tic-Tac-Toe. */
@main def searchDemo(): Unit =

  // ── Graph Search ───────────────────────────────────────────────
  println("=" * 50)
  println("Graph Search Demo")
  println("=" * 50)

  val graph = Graph(
    nodeDefs = List(
      ("Atlanta",    110, 520),
      ("Baltimore",  520, 360),
      ("Chicago",    310, 200),
      ("Denver",      75, 280),
      ("Erie",       480, 225),
      ("Fresno",      10, 400)
    ),
    edgeDefs = List(
      ("Atlanta",   "Baltimore"),
      ("Atlanta",   "Chicago"),
      ("Atlanta",   "Denver"),
      ("Baltimore", "Chicago"),
      ("Chicago",   "Erie"),
      ("Denver",    "Fresno"),
      ("Denver",    "Chicago")
    )
  )

  val startIdx = 0 // Atlanta
  val goalIdx  = 4 // Erie

  println(s"\nSearching from ${graph.nodes(startIdx).name} to ${graph.nodes(goalIdx).name}:")

  GraphSearch.depthFirst(graph, startIdx, goalIdx) match
    case Some(path) =>
      val names = path.map(graph.nodes(_).name)
      println(s"  DFS path: ${names.mkString(" → ")}")
    case None =>
      println("  DFS: no path found")

  GraphSearch.breadthFirst(graph, startIdx, goalIdx) match
    case Some(path) =>
      val names = path.map(graph.nodes(_).name)
      println(s"  BFS path: ${names.mkString(" → ")}")
    case None =>
      println("  BFS: no path found")

  // ── Maze Search ────────────────────────────────────────────────
  println()
  println("=" * 50)
  println("Maze Search Demo")
  println("=" * 50)

  val maze = Maze.generate(rows = 15, cols = 31)
  val mazeStart = Location(1, 1)
  val mazeGoal  = Location(13, 29)

  println(s"\nMaze (${maze.rows}×${maze.cols}):")
  println(maze.display())

  MazeSearch.breadthFirst(maze, mazeStart, mazeGoal) match
    case Some(path) =>
      println(s"\nBFS solution (${path.length} steps):")
      println(maze.display(path.toSet))
    case None =>
      println("\nNo path found!")

  // ── Tic-Tac-Toe ───────────────────────────────────────────────
  println()
  println("=" * 50)
  println("Tic-Tac-Toe: Computer (X) vs Computer (O)")
  println("=" * 50)

  var board = Board.empty
  var currentPlayer = Player.X

  while !board.isOver do
    Minimax.bestMove(board, currentPlayer) match
      case Some((r, c)) =>
        board = board.set(r, c, currentPlayer)
        println(s"\n$currentPlayer plays ($r, $c):")
        println(board.display)
        currentPlayer = currentPlayer.opponent
      case None => ()

  board.winner match
    case Some(p) => println(s"\n$p wins!")
    case None    => println("\nDraw!")
