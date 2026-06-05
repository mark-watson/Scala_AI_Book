// Copyright 2025-2026 Mark Watson. All rights reserved.

/** Maze generation and solving using DFS and BFS.
  *
  * A maze is a 2D grid where `true` means a wall (blocked) and
  * `false` means an open passage.
  */
case class Location(row: Int, col: Int)

case class Maze(grid: Vector[Vector[Boolean]], rows: Int, cols: Int):

  def isBlocked(loc: Location): Boolean =
    loc.row < 0 || loc.row >= rows ||
    loc.col < 0 || loc.col >= cols ||
    grid(loc.row)(loc.col)

  /** Four-directional neighbors that are in-bounds and open. */
  def openNeighbors(loc: Location): List[Location] =
    List(
      Location(loc.row - 1, loc.col),
      Location(loc.row + 1, loc.col),
      Location(loc.row, loc.col - 1),
      Location(loc.row, loc.col + 1)
    ).filterNot(isBlocked)

  /** ASCII display of the maze, optionally highlighting a path with '.' */
  def display(path: Set[Location] = Set.empty): String =
    (0 until rows).map: r =>
      (0 until cols).map: c =>
        val loc = Location(r, c)
        if path.contains(loc) then '.'
        else if grid(r)(c) then '#'
        else ' '
      .mkString
    .mkString("\n")

object Maze:
  /** Generate a random maze using recursive backtracking (DFS carving). */
  def generate(rows: Int, cols: Int, seed: Long = 42L): Maze =
    val rng = scala.util.Random(seed)
    val grid = Array.fill(rows, cols)(true) // start all walls

    def carve(r: Int, c: Int): Unit =
      grid(r)(c) = false
      val directions = rng.shuffle(List((0, 2), (0, -2), (2, 0), (-2, 0)))
      for (dr, dc) <- directions do
        val nr = r + dr
        val nc = c + dc
        if nr >= 0 && nr < rows && nc >= 0 && nc < cols && grid(nr)(nc) then
          grid(r + dr / 2)(c + dc / 2) = false // knock out wall between
          carve(nr, nc)

    carve(1, 1) // start from (1,1)
    Maze(grid.map(_.toVector).toVector, rows, cols)

object MazeSearch:

  /** Solve a maze using depth-first search. */
  def depthFirst(maze: Maze, start: Location, goal: Location): Option[List[Location]] =
    def search(loc: Location, visited: Set[Location]): Option[List[Location]] =
      if loc == goal then Some(List(goal))
      else
        maze.openNeighbors(loc)
          .filterNot(visited.contains)
          .view
          .flatMap(n => search(n, visited + n))
          .headOption
          .map(loc :: _)

    search(start, Set(start))

  /** Solve a maze using breadth-first search (shortest path). */
  def breadthFirst(maze: Maze, start: Location, goal: Location): Option[List[Location]] =
    import scala.collection.immutable.Queue

    @annotation.tailrec
    def search(queue: Queue[(Location, List[Location])], visited: Set[Location]): Option[List[Location]] =
      if queue.isEmpty then None
      else
        val ((loc, path), rest) = queue.dequeue
        if loc == goal then Some(path.reverse)
        else
          val nextLocs = maze.openNeighbors(loc).filterNot(visited.contains)
          val newVisited = visited ++ nextLocs
          val newEntries = nextLocs.map(n => (n, n :: path))
          search(rest.enqueueAll(newEntries), newVisited)

    search(Queue((start, List(start))), Set(start))
