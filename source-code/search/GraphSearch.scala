// Copyright 2025-2026 Mark Watson. All rights reserved.

/** Depth-first and breadth-first graph search, implemented functionally.
  *
  * Both return Option[List[Int]] — a path of node indices from start to goal,
  * or None if no path exists.
  */
object GraphSearch:

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
