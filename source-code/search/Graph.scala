// Copyright 2025-2026 Mark Watson. All rights reserved.
//> using scala 3.6.4

/** Immutable graph representation for search algorithms.
  *
  * Nodes have a name and (x, y) coordinates used to compute
  * edge distances. Edges are undirected pairs of node indices.
  */
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
