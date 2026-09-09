//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package ragdocs

// Splitter for RAG input docs. Paragraphs stay whole when short;
// long paragraphs split into overlapping word windows.
object Chunking:
  def split(text: String, maxWords: Int = 120, overlap: Int = 20): List[String] =
    val paras = text.split("\n\\s*\n").map(_.replaceAll("\\s+", " ").trim).filter(_.nonEmpty)
    paras.flatMap(p => window(p, maxWords, overlap)).toList

  private def window(text: String, maxWords: Int, overlap: Int): List[String] =
    val words = text.split(" ").toList
    if words.size <= maxWords then List(text)
    else
      val step = math.max(maxWords - overlap, 1)
      words.sliding(maxWords, step).map(_.mkString(" ")).toList
