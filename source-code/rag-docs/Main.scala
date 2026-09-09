//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package ragdocs

import java.nio.file.{Files, Paths}

@main def ragDemo(): Unit =
  val corpora = loadCorpora("data")
  val total = corpora.map(_.chunks.size).sum
  println(s"Loaded ${corpora.size} corpora with $total total chunks.")
  val question = "How do solar panels help cut carbon emissions from home charging of electric cars?"
  println(s"\nQuestion: $question")
  println(s"Sub-queries: ${RagPipeline.rewriteQuery(question).mkString(" | ")}")
  val (answerText, context) = RagPipeline.answer(corpora, question)
  println(s"\nUsed ${context.size} context chunks from: ${context.map(_.source).distinct.mkString(", ")}")
  println(s"\n===== ANSWER =====\n$answerText")

def loadCorpora(dir: String): List[Corpus] =
  val d = Paths.get(dir)
  if !Files.isDirectory(d) then return Nil
  Files.list(d).toArray.toList.collect {
    case p: java.nio.file.Path if p.toString.endsWith(".txt") =>
      val text = Files.readString(p)
      Corpus(p.getFileName.toString, Chunking.split(text).map(t => Chunk(p.getFileName.toString, t)))
  }
