//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package ragdocs

// Offline checks: chunking, ranking, rewrite, and prompt shape.
// Run with: scala-cli run . --main-class ragdocs.ragTest
@main def ragTest(): Unit =
  val energy = Chunk("energy.txt", "Solar panels convert sunlight to electricity. Home solar cuts grid demand and carbon emissions.")
  val cars = Chunk("cars.txt", "Electric cars charge from the grid or home chargers. Battery range grows each year.")
  val climate = Chunk("climate.txt", "Carbon dioxide traps heat. Cutting fossil fuel use slows warming.")
  val corpora = List(Corpus("docs", List(energy, cars, climate)))

  // Chunking: short text stays whole, long text splits with overlap.
  assert(Chunking.split("one two three").size == 1, "short text must stay whole")
  val long = (1 to 300).map(i => s"word$i").mkString(" ")
  val windows = Chunking.split(long, maxWords = 120, overlap = 20)
  assert(windows.size == 3, s"300 words in 120-word windows with step 100 must give 3 chunks, got ${windows.size}")
  assert(windows(1).contains("word101"), "windows must overlap")

  // Retrieval: solar question ranks the energy chunk first.
  val ranked = Retrieval.rank("solar panel electricity", corpora.head.chunks)
  assert(ranked.head._1 == energy, s"energy chunk must rank first, got ${ranked.head._1.source}")
  assert(Retrieval.rank("solar", Nil).isEmpty, "empty store ranks nothing")
  assert(Retrieval.topK("battery range", corpora.head.chunks, 1) == List(cars), "topK must return the car chunk")

  // Rewrite: compound questions split, simple ones pass through.
  val subs = RagPipeline.rewriteQuery("How do solar panels work and how do EVs charge?")
  assert(subs.size == 2, s"rewrite must split on 'and', got $subs")
  assert(RagPipeline.rewriteQuery("What is solar?").size == 1, "simple question stays whole")

  // Gather fans out across sub-queries and dedupes.
  val gathered = RagPipeline.gather(corpora, "solar panels and electric cars", topK = 1)
  assert(gathered.contains(energy) && gathered.contains(cars), s"gather must cover both parts, got ${gathered.map(_.source)}")

  // Prompt carries the question and the sources.
  val prompt = RagPipeline.buildPrompt("Q?", List(energy))
  assert(prompt.contains("Q?") && prompt.contains("energy.txt"), "prompt must hold question and source")

  println("All RAG tests passed.")
