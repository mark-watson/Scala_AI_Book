//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package microtransformer

import java.nio.file.{Files, Paths}

// Train on data/corpus.txt over sliding next-char windows, print
// loss per pass, print a greedy sample.
@main def microDemo(): Unit =
  val text = Files.readString(Paths.get("data/corpus.txt")).toLowerCase
  val chars = text.distinct.sorted
  val stoi = chars.zipWithIndex.toMap
  val itos = stoi.map(_.swap)
  val ids = text.map(stoi).toArray
  val blockSize = 8
  val model = MicroGPT(chars.size, blockSize)
  val starts = (0 until ids.length - blockSize).toArray
  for epoch <- 1 to 3 do
    var total = 0.0
    starts.foreach { at =>
      total += model.trainStep(
        ids.slice(at, at + blockSize),
        ids.slice(at + 1, at + blockSize + 1),
        0.05
      )
    }
    println(f"epoch $epoch mean loss ${total / starts.size}%.3f")
  println("\nSample:")
  println(model.generate("hello ".map(stoi).toArray, 60, stoi, itos))
