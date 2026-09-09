//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package microtransformer

// Offline checks. Run with: scala-cli run . --main-class microtransformer.transformerTest
@main def transformerTest(): Unit =
  def close(a: Double, b: Double, tol: Double, label: String): Unit =
    assert(math.abs(a - b) <= tol, s"$label: expected $b, got $a")

  // Autograd: z = a*b + a with a=2, b=3 gives dz/da = 4, dz/db = 2.
  val a = Value(2.0)
  val b = Value(3.0)
  val z = a * b + a
  z.backward()
  close(a.grad, 4.0, 1e-9, "dz/da")
  close(b.grad, 2.0, 1e-9, "dz/db")

  // Softmax rows sum to 1 and keep order.
  val probs = Value.softmax(Seq(Value(1.0), Value(2.0), Value(3.0)))
  close(probs.map(_.data).sum, 1.0, 1e-9, "softmax sums to 1")
  assert(probs.map(_.data).sliding(2).forall(p => p(0) < p(1)), "softmax keeps order")

  // Model: forward shape is (block, vocab); loss is finite.
  val text = "hello world, hello scala, hello there!"
  val chars = text.distinct.sorted
  val stoi = chars.zipWithIndex.toMap
  val itos = stoi.map(_.swap)
  val ids = text.map(stoi).toArray
  val model = MicroGPT(chars.size, 6, nEmbd = 8, nHead = 2, hidden = 16, seed = 7)
  val block = ids.take(6)
  val logits = model.forward(block)
  assert(logits.size == 6 && logits.forall(_.size == chars.size), "logit shape")
  val first = model.loss(block, block.tail :+ block.last).data
  assert(first.isFinite && first > 0.0, s"loss must be finite positive, got $first")

  // Training lowers the loss on this repetitive text.
  var last = first
  for step <- 1 to 25 do
    last = model.trainStep(block, block.tail :+ block.last, 0.05)
  assert(last < first, s"loss must fall, first=$first last=$last")

  // Generate returns only vocab chars at the asked length.
  val sample = model.generate("hello ".map(stoi).toArray, 20, stoi, itos)
  assert(sample.size == 26 && sample.forall(chars.contains), s"bad sample: $sample")

  println("All transformer tests passed.")
