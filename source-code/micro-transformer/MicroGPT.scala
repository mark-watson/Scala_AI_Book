//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package microtransformer

// Tiny char-level transformer: token embed, one causal attention
// layer with heads, RMSNorm, MLP, output head. Port of the model
// half of microgpt.lisp. Sizes stay small so a CPU trains it fast.
class MicroGPT(
    val vocabSize: Int,
    val blockSize: Int,
    nEmbd: Int = 8,
    nHead: Int = 2,
    hidden: Int = 16,
    seed: Long = 7
):
  private val rng = new scala.util.Random(seed)
  private def rand(rows: Int, cols: Int, scale: Double = 0.5): Array[Array[Value]] =
    Array.fill(rows, cols)(Value(rng.nextGaussian() * scale))
  private def randVec(n: Int): Array[Value] =
    Array.fill(n)(Value(1.0 + rng.nextGaussian() * 0.05))

  val tokenEmbed: Array[Array[Value]] = rand(vocabSize, nEmbd)
  val wq: Array[Array[Array[Value]]] = Array.fill(nHead)(rand(nEmbd / nHead, nEmbd, 0.3))
  val wk: Array[Array[Array[Value]]] = Array.fill(nHead)(rand(nEmbd / nHead, nEmbd, 0.3))
  val wv: Array[Array[Array[Value]]] = Array.fill(nHead)(rand(nEmbd / nHead, nEmbd, 0.3))
  val wo: Array[Array[Value]] = rand(nEmbd, nEmbd, 0.3)
  val ln1: Array[Value] = randVec(nEmbd)
  val w1: Array[Array[Value]] = rand(nEmbd, hidden, 0.3)
  val b1: Array[Value] = Array.fill(hidden)(Value(0.0))
  val w2: Array[Array[Value]] = rand(hidden, nEmbd, 0.3)
  val b2: Array[Value] = Array.fill(nEmbd)(Value(0.0))
  val ln2: Array[Value] = randVec(nEmbd)
  val wOut: Array[Array[Value]] = rand(vocabSize, nEmbd, 0.3)
  val bOut: Array[Value] = Array.fill(vocabSize)(Value(0.0))

  def parameters: Seq[Value] =
    tokenEmbed.flatten.toSeq ++ wq.flatten.flatten ++ wk.flatten.flatten ++
      wv.flatten.flatten ++ wo.flatten ++ ln1 ++ w1.flatten ++ b1 ++
      w2.flatten ++ b2 ++ ln2 ++ wOut.flatten ++ bOut

  private def matVec(m: Array[Array[Value]], x: Array[Value]): Array[Value] =
    m.map(row => row.zip(x).map { case (a, b) => a * b }.reduce(_ + _))

  private def rmsNorm(x: Array[Value], g: Array[Value]): Array[Value] =
    val ms = x.map(v => v * v).reduce(_ + _) / Value(x.size.toDouble)
    val inv = (ms + Value(1e-6)).pow(-0.5)
    x.zip(g).map { case (v, s) => v * inv * s }

  // Forward pass over one block. Returns per-position logits.
  def forward(ids: Array[Int]): Array[Array[Value]] =
    var x = ids.map(tokenEmbed(_).map(v => Value(v.data) + Value(0.0)))
    // Causal single-layer attention with residual.
    val headDim = nEmbd / nHead
    val attended = x.indices.map { t =>
      val outs = (0 until nHead).map { h =>
        val q = matVec(wq(h), x(t))
        val ks = (0 to t).map(s => matVec(wk(h), x(s)))
        val vs = (0 to t).map(s => matVec(wv(h), x(s)))
        val scores = ks.map(k => q.zip(k).map { case (a, b) => a * b }.reduce(_ + _) / Value(math.sqrt(headDim)))
        val weights = Value.softmax(scores)
        (0 until headDim).map { i =>
          weights.zip(vs).map { case (w, v) => w * v(i) }.reduce(_ + _)
        }.toArray
      }
      outs.foldLeft(Array.empty[Value])(_ ++ _)
    }.toArray
    x = x.zip(attended).map { case (xi, ai) =>
      val proj = matVec(wo, ai)
      xi.zip(proj).map { case (a, b) => a + b }
    }
    x = x.map(rmsNorm(_, ln1))
    // MLP with residual.
    x = x.zip(x.map { xi =>
      val h = matVec(w1, xi).zip(b1).map { case (a, b) => (a + b).relu() }
      matVec(w2, h).zip(b2).map { case (a, b) => a + b }
    }).map { case (a, b) => a.zip(b).map { case (p, q) => p + q } }
    x = x.map(rmsNorm(_, ln2))
    x.map(xi => matVec(wOut, xi).zip(bOut).map { case (a, b) => a + b })

  def loss(ids: Array[Int], targets: Array[Int]): Value =
    val logits = forward(ids)
    val terms = logits.zip(targets).map { case (row, t) =>
      val probs = Value.softmax(row)
      (probs(t) + Value(1e-9)).log() * Value(-1.0)
    }
    terms.reduce(_ + _) / Value(terms.size.toDouble)

  def trainStep(ids: Array[Int], targets: Array[Int], lr: Double): Double =
    parameters.foreach(_.grad = 0.0)
    val l = loss(ids, targets)
    l.backward()
    parameters.foreach(p => p.data -= lr * p.grad)
    l.data

  // Greedy next-char generation from a start string.
  def generate(start: Array[Int], n: Int, stoi: Map[Char, Int], itos: Map[Int, Char]): String =
    var ids = start.toList
    for _ <- 0 until n do
      val block = ids.takeRight(blockSize).toArray
      val logits = forward(block)
      val next = logits.last.map(_.data).zipWithIndex.maxBy(_._1)._2
      ids = ids :+ next
    ids.map(itos).mkString
