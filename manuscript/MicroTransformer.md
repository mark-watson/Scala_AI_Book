# A Tiny Transformer from Scratch

The neural nets chapter builds a one layer net by hand. This chapter climbs one rung: a char level transformer with real attention, trained on CPU in seconds. Token embeds, one causal attention layer with two heads, RMSNorm, a small MLP, and an output head, all on a scalar autograd engine you can read in full.

All code is in `source-code/micro-transformer`. Training text sits in `source-code/micro-transformer/data/corpus.txt`, twenty short lines of repeating toy English. Credit: the design ports Andrej Karpathy's microGPT, a minimal dependency-free GPT written in Python, first to Common Lisp and here to Scala:

```
hello world
hello scala
hello there
hello again world
scala runs on the jvm
...
```

## Autograd in Fifty Lines

Each `Value` holds a number, a gradient, parent links, and a backward step. Ops build the graph as they run. `backward` sorts the graph and replays each step in reverse. The full op set is add, multiply, power, exp, log, and ReLU:

```scala
class Value(var data: Double, var grad: Double = 0.0):
  private var children: Set[Value] = Set.empty
  private var backwardFn: () => Unit = () => ()

  def +(other: Value): Value =
    val a = this; val b = other
    result(data + b.data, Set(a, b), out => () => { a.grad += out.grad; b.grad += out.grad })
  def *(other: Value): Value =
    val a = this; val b = other
    result(data * b.data, Set(a, b), out => () => { a.grad += b.data * out.grad; b.grad += a.data * out.grad })
```

The closure must read the output grad, not the operand grad. Get that wrong and each weight learns zero, a fault the gradient test below catches at once. `softmax` rides on the same ops, so probs stay wired to the graph with no extra code:

```scala
def softmax(logits: Seq[Value]): Seq[Value] =
  val m = logits.map(_.data).max
  val exps = logits.map(v => (v - Value(m)).exp())
  val s = exps.reduce(_ + _)
  exps.map(_ / s)
```

## One Attention Layer, Fully Wired

`MicroGPT` holds eleven param groups: token embeds, per head query/key/value maps, output projection, two norms, two MLP maps plus biases, and the vocab head. With 8 wide embeds, 2 heads, and a 16 wide MLP the whole net fits in a few thousand scalars:

```scala
class MicroGPT(
    val vocabSize: Int,
    val blockSize: Int,
    nEmbd: Int = 8,
    nHead: Int = 2,
    hidden: Int = 16,
    seed: Long = 7
)
```

`forward` embeds each char, runs causal attention per head (each spot sees only past spots), concatenates the heads, projects and adds back, norms, runs the MLP with its own skip link, norms again, and scores each vocab char:

```scala
val scores = ks.map(k => q.zip(k).map { case (a, b) => a * b }.reduce(_ + _) / Value(math.sqrt(headDim)))
val weights = Value.softmax(scores)
(0 until headDim).map { i =>
  weights.zip(vs).map { case (w, v) => w * v(i) }.reduce(_ + _)
}.toArray
```

Loss is mean cross entropy on next char targets. `trainStep` clears grads, runs loss, backprops, and steps each param by learning rate:

```scala
def trainStep(ids: Array[Int], targets: Array[Int], lr: Double): Double =
  parameters.foreach(_.grad = 0.0)
  val l = loss(ids, targets)
  l.backward()
  parameters.foreach(p => p.data -= lr * p.grad)
  l.data
```

Generation is greedy: take the last block, score it, append the argmax char, repeat.

## Demo and Its Output

The demo in **micro-transformer/Main.scala** slides 8 char windows across the corpus with next char targets, trains three epochs at rate 0.05, and prints a 60 char sample from the seed `"hello "`:

```scala
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
```

It prints:

```
epoch 1 mean loss 2.597
epoch 2 mean loss 2.056
epoch 3 mean loss 1.804

Sample:
hello transhellllo transhellllo transhellllo transhellllo tran
```

Loss falls each pass and the sample echoes the training lines (`hello`, `trans...` from `transformer`). Greedy picks repeat hard, so the tail stutters on `lll`. That is honest output for a thousand param net, and it shows the mechanism works: the model found the corpus rhythm. At 13 ms per step the whole demo trains in under half a minute.

## Tests That Pin Learning

Five checks guard the core. Hand gradients on `z = a*b + a` with `a = 2, b = 3` must give exactly 4 and 2. Softmax rows must sum to one. Logits must shape `(block, vocab)`. Loss must fall over 25 fixed steps. Samples must hold only vocab chars at the asked length:

```scala
val z = a * b + a
z.backward()
close(a.grad, 4.0, 1e-9, "dz/da")
close(b.grad, 2.0, 1e-9, "dz/db")
...
assert(last < first, s"loss must fall, first=$first last=$last")
```

The loss check is the key test most from scratch nets skip: if loss cannot fall on a tiny repeat text, the wiring is wrong no matter how neat the code reads. The suite ends with:

```
All transformer tests passed.
```

Run the demo and the checks:

```bash
cd source-code/micro-transformer
scala-cli run . --main-class microtransformer.microDemo
scala-cli run . --main-class microtransformer.transformerTest
```

Grow it from here: swap greedy picks for temp sampling, widen the net, or train on your own notes and watch it quote them back.
