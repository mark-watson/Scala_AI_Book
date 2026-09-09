# A Tiny Transformer from Scratch

The neural nets chapter builds a one layer net by hand. This chapter climbs one rung: a char level transformer with real attention, trained on CPU in seconds. Token embeds, one causal attention layer with two heads, RMSNorm, a small MLP, and an output head, all on a scalar autograd engine you can read in full.

All code is in `source-code/micro-transformer`. Training text sits in `source-code/micro-transformer/data/corpus.txt`.

## Autograd in Fifty Lines

Each `Value` holds a number, a gradient, parent links, and a backward step. Ops build the graph as they run. `backward` sorts the graph and replays each step in reverse:

```scala
def *(other: Value): Value =
  val a = this; val b = other
  result(data * b.data, Set(a, b), out => () => { a.grad += b.data * out.grad; b.grad += a.data * out.grad })
```

The closure must read the output grad, not the operand grad. Get that wrong and each weight learns zero, a fault the gradient test below catches at once. `softmax` rides on the same ops, so probs stay wired to the graph with no extra code.

## One Attention Layer, Fully Wired

`MicroGPT.forward` embeds each char, runs causal attention per head (each spot sees only past spots), projects and adds back, norms, runs the MLP with its own skip link, norms again, and scores each vocab char. Loss is mean cross entropy on next char targets. `trainStep` clears grads, runs loss, backprops, and steps each param by learning rate:

```scala
def trainStep(ids: Array[Int], targets: Array[Int], lr: Double): Double =
  parameters.foreach(_.grad = 0.0)
  val l = loss(ids, targets)
  l.backward()
  parameters.foreach(p => p.data -= lr * p.grad)
  l.data
```

Three epochs over sliding windows of the sample text cut mean loss from 2.6 to 1.8, and greedy samples start to echo the training lines. At 13 ms per step on a laptop, the whole demo trains in under half a minute.

## Tests That Pin Learning

Four checks guard the core: hand gradients on `a*b + a`, softmax rows that sum to one, logit shape `(block, vocab)`, and loss that falls over 25 steps on fixed text. The last one is the key test most from scratch nets skip: if loss cannot fall on a tiny repeat text, the wiring is wrong no matter how neat the code reads.

Run the demo and the checks:

```bash
cd source-code/micro-transformer
scala-cli run . --main-class microtransformer.microDemo
scala-cli run . --main-class microtransformer.transformerTest
```

Grow it from here: swap greedy picks for temp sampling, widen the net, or train on your own notes and watch it quote them back.
