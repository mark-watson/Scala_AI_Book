# Backpropagation Neural Networks from Scratch

While modern AI often relies on massive libraries like PyTorch or TensorFlow, implementing a neural network from scratch is a powerful way to truly master the underlying mathematics of deep learning. In this chapter, we build a backpropagation neural network with one hidden layer in pure Scala 3, with zero dependencies.

Unlike traditional object-oriented implementations (which mutate weight arrays in place), we use a functional style: the network state is encapsulated in an immutable case class, and training methods return an updated copy of the network.

All code is in `source-code/neural-networks`.

## A Short History: Why We Need Hidden Layers

The artificial neuron dates to 1943, when Warren McCulloch and Walter Pitts modeled a nerve cell as a unit that sums its inputs and fires if the sum passes a threshold. In 1958 Frank Rosenblatt turned this into the **perceptron**, a trainable single-layer classifier. Progress stalled in 1969 when Marvin Minsky and Seymour Papert proved a hard limit in their book *Perceptrons*: a single layer can only separate classes with a straight line (a hyperplane), so it cannot learn the XOR function, whose true cases sit on opposite corners of a square. This result cooled research for years.

The way out is a **hidden layer**. Stack a second layer of neurons between input and output and the network can carve the input space into regions that no single line could, so XOR becomes learnable. The missing piece was a training method for the hidden weights. It arrived in 1986, when David Rumelhart, Geoffrey Hinton, and Ronald Williams popularized **backpropagation**, an efficient way to compute how every weight in a multi-layer network affects the error. That algorithm is what we build here, and it remains the engine under every modern deep network.

Hidden layers only help if the neurons are nonlinear. A neuron computes a weighted sum, and the sum of sums is still just a sum: stack any number of purely linear layers and the whole network collapses to a single linear map, no more powerful than one perceptron. Inserting a nonlinear **activation function** after each layer breaks this collapse and lets depth add real expressive power. In fact the **universal approximation theorem** (George Cybenko in 1989, Kurt Hornik in 1991) shows that a single hidden layer with enough nonlinear units can approximate any continuous function to any accuracy. Our XOR network is the smallest interesting instance of that theorem.

## Neural Network Representation

Our network is defined in **neural-networks/NeuralNetwork.scala** as a case class wrapping the dimensional configurations and weight matrices:

```scala
case class NeuralNetwork(
    numInputs: Int,
    numHidden: Int,
    numOutputs: Int,
    w1: Array[Array[Float]],
    w2: Array[Array[Float]],
    learningRate: Float = 0.5f
):
```

Here, `w1` represents the weights from the input layer to the hidden layer, and `w2` represents the weights from the hidden layer to the output layer. Each is a matrix: `w1` has shape `numInputs \times numHidden`$ and `w2` has shape `numHidden \times numOutputs`$. The whole network is nothing more than these two matrices plus their sizes, which is why the case class captures the entire model state.

To keep the code minimal, this network omits **bias** terms, the per-neuron constants that let an activation shift left or right independent of its inputs. Production networks always include them, and we would add a bias by giving each layer one extra input fixed at `1.0`. The XOR problem is still learnable without explicit biases because the four hidden units can cooperate to shape the decision boundary, as the demo output later confirms.

### Activation Function

We use the standard **Sigmoid** function to introduce non-linearity into the network, allowing it to learn non-linear decision boundaries:

```$
S(x) = \frac{1}{1 + e^{-x}}
```

The sigmoid squashes any real number into the range `(0, 1)`$, which makes its output easy to read as a probability or a soft on/off signal. Its derivative has a particularly convenient closed form when expressed in terms of the activated value `s = S(z)`$:

```$
S'(z) = S(z)\,\bigl(1 - S(z)\bigr) = s\,(1 - s)
```

This identity is the reason sigmoid was the default for decades: during backpropagation we already have the activation `s`$ in hand, so computing the gradient costs one multiply and one subtract, with no second call to `exp`. In code, we define the sigmoid and its derivative:

```scala
object NeuralNetwork:
  def sigmoid(x: Float): Float = 
    (1.0f / (1.0f + math.exp(-x).toFloat))

  /** Derivative of sigmoid where the input is already activated (s = sigmoid(z)). */
  def sigmoidPrime(s: Float): Float =
    s * (1.0f - s)
```

The convenient derivative comes at a cost worth knowing. Notice that `s\,(1 - s)`$ peaks at only `0.25`$ and falls to nearly zero when `s`$ is close to 0 or 1. In a deep network these small factors multiply together and the gradient shrinks toward nothing as it flows back through many layers, the **vanishing gradient** problem. This is why modern deep networks favor the ReLU activation, `\max(0, x)`$, whose gradient is a flat 1 for positive inputs. For a shallow network like ours, sigmoid works fine.

## Feedforward Pass

The forward pass runs input values through the network to compute the hidden activations and final outputs. We multiply the input vector by the weight matrix, sum them, and apply the sigmoid function:

```scala
  /** Forward pass: returns (hiddenActivations, outputActivations). */
  def forward(inputs: Array[Float]): (Array[Float], Array[Float]) =
    // input → hidden
    val hidden = Array.ofDim[Float](numHidden)
    for i <- 0 until numInputs; h <- 0 until numHidden do
      hidden(h) += inputs(i) * w1(i)(h)
    for h <- 0 until numHidden do
      hidden(h) = NeuralNetwork.sigmoid(hidden(h))

    // hidden → output
    val outputs = Array.ofDim[Float](numOutputs)
    for h <- 0 until numHidden; o <- 0 until numOutputs do
      outputs(o) += hidden(h) * w2(h)(o)
    for o <- 0 until numOutputs do
      outputs(o) = NeuralNetwork.sigmoid(outputs(o))

    (hidden, outputs)

  /** Run a single input through the network and return the outputs. */
  def recall(inputs: Array[Float]): Array[Float] = forward(inputs)._2
```

Each layer performs the same two steps: a **matrix-vector product** followed by an element-wise nonlinearity. The nested loop `hidden(h) += inputs(i) * w1(i)(h)` is a hand-written matrix multiply; a library like PyTorch would express the whole thing as `sigmoid(W1 · x)` and dispatch it to optimized BLAS routines, but the arithmetic is identical. We return both the hidden and output activations, because backpropagation needs the hidden activations again to compute the gradient, and recomputing them would waste work.

## Backpropagation and Weight Updates

Training means finding weights that make the outputs match the targets. We frame this as minimizing an **error function**, the squared difference between target and output summed over the output neurons:

```$
E = \frac{1}{2} \sum_{o} (t_o - o_o)^2
```

The error `E`$ is a surface over the space of all weights, and we want its lowest point. **Gradient descent** finds it by repeatedly stepping each weight a little in the direction that most reduces `E`$, which is the negative gradient `-\partial E / \partial w`$. The only hard part is computing that gradient for a weight buried in the hidden layer, and this is exactly what backpropagation does using the **chain rule** of calculus: the error's sensitivity to a hidden weight is the product of sensitivities along the path from that weight to the output. Working the chain rule through our two layers gives the three familiar update rules:

1. **Output Error**: Calculate the difference between targets and outputs, multiplied by the derivative of the activation function:

```$
\delta_{\text{output}} = (t - o) \cdot S'(o)
```

2. **Hidden Error**: Propagate the error backward through the weights `w_2`$ to the hidden layer. Each hidden neuron gets a share of the blame proportional to how strongly it fed each output:

```$
\delta_{\text{hidden}} = \left( \sum_{o} \delta_{\text{output}} \cdot w_2 \right) \cdot S'(h)
```

3. **Weight Update**: Create new weight matrices by adding the product of the errors, the activations, and the learning rate `\eta`$:

```$
w_{\text{new}} = w_{\text{old}} + \eta \cdot \delta \cdot \text{activation}
```

The `\delta`$ terms are the heart of the algorithm. Each one measures how much a neuron's total input should change to lower the error, and the weight update simply moves each weight in proportion to the error at its downstream neuron and the activation at its upstream neuron. The learning rate `\eta`$ scales every step: too small and training crawls, too large and the updates overshoot the minimum and the error can oscillate or diverge. Our demo uses `\eta = 0.8`$, which is aggressive but works for this tiny problem.

Here is how we implement this functionally in Scala:

```scala
  /** Train on one example. Returns (updatedNetwork, error). */
  def trainOne(inputs: Array[Float], targets: Array[Float]): (NeuralNetwork, Float) =
    val (hidden, outputs) = forward(inputs)

    // output errors
    val outputErrors = Array.ofDim[Float](numOutputs)
    for o <- 0 until numOutputs do
      outputErrors(o) = (targets(o) - outputs(o)) * NeuralNetwork.sigmoidPrime(outputs(o))

    // hidden errors (backprop)
    val hiddenErrors = Array.ofDim[Float](numHidden)
    for h <- 0 until numHidden do
      var sum = 0f
      for o <- 0 until numOutputs do sum += outputErrors(o) * w2(h)(o)
      hiddenErrors(h) = sum * NeuralNetwork.sigmoidPrime(hidden(h))

    // update weights (create new arrays)
    val newW2 = Array.tabulate(numHidden, numOutputs): (h, o) =>
      w2(h)(o) + learningRate * outputErrors(o) * hidden(h)
    val newW1 = Array.tabulate(numInputs, numHidden): (i, h) =>
      w1(i)(h) + learningRate * hiddenErrors(h) * inputs(i)

    val error = (0 until numOutputs).map(o => math.abs(targets(o) - outputs(o))).sum
    (copy(w1 = newW1, w2 = newW2), error)
```

The code maps one-to-one onto the three equations. Notice that `sigmoidPrime` receives the already-activated value (`outputs(o)`, `hidden(h)`), which is why the derivative is written as `s * (1 - s)` rather than in terms of the raw pre-activation. The functional design shows up in the last line: instead of mutating `w1` and `w2` in place, we build new arrays with `Array.tabulate` and return a fresh `NeuralNetwork` via `copy`. This makes the network immutable and each training step a pure function from old network to new network, which is easy to reason about and test. The trade-off is allocation: we create two new weight matrices per example. For large networks you would mutate in place to avoid that cost, but for teaching the algorithm the clarity is worth it. Note also that the returned `error` uses the absolute difference `|t - o|` purely as a human-readable progress metric; the value the training actually minimizes is the squared error whose gradient produced the delta rules above.

Because `trainOne` updates the weights after every single example, this is **stochastic (online) gradient descent**. The alternative, **batch** gradient descent, would accumulate the gradients over all examples and apply one averaged update per pass. Online updates are noisier but often converge faster and can escape shallow dips in the error surface. We also write a helper to train the network over a list of training examples for an entire epoch, folding the updated network through the inputs:

```scala
  /** Train on a list of (input, target) pairs for one epoch. */
  def trainEpoch(examples: List[(Array[Float], Array[Float])]): (NeuralNetwork, Float) =
    var net = this
    var totalError = 0f
    for (inputs, targets) <- examples do
      val (updated, err) = net.trainOne(inputs, targets)
      net = updated
      totalError += err
    (net, totalError / examples.size)
```

One subtlety of training deserves mention: the random starting weights. We initialize weights to small random values in `[-0.5, 0.5]`$ rather than to zero. If every weight started equal, every hidden neuron would compute the same output and receive the same gradient, so they would stay identical forever and the hidden layer would collapse to a single neuron. Random initialization **breaks this symmetry** and lets each hidden unit specialize.

## Demo: Learning the XOR Gate

The XOR logical gate is a classic AI benchmark because it is not linearly separable. As Minsky and Papert showed, a single-layer perceptron cannot learn it, which makes the hidden layer necessary and makes XOR the perfect smallest test of a real multi-layer network.

Our demo program initializes a network with 2 inputs, 4 hidden units, and 1 output, then trains it on the XOR truth table over 5000 epochs:

```scala
@main def neuralNetworkDemo(): Unit =
  val xorExamples = List(
    (Array(0f, 0f), Array(0f)),
    (Array(0f, 1f), Array(1f)),
    (Array(1f, 0f), Array(1f)),
    (Array(1f, 1f), Array(0f))
  )

  var net = NeuralNetwork.random(
    numInputs = 2, numHidden = 4, numOutputs = 1,
    learningRate = 0.8f
  )

  val numEpochs = 5000
  for epoch <- 1 to numEpochs do
    val (updated, error) = net.trainEpoch(xorExamples)
    net = updated
    if epoch % 500 == 0 || epoch == 1 then
      println(f"  Epoch $epoch%5d  avg error: $error%.6f")

  println("\nTrained network results:")
  for (input, target) <- xorExamples do
    val output = net.recall(input)
    println(f"  Input: [${input(0)}%.0f, ${input(1)}%.0f]  " +
            f"Target: ${target(0)}%.0f  Output: ${output(0)}%.4f")
```

Running this code with `scala-cli run .` outputs:

```text
==================================================
Neural Network: Learning XOR
==================================================
  Epoch     1  avg error: 0.505372
  Epoch   500  avg error: 0.380295
  Epoch  1000  avg error: 0.101740
  Epoch  1500  avg error: 0.046399
  Epoch  2000  avg error: 0.030560
  Epoch  2500  avg error: 0.023247
  Epoch  3000  avg error: 0.018939
  Epoch  3500  avg error: 0.016075
  Epoch  4000  avg error: 0.014022
  Epoch  4500  avg error: 0.012476
  Epoch  5000  avg error: 0.011265
```

The error curve tells the story of gradient descent. The first 500 epochs barely move: the network sits on a nearly flat part of the error surface while the hidden units slowly differentiate. Around epoch 1000 the error drops sharply as the network discovers a useful internal representation, then the improvement tapers as it settles into the minimum. This slow start followed by a fast drop is typical of small networks learning a nonlinear function. The final results confirm the network learned XOR:

```text
Trained network results:
  Input: [0, 0]  Target: 0  Output: 0.0104
  Input: [0, 1]  Target: 1  Output: 0.9892
  Input: [1, 0]  Target: 1  Output: 0.9890
  Input: [1, 1]  Target: 0  Output: 0.0124
```

As the output shows, the final errors drop to nearly 1%, and the model correctly outputs values extremely close to the targets. The network never sees the rule for XOR; it discovers, purely from four examples and the gradient of its own error, a set of weights that computes a function no single-layer perceptron ever could.
