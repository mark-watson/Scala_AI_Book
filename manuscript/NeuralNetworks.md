# Backpropagation Neural Networks from Scratch

While modern AI often relies on massive libraries like PyTorch or TensorFlow, implementing a neural network from scratch is a powerful way to truly master the underlying mathematics of deep learning. In this chapter, we build a backpropagation neural network with one hidden layer in pure Scala 3, with zero dependencies.

Unlike traditional object-oriented implementations (which mutate weight arrays in place), we use a functional style: the network state is encapsulated in an immutable case class, and training methods return an updated copy of the network.

All code is in `source-code/neural-networks`.

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

Here, `w1` represents the weights from the input layer to the hidden layer, and `w2` represents the weights from the hidden layer to the output layer.

### Activation Function

We use the standard **Sigmoid** function to introduce non-linearity into the network, allowing it to learn non-linear decision boundaries:

```latexmath
S(x) = \frac{1}{1 + e^{-x}}
```

In code, we define the sigmoid and its derivative (expressed in terms of the activated value `s = S(z)`):

```scala
object NeuralNetwork:
  def sigmoid(x: Float): Float = 
    (1.0f / (1.0f + math.exp(-x).toFloat))

  /** Derivative of sigmoid where the input is already activated (s = sigmoid(z)). */
  def sigmoidPrime(s: Float): Float =
    s * (1.0f - s)
```

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

## Backpropagation and Weight Updates

Backpropagation trains the network by computing the gradient of the error function with respect to the weights, and then updating the weights in the opposite direction of the gradient (gradient descent). 

1. **Output Error**: Calculate the difference between targets and outputs, multiplied by the derivative of the activation function:

   ```latexmath
   \delta_{\text{output}} = (t - o) \cdot S'(o)
   ```

2. **Hidden Error**: Propagate the error backward through the weights `w2` to the hidden layer:

   ```latexmath
   \delta_{\text{hidden}} = \left( \sum \delta_{\text{output}} \cdot w_2 \right) \cdot S'(h)
   ```

3. **Weight Update**: Create new weight matrices by adding the product of the errors, the activations, and the learning rate:

   ```latexmath
   w_{\text{new}} = w_{\text{old}} + \eta \cdot \delta \cdot \text{activation}
   ```

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

We also write a helper to train the network over a list of training examples for an entire epoch, folding over the inputs:

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

## Demo: Learning the XOR Gate

The XOR logical gate is a classic AI benchmark because it is not linearly separable. A single-layer perceptron cannot learn it, making a hidden layer necessary.

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

Trained network results:
  Input: [0, 0]  Target: 0  Output: 0.0104
  Input: [0, 1]  Target: 1  Output: 0.9892
  Input: [1, 0]  Target: 1  Output: 0.9890
  Input: [1, 1]  Target: 0  Output: 0.0124
```

As the output shows, the final errors drop to nearly 1%, and the model correctly outputs values extremely close to the targets.
