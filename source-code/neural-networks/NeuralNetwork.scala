// Copyright 2025-2026 Mark Watson. All rights reserved.
//> using scala 3.6.4

import scala.util.Random

/** Backpropagation neural network with one hidden layer.
  *
  * This is a from-scratch implementation — no libraries — matching
  * the Java version but using Scala idioms: the network state is
  * a case class, and training returns an updated copy.
  *
  * @param numInputs  number of input neurons
  * @param numHidden  number of hidden neurons
  * @param numOutputs number of output neurons
  * @param w1         weights from input → hidden  [numInputs][numHidden]
  * @param w2         weights from hidden → output  [numHidden][numOutputs]
  * @param learningRate step size for gradient descent
  */
case class NeuralNetwork(
    numInputs: Int,
    numHidden: Int,
    numOutputs: Int,
    w1: Array[Array[Float]],
    w2: Array[Array[Float]],
    learningRate: Float = 0.5f
):

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

  /** Train on a list of (input, target) pairs for one epoch (one pass through all examples). */
  def trainEpoch(examples: List[(Array[Float], Array[Float])]): (NeuralNetwork, Float) =
    var net = this
    var totalError = 0f
    for (inputs, targets) <- examples do
      val (updated, err) = net.trainOne(inputs, targets)
      net = updated
      totalError += err
    (net, totalError / examples.size)

object NeuralNetwork:

  def sigmoid(x: Float): Float = (1.0f / (1.0f + math.exp(-x).toFloat))

  /** Derivative of sigmoid where the input is already activated (i.e., s = sigmoid(z)). */
  def sigmoidPrime(s: Float): Float =
    s * (1.0f - s)


  /** Create a network with random weights in [-0.5, 0.5]. */
  def random(numInputs: Int, numHidden: Int, numOutputs: Int,
             learningRate: Float = 0.5f, seed: Long = 42L): NeuralNetwork =
    val rng = Random(seed)
    val w1 = Array.fill(numInputs, numHidden)(rng.nextFloat() - 0.5f)
    val w2 = Array.fill(numHidden, numOutputs)(rng.nextFloat() - 0.5f)
    NeuralNetwork(numInputs, numHidden, numOutputs, w1, w2, learningRate)

/** Demo: learn the XOR function. */
@main def neuralNetworkDemo(): Unit =
  println("=" * 50)
  println("Neural Network: Learning XOR")
  println("=" * 50)

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
