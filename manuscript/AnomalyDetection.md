# Anomaly Detection

Anomaly detection is the process of identifying data points, events, or observations that deviate significantly from the norm. It is widely used in fraud detection, system health monitoring, and medical diagnosis.

In this chapter, we implement a Gaussian Probability Density Function (PDF) anomaly detection algorithm in pure Scala 3 and apply it to the classic Wisconsin Breast Cancer dataset to detect malignant cell samples as anomalies.

All code is in `source-code/anomaly-detection`.

## How Gaussian Anomaly Detection Works

The Gaussian anomaly detection model assumes that normal features follow a normal (Gaussian) distribution. We train the model using mostly normal (non-anomalous) examples.

1. **Parameter Estimation**: For each feature `j`$, we compute the mean `\mu_j`$ and variance `\sigma_j^2`$ of the training examples:

```$
\mu_j = \frac{1}{m} \sum_{i=1}^{m} x_j^{(i)}
```

```$
\sigma_j^2 = \frac{1}{m} \sum_{i=1}^{m} \left( x_j^{(i)} - \mu_j \right)^2
```

2. **Probability Computation**: For a new input vector `x`$, we compute the probability `p(x)`$ using the Gaussian Probability Density Function:

```$
p(x_j) = \frac{1}{\sqrt{2\pi\sigma_j^2}} \, \exp\!\left( -\frac{(x_j - \mu_j)^2}{2\sigma_j^2} \right)
```

   The overall probability `p(x)`$ is the product of the probabilities of all features. In our implementation, we compute the average feature probability to prevent underflow:

```$
p(x) = \frac{1}{d} \sum_{j=1}^{d} p(x_j)
```

3. **Thresholding**: We flag an example as an anomaly if its probability is below a threshold parameter `\epsilon`$:

```$
p(x) < \epsilon
```

## Preprocessing the Data

Real-world data rarely follows a perfect Gaussian distribution. To improve model accuracy, we preprocess the features in **anomaly-detection/Main.scala** by applying a logarithmic transform to make the features look more Gaussian, followed by min-max scaling:

```scala
  val trainingData = (for line <- bufferedSource.getLines() yield
    val sarr = line.trim.split(",")
    val xs = sarr.take(10).map(_.toDouble)
    
    // Scale features down
    for i <- 0 until 9 do xs(i) = xs(i) * 0.1

    // Make the data look more like a Gaussian distribution:
    var min = 1e6
    var max = -1e6
    for i <- 0 until 9 do
      xs(i) = math.log(xs(i) + 1.2)
      if xs(i) < min then min = xs(i)
      if xs(i) > max then max = xs(i)
    
    val range = max - min
    for i <- 0 until 9 do
      xs(i) = if range > 0.0 then (xs(i) - min) / range else 0.0

    xs(9) = (xs(9) - 2.0) * 0.5 // make target output be [0,1] instead of [2,4]
    xs
  ).toArray
```

## Implementing Anomaly Detection

We encapsulate the model inside the `AnomalyDetection` class. When the class is instantiated, it splits the dataset into training (60%), cross-validation (28%), and testing (12%) partitions, keeping training data mostly free of anomalies:

```scala
class AnomalyDetection(
    val numFeatures: Int,
    val allExamples: Array[Array[Double]],
    seed: Long = 42L
):
  private val outcomeIndex = numFeatures - 1
  private val SQRT_2_PI = math.sqrt(2.0 * math.Pi)

  // Split datasets (60% training, 28% cross-validation, 12% testing)
  private val (trainingExamples, crossValidationExamples, testingExamples) =
    val training = collection.mutable.ArrayBuffer[Array[Double]]()
    val cv = collection.mutable.ArrayBuffer[Array[Double]]()
    val test = collection.mutable.ArrayBuffer[Array[Double]]()

    for row <- allExamples do
      if rng.nextDouble() < 0.6 then
        // Only keep normal (negative) examples in training, but allow a 10% leak of anomalies
        if row(outcomeIndex) < 0.5 || rng.nextDouble() < 0.1 then
          training.append(row)
      else if rng.nextDouble() < 0.7 then
        cv.append(row)
      else
        test.append(row)

    (training.toArray, cv.toArray, test.toArray)
```

The model estimates `\mu`$ and `\sigma^2`$ parameters from the training set, then tunes the threshold `\epsilon`$ by minimizing classification errors on the cross-validation set:

```scala
  /** Calculate average feature probability using Gaussian PDF. */
  private def p(x: Array[Double]): Double =
    var sum = 0.0
    // Skip target column at index outcomeIndex
    for nf <- 0 until numFeatures - 1 do
      val sigma = math.sqrt(sigmaSquared(nf))
      val diff = x(nf) - mu(nf)
      val exponent = -(diff * diff) / (2.0 * sigmaSquared(nf))
      sum += (1.0 / (SQRT_2_PI * sigma)) * math.exp(exponent)
    sum / numFeatures

  /** Tune epsilon hyperparameter using cross-validation data. */
  def train(): Unit =
    // Calculate sigmaSquared from training data using the computed mu values
    for nf <- 0 until numFeatures - 1 do
      val sum = trainingExamples.map(x => (x(nf) - mu(nf)) * (x(nf) - mu(nf))).sum
      sigmaSquared(nf) = sum / numTraining

    var bestErrorCount = Double.MaxValue
    for epsilonLoop <- 0 to 200 do
      val epsilon = 0.001 + 0.005 * epsilonLoop
      val errorCount = evaluateEpsilonOnCV(epsilon)
      if errorCount <= bestErrorCount then
        bestErrorCount = errorCount
        bestEpsilon = epsilon

    println(f"\n**** Best epsilon value = $bestEpsilon%.4f")
    test(bestEpsilon)
```

## Running the Anomaly Detector

When we run the project using `scala-cli run .`, the main application first displays ASCII histograms of the preprocessed features to confirm their distribution, then splits and trains the detector:

```text
==================================================
Anomaly Detection: Wisconsin Breast Cancer Data Set
==================================================

Feature Histograms:

Clump Thickness
0	148	█████████████████████████████
1	79	███████████████
2	144	████████████████████████████
3	118	███████████████████████
4	194	████████████████████████████████████████

Uniformity of Cell Size
0	374	████████████████████████████████████████████████████████████████████████████
1	66	█████████████
2	95	███████████████████
3	81	████████████████
4	67	█████████████
...

Training anomaly detector on 315 normal samples...
Cross-validation samples: 260
Testing samples: 108

**** Best epsilon value = 0.3560

 -- best epsilon = 0.356
 -- number of test examples = 108
 -- number of false positives = 5
 -- number of true positives = 35
 -- number of false negatives = 4
 -- number of true negatives = 64
 -- precision = 0.8750
 -- recall = 0.8974
 -- F1 = 0.8861

Model parameters:
  best epsilon = 0.3560
  num features = 10
```

With an F1 score of over 88% on the test set, the Gaussian PDF anomaly detector successfully identifies malignant cell samples as anomalies based solely on their deviation from normal cell configurations.
