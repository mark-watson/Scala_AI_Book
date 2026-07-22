# Anomaly Detection

Anomaly detection is the process of identifying data points, events, or observations that deviate significantly from the norm. It is widely used in fraud detection, system health monitoring, and medical diagnosis.

In this chapter, we implement a Gaussian Probability Density Function (PDF) anomaly detection algorithm in pure Scala 3 and apply it to the classic Wisconsin Breast Cancer dataset to detect malignant cell samples as anomalies.

All code is in `source-code/anomaly-detection`.

## Why Anomaly Detection, Not Classification

At first glance this looks like a classification problem: sort cell samples into benign and malignant. So why not train a classifier as we did with the neural network? The answer is the shape of the data you usually have. Anomalies are **rare and varied**. In fraud detection you may see millions of normal transactions and only a handful of frauds, and next month's fraud may look nothing like last month's. A supervised classifier needs many labeled examples of every class, and it learns only the kinds of anomalies it has already seen.

Anomaly detection turns the problem around. Instead of learning what an anomaly looks like, it learns what **normal** looks like, then flags anything that does not fit. We train almost entirely on normal examples, build a model of their distribution, and mark any new point with low probability under that model as an anomaly. This is a form of **semi-supervised** learning, sometimes called *novelty detection*: the training set is (mostly) one class, and the model detects departures from it. That is exactly why it suits fraud, intrusion detection, manufacturing defects, and rare-disease screening, where the "normal" class is abundant and the anomalies are few and unpredictable.

## How Gaussian Anomaly Detection Works

The Gaussian anomaly detection model assumes that normal features follow a normal (Gaussian) distribution. We train the model using mostly normal (non-anomalous) examples.

1. **Parameter Estimation**: For each feature `j`$, we compute the mean `\mu_j`$ and variance `\sigma_j^2`$ of the training examples:

```$
\mu_j = \frac{1}{m} \sum_{i=1}^{m} x_j^{(i)}
```

```$
\sigma_j^2 = \frac{1}{m} \sum_{i=1}^{m} \left( x_j^{(i)} - \mu_j \right)^2
```

These are not arbitrary formulas. They are the **maximum likelihood estimates** for a Gaussian: given the training data, they are the mean and variance that make that data most probable under a normal distribution. Note that the variance divides by `m`$ rather than `m - 1`$, which is the maximum likelihood form rather than the unbiased sample variance. For a training set of hundreds of examples the difference is negligible.

2. **Probability Computation**: For a new input vector `x`$, we score each feature with the Gaussian Probability Density Function, which measures how likely that feature value is under the fitted bell curve:

```$
p(x_j) = \frac{1}{\sqrt{2\pi\sigma_j^2}} \, \exp\!\left( -\frac{(x_j - \mu_j)^2}{2\sigma_j^2} \right)
```

   The textbook model then assumes the features are **independent** and multiplies the per-feature probabilities to get a joint probability `p(x) = \prod_j p(x_j)`$. That independence assumption is rarely true, but the model is robust enough that it works well in practice. Multiplying many probabilities together drives the product toward zero and risks numeric **underflow**, so our implementation instead averages the per-feature probabilities:

```$
p(x) = \frac{1}{d} \sum_{j=1}^{d} p(x_j)
```

   This average is no longer a true joint probability, but it preserves the property we care about: a sample that sits far from the norm on its features gets a low score, and a normal sample gets a high one. It trades theoretical purity for numeric stability, which is a sensible engineering choice for a from-scratch detector.

3. **Thresholding**: We flag an example as an anomaly if its score falls below a threshold parameter `\epsilon`$:

```$
p(x) < \epsilon
```

   The value of `\epsilon`$ sets the trade-off between catching anomalies and raising false alarms. A high `\epsilon`$ flags more samples (higher recall, more false positives); a low `\epsilon`$ flags fewer (higher precision, more misses). We do not guess it; we tune it, as shown below.

## Preprocessing the Data

The model assumes each feature is Gaussian, but real-world data rarely obliges. The Wisconsin features are integer scores from 1 to 10 and are skewed toward the low end, not bell-shaped at all. Feeding skewed data to a Gaussian model weakens it, so we reshape the features first. A **logarithmic transform** compresses a long right tail and pulls a skewed distribution closer to symmetric, after which **min-max scaling** maps every feature into `[0, 1]`$ so no single feature dominates the probability by virtue of its raw scale. We do this in **anomaly-detection/Main.scala**:

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

The dataset is the classic Wisconsin Breast Cancer set from the UCI repository: nine cell-measurement features (clump thickness, cell-size uniformity, and so on) plus an outcome column labeling each sample benign or malignant. The raw outcome uses 2 for benign and 4 for malignant, and the last line remaps it to 0 and 1 so we can treat malignant samples as the anomalies to detect. The small `+ 1.2` offset inside the logarithm keeps its argument safely positive.

## Implementing Anomaly Detection

We encapsulate the model inside the `AnomalyDetection` class. When the class is instantiated, it splits the dataset into three partitions, and the split is deliberately skewed to match the semi-supervised setup: training keeps almost only normal samples, while cross-validation and testing get a realistic mix of both classes:

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

The three-way split follows standard machine learning methodology, adapted for anomaly detection. The **training set** estimates `\mu`$ and `\sigma^2`$ and holds mostly normal data, so the model learns the shape of "normal". The **cross-validation set** contains labeled anomalies and is used to tune the one hyperparameter `\epsilon`$. The **test set**, untouched during tuning, gives an honest final measure of performance. Keeping tuning and testing separate is what stops us from fooling ourselves: a threshold chosen to look good on the test set would report an optimistic score that new data would not match.

The model estimates `\mu`$ and `\sigma^2`$ from the training set, then tunes the threshold `\epsilon`$ by scanning a range of values and keeping the one that misclassifies the fewest cross-validation examples:

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

The `train` method is a simple **grid search** over `\epsilon`$: it tries 201 evenly spaced thresholds and keeps whichever one makes the fewest mistakes on the cross-validation data. This brute-force scan is cheap because scoring the CV set is fast, and it avoids any assumption about where the best threshold lies.

### Measuring Success: Precision, Recall, and F1

Anomalies are rare, so plain **accuracy** is a trap. If only 5% of samples are malignant, a lazy detector that calls everything benign scores 95% accuracy while catching zero cancers. We need metrics that focus on the rare positive class. Every prediction falls into one of four cells of a **confusion matrix**, which the code counts with a clean pattern match:

```scala
      (isTargetAnomaly, isPredictedAnomaly) match
        case (true, true)   => truePositives += 1
        case (false, true)  => falsePositives += 1
        case (true, false)  => falseNegatives += 1
        case (false, false) => trueNegatives += 1
```

From these four counts we compute three standard scores. **Precision** asks: of the samples we flagged, how many were truly anomalies? **Recall** asks: of the truly anomalous samples, how many did we catch? They pull in opposite directions, so we summarize them with the **F1 score**, their harmonic mean, which stays low unless both are high:

```$
\text{precision} = \frac{TP}{TP + FP}, \quad \text{recall} = \frac{TP}{TP + FN}
```

```$
F_1 = \frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}
```

In a medical screen the balance matters: a false negative (missing a cancer) is far worse than a false positive (a needless follow-up test), so in practice you might tune `\epsilon`$ to favor recall. The F1 score gives us a single, honest number to compare against.

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

The confusion matrix in the output tells the full story. Of 108 test samples the detector caught 35 of the 39 malignant cases (recall 0.90) while raising only 5 false alarms among 69 benign cases (precision 0.88). With an F1 score over 88%, the Gaussian PDF anomaly detector successfully identifies malignant cell samples as anomalies based solely on their deviation from normal cell configurations. It never learns what cancer is; it learns what healthy cells look like and reports the samples that do not match.
