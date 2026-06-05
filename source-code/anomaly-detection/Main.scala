// Copyright 2025-2026 Mark Watson. All rights reserved.

package anomaly_detection

import scala.io.Source
import java.io.File

def printHistogram(
    title: String,
    values: Array[Array[Double]],
    indexToDisplay: Int,
    min: Double,
    max: Double,
    numBins: Int
): Unit =
  val bins = Array.ofDim[Int](numBins)
  for row <- values do
    if row != null then
      val x = row(indexToDisplay)
      var ind = (0.99 * (x - min) * numBins / max).toInt
      ind = math.max(0, math.min(ind, numBins - 1))
      bins(ind) += 1

  println(s"\n$title")
  for i <- 0 until numBins do
    println(s"$i\t${bins(i)}\t${"█" * (bins(i) / 5)}")

@main def anomalyDetectionDemo(): Unit =
  println("=" * 50)
  println("Anomaly Detection: Wisconsin Breast Cancer Data Set")
  println("=" * 50)

  val dataFile = "data/cleaned_wisconsin_cancer_data.csv"
  if !File(dataFile).exists() then
    println(s"Error: Data file not found at $dataFile")
    sys.exit(1)

  val bufferedSource = Source.fromFile(dataFile)
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
  bufferedSource.close()

  val NUM_HISTOGRAM_BINS = 5
  println("\nFeature Histograms:")
  printHistogram("Clump Thickness", trainingData, 0, 0.0, 1.0, NUM_HISTOGRAM_BINS)
  printHistogram("Uniformity of Cell Size", trainingData, 1, 0.0, 1.0, NUM_HISTOGRAM_BINS)
  printHistogram("Uniformity of Cell Shape", trainingData, 2, 0.0, 1.0, NUM_HISTOGRAM_BINS)
  printHistogram("Marginal Adhesion", trainingData, 3, 0.0, 1.0, NUM_HISTOGRAM_BINS)
  printHistogram("Single Epithelial Cell Size", trainingData, 4, 0.0, 1.0, NUM_HISTOGRAM_BINS)
  printHistogram("Bare Nuclei", trainingData, 5, 0.0, 1.0, NUM_HISTOGRAM_BINS)
  printHistogram("Bland Chromatin", trainingData, 6, 0.0, 1.0, NUM_HISTOGRAM_BINS)
  printHistogram("Normal Nucleoli", trainingData, 7, 0.0, 1.0, NUM_HISTOGRAM_BINS)
  printHistogram("Mitoses", trainingData, 8, 0.0, 1.0, NUM_HISTOGRAM_BINS)

  val detector = AnomalyDetection(10, trainingData)
  println(s"\nTraining anomaly detector on ${detector.numTraining} normal samples...")
  println(s"Cross-validation samples: ${detector.numCV}")
  println(s"Testing samples: ${detector.numTesting}")
  
  detector.train()

  println(s"\nModel parameters:")
  println(f"  best epsilon = ${detector.getBestEpsilon}%.4f")
  println(s"  num features = ${detector.muValues.length}")
