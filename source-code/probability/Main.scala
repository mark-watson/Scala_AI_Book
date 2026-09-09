//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package probability

// Medical screening case from Probability/examples/medical.lisp.
// A rare disease hits 0.1% of people. The test catches 99% of cases
// but gives false positives 5% of the time. Given a positive test,
// the chance of disease is only about 1.9%.
@main def probabilityDemo(): Unit =
  val prevalence = 0.001
  val sensitivity = 0.99
  val falsePositiveRate = 0.05

  def likelihood(hypothesis: String, evidence: String): Double =
    hypothesis match
      case "disease" => sensitivity
      case "healthy" => falsePositiveRate
      case h => throw new IllegalArgumentException(s"Unknown hypothesis: $h")

  val prior = BayesModel.fromPriors(Map("disease" -> prevalence, "healthy" -> (1.0 - prevalence)))
  val updated = BayesModel.update(prior, "positive-test", likelihood)

  println("=== Bayesian Analysis: Medical Screening Test ===")
  println("Priors:")
  prior.posteriors.foreach { case (h, p) => println(f"  P($h) = $p%.4f") }
  println("After a POSITIVE test result:")
  updated.posteriors.foreach { case (h, p) => println(f"  P($h | positive) = $p%.4f (${p * 100}%.2f %%)") }
  println(s"MAP hypothesis: ${updated.map._1}")
  println("Despite 99% sensitivity, a positive test gives only about 1.9%")
  println("chance of disease because the disease is so rare.")

  val (z, pValue) = Frequentist.zTestProportion(60, 100, 0.5)
  println(f"\nCoin toss check: 60 heads in 100 tosses vs fair coin: z = $z%.3f, p = $pValue%.4f")
  val (lo, hi) = Frequentist.wilsonInterval(60, 100)
  println(f"95%% Wilson interval for the bias: [$lo%.3f, $hi%.3f]")
