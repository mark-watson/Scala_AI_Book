//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package probability

// Offline checks for the probability toolkit. Run with: scala-cli run . --main-class probability.probabilityTest
@main def probabilityTest(): Unit =
  def close(a: Double, b: Double, tol: Double, label: String): Unit =
    assert(math.abs(a - b) <= tol, s"$label: expected $b, got $a")

  // Bayes: priors normalize, medical posterior is about 1.94%.
  val prior = BayesModel.fromPriors(Map("disease" -> 0.001, "healthy" -> 0.999))
  close(prior.posteriors.values.sum, 1.0, 1e-12, "priors sum to 1")
  val updated = BayesModel.update(prior, "positive-test",
    (h, _) => if h == "disease" then 0.99 else 0.05)
  close(updated.posterior("disease"), 0.01943, 1e-4, "medical posterior")
  assert(updated.map._1 == "healthy", "MAP stays healthy after one positive test")
  // Second positive test raises the chance a lot.
  val twice = BayesModel.update(updated, "positive-test",
    (h, _) => if h == "disease" then 0.99 else 0.05)
  assert(twice.posterior("disease") > updated.posterior("disease"), "second positive raises posterior")

  // Normal CDF spot checks.
  close(Frequentist.phiApprox(0.0), 0.5, 1e-9, "phi(0)")
  close(Frequentist.phiApprox(1.96), 0.975, 1e-3, "phi(1.96)")
  close(Frequentist.inversePhi(0.975), 1.96, 1e-3, "inversePhi(0.975)")

  // z-test on an obvious bias: 700 heads in 1000 tosses vs fair coin.
  val (z, p) = Frequentist.zTestProportion(700, 1000, 0.5)
  assert(z > 10.0 && p < 1e-9, s"biased coin must reject fair null, got z=$z p=$p")

  // Chi-squared on near-uniform dice rolls: high p-value keeps the null.
  val (_, _, pDice) = Frequentist.chiSquaredTest(
    Seq(16.0, 18.0, 15.0, 17.0, 16.0, 18.0), Seq(16.67, 16.67, 16.67, 16.67, 16.67, 16.67))
  assert(pDice > 0.5, s"fair dice must keep the null, got p=$pDice")

  // Wilson interval covers the observed rate and stays in [0, 1].
  val (lo, hi) = Frequentist.wilsonInterval(60, 100)
  assert(lo <= 0.6 && 0.6 <= hi && lo >= 0.0 && hi <= 1.0, s"bad Wilson interval [$lo, $hi]")

  // Correlation: perfect line gives r = 1, flat series gives 0.
  close(Correlation.pearson(Seq(1.0, 2.0, 3.0, 4.0), Seq(2.0, 4.0, 6.0, 8.0)), 1.0, 1e-12, "pearson line")
  close(Correlation.pearson(Seq(1.0, 1.0, 1.0), Seq(2.0, 4.0, 6.0)), 0.0, 1e-12, "pearson constant")

  println("All probability tests passed.")
