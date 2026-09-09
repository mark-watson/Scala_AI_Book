//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package probability

// Frequentist toolkit: z-tests, chi-squared tests, confidence intervals.
// Port of frequentist.lisp. A small p-value says the data would be
// unlikely if the null were true. It does not give P(null is false).
object Frequentist:

  // Standard normal CDF via Abramowitz and Stegun 26.2.17, error < 1.5e-7.
  def phiApprox(z: Double): Double =
    val p = 0.2316419
    val b1 = 0.319381530
    val b2 = -0.356563782
    val b3 = 1.781477937
    val b4 = -1.821255978
    val b5 = 1.330274429
    val az = math.abs(z)
    val t = 1.0 / (1.0 + p * az)
    val pdf = math.exp(-0.5 * az * az) / math.sqrt(2.0 * math.Pi)
    val cdf = 1.0 - pdf * (b1 * t + b2 * t * t + b3 * t * t * t + b4 * t * t * t * t + b5 * t * t * t * t * t)
    if z >= 0.0 then cdf else 1.0 - cdf

  // Chi-squared CDF via the Wilson-Hilferty transform, then phiApprox.
  def chiSquaredCdf(x: Double, df: Int): Double =
    if x <= 0.0 then 0.0
    else
      val k = df.toDouble
      val term = 2.0 / (9.0 * k)
      val z = (math.pow(x / k, 1.0 / 3.0) - (1.0 - term)) / math.sqrt(term)
      phiApprox(z)

  def zScore(observed: Double, expected: Double, stdDev: Double): Double =
    require(stdDev > 0.0, "stdDev must be positive.")
    (observed - expected) / stdDev

  // One-sample z-test for a binomial proportion. Returns (z, two-tailed p).
  def zTestProportion(successes: Int, n: Int, hypothesisedP: Double): (Double, Double) =
    val pHat = successes.toDouble / n
    val se = math.sqrt(hypothesisedP * (1.0 - hypothesisedP) / n)
    val z = (pHat - hypothesisedP) / se
    (z, math.min(2.0 * (1.0 - phiApprox(math.abs(z))), 1.0))

  // Pearson chi-squared goodness of fit. Returns (chi2, df, p).
  def chiSquaredTest(observed: Seq[Double], expected: Seq[Double]): (Double, Int, Double) =
    require(observed.size == expected.size, "Observed and expected must have the same length.")
    val chi2 = observed.zip(expected).map { case (o, e) =>
      if e == 0.0 then 0.0 else (o - e) * (o - e) / e
    }.sum
    val df = observed.size - 1
    (chi2, df, math.max(1.0 - chiSquaredCdf(chi2, df), 0.0))

  // Inverse normal CDF via Acklam's approximation, error < 1.2e-9.
  def inversePhi(p: Double): Double =
    require(p > 0.0 && p < 1.0, "p must lie strictly between 0 and 1.")
    val a = Array(-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    val b = Array(-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01)
    val c = Array(-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    val d = Array(7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)
    def poly(coef: Array[Double], x: Double): Double =
      coef.foldLeft(0.0)((acc, c) => acc * x + c)
    if p < 0.02425 then
      val q = math.sqrt(-2.0 * math.log(p))
      (((((c(0) * q + c(1)) * q + c(2)) * q + c(3)) * q + c(4)) * q + c(5)) /
        ((((d(0) * q + d(1)) * q + d(2)) * q + d(3)) * q + 1.0)
    else if p <= 0.97575 then
      val q = p - 0.5
      val r = q * q
      (((((a(0) * r + a(1)) * r + a(2)) * r + a(3)) * r + a(4)) * r + a(5)) * q /
        (((((b(0) * r + b(1)) * r + b(2)) * r + b(3)) * r + b(4)) * r + 1.0)
    else
      val q = math.sqrt(-2.0 * math.log(1.0 - p))
      -(((((c(0) * q + c(1)) * q + c(2)) * q + c(3)) * q + c(4)) * q + c(5)) /
        ((((d(0) * q + d(1)) * q + d(2)) * q + d(3)) * q + 1.0)

  // Wilson score interval for a binomial proportion. Returns (lo, hi).
  def wilsonInterval(successes: Int, n: Int, confidence: Double = 0.95): (Double, Double) =
    val z = inversePhi((1.0 + confidence) / 2.0)
    val pHat = successes.toDouble / n
    val denom = 1.0 + z * z / n
    val center = (pHat + z * z / (2.0 * n)) / denom
    val half = z * math.sqrt(pHat * (1.0 - pHat) / n + z * z / (4.0 * n * n)) / denom
    (math.max(center - half, 0.0), math.min(center + half, 1.0))
