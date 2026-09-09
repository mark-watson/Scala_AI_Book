//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package probability

// Correlation helpers. Port of correlation.lisp.
object Correlation:
  def mean(xs: Seq[Double]): Double =
    require(xs.nonEmpty, "Cannot take the mean of an empty series.")
    xs.sum / xs.size

  def covariance(xs: Seq[Double], ys: Seq[Double]): Double =
    require(xs.size == ys.size && xs.nonEmpty, "Series must be non-empty and equal in length.")
    val mx = mean(xs)
    val my = mean(ys)
    xs.zip(ys).map { case (x, y) => (x - mx) * (y - my) }.sum / xs.size

  def stdDev(xs: Seq[Double]): Double =
    math.sqrt(covariance(xs, xs))

  // Pearson r in [-1, 1]. Returns 0 when either series is constant.
  def pearson(xs: Seq[Double], ys: Seq[Double]): Double =
    val sx = stdDev(xs)
    val sy = stdDev(ys)
    if sx == 0.0 || sy == 0.0 then 0.0
    else covariance(xs, ys) / (sx * sy)
