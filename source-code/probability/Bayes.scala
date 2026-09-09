//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package probability

// Bayes model: a normalized map of hypothesis -> probability.
// Port of bayes.lisp: P(H|E) = P(E|H) * P(H) / sum_h P(E|h) * P(h)
final case class BayesModel(posteriors: Map[String, Double]):
  def posterior(hypothesis: String): Double =
    posteriors.getOrElse(
      hypothesis,
      throw new NoSuchElementException(s"Hypothesis $hypothesis not found in model.")
    )
  def map: (String, Double) = posteriors.maxBy(_._2)

object BayesModel:
  def fromPriors(priors: Map[String, Double]): BayesModel =
    val total = priors.values.sum
    require(total > 0.0, "All priors are zero, cannot normalize.")
    BayesModel(priors.view.mapValues(_ / total).toMap)

  def update(
      model: BayesModel,
      evidence: String,
      likelihood: (String, String) => Double
  ): BayesModel =
    val unnorm = model.posteriors.map { case (h, prior) => h -> likelihood(h, evidence) * prior }
    val marginal = unnorm.values.sum
    require(marginal > 0.0, "Marginal likelihood is zero, evidence is impossible under all hypotheses.")
    BayesModel(unnorm.view.mapValues(_ / marginal).toMap)
