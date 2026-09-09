# Probability: Bayes, Base Rates, and Tests

Many AI methods use probability, yet few developers study it direct. This chapter builds the two main schools in pure Scala 3: Bayesian update, which revises beliefs with facts, and frequentist tests, which check a null claim against facts. It closes with correlation, the most used and most abused number in facts work.

All code is in `source-code/probability`.

## Bayes Theorem in Seven Lines

A Bayes model is a map from claim to chance. Priors must sum to one, so the constructor scales them. Update then applies Bayes Theorem: multiply each prior by its likelihood, the chance of the seen facts under that claim, then scale again:

```scala
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
```

The `require` calls state the two ways Bayes fails: priors that sum to zero, and facts no claim allows. Both are user errors, so loud failure beats quiet NaN.

## The Medical Test That Fools Doctors

A rare disease hits 0.1% of people. A screen test spots 99% of cases but cries wolf 5% of the time. A patient tests positive. What is the chance she has the disease? Most people guess near 99%. Bayes says about 1.9%:

```scala
val prior = BayesModel.fromPriors(Map("disease" -> 0.001, "healthy" -> 0.999))
val updated = BayesModel.update(prior, "positive-test",
  (h, _) => if h == "disease" then 0.99 else 0.05)
// updated.posterior("disease") == 0.0194
```

The math is plain: true positives (0.001 times 0.99) drown in false positives (0.999 times 0.05). Base rates rule. A second positive test lifts the chance far, which the demo shows by updating twice. This case runs in `probability/Main.scala`.

## Frequentist Checks: z, Chi-Squared, Wilson

Bayes needs priors you may lack. Frequentist tools ask a narrower query: if the null claim held, how odd are these facts? In **probability/Frequentist.scala** we build the normal CDF by the Abramowitz and Stegun formula, the chi-squared CDF by the Wilson-Hilferty shift, then three checks on top:

```scala
val (z, pValue) = Frequentist.zTestProportion(60, 100, 0.5)
val (chi2, df, pDice) = Frequentist.chiSquaredTest(observed, expected)
val (lo, hi) = Frequentist.wilsonInterval(60, 100)
```

A small p says the facts would be rare under the null. It does not say the null is false with chance p. That swap is the most common error in applied stats, so state it each time you quote p.

The Wilson interval avoids the classic trap of the Wald interval, which collapses near 0 or 1. For 60 heads in 100 tosses it returns about [0.50, 0.69], a range you can read with no test lore.

## Correlation in Nine Lines

Pearson r falls out of covariance over both spreads. It returns 0 for flat series instead of NaN, since a flat line carries no trend to track:

```scala
def pearson(xs: Seq[Double], ys: Seq[Double]): Double =
  val sx = stdDev(xs)
  val sy = stdDev(ys)
  if sx == 0.0 || sy == 0.0 then 0.0
  else covariance(xs, ys) / (sx * sy)
```

High r means the series move as one. It never means one drives the other. The medical case proves why: test results track true cases well, yet one positive test still means scant chance of disease.

Run the demo and the checks:

```bash
cd source-code/probability
scala-cli run . --main-class probability.probabilityDemo
scala-cli run . --main-class probability.probabilityTest
```

The tests pin the 1.94% posterior, the normal CDF at 0 and 1.96, a biased coin reject, a fair dice keep, and r of 1.0 on a clean line.
