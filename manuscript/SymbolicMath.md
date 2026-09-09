# Symbolic Math: Differentiate and Integrate

Neural nets learn numbers through gradient steps, yet the gradient idea starts with pen and paper calculus. This chapter builds a small symbolic math core in Scala 3: terms, polynomials, exact differentiation, and exact integration. No numeric fits, no step error. The rules live in the types, and the compiler checks each rewrite.

All code is in `source-code/symbolic-math`.

## Terms and Polynomials as Case Classes

A term is a number times a variable to a whole power. A polynomial is a list of terms in one variable. Both are plain case classes, so match and copy do the whole work:

```scala
final case class Term(coefficient: Double, variable: Variable, exponent: Int)

final case class Polynomial(variable: Variable, terms: List[Term]):
  def normalize: Polynomial =
    val grouped = terms.groupBy(_.exponent).view.mapValues(_.map(_.coefficient).sum).toMap
    val kept = grouped.toList.filter(_._2 != 0.0).map { case (e, c) => Term(c, variable, e) }
    copy(terms = kept.sortBy(-_.exponent))
```

`normalize` joins like terms and drops zeros, so `3x + 2x` prints as `5x` and `x - x` prints as `0`. Each op builds a new value. Nothing mutates, which keeps long rewrite chains safe to read.

## Differentiation: Power Rule Plus Sum Rule

One term differentiates by the power rule: `d/dx(c x^n) = n c x^(n-1)`. Constants map to `None` and fall out of the sum. The full derivative maps that rule over all terms:

```scala
def differentiateTerm(term: Term): Option[Term] =
  if term.exponent == 0 then None
  else Some(Term(term.exponent * term.coefficient, term.variable, term.exponent - 1))

def differentiateN(poly: Polynomial, n: Int): Polynomial =
  require(n >= 0, "Order of derivative must be non-negative.")
  if n == 0 then poly else differentiateN(differentiate(poly), n - 1)
```

For `p = 3x^2 - x + 5` the code gives `p' = 6x - 1` and `p'' = 6`. Two helpers ride on top: `gradientAt` scores the slope at a point, and `criticalPoint` tests if the slope is near zero. The demo finds the vertex of the parabola at `x = 1/6`.

## Integration: Reverse Power Rule

Integration runs the power rule in reverse: the integral of `c x^n` is `(c/(n+1)) x^(n+1)`. Definite integrals then use the base theorem, `F(b) - F(a)`:

```scala
def integrateTerm(term: Term): Term =
  val e = term.exponent + 1
  Term(term.coefficient / e, term.variable, e)

def definite(poly: Polynomial, a: Double, b: Double): Double =
  val f = integrate(poly)
  f.evaluate(b) - f.evaluate(a)
```

The integral of `3x^2` from 0 to 2 is exactly 8. One test round trips the pair: differentiate the integral of `p` and get `p` back. If that assert holds, both rules agree.

Run the demo and the checks:

```bash
cd source-code/symbolic-math
scala-cli run . --main-class symbolicmath.symbolicMathDemo
scala-cli run . --main-class symbolicmath.symbolicMathTest
```

This core is small on purpose. It shows how sealed types plus match can carry math rules with no extra deps, a shape later chapters reuse for search states, grammar tags, and graph triples.
