//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package symbolicmath

// Symbolic differentiation of polynomials. Port of differentiation.lisp.
// Power rule: d/dx(c * x^n) = n*c * x^(n-1); sum rule applies per term.
object Differentiate:
  def differentiateTerm(term: Term): Option[Term] =
    if term.exponent == 0 then None
    else Some(Term(term.exponent * term.coefficient, term.variable, term.exponent - 1))

  def differentiate(poly: Polynomial): Polynomial =
    val terms = poly.terms.flatMap(differentiateTerm)
    if terms.isEmpty then Polynomial.zero(poly.variable)
    else Polynomial(poly.variable, terms).normalize

  def differentiateN(poly: Polynomial, n: Int): Polynomial =
    require(n >= 0, "Order of derivative must be non-negative.")
    if n == 0 then poly else differentiateN(differentiate(poly), n - 1)

  def gradientAt(poly: Polynomial, x: Double): Double =
    differentiate(poly).evaluate(x)

  def criticalPoint(poly: Polynomial, x: Double, tol: Double = 1e-9): Boolean =
    math.abs(gradientAt(poly, x)) <= tol
