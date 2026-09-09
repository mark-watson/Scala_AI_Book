//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package symbolicmath

// Symbolic integration of polynomials. Port of integration.lisp.
// Reverse power rule: integral c * x^n dx = (c/(n+1)) * x^(n+1).
// Definite integrals use the fundamental theorem: F(b) - F(a).
object Integrate:
  def integrateTerm(term: Term): Term =
    val e = term.exponent + 1
    Term(term.coefficient / e, term.variable, e)

  def integrate(poly: Polynomial): Polynomial =
    Polynomial(poly.variable, poly.terms.map(integrateTerm)).normalize

  def integrateN(poly: Polynomial, n: Int): Polynomial =
    require(n >= 0, "Order of integral must be non-negative.")
    if n == 0 then poly else integrateN(integrate(poly), n - 1)

  def definite(poly: Polynomial, a: Double, b: Double): Double =
    val f = integrate(poly)
    f.evaluate(b) - f.evaluate(a)
