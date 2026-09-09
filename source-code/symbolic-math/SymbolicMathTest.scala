//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package symbolicmath

// Offline checks. Run with: scala-cli run . --main-class symbolicmath.symbolicMathTest
@main def symbolicMathTest(): Unit =
  def close(a: Double, b: Double, tol: Double, label: String): Unit =
    assert(math.abs(a - b) <= tol, s"$label: expected $b, got $a")

  val x = Variable("x")
  // p = 3x^2 - x + 5, p' = 6x - 1, p'' = 6
  val p = Polynomial(x, List(Term(3.0, x, 2), Term(-1.0, x, 1), Term(5.0, x, 0))).normalize
  assert(Differentiate.differentiate(p).toString == "6x - 1", s"p' wrong: ${Differentiate.differentiate(p)}")
  assert(Differentiate.differentiateN(p, 2).toString == "6", s"p'' wrong: ${Differentiate.differentiateN(p, 2)}")
  assert(Differentiate.differentiateN(p, 0) == p, "zeroth derivative is identity")

  // Constants differentiate to the zero polynomial.
  val c = Polynomial.constant(x, 5.0)
  assert(Differentiate.differentiate(c).terms.isEmpty, "d/dx(5) must be zero")

  // ∫ 3x^2 dx = x^3; ∫[0,2] 3x^2 dx = 8.
  val q = Polynomial(x, List(Term(3.0, x, 2)))
  assert(Integrate.integrate(q).toString == "1x^3", s"integral wrong: ${Integrate.integrate(q)}")
  close(Integrate.definite(q, 0.0, 2.0), 8.0, 1e-9, "definite integral")

  // Round trip: differentiate(integrate(p)) == p.
  assert(Differentiate.differentiate(Integrate.integrate(p)) == p, "d/dx ∫ p dx must equal p")

  // Numeric checks: p(2) = 15, gradient at x=1 is 5, x=1/6 is critical.
  close(p.evaluate(2.0), 15.0, 1e-9, "evaluate")
  close(Differentiate.gradientAt(p, 1.0), 5.0, 1e-9, "gradient")
  assert(Differentiate.criticalPoint(p, 1.0 / 6.0), "vertex of parabola is critical")

  println("All symbolic math tests passed.")
