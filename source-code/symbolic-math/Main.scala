//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package symbolicmath

@main def symbolicMathDemo(): Unit =
  val x = Variable("x")
  // p = 3x^2 - x + 5
  val p = Polynomial(x, List(Term(3.0, x, 2), Term(-1.0, x, 1), Term(5.0, x, 0))).normalize
  println(s"p        = $p")
  println(s"p'       = ${Differentiate.differentiate(p)}")
  println(s"p''      = ${Differentiate.differentiateN(p, 2)}")
  println(s"int(p)   = ${Integrate.integrate(p)} + C")
  println(f"int[0,2] = ${Integrate.definite(p, 0.0, 2.0)}%.1f")
  println(f"p(2)     = ${p.evaluate(2.0)}%.1f")
