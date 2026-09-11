# Symbolic Math: Differentiate and Integrate

Neural networks operate on numeric parameters adjusted by gradient descent. Symbolic mathematics, by contrast, operates by manipulating symbolic expressions.

This chapter implements a small symbolic mathematics core in Scala 3: terms, polynomials, exact differentiation, and exact integration. There are no numeric approximations, no step size to tune, and no truncation error. The rules of calculus are encoded directly, and every result is an exact expression.

All code is in `source-code/symbolic-math`.

## Two Ways to Do Calculus on a Computer

Most calculus software used in machine learning is **numeric**. It represents a function as code and estimates its slope by perturbing the input by a small amount:

```$
f'(x) \approx \frac{f(x+h) - f(x)}{h}
```

This **finite difference** is simple to implement and applies to any function that can be evaluated, which is why it is ubiquitous in numerical work. The limitation is the approximation itself. If `h`$ is too large, curvature biases the estimate. If `h`$ is too small, cancellation between two nearly equal numbers destroys precision. No single value of `h`$ is suitable for every function, and the result is always a decimal rather than a formula.

**Symbolic** calculus takes the opposite approach. It represents an expression as data, a tree of operators and symbols, and rewrites that tree using the standard rules of calculus. Differentiating `3x^2 - x + 5`$ returns the exact expression `6x - 1`$, not a table of slopes. Evaluating that expression at a point is a separate, optional step. Because the manipulation is exact, symbolic methods expose structure that numeric methods conceal: the location of zeros of the derivative, the form of the antiderivative, and the relationship between two expressions.

Symbolic manipulation is one of the oldest branches of artificial intelligence. Systems such as Macsyma at MIT in the 1960s, and later Mathematica, Maple, and SymPy, were built to perform algebra symbolically, and much of that lineage was written in Lisp. The code in this chapter is a direct port of a small Common Lisp calculus package, and it follows the same principle: represent the mathematics as data, then apply rewrite rules.

For AI in particular, symbolic differentiation is the precursor to **automatic differentiation**, the technique underlying every deep learning framework. Automatic differentiation applies the same rules (power, sum, product, chain) mechanically to a program while carrying numeric values through the computation. Understanding the symbolic core makes the autodiff engine easier to reason about. The implementation here is deliberately small, and the chapter closes by noting what a full computer algebra system would add.

## Terms and Polynomials as Data

We start with the objects we can represent exactly. A **term** is a coefficient times a variable raised to a whole-number power:

```$
\text{term} = c \cdot x^{n}
```

A **polynomial** in one variable is a sum of such terms:

```$
p(x) = \sum_{k=0}^{n} a_k x^{k}
```

This representation covers a substantial part of introductory calculus and makes every rule exact. It consists of two case classes: one for a term and one for the polynomial that holds a list of terms. The complete data model, in `Expr.scala`, is as follows:

```scala
//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package symbolicmath

// Core data model. Port of data.lisp: a polynomial in one variable is
// a list of terms, each term is coefficient * variable ^ exponent.
final case class Variable(name: String)

final case class Term(coefficient: Double, variable: Variable, exponent: Int):
  def negate: Term = copy(coefficient = -coefficient)
  def scale(k: Double): Term = copy(coefficient = coefficient * k)
  override def toString: String =
    val c = if coefficient == coefficient.toLong then coefficient.toLong.toString else coefficient.toString
    exponent match
      case 0 => s"$c"
      case 1 => s"${c}${variable.name}"
      case n => s"${c}${variable.name}^$n"

final case class Polynomial(variable: Variable, terms: List[Term]):
  def degree: Int = terms.map(_.exponent).maxOption.getOrElse(0)
  def normalize: Polynomial =
    val grouped = terms.groupBy(_.exponent).view.mapValues(_.map(_.coefficient).sum).toMap
    val kept = grouped.toList.filter(_._2 != 0.0).map { case (e, c) => Term(c, variable, e) }
    copy(terms = kept.sortBy(-_.exponent))
  def add(other: Polynomial): Polynomial =
    require(variable == other.variable, "Cannot add polynomials in different variables.")
    copy(terms = terms ++ other.terms).normalize
  def subtract(other: Polynomial): Polynomial = add(other.copy(terms = other.terms.map(_.negate)))
  def scale(k: Double): Polynomial = copy(terms = terms.map(_.scale(k))).normalize
  def evaluate(x: Double): Double =
    terms.map(t => t.coefficient * math.pow(x, t.exponent)).sum
  override def toString: String =
    if terms.isEmpty then "0"
    else terms.mkString(" + ").replace("+ -", "- ")

object Polynomial:
  def zero(v: Variable): Polynomial = Polynomial(v, Nil)
  def constant(v: Variable, c: Double): Polynomial = Polynomial(v, List(Term(c, v, 0))).normalize
  def identity(v: Variable): Polynomial = Polynomial(v, List(Term(1.0, v, 1)))
```

The design mirrors the algebra. `Variable` is a name, which allows the code to reject an attempt to add a polynomial in `x`$ to one in `y`$ rather than producing an invalid result. `Term` bundles the three values that define a monomial and provides `negate` and `scale`, the only two operations a term requires in this chapter. `Polynomial` holds the variable and a list of terms, and its companion object provides the three constants the algebra requires: zero, an arbitrary constant, and the identity `x`$.

### Canonical Form and the Role of `normalize`

The central method is `normalize`, and it must be understood before any calculus is attempted. When polynomials are added or scaled, the term list can contain **like terms**, terms with the same exponent, and terms whose coefficients cancel to zero. The raw list is correct but not canonical, so two polynomials that are mathematically equal may have different term lists. This breaks structural equality, and `d/dx(int(p)) == p` would fail even when the mathematics is correct.

`normalize` restores a **canonical form** in three steps:

1. `groupBy(_.exponent)` collects terms that share an exponent.
2. `mapValues(_.map(_.coefficient).sum)` adds their coefficients, so `3x + 2x`$ becomes `5x`$.
3. The result is filtered to drop any coefficient equal to zero, then sorted by descending exponent.

After normalization, `x - x`$ becomes the empty term list, which prints as `0`, and `3x + 2x`$ prints as `5x`$. Sorting by descending exponent places the highest power first, the conventional order for polynomials. This is what allows the test suite to compare whole polynomials with `==`: equal polynomials have equal normalized term lists.

The remainder of the class is small and purely functional. `degree` reports the largest exponent, or zero for the zero polynomial. `add` verifies that both polynomials share a variable, concatenates the term lists, and normalizes. `subtract` negates the second polynomial and adds. `scale` multiplies every coefficient. `evaluate` computes the numeric value at a point `x`$ using `math.pow`, which handles fractional and negative inputs correctly because the exponent is an integer. Every method returns a new value and leaves the receiver unchanged, so a sequence of rewrites is safe to reason about.

The `toString` implementations contain two notable details. First, a coefficient that is a whole number is printed without a decimal point, so `5.0` prints as `5` while `0.5` prints as `0.5`. Second, a term of exponent 0 prints as the bare coefficient, exponent 1 prints as `6x`, and higher powers print as `3x^2`. The polynomial joins terms with `" + "` and then replaces `"+ -"` with `"- "`, so negative coefficients appear as subtraction. This substitution is why `6x - 1` appears instead of `6x + -1`.

## Differentiation: The Power Rule and Linearity

The derivative measures the instantaneous rate of change, defined by the limit

```$
f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
```

For a power of the variable, expanding `(x+h)^n`$ with the binomial theorem and letting `h`$ go to zero leaves exactly one surviving term, which gives the **power rule**:

```$
\frac{d}{dx}\left(x^{n}\right) = n x^{n-1}
```

A constant multiple pulls out, and a sum differentiates term by term, the **sum rule**:

```$
\frac{d}{dx}\left(c\,x^{n}\right) = n c\,x^{n-1}
\qquad\qquad
\frac{d}{dx}\left(f + g\right) = f' + g'
```

Together these two rules say differentiation is a **linear operator**, and they are all a polynomial needs. The code applies them per term, then collects the results. Here is `Differentiate.scala` in full:

```scala
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
```

`differentiateTerm` implements the power rule directly. It multiplies the coefficient by the exponent and decrements the exponent. A constant term has exponent zero, and its derivative is zero, so the function returns `None` rather than a term with a zero coefficient. That choice lets `differentiate` use `flatMap`, which transforms each term and discards the `None` results in a single pass. The `Option` return type expresses that a term either survives differentiation or vanishes.

`differentiate` applies that rule to every term and normalizes the survivors. The explicit `if terms.isEmpty` branch returns the zero polynomial when the input was constant; `normalize` would reach the same state, but the branch makes the intent clear. `differentiateN` applies the operator `n`$ times by recursion, with a `require` guard that rejects a negative order. Its base case returns the polynomial unchanged, since the zeroth derivative of a function is the function itself.

The last two methods connect the symbolic representation to numeric values. `gradientAt` differentiates symbolically and then evaluates the exact derivative at a point, so the slope is computed from a formula rather than a finite difference. `criticalPoint` tests whether that slope is zero within a tolerance, the condition for a local maximum, minimum, or saddle point. For `p(x) = 3x^2 - x + 5`$ the derivative is `p'(x) = 6x - 1`$, which is zero at `x = 1/6`$, the vertex of the upward parabola. The test suite checks that point explicitly.

## Integration: The Reverse Power Rule and the Fundamental Theorem

Integration runs the power rule backward. Since differentiating `x^{n+1}`$ multiplies by `n+1`$, integrating must divide by it:

```$
\int c\,x^{n}\,dx = \frac{c}{n+1}\,x^{n+1} + C
\qquad\qquad n \neq -1
```

The added `C`$ is the **constant of integration**. Any constant differentiates to zero, so an antiderivative is defined only up to an additive constant, and a complete answer must state it. The code returns the antiderivative with `C = 0`$ and lets the caller add `C`$ when printing. For polynomials with non-negative exponents, `n+1`$ is at least one, so the division is always defined. Like differentiation, integration is linear and applies term by term.

A **definite** integral goes from a lower limit `a`$ to an upper limit `b`$. The **fundamental theorem of calculus** says the answer is the antiderivative evaluated at the endpoints:

```$
\int_a^b f(x)\,dx = F(b) - F(a)
```

The constant of integration cancels in the subtraction, which is why the code never has to track it. Here is `Integrate.scala` in full:

```scala
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
```

`integrateTerm` increments the exponent and divides the coefficient by the new exponent. It always returns a `Term`, never an `Option`, because no polynomial term integrates to zero. `integrate` applies it across the terms and normalizes. `integrateN` repeats the operation for higher antiderivatives, with the same non-negative guard as `differentiateN`. `definite` integrates once to obtain an antiderivative, then subtracts its value at `a`$ from its value at `b`$, following the theorem directly.

The two operators are inverse in one direction. Differentiating an antiderivative returns the original polynomial, so the test suite asserts `Differentiate.differentiate(Integrate.integrate(p)) == p`. The reverse order holds only up to the constant: integrating a derivative returns the original plus an unknown `C`$, so no such equality is asserted in the other direction. This asymmetry is precisely why the constant of integration exists.

## The Demo

`Main.scala` exercises every operation on a single polynomial:

```scala
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
```

The polynomial is constructed from an explicit list of terms and normalized, so the printed order is canonical. The demo prints the polynomial, its first and second derivatives, its indefinite integral (with the `+ C` appended by the print statement rather than the library), a definite integral, and a point evaluation. Running it from the example directory:

```bash
cd source-code/symbolic-math
scala-cli run . --main-class symbolicmath.symbolicMathDemo
```

produces:

```text
p        = 3x^2 - 1x + 5
p'       = 6x - 1
p''      = 6
int(p)   = 1x^3 - 0.5x^2 + 5x + C
int[0,2] = 16.0
p(2)     = 15.0
```

## The Test Suite

The project includes a self-checking test file that requires no test framework. It uses plain `assert` calls and a small `close` helper for floating point comparisons:

```scala
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
```

The checks target the important behavior rather than coverage. The first two confirm the power rule and the sum rule on a concrete polynomial. The zeroth-derivative check confirms the base case of the recursion. The constant check confirms that a term of exponent zero disappears entirely, leaving an empty term list. The integration checks confirm the reverse power rule symbolically and the fundamental theorem numerically. The round trip is the strongest test in the file: it exercises both operators across all three terms and would catch an error in either rule. The final numeric checks connect the symbolic derivative back to the geometry, confirming `p(2) = 15`$, `p'(1) = 5`$, and that `x = 1/6`$ is a critical point.

The `close` helper compares doubles within a tolerance rather than for exact equality. This is the correct approach for floating point arithmetic: `1/6`$ is not exactly representable, and an antiderivative such as `(2/3)x^3`$ produces a repeating decimal. Exact comparison would fail on rounding error unrelated to the logic. The tolerance is `1e-9`$ for these small numbers, loose enough to absorb representation error and tight enough to detect a genuine mistake.

Run it with:

```bash
scala-cli run . --main-class symbolicmath.symbolicMathTest
```

and it prints one line:

```text
All symbolic math tests passed.
```

## Interpreting the Output

The demo output demonstrates that the rules agree with hand calculus.

`p = 3x^2 - 1x + 5` is the input polynomial. The `1x` is the coefficient-one term printed literally, a minor cosmetic limitation of `toString`. The mathematics is correct; a more careful printer would omit the `1` and write `x`, which is left as an exercise.

`p' = 6x - 1` applies the power rule to each term: `3x^2`$ gives `6x`$, `-x`$ gives `-1`$, and the constant `5`$ gives nothing. The sum of those results is the exact derivative, with no decimal approximation.

`p'' = 6` differentiates once more. The `6x`$ term becomes `6`$ and the `-1`$ term disappears. The second derivative of a quadratic is constant, the algebraic statement that its slope changes at a constant rate.

`int(p) = 1x^3 - 0.5x^2 + 5x + C` reverses the process term by term: `3x^2`$ integrates to `x^3`$, `-x`$ to `-0.5x^2`$, and `5`$ to `5x`$. The `+ C`$ is appended by the demo because the library returns the `C = 0`$ antiderivative. Omitting `C`$ is common in software but a frequent source of error, so the print statement makes it explicit.

`int[0,2] = 16.0` evaluates the antiderivative at the endpoints. Substituting `x = 2`$ gives `8 - 2 + 10 = 16`$ and substituting `x = 0`$ gives `0`$, so the definite integral is `16`$. This is the signed area under the curve between the two limits. The test file checks the simpler case `\int_0^2 3x^2\,dx = 8`$ with the same machinery.

`p(2) = 15.0` is a direct evaluation: `3(4) - 2 + 5 = 15`$. This value does not involve calculus and confirms that `evaluate` agrees with arithmetic.

Taken together, the output shows the three requirements of a computer algebra system: rewrite an expression exactly, evaluate the result at a point, and remain consistent across inverse operations. The round-trip test verifies the last property.

## What This Core Does Not Do

The limits of a small system are as important as its features, because they define the starting point for any larger implementation.

- **One variable only.** Every polynomial carries a single `Variable`, and `add` refuses to mix variables. Multivariate polynomials would need a term to hold a monomial over several variables, and differentiation would become a partial derivative that names the variable to differentiate.
- **Polynomials only.** There is no product rule, quotient rule, or chain rule, because there are no products, quotients, or compositions in the data model. Adding those rules means replacing the flat term list with a general expression tree whose nodes are `Add`, `Mul`, `Pow`, `Sin`, and so on. The pattern matching style carries over unchanged.
- **No simplification beyond collecting like terms.** `normalize` combines coefficients, but it does not expand `(x+1)^2`$, factor `x^2 - 1`$, or cancel rational expressions. Each of those is a separate rewrite system.
- **The exponent is not constrained.** Nothing stops a caller from constructing `Term(1.0, x, -1)`$. Differentiation handles it, but integration divides by `exponent + 1`$ and would divide by zero. A `require` in the `Term` constructor or a check in `integrateTerm` would close the gap.
- **Floating point coefficients.** Coefficients are `Double`, so a coefficient like `1/3`$ is inexact. Collecting like terms compares against `0.0` with exact equality, which is safe for the halves and thirds produced by these rules but could misjudge a true cancellation that arrives by two rounding paths. A rational number type would make the algebra exact.

Each of these is a deliberate trade. The value of the small core is that every limitation is visible within a few dozen lines, and each has a clear path to a more capable implementation.

## Wrap Up

This chapter implemented a working computer algebra core in pure Scala with no dependencies. The components are small, but together they cover the full path from representation to verification.

- **Terms and polynomials as case classes** turn algebra into data. `normalize` restores a canonical form, which is what makes exact structural equality, and therefore round-trip testing, possible.
- **Differentiation** is the power rule plus linearity. `differentiateTerm` returns an `Option` so vanishing constants are removed by `flatMap`, and `differentiateN` composes the rule for higher orders.
- **Integration** is the reverse power rule plus the fundamental theorem of calculus. The constant of integration is dropped because the definite integral cancels it, and the round trip `differentiate(integrate(p)) == p` verifies that the two rules are inverse.
- **The test suite** verifies each rule with a concrete value and one cross-check, using a tolerance for the floating point comparisons that exact equality cannot survive.

The broader point concerns representation. Here calculus is not a collection of formulas to memorize but a set of rewrite rules acting on a type. The same structure, small immutable value types with companion-object operations and pure functions, reappears later in this book for search states, grammar tags, and graph triples. When a later chapter addresses automatic differentiation or a neural-symbolic system, the symbolic component is the machinery built here, scaled up.

## Optional Practice Problems

1. **Nicer printing.** The demo prints `1x^3` and `1x` where a mathematician writes `x^3` and `x`. Modify `Term.toString` so a coefficient of `1` is omitted (except for a constant term, where it must still print), and update the assertion in `symbolicMathTest` that expects `1x^3`.

2. **Guard the exponent.** Add `require(exponent >= 0, "Exponent must be non-negative.")` to the `Term` case class. Confirm the existing tests still pass, then try constructing `Term(1.0, Variable("x"), -1)` and report the error. Explain why the guard is better placed on the type than inside `Integrate.integrateTerm`.

3. **Multiply two polynomials.** Add `def multiply(other: Polynomial): Polynomial` to `Polynomial`. Multiply term by term, add the exponents and multiply the coefficients, and normalize the result. Test it on `(x + 1)(x - 1)`$ and confirm the answer is `x^2 - 1`$. Why can you not express this with the existing `add` and `scale` alone?

4. **Check the derivative against a finite difference.** Write a function `finiteDifference(p: Polynomial, x: Double, h: Double): Double` that computes `(p(x+h) - p(x))/h`$, and compare it to `Differentiate.gradientAt`$ for `p(x) = 3x^2 - x + 5`$ at `x = 1`$ with `h = 1e-4`$, `1e-6`$, and `1e-8`$. Report the error at each step and explain why it eventually gets worse as `h`$ shrinks.

5. **Numeric integration for comparison.** Implement a Riemann sum or Simpson's rule that approximates `\int_a^b p(x)\,dx`$ from evaluations of `p`$. Compare it to `Integrate.definite` for `p(x) = 3x^2 - x + 5`$ on `[0, 2]`$ and report the difference. Which method would you trust for a polynomial and why?

6. **Second derivative test.** Add a function `classifyCriticalPoint(p: Polynomial, x: Double): String` that returns `"minimum"`, `"maximum"`, or `"inflection"` using the sign of the second derivative. Test it at `x = 1/6`$ for `p(x) = 3x^2 - x + 5`$ and confirm it reports a minimum.

7. **Higher-order round trip.** Assert that `differentiateN(integrateN(p, k), k) == p`$ for `k = 1, 2, 3`$ on a polynomial of degree at least three. Does the identity hold for every `k`$? Explain the role of the dropped constants.

8. **A general expression tree.** Replace the flat polynomial with a `sealed trait Expr` and cases `Const`, `Var`, `Add`, `Mul`, and `Pow`. Implement `derivative` using the sum, product, and power rules. Test it on `(x + 1)(x + 1)`$ and compare the result to differentiating `x^2 + 2x + 1`$ directly. What extra simplification would you need to make the two answers print identically?
