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
