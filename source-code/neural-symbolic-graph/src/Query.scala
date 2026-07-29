package nsk

import Unify.{Env, empty, unify, unifyTriple, resolve}

/** The result of a query: one row of `(name, value)` bindings per solution.
  *
  * The `toString` view mirrors the Common Lisp `query-result` printer so the
  * REPL shows a readable table.
  */
case class QueryResult(solutions: List[List[(String, Term)]]):
  override def toString: String =
    if solutions.isEmpty then "#<no solutions>"
    else if solutions == List(Nil) then "yes"
    else
      solutions
        .map(sol => sol.map((k, v) => s"$k=$v").mkString(", "))
        .mkString("\n")

/** Pattern matching, conjunction, and the neural fallback.
  *
  * The active graph and the neural configuration arrive as `using` parameters.
  * They are the Scala stand-in for the Common Lisp special variables `*graph*`
  * and `*ollama-url*`: a caller supplies a local `given` to rebind them, just
  * as Lisp code rebinds a special with `let`.
  */
object Query:

  /** Replace bound variables in a pattern with their values, for indexing. */
  private def ground(pat: Triple, env: Env): Triple =
    def g(t: Term): Term = t match
      case v: Term.Var => resolve(v, env)
      case _           => t
    Triple(g(pat.s), g(pat.p), g(pat.o))

  /** Every environment that satisfies PAT under ENV. A neural predicate first
    * tries a plain symbolic match on its bare name, then asks the model. */
  def matchPattern(pat: Triple, env: Env)(using graph: Graph, cfg: NeuralConfig): List[Env] =
    val neuralp = pat.p.isInstanceOf[Term.Neural]
    val spat = pat.p match
      case Term.Neural(inner) => pat.copy(p = inner)
      case _                  => pat
    val gpat    = ground(spat, env)
    val results = graph.candidates(gpat).flatMap(tr => unifyTriple(spat, tr, env))
    if results.isEmpty && neuralp then neuralMatch(gpat, spat, env).toList
    else results

  private def neuralMatch(grounded: Triple, spat: Triple, env: Env)(using
      cfg: NeuralConfig
  ): Option[Env] =
    (grounded.s, spat.o) match
      case (subject, target: Term.Var) if Term.concrete(subject) =>
        Neural
          .queryFallback(subject, grounded.p)
          .flatMap(answer => unify(target, Neural.sanitizeToKeyword(answer), Some(env)))
      case _ => None

  /** Prove PATTERNS as a conjunction, threading environments forward. */
  def prove(patterns: List[Triple], env: Env)(using Graph, NeuralConfig): List[Env] =
    patterns match
      case Nil          => List(env)
      case pat :: rest  => matchPattern(pat, env).flatMap(e => prove(rest, e))

  def runQuery(patterns: List[Triple], vars: List[Term.Var])(using
      Graph,
      NeuralConfig
  ): QueryResult =
    val sols = prove(patterns, empty).map(env => vars.map(v => v.name -> resolve(v, env)))
    QueryResult(sols.distinct)

  def matchTriple(pat: Triple)(using Graph, NeuralConfig): QueryResult =
    runQuery(List(pat), collectVars(pat))

  def matchTriple(s: Term, p: Term, o: Term)(using Graph, NeuralConfig): QueryResult =
    matchTriple(Triple(s, p, o))

  /** Run a query form produced by the reader or the `nsk` interpolator. */
  def run(form: Form)(using Graph, NeuralConfig): QueryResult = form match
    case Form.Pat(t)      => matchTriple(t)
    case Form.Ask(vs, cs) => runQuery(cs, vs)
    case Form.Cmd(n, _)   => throw IllegalArgumentException(s"not a query: $n")

  private def collectVars(pat: Triple): List[Term.Var] =
    pat.toList.flatMap {
      case v: Term.Var        => List(v)
      case Term.Neural(inner) => inner match { case v: Term.Var => List(v); case _ => Nil }
      case _                  => Nil
    }.distinct
