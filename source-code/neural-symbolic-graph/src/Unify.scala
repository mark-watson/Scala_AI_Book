package nsk

/** Unification in the Norvig/PAIP style.
  *
  * The environment is a map from logic variable to value. Common Lisp uses a
  * `+FAIL+` sentinel so that `NIL` can stand for the empty successful
  * environment; Scala models the same idea more directly with `Option`, where
  * `None` is failure and `Some(env)` is success.
  */
object Unify:
  type Env = Map[Term.Var, Term]

  val empty: Env = Map.empty

  def unify(x: Term, y: Term, env: Option[Env] = Some(empty)): Option[Env] =
    env.flatMap: e =>
      (x, y) match
        case _ if x == y      => Some(e)
        case (v: Term.Var, _) => unifyVar(v, y, e)
        case (_, v: Term.Var) => unifyVar(v, x, e)
        case _                => None

  private def unifyVar(v: Term.Var, x: Term, e: Env): Option[Env] =
    e.get(v) match
      case Some(bound) => unify(bound, x, Some(e))
      case None        => Some(e + (v -> x))

  /** Unify a pattern triple against a stored triple, position by position. */
  def unifyTriple(pat: Triple, tr: Triple, env: Env): Option[Env] =
    unify(pat.s, tr.s, Some(env))
      .flatMap(e => unify(pat.p, tr.p, Some(e)))
      .flatMap(e => unify(pat.o, tr.o, Some(e)))

  /** Replace bound variables in X with their values, recursively. */
  def resolve(x: Term, e: Env): Term = x match
    case v: Term.Var => e.get(v).map(resolve(_, e)).getOrElse(v)
    case _           => x
