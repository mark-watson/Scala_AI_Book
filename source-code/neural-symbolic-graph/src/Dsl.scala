package nsk

import scala.language.dynamics
import scala.language.implicitConversions

/** An embedded Scala DSL that recreates the Common Lisp reader-macro syntax.
  *
  * Import it with `import nsk.dsl.*`. It offers two ways to write queries:
  *
  *   - the `nsk"..."` string interpolator, which parses the exact Lisp
  *     surface syntax (this is the closest analog of a reader macro);
  *   - native Scala combinators: `v.who` for a logic variable, `~pred` for a
  *     neural predicate, a `String -> keyword` conversion, and `ask`.
  */
object dsl:

  /** `?who` becomes `v.who`. `scala.Dynamic` turns the field name into a
    * logic variable at compile time, which is how Common Lisp's `?` reader
    * macro turns the following symbol into `(logic-var 'who)`. */
  object v extends Dynamic:
    def selectDynamic(name: String): Term.Var       = Term.Var(name)
    def applyDynamic(name: String)(): Term.Var      = Term.Var(name)

  extension (sc: StringContext)
    /** `:mark` becomes `kw"mark"`. */
    def kw(args: Any*): Term.Kw =
      Term.Kw(sc.s(args*).trim.toUpperCase.replace(' ', '-'))

    /** The NSK reader as a string interpolator: `nsk"[?who :wrote :nsk]"`.
      * A spliced `Term` round-trips through its `toString`, which the reader
      * parses back. */
    def nsk(args: Any*): Form = Reader.readForm(sc.s(args*))

  /** `~pred` wraps a term as a neural predicate. Scala allows `unary_~` as a
    * prefix operator, so this reads exactly like the Lisp `~`. */
  extension (t: Term) def unary_~ : Term.Neural = Term.Neural(t)

  /** A plain Scala string is an atom in query positions, so `"wrote"` means
    * the keyword `:wrote`. Use `Term.Str(...)` for a real string value. */
  given Conversion[String, Term] = s => Term.Kw(s.toUpperCase)

  /** Build a triple pattern: `t(v.who, "wrote", "nsk")`. */
  def t(s: Term, p: Term, o: Term): Triple = Triple(s, p, o)

  /** Datalog-style query written in native Scala. */
  def ask(vars: Term.Var*)(clauses: Triple*)(using Graph, NeuralConfig): QueryResult =
    Query.runQuery(clauses.toList, vars.toList)

  /** Run a parsed query form. */
  extension (f: Form) def run(using Graph, NeuralConfig): QueryResult = Query.run(f)
