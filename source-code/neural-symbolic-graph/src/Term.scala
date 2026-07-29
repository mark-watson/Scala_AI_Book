package nsk

/** A term in a triple or a query pattern.
  *
  * Keyword names are stored upper-case, the way the Common Lisp reader
  * upcases `:mark` to `:MARK`. The three `toString`/`display`/`jsonText`
  * views mirror Common Lisp's `~S`, `~A`, and the JSON rendering.
  */
enum Term:
  case Var(name: String)     // logic variable:   ?who
  case Kw(name: String)      // symbolic atom:     :mark   (name is UPPER-CASE)
  case Str(value: String)    // string literal:    "hello"
  case Num(value: Long)      // integer literal:   42
  case Neural(inner: Term)   // neural predicate:  ~:codes-in

  /** Readable form (Common Lisp `~S`). Round-trips back through the reader. */
  override def toString: String = this match
    case Var(n)    => s"?$n"
    case Kw(n)     => s":$n"
    case Str(s)    => "\"" + s + "\""
    case Num(n)    => n.toString
    case Neural(t) => s"~$t"

  /** Plain form (Common Lisp `~A`): a keyword without its colon. */
  def display: String = this match
    case Kw(n)     => n
    case Str(s)    => s
    case Num(n)    => n.toString
    case Var(n)    => s"?$n"
    case Neural(t) => s"~${t.display}"

  /** Lower-case text used for JSON output. */
  def jsonText: String = this match
    case Kw(n)     => n.toLowerCase
    case Str(s)    => s
    case Num(n)    => n.toString
    case Var(n)    => s"?$n"
    case Neural(t) => t.jsonText

object Term:
  /** True when TERM is a concrete value usable as an index key. */
  def concrete(t: Term): Boolean = t match
    case _: Var | _: Neural => false
    case _                  => true

/** A subject-predicate-object triple. */
case class Triple(s: Term, p: Term, o: Term):
  def toList: List[Term]      = List(s, p, o)
  override def toString: String = s"[$s $p $o]"
