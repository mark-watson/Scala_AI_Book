# Building a Neural-Symbolic Knowledge Graph Engine in Scala

Artificial intelligence has two main traditions. The symbolic tradition
stores knowledge as explicit facts and rules, then reasons over them with logic.
It is precise, and it can show every step of an inference. Its weakness is
coverage: it knows only what someone has told it. The neural tradition learns
patterns from data. A large language model can answer a question it has never
seen, yet it cannot prove its answer, and it states a wrong fact with the same
confidence as a right one.

A neural-symbolic system joins the two. It keeps a symbolic store for the facts
it holds for certain, and it calls a neural model for the gaps. This chapter
builds one such system, a knowledge graph engine named NSK. NSK stores facts as
triples, answers queries by logical unification, and falls back to a local
language model when the symbolic store has no answer.

Dear reader, this engine is a port of a Common Lisp program in my Common Lisp book. The Lisp version leans on reader
macros for a compact query syntax. We recreate that syntax in Scala 3, and along
the way we meet a set of Scala features worth knowing: enums for the
data model, `Option` for search that can fail, context parameters for ambient
configuration, and a string interpolator that stands in for a reader macro. The
whole core uses the standard library and no outside dependency.

## What we are going to build

By the end of the chapter you will have a program that:

- stores subject-predicate-object triples in memory, with two hash indices for
  fast lookup;
- saves every change to an append-only log and rebuilds its state on restart;
- answers Datalog-style queries by unification, joining several patterns into
  one result;
- asks a local Ollama model to fill a missing fact when a query marks a
  predicate as neural;
- runs as an interactive REPL and as a small REST server.

We start with the ideas, then read the code file by file.

## Background: three ideas you need first

### Knowledge as triples

The oldest and most durable way to store facts is the triple: a subject, a
predicate, and an object. The triple `(:mark :wrote :nsk)` reads as the sentence
"Mark wrote NSK". This is the
model behind RDF and behind most knowledge graphs in production today. Connect
enough triples and you have a graph, where the subjects and objects are nodes and
the predicates are labeled edges.

A triple is easy to store, easy to index, and easy to match. NSK writes a
keyword atom with a leading colon, so `:mark` is the atom named `mark`. Names
are held in upper case, the way a Common Lisp reader upcases a symbol, so the
atom prints as `:MARK`.

### Querying by unification

A query is a triple with holes in it. We write a hole as a logic variable, a
name with a leading question mark. The pattern `[?who :wrote :nsk]` asks "who
wrote NSK?". The pattern is not a fact; it is a request to find every fact that
fits its shape.

To match a pattern against a fact is to find a substitution, a map
`\theta`$ from variables to terms, that makes the two triples equal:

```$
\mathrm{pat}\,\theta = \mathrm{fact}
```

For `[?who :wrote :nsk]` against the fact `[:mark :wrote :nsk]`, the substitution
is `\theta = \{\,\text{?who} \mapsto \text{:mark}\,\}`$. The algorithm that
finds the most general such `\theta`$, or reports that none exists, is
*unification*. It is the engine inside Prolog, inside type inference, and inside
NSK.

A real query has more than one pattern. The query "who wrote NSK and also codes
in Lisp?" is a conjunction of two patterns:

```
(ask (?a) [?a :wrote :nsk] [?a :codes-in :lisp])
```

We solve a conjunction by threading substitutions forward: solve the first
pattern, and for each substitution it yields, carry those bindings into the
second pattern. Written as a recurrence over a list of clauses, with `\theta`$
the substitution so far:

```$
\mathrm{prove}(\langle\,\rangle, \theta) = \{\theta\}, \qquad
\mathrm{prove}(c :: \mathit{rest}, \theta) =
  \bigcup_{\theta' \in \mathrm{match}(c,\, \theta)} \mathrm{prove}(\mathit{rest}, \theta')
```

The base case, an empty clause list, returns the current substitution as the one
solution. This recurrence maps line for line onto the Scala we will read in the
query engine.

### The neural fallback

Symbolic matching answers only from stored facts. NSK adds one move. A query may
mark a predicate as *neural* with a leading tilde, as in `[:mark ~:codes-in
?lang]`. NSK first tries a plain symbolic match on the bare predicate. If the
store has no answer, and only then, it asks a language model to infer the
missing object. The model returns text like "Common Lisp"; NSK turns that text
into the keyword `:COMMON-LISP` and binds it to the variable.

This is the neural-symbolic join in one sentence: symbolic first for precision,
neural second for coverage.

## How the pieces connect

The program is small. Each file holds one idea.

```
project.scala      scala-cli build config (Scala version, main class)
src/Term.scala     terms and triples, the data model
src/Unify.scala    Norvig-style unification over Option
src/Graph.scala    the triplestore: indices and the append-only log
src/Reader.scala   the NSK reader: text to an abstract syntax tree
src/Query.scala    pattern matching, conjunction, and the neural fallback
src/Neural.scala   the Ollama client and text-to-triples
src/Json.scala     a self-contained JSON model, parser, and writer
src/Dsl.scala      the nsk"..." interpolator and the native Scala DSL
src/Repl.scala     the interactive read-eval-print loop
src/Server.scala   the REST server on the JDK HTTP server
src/Main.scala     the command-line entry point
Tests.scala        the test suite
```

Data flows one way for a query. Text enters through the REPL, the interpolator,
or an HTTP request. The reader turns text into a `Form`. The query engine turns
a `Form` into a search over the graph. Unification decides each match. If a
neural predicate finds nothing, the neural layer calls the model. The result
prints as a small table or as JSON.

We read the files in that order, so each one builds on the last.

## The build file

The project uses `scala-cli`, which needs no separate build tool. A single
directive file names the Scala version and the entry point.

```scala
//> using scala 3.8.4
//> using mainClass nsk.main

// NSK: Neural-Symbolic Knowledge Graph engine, ported from Common Lisp.
//
// The core has no external dependencies. JSON is hand-rolled, the Ollama
// client uses the JDK HttpClient, and the REST server uses the JDK's
// built-in com.sun.net.httpserver. Build and run with scala-cli:
//
//   scala-cli run .                         start the REPL
//   scala-cli run . -- --serve --port 8800  start the REST server
//   scala-cli run . --main-class nsk.test   run the test suite
```

The `//> using` lines are scala-cli directives. They pin the compiler to Scala
3.8.4 and set the default main class. Nothing else is needed to compile or run.

## The data model: terms and triples

Everything in NSK is a `Term` or a `Triple`. A term is one of five things: a
logic variable, a keyword atom, a string, an integer, or a neural wrapper around
another term. The neural wrapper is how a tilde predicate carries its "ask the
model" flag through the engine.

Before the code, here is what each term looks like on the page:

```
?who          a logic variable
:mark         a keyword atom (stored upper case as MARK)
"hello"       a string value
42            an integer
~:codes-in    a neural predicate wrapping the atom :codes-in
```

A `Term` is a closed set of five cases, so it is a natural Scala 3 `enum`. The
compiler then checks that every `match` covers all five, a guarantee the Lisp
`ecase` gives only at run time. The type carries three views of itself, one per
output format the engine needs: `toString` for the readable form that the reader
can parse back, `display` for a plain form without the colon, and `jsonText`
for lower-case JSON output. These three mirror the Common Lisp `~S`, `~A`, and
JSON renderings.

```scala
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
```

Two details pay off later. First, `Term.concrete` returns true only for terms
that can serve as an index key, so a variable and a neural wrapper are not
concrete. The graph uses this to decide which index to consult. Second, the
`toString` of a term prints the exact surface syntax the reader reads back. That
round-trip is what lets the log file and the string interpolator reuse one
parser.

## Unification

Unification is the central operation of the query engine. It takes two terms and an
environment, and it returns either a larger environment that makes the terms
equal or a signal that no such environment exists. The environment is a map from
variable to value, `\theta : \mathrm{Var} \to \mathrm{Term}`$.

Common Lisp represents failure with a special `+FAIL+` sentinel, because in Lisp
`NIL` already means the empty successful environment and cannot double as
failure. Scala has a cleaner tool for "a value or nothing": `Option`. Here
`Some(env)` is success and `None` is failure. The whole engine then chains with
`flatMap` and reads as a short pipeline.

```scala
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
```

Read `unify` case by case. If the two terms are already equal, the environment
is unchanged. If either term is a variable, `unifyVar` handles it: a variable
already bound must unify with its stored value, and an unbound variable takes the
other term as its new value. Anything else fails. `unifyTriple` runs `unify`
three times, once per position, and threads the growing environment through with
`flatMap`, so a failure at any position stops the chain. `resolve` walks a chain
of bindings to the final value, which is how we read an answer out of an
environment.

The whole file is under forty lines. That economy comes straight from using
`Option` instead of a sentinel: the failure case never needs a name, because
`None` and `flatMap` carry it.

## The triplestore

The graph holds triples and answers "which triples could match this pattern?".
It keeps two hash indices, one keyed by subject and one keyed by object, so a
query with a known subject never scans the whole store. A concrete subject makes
subject lookup average `O(1)`$, and we then unify against only the triples that
share that subject rather than all `n`$ of them.

The graph is also durable. Every add and every delete appends one line to a log
file. On startup NSK replays the log to rebuild state. The log uses the NSK
surface syntax, so the same reader that serves the REPL replays it. There is no
`eval`, so a log file cannot run code, which makes replay safe by construction.

Here is a sample log, the input the replay code consumes:

```
(:add :mark :wrote :nsk)
(:add :jane :wrote :book)
(:add :mark :codes-in :lisp)
(:del :jane :wrote :book)
```

Replaying those four lines yields a graph with two live triples: Jane's line was
added and then deleted. The code below keeps the indices and the log in step and
provides the replay.

```scala
package nsk

import java.io.{BufferedReader, File, FileReader, FileWriter, PrintWriter}
import scala.collection.mutable

/** The triplestore: two hash indices plus an append-only log.
  *
  * Triples index by subject and by object for fast pattern narrowing. State
  * survives restarts through a log of forms that is replayed on startup. The
  * log uses the NSK surface syntax, so [[Reader]] replays it and no `eval` is
  * involved, which makes replay safe by construction.
  */
final class Graph private (val logPath: Option[String]):
  private val spo  = mutable.Map.empty[Term, List[Triple]] // subject -> triples
  private val osp  = mutable.Map.empty[Term, List[Triple]] // object  -> triples
  private var live = List.empty[Triple]                    // every live triple, newest first
  private var n    = 0
  private var log  = Option.empty[PrintWriter]

  def count: Int                    = n
  def all: List[Triple]             = live.reverse         // insertion order
  def spoIndex(k: Term): List[Triple] = spo.getOrElse(k, Nil)
  def ospIndex(k: Term): List[Triple] = osp.getOrElse(k, Nil)

  private def index(tr: Triple): Unit =
    spo(tr.s) = tr :: spoIndex(tr.s)
    osp(tr.o) = tr :: ospIndex(tr.o)
    live = tr :: live
    n += 1

  private def unindex(tr: Triple): Unit =
    spo(tr.s) = spoIndex(tr.s).filterNot(_ == tr)
    osp(tr.o) = ospIndex(tr.o).filterNot(_ == tr)
    live = live.filterNot(_ == tr)
    n -= 1

  def present(tr: Triple): Boolean = spoIndex(tr.s).contains(tr)

  private def writeLog(op: String, tr: Triple): Unit =
    log.foreach { w => w.println(Graph.entry(op, tr)); w.flush() }

  /** Add a triple and append it to the log. Duplicates are ignored. */
  def add(s: Term, p: Term, o: Term): Triple =
    val tr = Triple(s, p, o)
    if !present(tr) then { index(tr); writeLog("add", tr) }
    tr

  /** Remove a triple and record the deletion in the log. */
  def remove(s: Term, p: Term, o: Term): Triple =
    val tr = Triple(s, p, o)
    if present(tr) then { unindex(tr); writeLog("del", tr) }
    tr

  /** Triples that could match PATTERN, narrowed by the indices. */
  def candidates(pat: Triple): List[Triple] =
    if Term.concrete(pat.s) then spoIndex(pat.s)
    else if Term.concrete(pat.o) then ospIndex(pat.o)
    else live

  private def applyEntry(form: Form): Unit = form match
    case Form.Cmd("add", List(s, p, o)) =>
      val tr = Triple(s, p, o); if !present(tr) then index(tr)
    case Form.Cmd("del", List(s, p, o)) =>
      val tr = Triple(s, p, o); if present(tr) then unindex(tr)
    case _ => ()

  private def replay(path: String): Unit =
    val f = File(path)
    if f.exists then
      val in = BufferedReader(FileReader(f))
      try
        var line = in.readLine()
        while line != null do
          val t = line.trim
          if t.nonEmpty then applyEntry(Reader.readForm(t))
          line = in.readLine()
      finally in.close()

  private def openLog(path: String): Unit =
    log = Some(PrintWriter(FileWriter(path, /* append = */ true)))

  def flush(): Unit = log.foreach(_.flush())

  def close(): Unit =
    log.foreach { w => w.flush(); w.close() }
    log = None

object Graph:
  /** A fresh in-memory graph with no backing log. */
  def inMemory: Graph = new Graph(None)

  /** Open (or create) the store at PATH, replay its log, keep it open. */
  def open(path: String): Graph =
    val g = new Graph(Some(path))
    g.replay(path)
    g.openLog(path)
    g

  private def entry(op: String, tr: Triple): String =
    s"(:$op ${source(tr.s)} ${source(tr.p)} ${source(tr.o)})"

  private def source(t: Term): String = t match
    case Term.Kw(name) => ":" + name.toLowerCase
    case Term.Str(s)   => "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"") + "\""
    case Term.Num(x)   => x.toString
    case other         => other.toString
```

The key method for queries is `candidates`. Given a pattern, it picks the
narrowest index it can: the subject index if the subject is concrete, else the
object index if the object is concrete, else the full live list. The query
engine then unifies the pattern against only those candidates. Note the two
constructors on the companion: `inMemory` makes a graph with no log, which the
tests use, and `open` replays a log then keeps it open for appends.

The `source` helper serializes a term for the log. It writes keywords in lower
case with a colon and escapes quotes in strings, which produces the exact syntax
the reader reads back on replay.

## The reader

The reader turns text into an abstract syntax tree. In Common Lisp this job
belongs to reader macros, small functions the reader runs when it meets a
trigger character. Scala has no reader macros, so we write a small recursive
descent parser and expose it later through a string interpolator, which is the
closest Scala analog.

The reader accepts three kinds of top-level form. Here is a sample of each, the
input the parser consumes:

```
(:add :mark :wrote :nsk)                            a command
[?who :wrote :nsk]                                  a single pattern
(ask (?a) [?a :wrote :nsk] [?a :codes-in :lisp])    a conjunctive query
```

Inside a form, the reader recognizes the term prefixes from the data model:
`?name` for a variable, `:name` for a keyword, `~term` for a neural wrapper, a
double quote for a string, and a digit or minus sign for a number. A `Form` is
another closed set, so it too is an `enum`.

The parser raises `IncompleteInput` when the text ends in the middle of a form.
The REPL treats that as "read another line", the way a Lisp listener waits for a
closing paren. A separate `readForms` reads as many complete forms as it can and
reports how far it got, which lets the REPL buffer a partial form across lines.

```scala
package nsk

import scala.collection.mutable.ListBuffer

/** A parsed top-level form. */
enum Form:
  case Cmd(name: String, args: List[Term])          // (:add s p o) or a bare :count
  case Pat(triple: Triple)                          // [s p o]
  case Ask(vars: List[Term.Var], clauses: List[Triple]) // (ask (?a) [..] [..])

/** Thrown when the input runs out in the middle of a form. The REPL treats
  * this as "read another line", the way a Lisp reader waits for more input. */
final class IncompleteInput extends RuntimeException("incomplete input")

/** Thrown on malformed input. */
final class ReadError(msg: String) extends RuntimeException(msg)

/** The NSK reader.
  *
  * This is the Scala answer to the Common Lisp reader macros. It turns the
  * same surface syntax into an abstract syntax tree:
  *
  *   ?name    -> Term.Var(name)
  *   :name    -> Term.Kw(NAME)          (upper-cased, as in Common Lisp)
  *   ~term    -> Term.Neural(term)
  *   [s p o]  -> Form.Pat(Triple(...))
  *   (:op ..) -> Form.Cmd(op, args)
  *   (ask ..) -> Form.Ask(vars, clauses)
  */
object Reader:
  /** Read exactly one form from S (trailing text is ignored). */
  def readForm(s: String): Form =
    val c = Cursor(s)
    c.form()

  /** Read as many complete forms as possible; return them and the index of
    * the first unconsumed character (an incomplete trailing form is left). */
  def readForms(s: String): (List[Form], Int) =
    val c    = Cursor(s)
    val out  = ListBuffer.empty[Form]
    var more = true
    while more do
      c.skipWs()
      if c.atEnd then more = false
      else
        val mark = c.pos
        try out += c.form()
        catch
          case _: IncompleteInput =>
            c.pos = mark
            more = false
    (out.toList, c.pos)

  private final class Cursor(s: String):
    var pos = 0

    def atEnd: Boolean = pos >= s.length
    def peek: Char     = s(pos)

    def skipWs(): Unit = while pos < s.length && s(pos).isWhitespace do pos += 1

    private def expect(ch: Char): Unit =
      skipWs()
      if atEnd then throw IncompleteInput()
      if s(pos) == ch then pos += 1 else throw ReadError(s"expected '$ch' at $pos")

    private def tokenChars(): String =
      val start = pos
      while pos < s.length && !s(pos).isWhitespace && !"()[]".contains(s(pos)) do pos += 1
      if pos == start then throw ReadError(s"empty token at $pos")
      s.substring(start, pos)

    /** A single term. */
    def term(): Term =
      skipWs()
      if atEnd then throw IncompleteInput()
      peek match
        case '?'                        => pos += 1; Term.Var(tokenChars())
        case ':'                        => pos += 1; Term.Kw(tokenChars().toUpperCase)
        case '~'                        => pos += 1; Term.Neural(term())
        case '"'                        => Term.Str(stringLiteral())
        case c if c == '-' || c.isDigit => number()
        case c                          => throw ReadError(s"unexpected term char '$c' at $pos")

    private def number(): Term =
      val start = pos
      if peek == '-' then pos += 1
      while pos < s.length && s(pos).isDigit do pos += 1
      Term.Num(s.substring(start, pos).toLong)

    private def stringLiteral(): String =
      expect('"')
      val sb   = StringBuilder()
      var done = false
      while !done do
        if atEnd then throw IncompleteInput()
        val ch = s(pos); pos += 1
        ch match
          case '"'  => done = true
          case '\\' =>
            if atEnd then throw IncompleteInput()
            val e = s(pos); pos += 1
            sb.append(e match
              case 'n'   => '\n'
              case 't'   => '\t'
              case 'r'   => '\r'
              case other => other
            )
          case c => sb.append(c)
      sb.toString

    /** A [s p o] triple pattern. */
    def triple(): Triple =
      expect('[')
      val terms = termsUntil(']')
      if terms.length != 3 then
        throw ReadError(s"triple pattern needs exactly three terms, got ${terms.length}")
      Triple(terms(0), terms(1), terms(2))

    private def termsUntil(close: Char): List[Term] =
      val out  = ListBuffer.empty[Term]
      var done = false
      while !done do
        skipWs()
        if atEnd then throw IncompleteInput()
        if peek == close then { pos += 1; done = true }
        else out += term()
      out.toList

    private def clausesUntil(close: Char): List[Triple] =
      val out  = ListBuffer.empty[Triple]
      var done = false
      while !done do
        skipWs()
        if atEnd then throw IncompleteInput()
        peek match
          case c if c == close => pos += 1; done = true
          case '['             => out += triple()
          case _               => throw ReadError(s"ask clause is not a [triple] at $pos")
      out.toList

    private def bareWord(): String =
      skipWs()
      val start = pos
      while pos < s.length && !s(pos).isWhitespace && !"()[]".contains(s(pos)) do pos += 1
      s.substring(start, pos)

    /** A top-level form. */
    def form(): Form =
      skipWs()
      if atEnd then throw IncompleteInput()
      peek match
        case '(' => listForm()
        case '[' => Form.Pat(triple())
        case ':' => pos += 1; Form.Cmd(tokenChars().toLowerCase, Nil)
        case c   => throw ReadError(s"unexpected char '$c' at $pos")

    private def listForm(): Form =
      expect('(')
      skipWs()
      if atEnd then throw IncompleteInput()
      if peek == ':' then
        pos += 1
        val name = tokenChars().toLowerCase
        Form.Cmd(name, termsUntil(')'))
      else
        bareWord() match
          case "ask" =>
            skipWs(); expect('(')
            val vars    = termsUntil(')').collect { case v: Term.Var => v }
            val clauses = clausesUntil(')')
            Form.Ask(vars, clauses)
          case head => throw ReadError(s"unknown list head '$head'")
```

The `term` method is the whole surface syntax in one `match`. It reads the first
character and dispatches: a question mark starts a variable, a colon starts a
keyword upcased to match the Lisp reader, a tilde recursively reads the wrapped
term, a quote reads a string, and a digit or minus reads a number. The
`termsUntil` and `clausesUntil` helpers read a run of items up to a closing
bracket. `form` handles the three top-level shapes: a parenthesized list, a
bracketed pattern, and a bare keyword command such as `:count`.

The one subtle point is `readForms`. It reads forms until it hits end of input
or an incomplete form. On `IncompleteInput` it rewinds the cursor to the start
of the partial form and stops, returning the complete forms and the cursor
position. The REPL keeps the leftover text and prepends the next line to it.

## The query engine

Now the pieces combine. The query engine takes a `Form`, runs the pattern search
over the graph, and applies the neural fallback. This is a direct transcription
of the `\mathrm{prove}`$ recurrence from the background section.

The active graph and the neural configuration arrive as `using` parameters. This
is the Scala stand-in for the Common Lisp special variables `*graph*` and
`*ollama-url*`. In Lisp a caller rebinds a special with `let`; in Scala a caller
supplies a local `given`, and the compiler threads it through every call that
declares a matching `using` clause. The benefit over a global variable is static
safety: a function that needs a graph will not compile without one in scope.

```scala
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
```

Follow one query through `matchPattern`. It first checks whether the predicate
is neural and, if so, strips the tilde to get a plain symbolic pattern `spat`. It
grounds that pattern by substituting any already-bound variables, so the graph
can use its indices. It asks the graph for candidates and unifies the pattern
against each. If the symbolic search finds nothing and the predicate was neural,
only then does it call `neuralMatch`.

`neuralMatch` fires only in the useful shape: a concrete subject and a variable
object, "given this subject and predicate, what is the object?". It calls the
model through `Neural.queryFallback`, turns the answer into a keyword, and unifies
that keyword with the target variable.

`prove` is the recurrence made literal. An empty pattern list yields one
solution, the current environment. Otherwise it matches the first pattern and,
for each resulting environment, recurses on the rest with `flatMap`. `runQuery`
runs `prove` from the empty environment, then reads each requested variable out
of each solution with `resolve`, and drops duplicates. The `QueryResult` printer
renders one row per solution, so `[?who :wrote :nsk]` with two authors prints two
lines.

## The neural layer

The neural layer is the only part that reaches outside the process. It speaks to
a local Ollama daemon over HTTP through the JDK's own `HttpClient`, so the core
still carries no external dependency.

Two prompts drive it. One asks the model to infer a single object for a subject
and predicate. The other asks the model to extract triples from free text. Both
demand a JSON reply, which keeps parsing simple and predictable.

Here is the request NSK sends to infer an object, the data the model receives:

```json
{
  "model": "qwen3.5:4b",
  "system": "You are a graph database inference node. Given a Subject and a Predicate, infer the single most likely Object. Reply ONLY as JSON: {\"result\": \"value\"}.",
  "prompt": "Subject: mark. Predicate: favorite-language. What is the Object?",
  "format": "json",
  "stream": false
}
```

Ollama wraps the model's answer in an envelope. The field we want is `response`,
which itself holds a JSON string:

```json
{
  "model": "qwen3.5:4b",
  "created_at": "2026-07-28T12:00:00Z",
  "response": "{\"result\": \"Common Lisp\"}",
  "done": true
}
```

So NSK parses twice: once to pull out `response`, and once to parse that inner
string and read `result`. The value "Common Lisp" then becomes the keyword
`:COMMON-LISP`. The code below does exactly that, and guards every network call,
so a missing daemon degrades to "no answer" rather than a crash.

```scala
package nsk

import java.net.URI
import java.net.http.{HttpClient, HttpRequest, HttpResponse}
import java.time.Duration

/** Configuration for the neural layer.
  *
  * Passed as a `given`, this replaces the Common Lisp specials `*ollama-url*`,
  * `*ollama-model*`, and `*ollama-timeout*`. A test rebinds it with a local
  * `given`, the way Lisp code rebinds a special with `let`.
  */
case class NeuralConfig(
    url: String = "http://localhost:11434",
    model: String = "qwen3.5:4b",
    timeoutSeconds: Int = 60
)

object NeuralConfig:
  given default: NeuralConfig = NeuralConfig()

/** The neural integration layer (a local Ollama daemon).
  *
  * When a symbolic query fails on a `~` predicate, NSK asks the model to infer
  * the missing object. The same layer turns free text into triples. HTTP goes
  * through the JDK's own `HttpClient`, so the core keeps no dependency on an
  * external HTTP library.
  */
object Neural:
  private val systemInference =
    "You are a graph database inference node. Given a Subject and a Predicate, " +
      "infer the single most likely Object. Reply ONLY as JSON: {\"result\": \"value\"}."

  private val systemExtraction =
    "You extract knowledge-graph triples from text. Reply ONLY as JSON of the form " +
      "{\"triples\": [{\"subject\": \"..\", \"predicate\": \"..\", \"object\": \"..\"}]}. " +
      "Use short lower-case tokens."

  /** Readable label for a subject or predicate term. */
  def termLabel(t: Term): String = t match
    case Term.Neural(inner) => termLabel(inner)
    case Term.Kw(name)      => name.toLowerCase
    case Term.Str(s)        => s
    case other              => other.display

  /** Convert an LLM string such as "Common Lisp" into the keyword `:COMMON-LISP`. */
  def sanitizeToKeyword(s: String): Term.Kw =
    val trimSet = " \t\n\r.,".toSet
    val trimmed = s.dropWhile(trimSet).reverse.dropWhile(trimSet).reverse
    Term.Kw(trimmed.toUpperCase.replace(' ', '-'))

  private def post(url: String, body: String, cfg: NeuralConfig): String =
    val client = HttpClient
      .newBuilder()
      .connectTimeout(Duration.ofSeconds(cfg.timeoutSeconds))
      .build()
    val req = HttpRequest
      .newBuilder(URI.create(url))
      .timeout(Duration.ofSeconds(cfg.timeoutSeconds))
      .header("Content-Type", "application/json")
      .POST(HttpRequest.BodyPublishers.ofString(body))
      .build()
    client.send(req, HttpResponse.BodyHandlers.ofString()).body()

  /** Send a /api/generate request and return the model's raw response string. */
  private def generate(prompt: String, system: String)(using cfg: NeuralConfig): String =
    val payload = Json
      .obj(
        "model"  -> Json.Str(cfg.model),
        "system" -> Json.Str(system),
        "prompt" -> Json.Str(prompt),
        "format" -> Json.Str("json"),
        "stream" -> Json.Bool(false)
      )
      .render
    val raw = post(s"${cfg.url}/api/generate", payload, cfg)
    Json.parse(raw)("response").flatMap(_.str).getOrElse("")

  /** Ask the model to infer the object for (SUBJECT PREDICATE). Returns the
    * answer string, or None when the daemon is unreachable or gives nothing. */
  def queryFallback(subject: Term, predicate: Term)(using NeuralConfig): Option[String] =
    try
      val prompt =
        s"Subject: ${termLabel(subject)}. Predicate: ${termLabel(predicate)}. What is the Object?"
      val response = generate(prompt, systemInference)
      val inner    = try Json.parse(response) catch case _: Throwable => Json.Null
      inner("result").flatMap(_.str).filter(_.nonEmpty).orElse(Some(response).filter(_.nonEmpty))
    catch
      case e: Throwable =>
        Console.err.println(s"; neural fallback unavailable: ${e.getMessage}")
        None

  /** Use the model to parse TEXT into a list of keyword triples. */
  def textToTriples(text: String)(using NeuralConfig): List[Triple] =
    try
      val response = generate(text, systemExtraction)
      val rows = Json.parse(response)("triples").collect { case Json.Arr(items) => items }
        .getOrElse(Nil)
      rows.flatMap { row =>
        for
          s <- row("subject").flatMap(_.str)
          p <- row("predicate").flatMap(_.str)
          o <- row("object").flatMap(_.str)
        yield Triple(sanitizeToKeyword(s), sanitizeToKeyword(p), sanitizeToKeyword(o))
      }
    catch
      case e: Throwable =>
        Console.err.println(s"; extraction unavailable: ${e.getMessage}")
        Nil

  /** Extract triples from TEXT and add them to the graph. */
  def ingestText(text: String)(using g: Graph, cfg: NeuralConfig): List[Triple] =
    val triples = textToTriples(text)
    triples.foreach(tr => g.add(tr.s, tr.p, tr.o))
    triples
```

`sanitizeToKeyword` is the small conversion between the neural and symbolic
sides. It trims
whitespace and trailing punctuation, upcases, and turns spaces into hyphens, so
the model's "Common Lisp" becomes the atom `:COMMON-LISP` that the symbolic side
can store and match. `queryFallback` wraps the whole call in a `try`, so when the
daemon is down it prints one diagnostic to standard error and returns `None`. A
neural query then simply yields no solution, which is the behavior we want in a
system that must keep running without a model.

`ingestText` closes the loop the other way: it asks the model to read free text
into triples and adds each one to the graph. Text goes in, structured facts come
out, and from then on the symbolic engine can query them.

## Supporting infrastructure

Four files carry the plumbing: JSON, the REPL, the REST server, and the
command-line entry point. They hold no new AI ideas, but the program does not run
without them, so we read each in turn.

### JSON

NSK keeps its own JSON so the core needs no library, the same choice the Lisp
version makes. JSON is a closed set of six shapes, so once again an `enum` fits.
The parser is a small recursive descent reader; the writer walks the tree and
escapes strings.

```scala
package nsk

/** A small, self-contained JSON model, parser, and writer.
  *
  * NSK keeps its own JSON code so the core has no external dependency, the
  * same choice the Common Lisp version makes. The Lisp writer takes a tagged
  * list and the reader returns alists; here JSON is a proper algebraic data
  * type, which is the idiomatic Scala model.
  */
enum Json:
  case Str(value: String)
  case Num(value: Double)
  case Bool(value: Boolean)
  case Null
  case Arr(items: List[Json])
  case Obj(fields: List[(String, Json)])

  def render: String =
    val sb = StringBuilder()
    Json.write(this, sb)
    sb.toString

  /** Look up KEY in an object, the analog of Common Lisp `json-get`. */
  def apply(key: String): Option[Json] = this match
    case Obj(fs) => fs.collectFirst { case (k, v) if k == key => v }
    case _       => None

  def str: Option[String]    = this match { case Str(s) => Some(s); case _ => None }
  def double: Option[Double] = this match { case Num(n) => Some(n); case _ => None }

object Json:
  def obj(fields: (String, Json)*): Json = Obj(fields.toList)
  def arr(items: Json*): Json            = Arr(items.toList)

  final class JsonError(msg: String) extends RuntimeException(msg)

  def parse(input: String): Json =
    val p = Parser(input)
    val v = p.value()
    p.skipWs()
    v

  // ----- writer -----------------------------------------------------------

  private def write(v: Json, sb: StringBuilder): Unit = v match
    case Str(s)  => writeString(s, sb)
    case Num(n)  => if n == n.toLong.toDouble then sb.append(n.toLong) else sb.append(n)
    case Bool(b) => sb.append(if b then "true" else "false")
    case Null    => sb.append("null")
    case Arr(items) =>
      sb.append('[')
      items.zipWithIndex.foreach { (it, i) => if i > 0 then sb.append(','); write(it, sb) }
      sb.append(']')
    case Obj(fields) =>
      sb.append('{')
      fields.zipWithIndex.foreach { case ((k, vv), i) =>
        if i > 0 then sb.append(',')
        writeString(k, sb); sb.append(':'); write(vv, sb)
      }
      sb.append('}')

  private def writeString(s: String, sb: StringBuilder): Unit =
    sb.append('"')
    s.foreach {
      case '"'          => sb.append("\\\"")
      case '\\'         => sb.append("\\\\")
      case '\n'         => sb.append("\\n")
      case '\r'         => sb.append("\\r")
      case '\t'         => sb.append("\\t")
      case '\b'         => sb.append("\\b")
      case '\f'         => sb.append("\\f")
      case c if c < ' ' => sb.append("\\u%04x".format(c.toInt))
      case c            => sb.append(c)
    }
    sb.append('"')

  // ----- reader -----------------------------------------------------------

  private final class Parser(s: String):
    private var pos = 0

    private def fail(msg: String): Nothing = throw JsonError(s"$msg at $pos")
    private def atEnd: Boolean             = pos >= s.length
    private def peek: Char                 = s(pos)
    private def next(): Char               = { val c = s(pos); pos += 1; c }

    def skipWs(): Unit =
      while pos < s.length && " \t\n\r".contains(s(pos)) do pos += 1

    def value(): Json =
      skipWs()
      if atEnd then fail("unexpected end of input")
      peek match
        case '{'                       => obj()
        case '['                       => arr()
        case '"'                       => Str(string())
        case 't'                       => literal("true", Bool(true))
        case 'f'                       => literal("false", Bool(false))
        case 'n'                       => literal("null", Null)
        case c if c == '-' || c.isDigit => number()
        case c                         => fail(s"unexpected char $c")

    private def literal(text: String, v: Json): Json =
      text.foreach(e => if atEnd || next() != e then fail(s"bad literal, expected $text"))
      v

    private def obj(): Json =
      next() // {
      skipWs()
      if !atEnd && peek == '}' then { next(); return Obj(Nil) }
      val fields = List.newBuilder[(String, Json)]
      var more   = true
      while more do
        skipWs()
        val key = string()
        skipWs()
        if atEnd || next() != ':' then fail("expected : after object key")
        fields += (key -> value())
        skipWs()
        next() match
          case ',' => ()
          case '}' => more = false
          case _   => fail("expected , or } in object")
      Obj(fields.result())

    private def arr(): Json =
      next() // [
      skipWs()
      if !atEnd && peek == ']' then { next(); return Arr(Nil) }
      val items = List.newBuilder[Json]
      var more  = true
      while more do
        items += value()
        skipWs()
        next() match
          case ',' => ()
          case ']' => more = false
          case _   => fail("expected , or ] in array")
      Arr(items.result())

    private def string(): String =
      if atEnd || next() != '"' then fail("expected a string")
      val sb   = StringBuilder()
      var done = false
      while !done do
        if atEnd then fail("unterminated string")
        next() match
          case '"'  => done = true
          case '\\' =>
            next() match
              case '"'  => sb.append('"')
              case '\\' => sb.append('\\')
              case '/'  => sb.append('/')
              case 'b'  => sb.append('\b')
              case 'f'  => sb.append('\f')
              case 'n'  => sb.append('\n')
              case 'r'  => sb.append('\r')
              case 't'  => sb.append('\t')
              case 'u'  => sb.append(hex(4).toChar)
              case _    => fail("bad string escape")
          case c => sb.append(c)
      sb.toString

    private def hex(n: Int): Int =
      var v = 0
      for _ <- 0 until n do
        val d = Character.digit(next(), 16)
        if d < 0 then fail("bad \\u escape")
        v = v * 16 + d
      v

    private def number(): Json =
      val start = pos
      if peek == '-' then next()
      while !atEnd && (peek.isDigit || "+-.eE".contains(peek)) do next()
      Num(s.substring(start, pos).toDouble)
```

The `apply(key)` method gives the neat `json("response")` lookup the neural layer
relies on, and `str` and `double` pull a typed value out of a node. Because
`apply` returns an `Option`, a chain of lookups uses `flatMap` and never throws
on a missing field.

### The REPL

The read-eval-print loop reads a form and interprets it. Common Lisp reads a form
and then calls `eval`; Scala has no `eval`, so the loop dispatches on the `Form`
type instead. That is a little more code and a much smaller attack surface, since
the only things the loop can do are the ones we wrote.

```scala
package nsk

import java.io.{BufferedReader, ByteArrayOutputStream, PrintStream}
import java.nio.charset.StandardCharsets.UTF_8

/** The interactive NSK read-eval-print loop.
  *
  * Common Lisp reads a form and then `eval`s it. Scala has no `eval`, so the
  * loop instead reads a [[Form]] with [[Reader]] and interprets it directly.
  * That is a little more code but keeps the language surface small and safe.
  */
object Repl:
  val banner: String =
    "NSK: Neural-Symbolic Knowledge Graph Engine\nType :help for commands, :quit to exit."

  val help: String =
    """Commands:
      |  :help                 show this help
      |  :facts                list every triple
      |  :count                show the triple count
      |  :add s p o            add a triple, e.g. (:add :mark :wrote :nsk)
      |  :del s p o            remove a triple
      |  :ingest "text"        extract triples from text with the model
      |  :save                 flush the log to disk
      |  :quit                 leave the REPL
      |
      |Queries use the NSK syntax:
      |  [?who :wrote :nsk]                       one pattern
      |  (ask (?a) [?a :wrote :nsk] [?a :codes-in :lisp])
      |  [:mark ~:codes-in ?lang]                 ~ falls back to the model""".stripMargin

  /** Evaluate one form. Returns true to keep looping, false to quit. */
  def eval(form: Form, out: PrintStream)(using Graph, NeuralConfig): Boolean =
    form match
      case Form.Cmd(name, args) => command(name, args, out)
      case Form.Pat(_) | Form.Ask(_, _) =>
        out.println(Query.run(form)); true

  private def command(name: String, args: List[Term], out: PrintStream)(using
      g: Graph,
      cfg: NeuralConfig
  ): Boolean =
    name match
      case "quit" | "exit" => false
      case "help"          => out.println(help); true
      case "count"         => out.println(s"${g.count} triples"); true
      case "save"          => g.flush(); out.println("saved."); true
      case "facts" =>
        g.all.foreach(tr => out.println(s"  ${tr.s.display}  ${tr.p.display}  ${tr.o.display}"))
        out.println(s"(${g.count} triples)"); true
      case "add" =>
        args match
          case List(s, p, o) =>
            g.add(s, p, o); out.println(s"added ${s.display} ${p.display} ${o.display}")
          case _ => out.println("usage: (:add s p o)")
        true
      case "del" =>
        args match
          case List(s, p, o) =>
            g.remove(s, p, o); out.println(s"removed ${s.display} ${p.display} ${o.display}")
          case _ => out.println("usage: (:del s p o)")
        true
      case "ingest" =>
        args match
          case List(Term.Str(text)) =>
            val added = Neural.ingestText(text)
            out.println(s"ingested ${added.length} triple${if added.length == 1 then "" else "s"}")
          case _ => out.println("usage: (:ingest \"text\")")
        true
      case other => out.println(s"unknown command: $other"); true

  /** Process every complete form in INPUT, returning captured output. Used by
    * the tests and by anyone who wants to script the REPL. */
  def feed(input: String)(using Graph, NeuralConfig): String =
    val buf = ByteArrayOutputStream()
    val out = PrintStream(buf, true, UTF_8)
    val (forms, _) = Reader.readForms(input + "\n")
    forms.foreach(f => eval(f, out))
    buf.toString(UTF_8)

  /** The interactive loop, reading line by line and buffering partial forms. */
  def run(in: BufferedReader, out: PrintStream)(using Graph, NeuralConfig): Unit =
    out.println(banner)
    var buffer  = ""
    var running = true
    while running do
      out.print("nsk> "); out.flush()
      val line = in.readLine()
      if line == null then running = false
      else
        buffer += line + "\n"
        val (forms, consumed) = Reader.readForms(buffer)
        buffer = buffer.substring(consumed)
        val it = forms.iterator
        while running && it.hasNext do
          try if !eval(it.next(), out) then running = false
          catch case e: Throwable => out.println(s"; error: ${e.getMessage}")
    out.println("Bye.")
```

`feed` is worth noting: it runs a whole script and returns the captured output as
a string. The tests use it to drive the REPL without a terminal, and it is what
makes the loop testable. The interactive `run` buffers text across
lines, so a query split over several lines still parses once it is complete.

### The REST server

The `--serve` flag starts a small HTTP server on the JDK's built-in
`com.sun.net.httpserver`, so there is still no outside dependency. It exposes two
routes: `POST /query` and `GET /health`.

A query request is a JSON object with three optional fields. A field that is
`null` or missing means "any", so it becomes a variable. A field that starts with
a question mark is a named variable. Any other string is a keyword. Here is a
sample request and the reply it produces:

```json
{"subject": null, "predicate": "wrote", "object": "nsk"}
```

```json
{"count":1,"results":[{"subject":"mark"}]}
```

The server maps the JSON fields to terms, runs the same `Query.matchTriple` the
REPL uses, and renders the solutions back to JSON.

```scala
package nsk

import com.sun.net.httpserver.{HttpExchange, HttpServer}
import java.net.InetSocketAddress
import java.nio.charset.StandardCharsets.UTF_8

/** An optional REST server for the `--serve` flag.
  *
  * Common Lisp loads Hunchentoot on demand; here the server uses the JDK's own
  * `com.sun.net.httpserver`, so it needs no external dependency at all.
  */
object Server:
  @volatile private var server: Option[HttpServer] = None

  /** A request field maps to a graph term: null or missing means "any", a
    * value starting with `?` is a variable, any other string is a keyword. */
  def fieldToTerm(value: Option[Json], role: String): Term = value match
    case None | Some(Json.Null)                        => Term.Var(role)
    case Some(Json.Str(s)) if s.startsWith("?")        => Term.Var(s.drop(1))
    case Some(Json.Str(s))                             => Neural.sanitizeToKeyword(s)
    case Some(other)                                   => Term.Str(other.render)

  def termJson(t: Term): String = t match
    case Term.Kw(name)  => name.toLowerCase
    case Term.Str(s)    => s
    case Term.Var(n)    => s"?$n"
    case Term.Num(x)    => x.toString
    case Term.Neural(i) => termJson(i)

  def solutionToJson(solution: List[(String, Term)]): Json =
    Json.Obj(solution.map((k, v) => k -> Json.Str(termJson(v))))

  private def body(ex: HttpExchange): String =
    new String(ex.getRequestBody.readAllBytes(), UTF_8)

  private def respond(ex: HttpExchange, json: String): Unit =
    val bytes = json.getBytes(UTF_8)
    ex.getResponseHeaders.set("Content-Type", "application/json")
    ex.sendResponseHeaders(200, bytes.length.toLong)
    val os = ex.getResponseBody
    os.write(bytes)
    os.close()

  private def handleQuery(ex: HttpExchange)(using Graph, NeuralConfig): Unit =
    val req = try Json.parse(body(ex)) catch case _: Throwable => Json.Null
    val s   = fieldToTerm(req("subject"), "subject")
    val p   = fieldToTerm(req("predicate"), "predicate")
    val o   = fieldToTerm(req("object"), "object")
    val sols = Query.matchTriple(s, p, o).solutions
    respond(
      ex,
      Json
        .obj(
          "count"   -> Json.Num(sols.length),
          "results" -> Json.Arr(sols.map(solutionToJson))
        )
        .render
    )

  private def handleHealth(ex: HttpExchange)(using g: Graph, cfg: NeuralConfig): Unit =
    respond(
      ex,
      Json
        .obj(
          "status"  -> Json.Str("ok"),
          "triples" -> Json.Num(g.count),
          "model"   -> Json.Str(cfg.model)
        )
        .render
    )

  /** Start serving /query and /health on PORT. */
  def start(port: Int)(using Graph, NeuralConfig): HttpServer =
    stop()
    val s = HttpServer.create(InetSocketAddress(port), 0)
    s.createContext("/query", (ex: HttpExchange) => handleQuery(ex))
    s.createContext("/health", (ex: HttpExchange) => handleHealth(ex))
    s.setExecutor(null)
    s.start()
    server = Some(s)
    s

  /** Stop the running server, if any. */
  def stop(): Unit =
    server.foreach(_.stop(0))
    server = None
```

The `createContext` calls pass a lambda where an `HttpHandler` is expected. Scala
turns the lambda into the single-method interface, so the handler stays a one
liner. The rest of the file is straight translation between JSON and terms.

### The entry point

`Main.scala` parses flags, opens the store, and then either serves or drops into
the REPL. It sets up the two `given` values, the graph and the neural config,
that every query function needs.

```scala
package nsk

import java.io.{BufferedReader, InputStreamReader}

/** Command-line entry point. Parse flags, then serve or drop into the REPL.
  * The object is named `App` (not `Main`) so its class file does not collide
  * with the `main.class` from `@main` on a case-insensitive filesystem.
  */
@main def main(args: String*): Unit = App.run(args.toArray)

object App:
  val version = "NSK 1.0.0"

  def run(args: Array[String]): Unit =
    if args.contains("--help") then
      println(usage)
    else
      val dbPath = flagValue(args, "--db").getOrElse("nsk-graph.log")
      given Graph        = Graph.open(dbPath)
      given NeuralConfig = NeuralConfig()
      try
        if args.contains("--serve") then
          val port = flagValue(args, "--port").map(_.toInt).getOrElse(8800)
          Server.start(port)
          println(s"$version serving on http://localhost:$port  (Ctrl-C to stop)")
          while true do Thread.sleep(3600000)
        else
          Repl.run(BufferedReader(InputStreamReader(System.in)), System.out)
      catch case e: Throwable => Console.err.println(s"fatal: ${e.getMessage}")
      finally summon[Graph].close()

  private def flagValue(args: Array[String], flag: String): Option[String] =
    val i = args.indexOf(flag)
    Option.when(i >= 0 && i + 1 < args.length)(args(i + 1))

  def usage: String =
    s"""$version
       |
       |Usage: nsk [options]
       |
       |  (no options)     start the interactive REPL
       |  --serve          start the REST server instead of the REPL
       |  --port N         server port (default 8800)
       |  --db PATH        transaction log path (default nsk-graph.log)
       |  --help           show this message
       |
       |Wrap the REPL with rlwrap for history and line editing:
       |  rlwrap scala-cli run .""".stripMargin
```

The two `given` lines put the graph and config in scope for the whole `try`
block, so every call to `Repl.run` or `Server.start` finds them through their
`using` clauses. The `finally` closes the log so the last writes reach disk.

## The embedded DSL: recreating reader macros in Scala

The Common Lisp original uses macros twice: reader macros for `?var`,
`[s p o]`, and `~pred`, and a `defmacro` for `ask`. Scala has no reader macros
and no `defmacro`, yet the port can offer the same syntax. This section is the
most interesting part for Scala programmers, because it shows five features
working together to give a Lisp-like surface.

First, the target. From Scala code we want to write a query two ways: as the
original Lisp text, and as native Scala combinators. Both should produce the same
`Form` or `QueryResult`:

```scala
import nsk.dsl.{*, given}

given Graph = Graph.inMemory
summon[Graph].add("mark", "wrote", "nsk")

nsk"[?who :wrote :nsk]".run                        // the Lisp text
ask(v.who)(t(v.who, "wrote", "nsk"))               // native Scala
```

Here is the DSL that makes both lines legal.

```scala
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
```

Five features carry the syntax:

1. **A string interpolator as the reader macro.** `nsk"[?who :wrote :nsk]"` calls
   a `StringContext` extension that runs the same `Reader` the REPL uses. One
   parser serves the REPL, log replay, and this interpolator. A spliced `Term`
   round-trips, because a term's `toString` prints the exact surface syntax the
   reader reads back.

2. **`scala.Dynamic` for `?var`.** The Lisp `?` macro reads the next symbol and
   wraps it. `object v extends Dynamic` does the same with a field access, so
   `v.who` becomes `Term.Var("who")` at compile time.

3. **`unary_~` for the neural predicate.** Scala lets a type define `unary_~`, so
   `~` reads as a prefix operator, exactly like the Lisp `~`. Now `~kw"codes-in"`
   is a neural predicate.

4. **`given Conversion` for keyword atoms.** Lisp writes `:mark` for an atom.
   Scala has no atom literal, so a `given Conversion` lets a plain string stand
   in for one in query positions. A wildcard `import nsk.dsl.*` does not bring in
   a `given`, so callers import with `import nsk.dsl.{*, given}`.

5. **`using` parameters for the special variables.** Lisp threads the active
   graph and the Ollama settings through `*graph*` and `*ollama-url*`. Scala's
   context parameters do the same job with static safety, as we saw in the query
   engine.

Two more choices complete the idiom: every closed data model is an `enum`, which
gives exhaustive `match` checking the Lisp `ecase` gives only at run time; and
unification returns `Option` in place of the `+FAIL+` sentinel, which lets the
whole engine chain with `flatMap`.

## Running the engine

The project ships a `Makefile` that wraps scala-cli. It passes `--jvm system` so
scala-cli uses the JDK already on your path rather than downloading one.

```bash
make compile   # type-check every source file
make test      # run the test suite
make run       # start the interactive REPL
make serve     # start the REST server on port 8800
```

Start the REPL and add a few facts. The session below shows the exact output you
will see. Recall that keyword names are stored upper case, so a fact echoes back
in upper case, and a query result prints the atom with its colon.

```
$ make run
NSK: Neural-Symbolic Knowledge Graph Engine
Type :help for commands, :quit to exit.
nsk> (:add :mark :wrote :nsk)
added MARK WROTE NSK
nsk> (:add :jane :wrote :nsk)
added JANE WROTE NSK
nsk> (:add :mark :codes-in :lisp)
added MARK CODES-IN LISP
nsk> [?who :wrote :nsk]
who=:MARK
who=:JANE
nsk> (ask (?a) [?a :wrote :nsk] [?a :codes-in :lisp])
a=:MARK
nsk> :count
3 triples
nsk> :quit
Bye.
```

The single pattern `[?who :wrote :nsk]` returns two rows, because two people
wrote NSK. The conjunctive `ask` returns one row, because only Mark both wrote
NSK and codes in Lisp. That narrowing from two rows to one is the join at work.

If you have Ollama running with the `qwen3.5:4b` model, a neural predicate fills
a fact the store does not hold:

```
nsk> [:mark ~:favorite-language ?lang]
lang=:COMMON-LISP
```

NSK found no `:favorite-language` triple, so it asked the model, which answered
"Common Lisp", and NSK bound the sanitized keyword to `?lang`. With no daemon
running, the same query prints one diagnostic line to standard error and returns
no rows:

```
nsk> [:mark ~:favorite-language ?lang]
; neural fallback unavailable: Connection refused
#<no solutions>
```

The REST server runs the same engine over HTTP:

```
$ make serve
NSK 1.0.0 serving on http://localhost:8800  (Ctrl-C to stop)
```

```
$ curl -s http://localhost:8800/query \
    -H 'Content-Type: application/json' \
    -d '{"subject": null, "predicate": "wrote", "object": "nsk"}'
{"count":2,"results":[{"subject":"mark"},{"subject":"jane"}]}

$ curl -s http://localhost:8800/health
{"status":"ok","triples":3,"model":"qwen3.5:4b"}
```

## Testing

The suite lives in one file and uses a tiny hand-rolled harness, in the spirit of
the Common Lisp `check` macro. A `check` runs a boolean and records a pass, a
fail, or an error. There is no test framework and so no dependency.

```scala
package nsk

import nsk.dsl.{*, given} // the `given` selector imports the String -> Term conversion
import scala.language.implicitConversions
import java.nio.file.{Files, Path}

/** A tiny zero-dependency test harness, in the spirit of the Common Lisp
  * `check` macro. Run with:  scala-cli run . --main-class nsk.test
  */
object T:
  var pass = 0
  var fail = 0

  def section(title: String): Unit = println(s"\n== $title ==")

  def check(label: String)(cond: => Boolean): Unit =
    try
      if cond then { pass += 1; println(s"  ok   $label") }
      else { fail += 1; println(s"  FAIL $label") }
    catch case e: Throwable => { fail += 1; println(s"  ERR  $label  <$e>") }

@main def test(): Unit =
  import T.*

  section("unification")
  check("a and b do not unify")(Unify.unify(kw"a", kw"b").isEmpty)
  check("a unifies with a")(Unify.unify(kw"a", kw"a").isDefined)
  check("variable binds to value") {
    val x = v.x
    Unify.unify(x, kw"mark").flatMap(_.get(x)).contains(kw"mark")
  }
  check("logic vars are equal by name")(v.x == v.x)
  check("an already-bound var will not rebind") {
    val x   = v.x
    val env = Unify.unify(x, kw"mark")
    Unify.unify(x, kw"jane", env).isEmpty
  }
  check("resolve follows a binding") {
    val x = v.x
    Unify.resolve(x, Unify.unify(x, kw"mark").get) == kw"mark"
  }

  section("store")
  locally {
    val g = Graph.inMemory
    g.add("mark", "wrote", "nsk")
    g.add("jane", "wrote", "book")
    g.add("mark", "codes-in", "lisp")
    g.add("mark", "wrote", "nsk") // duplicate, ignored
    check("duplicates are ignored")(g.count == 3)
    check("subject index works")(g.spoIndex(kw"mark").length == 2)
    check("object index works")(g.ospIndex(kw"book").length == 1)
    check("insertion order kept")(g.all.head == t("mark", "wrote", "nsk"))
    g.remove("mark", "wrote", "nsk")
    check("remove updates the count")(g.count == 2)
    check("remove clears the index")(!g.spoIndex(kw"mark").contains(t("mark", "wrote", "nsk")))
  }

  section("persistence (replay)")
  locally {
    val path = "nsk-test-tmp.log"
    Files.deleteIfExists(Path.of(path))
    locally {
      val g = Graph.open(path)
      g.add("a", "b", "c")
      g.add("d", "e", "f")
      g.remove("a", "b", "c")
      g.close()
    }
    val g2 = Graph.open(path)
    check("replay reaches the right count")(g2.count == 1)
    check("replay keeps live triples")(g2.all.head == t("d", "e", "f"))
    check("replayed deletion sticks")(!g2.present(t("a", "b", "c")))
    g2.close()
    Files.deleteIfExists(Path.of(path))
  }

  section("reader macros")
  nsk"[?person :wrote :nsk]" match
    case Form.Pat(tr) =>
      check("[ ] reads as a pattern")(true)
      check("?x reads as a logic var")(tr.s == v.person)
      check("keyword reads upper-case")(tr.p == Term.Kw("WROTE"))
      check("third slot")(tr.o == Term.Kw("NSK"))
    case _ => check("[ ] reads as a pattern")(false)
  nsk"[?a ~:codes-in ?l]" match
    case Form.Pat(tr) =>
      check("~pred reads as a neural predicate")(tr.p == Term.Neural(Term.Kw("CODES-IN")))
    case _ => check("~pred reads as a neural predicate")(false)

  section("query engine")
  locally {
    given Graph = Graph.inMemory
    summon[Graph].add("mark", "wrote", "nsk")
    summon[Graph].add("jane", "wrote", "nsk")
    summon[Graph].add("mark", "codes-in", "lisp")
    val one = nsk"[?who :wrote :nsk]".run.solutions
    check("two authors wrote nsk")(one.length == 2)
    check("mark is one")(one.exists(_.exists((_, value) => value == kw"mark")))
    val joined = nsk"(ask (?a) [?a :wrote :nsk] [?a :codes-in :lisp])".run.solutions
    check("join narrows to one author")(joined.length == 1)
    check("the author is mark")(joined.head.head._2 == kw"mark")
  }

  section("json")
  check("strings are escaped")(Json.Str("a\"b").render == "\"a\\\"b\"")
  check("objects parse to fields") {
    Json.parse("""{"x":1,"y":true}""") == Json.Obj(List("x" -> Json.Num(1), "y" -> Json.Bool(true)))
  }
  check("arrays parse to lists") {
    Json.parse("[1,2,3]") == Json.Arr(List(Json.Num(1), Json.Num(2), Json.Num(3)))
  }
  locally {
    val inner = """{"result": "Common Lisp"}"""
    val outer = Json
      .obj("model" -> Json.Str("m"), "response" -> Json.Str(inner), "done" -> Json.Bool(true))
      .render
    val response = Json.parse(outer)("response").flatMap(_.str).get
    val result   = Json.parse(response)("result").flatMap(_.str).get
    check("extract result from an Ollama-style reply")(result == "Common Lisp")
    check("sanitize to keyword")(Neural.sanitizeToKeyword(result) == Term.Kw("COMMON-LISP"))
  }
  check("sanitize trims and upcases") {
    Neural.sanitizeToKeyword("  Common Lisp. ") == Term.Kw("COMMON-LISP")
  }

  section("server helpers")
  check("null field becomes a variable")(Server.fieldToTerm(None, "subject").isInstanceOf[Term.Var])
  check("?who field becomes a variable")(Server.fieldToTerm(Some(Json.Str("?who")), "subject") == v.who)
  check("plain field becomes a keyword")(Server.fieldToTerm(Some(Json.Str("mark")), "subject") == kw"mark")
  check("keyword renders as lower-case text")(Server.termJson(kw"mark") == "mark")
  locally {
    given Graph = Graph.inMemory
    summon[Graph].add("mark", "wrote", "nsk")
    val sol  = Query.matchTriple(kw"mark", kw"wrote", v.o).solutions.head
    val json = Server.solutionToJson(sol).render
    check("solution renders to json")(json.contains("nsk"))
  }

  section("neural fallback (no daemon)")
  locally {
    given Graph        = Graph.inMemory
    given NeuralConfig = NeuralConfig(url = "http://127.0.0.1:9", timeoutSeconds = 2)
    summon[Graph].add("mark", "wrote", "nsk")
    val sols = nsk"(ask (?l) [:mark ~:codes-in ?l])".run.solutions
    check("a ~ query fails cleanly when Ollama is down")(sols.isEmpty)
  }

  section("repl (scripted)")
  locally {
    given Graph = Graph.inMemory
    val out = Repl.feed(
      """(:add :mark :wrote :nsk)
        |(:add :mark :codes-in :lisp)
        |[?who :wrote :nsk]
        |(ask (?a) [?a :wrote :nsk] [?a :codes-in :lisp])
        |:count""".stripMargin
    )
    check("repl :add reports the addition")(out.contains("added"))
    check("repl runs a single pattern")(out.contains("who=:MARK"))
    check("repl runs an ask join")(out.contains("a=:MARK"))
    check("repl :count is correct")(out.contains("2 triples"))
    check("repl mutated the graph")(summon[Graph].count == 2)
  }

  println("\n==================================")
  println(s"NSK tests: $pass passed, $fail failed")
  println("==================================")
  if fail > 0 then System.exit(1)
```

Run it with `make test`. The harness prints one line per check and a summary at
the end. The neural section points the config at a dead port, so it runs with no
model and confirms that a neural query fails cleanly. You will see:

```
$ make test

== unification ==
  ok   a and b do not unify
  ok   a unifies with a
  ok   variable binds to value
  ok   logic vars are equal by name
  ok   an already-bound var will not rebind
  ok   resolve follows a binding

== store ==
  ok   duplicates are ignored
  ok   subject index works
  ok   object index works
  ok   insertion order kept
  ok   remove updates the count
  ok   remove clears the index

== persistence (replay) ==
  ok   replay reaches the right count
  ok   replay keeps live triples
  ok   replayed deletion sticks

== reader macros ==
  ok   [ ] reads as a pattern
  ok   ?x reads as a logic var
  ok   keyword reads upper-case
  ok   third slot
  ok   ~pred reads as a neural predicate

== query engine ==
  ok   two authors wrote nsk
  ok   mark is one
  ok   join narrows to one author
  ok   the author is mark

== json ==
  ok   strings are escaped
  ok   objects parse to fields
  ok   arrays parse to lists
  ok   extract result from an Ollama-style reply
  ok   sanitize to keyword
  ok   sanitize trims and upcases

== server helpers ==
  ok   null field becomes a variable
  ok   ?who field becomes a variable
  ok   plain field becomes a keyword
  ok   keyword renders as lower-case text
  ok   solution renders to json

== neural fallback (no daemon) ==
  ok   a ~ query fails cleanly when Ollama is down

== repl (scripted) ==
  ok   repl :add reports the addition
  ok   repl runs a single pattern
  ok   repl runs an ask join
  ok   repl :count is correct
  ok   repl mutated the graph

==================================
NSK tests: 41 passed, 0 failed
==================================
```

## Interpreting the results

Read the output as a set of claims about the engine, not just a list of ticks.

The **unification** checks prove the core is sound. Two different atoms do not
unify, an atom unifies with itself, a variable binds once and will not silently
rebind to a second value, and `resolve` reads a binding back out. If any of these
failed, every query would be suspect, so they come first.

The **store** and **persistence** checks prove the graph is correct and durable.
Duplicates do not inflate the count, both indices return the right triples, and,
after a write then a delete then a restart, replay reaches exactly the surviving
triple. This is the guarantee that lets you stop and restart NSK without losing
or double-counting facts.

The **reader** and **query** checks prove the surface syntax and the search agree
with the theory. The join check is the important one: `[?who :wrote :nsk]` yields
two authors, and adding the second clause `[?a :codes-in :lisp]` narrows the
answer to one. That is the conjunctive `\mathrm{prove}`$ recurrence returning
exactly the intersection, which is the whole point of a Datalog-style query.

The **neural fallback** check is the neural-symbolic contract in one line. With
no model reachable, a `~` query returns no solutions and does not throw. The
system stays up without a model and gains coverage with one. That is the property
that makes a neural-symbolic design safe to deploy: the neural part is an
enhancement, never a dependency for basic function.

A green suite of 41 checks means the symbolic engine is correct on its own, the
persistence layer is sound, the reader accepts the intended syntax, and the
neural layer fails safe. With Ollama present, the same engine also answers
questions no one entered, which is the capability a purely symbolic store can
never have.

## Wrap Up

We built a working neural-symbolic knowledge graph engine and kept the whole core
on the standard library. The symbolic half stores triples, indexes them two ways,
and answers conjunctive queries by unification, with an append-only log for
durability. The neural half calls a local model, but only to fill a gap the
symbolic store cannot, and it fails safe when the model is absent. The two halves
come together in `matchPattern`, where "symbolic first, neural second" is four
lines of Scala.

Along the way the port showed how Scala 3 matches the expressive syntax of Common
Lisp macros without macros: a `StringContext` interpolator stands in for a reader
macro, `scala.Dynamic` and `unary_~` give the `?var` and `~pred` prefixes, a
`given Conversion` supplies keyword literals, and `using` parameters replace Lisp
special variables with static safety. Closed data models became `enum`s with
exhaustive matching, and search that can fail returned `Option` instead of a
sentinel. The result is a program that reads like a short pipeline and that the
compiler checks at every step.

The design scales to real use. Swap the in-memory indices for a database, point
the neural layer at a larger model, and the query surface does not change. The
pattern, a trusted symbolic store with a neural fallback for coverage, is one of
the most useful approaches in applied AI today, and you now have a small, complete
example of it to build on.

## Optional Practice Problems

These exercises extend the engine you just read. Each names the file to start in.
They rise in difficulty, from a warm-up to a small research project.

1. **A predicate index.** `Graph.candidates` narrows by subject or object but
   never by predicate, so a query like `[?s :wrote ?o]` scans every live triple.
   Add a third index keyed by predicate in `Graph.scala`, and extend
   `candidates` to use it when only the predicate is concrete. Add a test in
   `Tests.scala` that proves the new index returns the right triples.

2. **List every predicate.** Add a REPL command `:preds` to `Repl.scala` that
   prints each distinct predicate in the graph, one per line. Reuse `Term.display`
   for the output, and match the style of the existing `:facts` command.

3. **Wildcard delete.** The current `(:del s p o)` removes one exact triple. Add
   a `(:del-where ...)` command that accepts variables and deletes every triple
   that matches the pattern. Hint: run the pattern through `Query.matchTriple`
   first, resolve each solution back to a ground triple, then call
   `Graph.remove`. Be careful to collect the matches before you start deleting.

4. **Write-through neural facts.** Right now a neural answer is used once and
   thrown away. Change the neural path so that when the model infers an object,
   NSK also adds the new triple to the graph, so the next identical query is a
   fast symbolic hit. Decide where this belongs: `Query.neuralMatch` has the
   answer and the graph is a `using` parameter. Add a test that a repeated `~`
   query finds the cached fact on the second run without the model.

5. **A `not` clause.** Extend the reader and the query engine with negation as
   failure, so `(ask (?a) [?a :wrote :nsk] (not [?a :codes-in :lisp]))` returns
   authors of NSK who do not code in Lisp. In `Reader.scala` accept a `(not
   [..])` clause; in `Query.scala` a `not` clause succeeds with the current
   environment when its inner pattern has no solution, and fails otherwise.
   Think about why the inner pattern should be ground by the time you test it.

6. **Provenance for neural facts.** A fact from the model is less certain than one
   a user typed. Extend the store so a triple can carry a source tag, `:user` or
   `:neural`, and add a REPL command that lists only the neural-derived facts.
   The `Term.Neural` wrapper already marks neural terms; decide whether
   provenance belongs on the term, on the triple, or in a side table, and defend
   your choice in a comment.

7. **A batch query endpoint.** Add a `POST /batch` route to `Server.scala` that
   accepts a JSON array of query objects and returns an array of result objects
   in the same order. Reuse `fieldToTerm` and `solutionToJson`. Add a health
   field that reports whether the Ollama daemon answered a probe request, so a
   caller can tell in advance whether neural queries will work.

8. **Rules, not just facts.** This is the large one. Add Datalog rules with a
   head and a body, so a user can state `(:rule [?x :ancestor ?z] [?x :parent ?y]
   [?y :ancestor ?z])` and have `:ancestor` computed by forward or backward
   chaining. Store rules apart from facts, extend `Query.prove` to expand a rule
   when a pattern matches its head, and guard against infinite recursion on a
   cyclic graph. Compare your approach to the neural fallback: one derives new
   facts by logic, the other by a model, and a mature engine offers both.
