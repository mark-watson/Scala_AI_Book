# NSK: Neural-Symbolic Knowledge Graph Engine (Scala)

NSK is a hybrid knowledge graph engine. It stores subject-predicate-object
triples, answers Datalog-style queries by unification, and falls back to a
local language model through Ollama when a strict symbolic match is missing.

This is a Scala 3 port of the Common Lisp NSK engine. The Common Lisp version
uses reader macros for a nice query syntax. This port recreates that syntax
with Scala 3 features. The last section, [Scala tricks](#scala-tricks), lists
each one.

## Features

- **Triplestore** with two hash indices (by subject and by object) for fast
  pattern matching.
- **Durable storage** through an append-only log. NSK replays the log on
  startup to rebuild the graph.
- **Query syntax** in two forms: an `nsk"..."` string interpolator that parses
  the original Lisp surface syntax, and native Scala combinators.
- **Unification** in the Norvig/PAIP style, with `Option` for success or
  failure.
- **Neural fallback** to a local Ollama model. When a `~` predicate finds no
  symbolic match, NSK asks the model for the missing object.
- **REST server** through the JDK's own HTTP server.
- **Zero dependencies** in the core. JSON, the Ollama client, and the web
  server all use the standard library.

## Requirements

- [scala-cli](https://scala-cli.virtuslab.org) and a JDK 17 or newer.
- Optional, for the neural layer: [Ollama](https://ollama.com) running locally
  with the `qwen3.5:4b` model.

The graph engine, the query language, and persistence work with no Ollama
installed.

## Run it

With the Makefile:

```bash
make compile   # type-check every source file
make test      # run the test suite
make run       # start the interactive REPL
make serve     # start the REST server on port 8800
```

Or call scala-cli directly:

```bash
scala-cli run .                          # start the interactive REPL
scala-cli run . -- --serve --port 8800   # start the REST server
scala-cli run . --main-class nsk.test    # run the test suite
```

The Makefile passes `--jvm system`, so scala-cli uses the JDK already on your
PATH instead of downloading one.

## Quick start

```
NSK: Neural-Symbolic Knowledge Graph Engine
Type :help for commands, :quit to exit.
nsk> (:add :mark :wrote :nsk)
added MARK WROTE NSK
nsk> (:add :mark :codes-in :lisp)
added MARK CODES-IN LISP
nsk> [?who :wrote :nsk]
who=:MARK
nsk> (ask (?a) [?a :wrote :nsk] [?a :codes-in :lisp])
a=:MARK
nsk> :quit
Bye.
```

## Query syntax

Two forms produce the same query. The first parses the original Lisp text; the
second is native Scala.

| Lisp text (`nsk"..."`)   | Native Scala            | Meaning              |
| ------------------------ | ----------------------- | -------------------- |
| `?person`                | `v.person`              | a logic variable     |
| `:mark`                  | `kw"mark"` or `"mark"`  | a symbolic atom      |
| `~:codes-in`             | `~kw"codes-in"`         | a neural relation    |
| `[?p :wrote :nsk]`       | `t(v.p, "wrote", "nsk")`| one triple pattern   |

From Scala code:

```scala
import nsk.dsl.{*, given} // the `given` selector pulls in the String -> Term conversion

given Graph = Graph.inMemory
summon[Graph].add("mark", "wrote", "nsk")
summon[Graph].add("mark", "codes-in", "lisp")

// The reader as a string interpolator (the closest analog of a reader macro):
nsk"[?who :wrote :nsk]".run
// who=:MARK

nsk"(ask (?a) [?a :wrote :nsk] [?a :codes-in :lisp])".run
// a=:MARK

// The same query in native Scala combinators:
ask(v.a)(t(v.a, "wrote", "nsk"), t(v.a, "codes-in", "lisp"))
// a=:MARK
```

A plain Scala string is an atom, so `"wrote"` means the keyword `:wrote`. Use
`Term.Str("...")` for a real string value.

## Commands

| Command             | Effect                                    |
| ------------------- | ----------------------------------------- |
| `:help`             | show the command list                     |
| `(:add s p o)`      | add a triple                              |
| `(:del s p o)`      | remove a triple                           |
| `(:ingest "text")`  | extract triples from text with the model  |
| `:facts`            | list every triple                         |
| `:count`            | show the triple count                     |
| `:save`             | flush the log to disk                     |
| `:quit`             | leave the REPL                            |

## Storage

Every `add` and `del` appends one form to the log, then flushes it:

```
(:add :mark :wrote :nsk)
(:add :mark :codes-in :lisp)
(:del :mark :wrote :nsk)
```

On startup NSK reads the log in order and replays it, so the in-memory graph
matches the last saved state. The reader has no `eval`, so a log file cannot
run code.

The default log is `nsk-graph.log` in the working directory. Change it with the
`--db` flag.

## Neural layer

NSK talks to Ollama at `http://localhost:11434` and targets the `qwen3.5:4b`
model. Change these with a `given NeuralConfig`:

```scala
given NeuralConfig = NeuralConfig(url = "http://localhost:11434", model = "qwen3.5:4b")
```

A `~` predicate first tries a plain symbolic match. If none exists and the
daemon is up, NSK asks the model and turns the answer into a keyword, so
`"Common Lisp"` becomes `:COMMON-LISP`. When the daemon is down, the query
returns no solutions and prints a short note.

## REST server

```
scala-cli run . -- --serve --port 8800
```

`POST /query` with a JSON body. A field that is `null` or starts with `?` is a
variable; any other string becomes a keyword.

```
curl -s http://localhost:8800/query \
  -H 'Content-Type: application/json' \
  -d '{"subject": null, "predicate": "wrote", "object": "nsk"}'
```

```json
{"count":1,"results":[{"subject":"mark"}]}
```

`GET /health` returns the triple count and the model name.

## Project layout

```
Makefile           common tasks: compile, test, run, serve
project.scala      scala-cli build config (Scala version, main class)
src/Term.scala     terms and triples
src/Json.scala     self-contained JSON model, parser, writer
src/Unify.scala    Norvig-style unification over Option
src/Graph.scala    triplestore, indices, log persistence
src/Reader.scala   the NSK reader: forms and terms from text
src/Dsl.scala      the nsk"..." interpolator and the native DSL
src/Query.scala    pattern matching, neural fallback, query result
src/Neural.scala   Ollama client, inference, text-to-triples
src/Repl.scala     interactive read-eval-print loop
src/Server.scala   REST server on the JDK HTTP server
src/Main.scala     command-line entry point
Tests.scala        test suite
```

## Scala tricks

The Common Lisp engine reaches for macros twice: reader macros for the query
syntax (`?var`, `[s p o]`, `~pred`) and a defmacro `ask`. Here is how the Scala
port gets the same feel.

### 1. A string interpolator as the reader macro

A Common Lisp reader macro turns text into forms at read time. The Scala
analog is a custom string interpolator. `nsk"[?who :wrote :nsk]"` calls a
`StringContext` extension that runs the same parser the REPL uses:

```scala
extension (sc: StringContext)
  def nsk(args: Any*): Form = Reader.readForm(sc.s(args*))
```

So one parser, [`Reader`](src/Reader.scala), serves three roles: the REPL, log
replay, and the interpolator. A spliced `Term` round-trips, because a term's
`toString` prints the exact surface syntax the reader reads back.

### 2. `scala.Dynamic` for `?var`

The Lisp `?` macro reads the next symbol and wraps it. Scala's `Dynamic` does
the same with a field access: `v.who` becomes `Term.Var("who")` at compile
time.

```scala
object v extends Dynamic:
  def selectDynamic(name: String): Term.Var = Term.Var(name)
```

### 3. `unary_~` for the neural predicate

Scala lets a type define `unary_~`, so `~` reads as a prefix operator, exactly
like the Lisp `~`:

```scala
extension (t: Term) def unary_~ : Term.Neural = Term.Neural(t)
```

Now `~kw"codes-in"` is a neural predicate.

### 4. `given Conversion` for keyword atoms

Lisp writes `:mark` for an atom. Scala has no atom literal, so a `given
Conversion` lets a plain string stand in for one in query positions:

```scala
given Conversion[String, Term] = s => Term.Kw(s.toUpperCase)
```

Now `t(v.who, "wrote", "nsk")` reads like the Lisp `[?who :wrote :nsk]`. A
wildcard `import nsk.dsl.*` does not bring in a `given`, so callers import the
DSL with `import nsk.dsl.{*, given}`.

### 5. `using` parameters for the special variables

Common Lisp threads the active graph and the Ollama settings through the
special variables `*graph*` and `*ollama-url*`, and rebinds them with `let`.
Scala's context parameters do the same job with static safety. The query and
store functions take `(using Graph, NeuralConfig)`, and a caller rebinds them
with a local `given`:

```scala
given NeuralConfig = NeuralConfig(url = "http://127.0.0.1:9", timeoutSeconds = 2)
nsk"(ask (?l) [:mark ~:codes-in ?l])".run   // uses the local config
```

### 6. `enum` for the data model

`Term`, `Json`, and `Form` are `enum` algebraic data types. The compiler checks
that every `match` covers each case, which the Lisp `cond` and `ecase` forms
check only at run time.

### 7. `Option` in place of the `+FAIL+` sentinel

Lisp unification returns a `+FAIL+` sentinel and lets `NIL` mean the empty
successful environment. Scala models both with one type: `Some(env)` for
success and `None` for failure. The whole engine then chains with `flatMap`
and reads as a pipeline of small steps.

### 8. A zero-dependency core on the JDK

The Lisp core hand-rolls JSON to avoid a dependency, loads an HTTP client only
when present, and loads the web server on demand. The Scala port keeps the same
zero-dependency goal with the standard library: a hand-rolled
[`Json`](src/Json.scala), the JDK `HttpClient` for Ollama, and
`com.sun.net.httpserver` for the REST server.

## License

Apache-2.0.
