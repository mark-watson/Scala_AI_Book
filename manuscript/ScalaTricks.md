# Scala 3 Conventions and Tricks Used in This Book

Every example in this book is written in Scala 3 and runs with `scala-cli`. The code leans on a small set of language features that repeat in nearly every chapter. If you know another language but are new to Scala 3, read this chapter first. Each section shows a snippet taken from the book's own source code.

## Running an Example with scala-cli

Each project is a directory of `.scala` files with no `build.sbt`. Build settings live at the top of a file as `//> using` directives:

```scala
//> using scala 3.6.4
//> using dep com.lihaoyi::requests:0.9.3
//> using dep com.lihaoyi::ujson:4.4.3
```

The first line pins the compiler version. The `dep` lines declare library dependencies using Maven coordinates. The `::` between group and artifact means "use the version built for the Scala version in use". A project-wide `project.scala` file can hold directives shared by every file.

Run a project from its directory:

```bash
scala-cli run .
```

## Indentation Instead of Braces

Scala 3 lets you drop braces. Blocks are defined by indentation, and keywords replace the old punctuation:

```scala
if b.isOver then score(b, player, depth)
else if isMaximizing then
  boundary:
    var best = Int.MinValue
    for (r, c) <- b.emptyCells do
      best = math.max(best, step(r, c))
    best
```

The pieces to recognize:

- `if cond then ... else ...` replaces `if (cond) { ... } else { ... }`.
- `for ... do ...` and `while ... do ...` replace the braced loop body.
- `match` cases sit in an indented block, no braces required.
- `method:` passes an indented block as the last argument. In `source-code/search/TicTacToe.scala:52` the code reads `(0 until 3).map: r => ...`.

Braces still work and a few ported files use them. Follow the style of the file you edit.

## Top-Level Definitions and @main

Scala 3 allows methods and values at the top of a file, outside any object. A program entry point is marked with `@main`:

```scala
@main def toolsDemo(args: String*): Unit =
  BuiltinTools.registerAll()
  val model = if args.nonEmpty then args(0) else "mistral"
```

`@main` generates a JVM `main` method, so the name of the method becomes the program name. Parameters become command-line arguments, and a default value makes the argument optional. You can define several `@main` methods in one project and pick one with `scala-cli run . --main-class <name>`.

## Case Classes and copy

A `case class` gives you a constructor, accessors, `equals`, `hashCode`, and a readable `toString` for free. The examples use them as immutable records:

```scala
final case class Term(coefficient: Double, variable: Variable, exponent: Int):
  def negate: Term = copy(coefficient = -coefficient)
  def scale(k: Double): Term = copy(coefficient = coefficient * k)
```

`copy` builds a changed duplicate. Named arguments let you change one field and leave the rest. Constructor parameters can have defaults, so callers may omit them:

```scala
case class NeuralNetwork(
    numInputs: Int,
    numHidden: Int,
    numOutputs: Int,
    w1: Array[Array[Float]],
    w2: Array[Array[Float]],
    learningRate: Float = 0.5f
):
```

The neural network chapter trains by returning `copy(w1 = newW1, w2 = newW2)` instead of mutating the weights in place. Prefer this style: return a new value, keep the old one intact.

## Enums as Data Types

A Scala 3 `enum` can be a simple set of constants or a full algebraic data type whose cases carry data:

```scala
enum Term:
  case Var(name: String)     // logic variable:   ?who
  case Kw(name: String)      // symbolic atom:     :mark   (name is UPPER-CASE)
  case Str(value: String)    // string literal:    "hello"
  case Num(value: Long)      // integer literal:   42
  case Neural(inner: Term)   // neural predicate:  ~:codes-in
```

Enums can also declare fields and methods. This is used to attach a wire value to each case:

```scala
enum Role(val value: String):
  case System extends Role("system")
  case User extends Role("user")
  case Assistant extends Role("assistant")
```

`Role.values` returns all cases, which makes it easy to parse a string back into an enum.

## Pattern Matching

`match` tests a value against patterns and picks the first that fits. It replaces long `if`/`else` chains and `switch` statements, and it can see inside data:

```scala
board.winner match
  case Some(p) if p == player => 10 - depth
  case Some(_)                => depth - 10
  case None                   => 0
```

Useful pattern forms used in the book:

- Case-class patterns bind their fields: `case Filled(p) => p.toString`.
- Alternatives share a body: `case _: Var | _: Neural => false`.
- Guards add a condition: `case line if line.forall(...) => Player.X`.
- Cons patterns drive recursion: `case Nil => ...; case pat :: rest => ...`.
- Tuple patterns split pairs: `case (isTarget, isPredicted) => ...`.

The compiler warns when a match is not exhaustive, so keep patterns complete.

## Options and Either

Scala uses `Option` for a value that may be absent, instead of `null`:

```scala
Json.parse(raw)("response").flatMap(_.str).getOrElse("")
```

`Some(x)` holds a value, `None` is empty. Methods like `map`, `flatMap`, `filter`, and `getOrElse` chain without null checks. `Option.when(cond)(value)` builds an `Option` from a condition.

Use `Either` for an operation that can fail with a reason. By convention `Right` is success and `Left` is the error:

```scala
def getChat(messages: List[ChatMessage], model: String): Either[String, String] =
  try
    val response = requests.post(...)
    if response.statusCode == 200 then Right(content)
    else Left(s"Ollama API request failed (HTTP ${response.statusCode})")
  catch
    case e: Exception => Left(s"Error connecting to Ollama: ${e.getMessage}")
```

## For-Comprehensions

A for-comprehension builds a result from one or more generators. With `yield` it returns a collection:

```scala
def emptyCells: List[(Int, Int)] =
  (for
    r <- 0 until 3
    c <- 0 until 3
    if cells(r * 3 + c) == Cell.Empty
  yield (r, c)).toList
```

The same syntax works over `Option` and stops at the first `None`:

```scala
for
  s <- row("subject").flatMap(_.str)
  p <- row("predicate").flatMap(_.str)
  o <- row("object").flatMap(_.str)
yield Triple(sanitize(s), sanitize(p), sanitize(o))
```

Use `for ... do ...` without `yield` when you want a loop with side effects, such as printing or updating a mutable variable.

## Collections

The standard collections do most of the work. The idioms that appear most often:

```scala
terms.groupBy(_.exponent).view.mapValues(_.map(_.coefficient).sum).toMap
words.sliding(maxWords, step).map(_.mkString(" ")).toList
Array.tabulate(numHidden, numOutputs): (h, o) => w2(h)(o) + delta
population.sortBy(-_.fitness)
lines.collectFirst { case line if line.forall(ok) => Player.X }
```

Read them as a pipeline: transform a collection into another collection. `groupBy` buckets by a key, `view.mapValues` changes values lazily, `sliding` makes overlapping windows, `tabulate` builds an array by index, and `sortBy(-_.fitness)` sorts high to low by negating the key.

## String Interpolation

An `s"..."` string substitutes expressions with `$name` or `${expression}`. An `f"..."` string adds printf-style format specifiers:

```scala
println(f"  Epoch $epoch%5d  avg error: $error%.6f")
```

Triple-quoted strings span lines and do not need escaped quotes. `stripMargin` removes the leading `|` so the text lines up in the source:

```scala
s"""|SELECT ?p WHERE {
    |  $entity1Uri ?p $entity2Uri .
    |} LIMIT 10
    |""".stripMargin
```

## Extension Methods, Givens, and Context Parameters

An extension method adds a method to a type you do not own:

```scala
extension (sc: StringContext)
  def kw(args: Any*): Term.Kw =
    Term.Kw(sc.s(args*).trim.toUpperCase.replace(' ', '-'))
```

That snippet defines the custom `kw"mark"` interpolator. A `given` provides a value the compiler passes to methods automatically, and a `using` parameter receives it:

```scala
given default: NeuralConfig = NeuralConfig()

def prove(patterns: List[Triple], env: Env)(using Graph, NeuralConfig): List[Env] =
  ...
```

Call `summon[NeuralConfig]` to fetch the current given, or define a local `given` to override it inside a block. These are Scala 3's replacement for Scala 2 implicits.

## Operator Overloading

Any method whose name is a symbol can be used as an operator. The autograd engine in the transformer chapter defines arithmetic on a `Value` class so the math reads naturally:

```scala
def +(other: Value): Value = ...
def *(other: Value): Value = ...
def unary_- : Value = this * Value(-1.0)
def -(other: Value): Value = this + (-other)
```

`unary_-` enables the prefix minus, and `unary_~` (used in the neural-symbolic chapter) enables a prefix tilde. Operators are ordinary methods, so `a + b` is the same as `a.+(b)`.

## Early Exit with boundary and break

Scala discourages `return`. To leave a loop early and produce a value, wrap the code in `boundary` and call `break`:

```scala
import scala.util.boundary, boundary.break
boundary:
  for (r, c) <- board.emptyCells do
    val s = minimax(board.set(r, c, player), false, depth + 1, alpha, beta)
    best = math.max(best, s)
    if beta <= best then break(best)
  best
```

`break(value)` exits the nearest enclosing `boundary` and returns `value`. This is how the alpha-beta pruning in the search chapter stops exploring a branch.

## inline and @tailrec

Two annotations appear in performance-sensitive code. `inline` asks the compiler to paste the method body at each call site, which removes a call for small helpers:

```scala
inline def pieceType(p: Int): Int = p & TYPE_MASK
inline def opponent(c: Int): Int = if (c == WHITE) BLACK else WHITE
```

`@annotation.tailrec` tells the compiler to check that a recursive method is tail-recursive, so it compiles to a loop and cannot overflow the stack:

```scala
@annotation.tailrec
def search(queue: Queue[(Int, List[Int])], visited: Set[Int]): Option[List[Int]] =
  ...
```

If the method is not tail-recursive, the compiler reports an error. Use it whenever a loop is written as recursion.

## A Note on DSLs

The neural-symbolic chapter goes further and builds a small query language. It uses `scala.Dynamic` so that `v.who` becomes a logic variable, a custom `nsk"..."` interpolator, and a `given Conversion` so a plain string stands for a keyword. The Laminar web chapter uses a similar style, with `:=` for attributes and `<--` for signals. You do not need to know how these are built to use them. Read the DSL section in those chapters when you get there.

## Where to Go Next

With these conventions in mind, the rest of the book should read cleanly. Each chapter lists the source directory it uses, and each project runs on its own with `scala-cli run .`.
