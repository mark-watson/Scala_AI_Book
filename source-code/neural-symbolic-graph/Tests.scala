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
