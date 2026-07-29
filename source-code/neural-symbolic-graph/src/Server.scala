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
