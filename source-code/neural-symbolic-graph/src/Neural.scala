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
