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
