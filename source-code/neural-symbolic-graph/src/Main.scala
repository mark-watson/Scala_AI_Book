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
