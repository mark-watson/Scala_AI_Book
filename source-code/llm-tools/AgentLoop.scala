//> using scala 3.6.4
//> using dep com.lihaoyi::requests:0.9.3
//> using dep com.lihaoyi::ujson:4.4.3
// Copyright 2026 Mark Watson. All rights reserved.

package llmtools

// Agent loop: prompt the model with the tool list, parse CALL lines,
// run tools, feed results back, repeat until the model prints FINAL.
// The CALL/FINAL line protocol keeps the loop model-agnostic: any
// chat model that follows text orders can drive it.
object AgentLoop:
  val callPrefix = "CALL:"
  val finalPrefix = "FINAL:"

  def systemPrompt(task: String): String =
    s"""You solve tasks by calling tools. Tools:
${ToolRegistry.promptBlock}
Reply with one $callPrefix line per tool call, shaped like:
CALL: tool-name | arg1 | arg2
or give the answer shaped like:
FINAL: <answer text>
Task: $task"""

  // Parse one model reply into tool calls and an optional final answer.
  def parseReply(reply: String): (List[(String, List[String])], Option[String]) =
    val calls = reply.linesIterator.toList.collect {
      case line if line.trim.startsWith(callPrefix) =>
        val parts = line.trim.stripPrefix(callPrefix).split("\\|").map(_.trim).toList
        (parts.head, parts.tail)
    }
    val finalAnswer = reply.linesIterator.toList
      .find(_.trim.startsWith(finalPrefix))
      .map(_.trim.stripPrefix(finalPrefix).trim)
    (calls, finalAnswer)

  // Run one task. Returns the final answer text.
  def run(task: String, model: String = "mistral", maxSteps: Int = 5): String =
    import scala.util.boundary, boundary.break
    boundary:
      var transcript = systemPrompt(task)
      for _ <- 1 to maxSteps do
        val reply = askModel(transcript, model)
        val (calls, finalAnswer) = parseReply(reply)
        finalAnswer match
          case Some(answer) => break(answer)
          case None =>
            val results = calls.map { case (name, args) =>
              val out =
                try ToolRegistry.execute(name, args)
                catch case e: Exception => s"ERROR: ${e.getMessage}"
              s"Result of $name: $out"
            }
            transcript += s"\nModel said:\n$reply\n${results.mkString("\n")}\nContinue with CALL or FINAL lines."
      throw new IllegalStateException(s"No FINAL answer within $maxSteps steps.")

  private def askModel(prompt: String, model: String): String =
    val payload = ujson.Obj("model" -> model, "prompt" -> prompt, "stream" -> false).render()
    val response = requests.post(
      "http://localhost:11434/api/generate",
      headers = Map("Content-Type" -> "application/json"),
      data = payload,
      readTimeout = 120000
    )
    ujson.read(response.text())("response").str
