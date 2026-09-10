# LLM Tool Use Without a Framework

The AgentScope chapter drives tools through a full agent SDK. That SDK supplies a ReAct loop, a JSON schema for every tool, and the message plumbing between them. At times that is more than you need. This chapter builds the same capability in pure Scala 3: a registry of named tools, three safe builtins, and a loop that lets any chat model call them through plain text. No SDK and no JSON schema dance. The model writes `CALL:` lines, the code runs them, and the results flow back in.

All code is in `source-code/llm-tools`.

## What Tool Use Actually Is

A chat model is a text function. You give it text and it returns text. It cannot read a file, run a query, or check a clock. It can only predict the next token. Tool use removes that limit with a loop that lives in your program, not in the model. Each round has three parts, the same Reason, Act, Observe cycle the AgentScope chapter described:

1. The host sends the model the task plus a description of the tools it may use.
2. The model replies with text. If it wants a tool, the text names the tool and its arguments. The model does not run anything.
3. The host parses that text, runs the named tool, and appends the result to the conversation. It then sends the longer conversation back to the model.
4. The loop repeats until the model replies with a final answer.

The model's only job is to produce the next piece of text. The host's job is to read that text, decide whether it is a tool request or an answer, and act on it. The model chooses the tool and its arguments; the host runs it and reports back.

Two choices define such a loop: how the model asks for a tool, and how the host detects the request. A framework chooses structured function calling, where the model returns JSON that matches a schema the host supplied. We choose a plain text line protocol. The reason is practical. Small local models print clean lines far more often than they print valid JSON. A line such as `CALL: read-file | notes.txt` is easy for a four billion parameter model to produce and easy for the host to parse. The trade-off is less validation up front, so the parser and a fixed tool list have to carry the safety.

The description text you give the model is not documentation for humans. It is the exact information the model uses to decide when to call a tool. Writing a clear, short description is the real work of building an agent.

## A Tool Registry in Thirty Lines

Every tool needs the same four things: a name the model will write, a one line description so the model knows when to use it, the names of its parameters, and a function that does the work. A case class holds them in **Tool.scala**:

```scala
final case class ToolDefinition(
    name: String,
    description: String,
    parameters: List[String],
    run: List[String] => String
)

object ToolRegistry:
  private val tools = scala.collection.mutable.LinkedHashMap[String, ToolDefinition]()

  def register(tool: ToolDefinition): Unit = tools(tool.name) = tool

  def list: List[ToolDefinition] = tools.values.toList

  def execute(name: String, args: List[String]): String =
    val tool = tools.getOrElse(name, throw new NoSuchElementException(s"Tool $name not found."))
    if args.size != tool.parameters.size then
      throw new IllegalArgumentException(
        s"Tool $name needs ${tool.parameters.size} args (${tool.parameters.mkString(", ")}), got ${args.size}."
      )
    tool.run(args)

  // Render the registry as a prompt block the model can read.
  def promptBlock: String =
    list.map(t => s"- ${t.name}(${t.parameters.mkString(", ")}): ${t.description}").mkString("\n")
```

`run` takes a list of strings and returns a string. That uniformity is the point. Files, clocks, and web calls all fit the same shape. If a tool needs a number or a date, parse it at the edge inside `run`.

`LinkedHashMap` keeps insertion order, so `promptBlock` lists tools in the order you registered them, which keeps the prompt stable from run to run. `execute` does two checks before it runs anything. It fails loud on an unknown name and on the wrong number of arguments. Both checks matter. The model will make mistakes, and a clear exception message becomes useful text that the loop can feed back.

`promptBlock` renders the registry as a prompt block. This is the only place the model learns what tools exist, and the format mirrors the `CALL` line, so the description and the call syntax agree.

## Three Safe Builtins

**BuiltinTools.scala** ports the Lisp originals from the `llm-tools` and `cl-llm-agent` projects:

```scala
object BuiltinTools:
  val currentTime: ToolDefinition = ToolDefinition(
    "current-time",
    "Return the current local date and time in ISO 8601 format. Takes no args.",
    Nil,
    _ => ZonedDateTime.now(ZoneId.systemDefault()).format(DateTimeFormatter.ISO_OFFSET_DATE_TIME)
  )

  val listFiles: ToolDefinition = ToolDefinition(
    "list-files",
    "List files in a directory, skipping hidden files. Args: directory-path.",
    List("directory-path"),
    args =>
      val dir = Paths.get(args.head)
      if !Files.isDirectory(dir) then throw new IllegalArgumentException(s"Directory not found: ${args.head}")
      Files.list(dir).toArray.toList
        .collect { case p: java.nio.file.Path => p.getFileName.toString }
        .filterNot(n => n.startsWith(".") || n.endsWith("~"))
        .sorted
        .mkString("\n")
  )

  val readFile: ToolDefinition = ToolDefinition(
    "read-file",
    "Return the text of a file. Args: file-path.",
    List("file-path"),
    args =>
      val p = Paths.get(args.head)
      if !Files.isRegularFile(p) then throw new IllegalArgumentException(s"File not found: ${args.head}")
      Files.readString(p)
  )

  def registerAll(): Unit =
    ToolRegistry.register(currentTime)
    ToolRegistry.register(listFiles)
    ToolRegistry.register(readFile)
```

`current-time` takes no args and returns an ISO 8601 stamp with the machine's offset. `list-files` takes a directory path, checks that it exists, lists one entry per line, and drops hidden files (names starting with a dot) and editor backups (names ending with a tilde). It sorts the result so the same directory always yields the same text, which helps when you compare runs. `read-file` takes a file path, checks that it is a regular file, and returns the whole text.

All three are read only. That is on purpose. An agent that can only read cannot wipe your disk while you test prompts. Add write tools later, and when you do, put them behind an allow list of paths.

`registerAll` puts the three into the registry. The demo and the tests both call it once at startup.

## The CALL and FINAL Loop

The loop is the core of the example. It sends a prompt, reads the reply, and looks for two line shapes. `CALL: name | arg` runs a tool. `FINAL: text` ends the run. Everything else in the reply is ignored, so the model may add a sentence of reasoning if it wants. **AgentLoop.scala** declares the `requests` and `ujson` dependencies and starts with the prompt and the parser:

```scala
//> using dep com.lihaoyi::requests:0.9.3
//> using dep com.lihaoyi::ujson:4.4.3

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
```

`parseReply` works line by line. A line that starts with `CALL:` is trimmed, then split on the pipe. The first part is the tool name and the rest are arguments. A line that starts with `FINAL:` yields the answer text. Two details make this forgiving. The trim runs before the split, so a trailing space is harmless. Scala's `split` also drops a trailing empty field, so a model that writes `CALL: current-time |` still sends zero arguments, which is what `current-time` expects.

`run` ties it together:

```scala
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
```

The transcript starts as the system prompt. Each round asks the model, parses the reply, and checks for a final answer. If there is one, `boundary` and `break` return it at once. If not, every call runs and its output joins the transcript under a `Result of` line. The next round sees the task, its own earlier reply, and the tool results. That growing transcript is the model's whole memory of the session, so the model does not need any memory of its own.

The `try` around each call is important. A tool that raises does not crash the loop. Its message becomes `ERROR:` text that the model reads on the next round. A missing file or a wrong argument count then becomes something the model can react to, either by fixing the arguments or by telling the user what went wrong. This is the same shape the book's coding agent uses.

`maxSteps` bounds the loop. Without it, a model that never prints `FINAL:` would spin forever. Five steps is enough for the small tasks here.

The last piece talks to the model:

```scala
  private def askModel(prompt: String, model: String): String =
    val payload = ujson.Obj("model" -> model, "prompt" -> prompt, "stream" -> false).render()
    val response = requests.post(
      "http://localhost:11434/api/generate",
      headers = Map("Content-Type" -> "application/json"),
      data = payload,
      readTimeout = 120000
    )
    ujson.read(response.text())("response").str
```

It POSTs to Ollama's `/api/generate` endpoint with `stream` set to false, so the server buffers the whole reply and returns one JSON object. The loop reads the `response` field. Any chat model with the same endpoint, or a different client that returns text, can drive this loop unchanged. That is what model agnostic means here.

## The Demo

**Main.scala** registers the builtins, prints the prompt block, runs two tools directly to show they work, then hands a task to the loop:

```scala
@main def toolsDemo(args: String*): Unit =
  BuiltinTools.registerAll()
  println("Registered tools:")
  println(ToolRegistry.promptBlock)
  println("\n--- current-time ---")
  println(ToolRegistry.execute("current-time", Nil))
  println("\n--- list-files . ---")
  println(ToolRegistry.execute("list-files", List(".")))
  val model = if args.nonEmpty then args(0) else "mistral"
  val task = "What time is it and what files sit here?"
  println(s"\n--- agent run (needs local Ollama, model $model): $task ---")
  try println("FINAL: " + AgentLoop.run(task, model = model))
  catch case e: Exception => println(s"[skip live loop] ${e.getMessage}")
```

The optional command-line argument lets you point the live loop at any model you have pulled. With no argument it uses `mistral`, the default in `AgentLoop.run`.

The direct calls need no model, so they run even when Ollama is down:

```
Registered tools:
- current-time(): Return the current local date and time in ISO 8601 format. Takes no args.
- list-files(directory-path): List files in a directory, skipping hidden files. Args: directory-path.
- read-file(file-path): Return the text of a file. Args: file-path.

--- current-time ---
2026-09-10T13:54:06.577059-07:00

--- list-files . ---
AgentLoop.scala
BuiltinTools.scala
Main.scala
Makefile
README.md
Tool.scala
ToolTest.scala

--- agent run (needs local Ollama, model mistral): What time is it and what files sit here? ---
[skip live loop] Request to http://localhost:11434/api/generate failed with status code 404
{"error":"model 'mistral' not found"}
```

The `404` is expected when the default model is not pulled. `AgentLoop.run` defaults to `mistral`, so pull it, or pass a model you already have as the demo's argument. With a small model such as `qwen3.5:4b` pulled, run the demo like this:

```bash
scala-cli run . --main-class llmtools.toolsDemo -- qwen3.5:4b
```

The same task then drives the loop and prints:

```
FINAL: It is 2026-09-10T13:57:16 (UTC-7, Pacific Time) and the files here are: AgentLoop.scala,
BuiltinTools.scala, Main.scala, Makefile, README.md, Tool.scala, and ToolTest.scala
```

Behind that answer the loop made two rounds. The first reply was:

```
CALL: current-time |
CALL: list-files | .
```

The host ran both, put the two results in the transcript, and asked again. The second reply was the `FINAL:` line above. The model chose which tools to call and what arguments to pass, and it merged two results into one sentence. We wrote no code for that part. Small models vary from run to run, so the exact wording changes, but the shape stays the same: one round of calls, then one final answer.

The error path matters just as much. Asked to read a file that does not exist, the model calls `read-file`, the tool raises, and the loop feeds back `ERROR: File not found: nope.txt`. On the next round the model answers:

```
FINAL: The file nope.txt was not found. It doesn't exist in the directory being searched.
```

A framework would surface that failure as a tool result too. Doing it by hand shows the mechanism plainly: the error is just more text in the transcript.

## Offline Tests

**ToolTest.scala** runs on a temp directory and needs no model:

```scala
@main def toolsTest(): Unit =
  BuiltinTools.registerAll()

  // Registry holds the three builtins.
  val names = ToolRegistry.list.map(_.name).toSet
  assert(names == Set("current-time", "list-files", "read-file"), s"registry wrong: $names")

  // Unknown tool and wrong arity raise.
  try { ToolRegistry.execute("nope", Nil); assert(false, "unknown tool must raise") }
  catch case _: NoSuchElementException => ()
  try { ToolRegistry.execute("read-file", Nil); assert(false, "wrong arity must raise") }
  catch case _: IllegalArgumentException => ()

  // current-time returns an ISO-8601 stamp.
  val stamp = ToolRegistry.execute("current-time", Nil)
  assert(stamp.matches("\\d{4}-\\d{2}-\\d{2}T.*"), s"bad timestamp: $stamp")

  // list-files sees this dir, skips hidden files; read-file round trips.
  val dir = Files.createTempDirectory("tools-test")
  Files.writeString(dir.resolve("a.txt"), "hello")
  Files.writeString(dir.resolve(".hidden"), "x")
  val listing = ToolRegistry.execute("list-files", List(dir.toString))
  assert(listing.contains("a.txt") && !listing.contains(".hidden"), s"bad listing: $listing")
  assert(ToolRegistry.execute("read-file", List(dir.resolve("a.txt").toString)) == "hello", "read-file wrong")
  try { ToolRegistry.execute("read-file", List(dir.resolve("missing.txt").toString)); assert(false, "missing file must raise") }
  catch case _: IllegalArgumentException => ()

  // Reply parser: CALL lines become calls, FINAL line becomes the answer.
  val (calls, fin) = AgentLoop.parseReply("CALL: read-file | a.txt\nFINAL: done here")
  assert(calls == List(("read-file", List("a.txt"))), s"calls wrong: $calls")
  assert(fin.contains("done here"), s"final wrong: $fin")
  val (noCalls, noFin) = AgentLoop.parseReply("just thinking out loud")
  assert(noCalls.isEmpty && noFin.isEmpty, "plain text parses to nothing")

  // Prompt block names every tool so the model can see them.
  assert(AgentLoop.systemPrompt("t").contains("read-file"), "system prompt must list tools")

  println("All tool tests passed.")
```

The checks pin each part of the design. They confirm the registry holds exactly the three builtins; that an unknown tool raises `NoSuchElementException` and a wrong argument count raises `IllegalArgumentException`; that `current-time` returns a stamp matching `YYYY-MM-DDTHH:MM:SS`; that `list-files` sees a written file but skips a hidden one; that `read-file` round trips a written file and raises on a missing one; that `parseReply` turns a `CALL:` line into a call, a `FINAL:` line into the answer, and plain text into nothing; and that the system prompt names every tool. The last check is the one that guards the contract between the prompt and the parser. If you rename a tool or drop it from the prompt, the test fails before the model ever sees it.

Run the demo and the checks:

```bash
cd source-code/llm-tools
scala-cli run . --main-class llmtools.toolsDemo
scala-cli run . --main-class llmtools.toolsTest
```

The checks end with:

```
All tool tests passed.
```

## Wrap Up

This chapter built a tool loop with no framework. The pieces are small: a case class for a tool, a map for the registry, three read only builtins, and a loop that parses two line prefixes. The model writes `CALL:` lines to act and a `FINAL:` line to stop. The host runs the tools and feeds the results back as text, so a failure becomes something the model can see and correct on the next round.

The text protocol is not the only way. The AgentScope chapter used structured function calling, where the model returns JSON that matches a schema. Structured calls are more robust when the model supports them well. A plain line protocol needs no schema and works with small local models that struggle with JSON. Knowing both lets you pick the lighter tool when it fits, and it shows what a framework does for you when you would rather not write it yourself.

## Optional Practice Problems

1. **Add a tool.** Register a `word-count` tool that takes a file path and returns the number of lines and words. Add a test that writes a file and checks the counts, then ask the agent for the word count of a file in this directory.

2. **Multiple arguments.** Add a `grep-file` tool with two parameters, `pattern` and `file-path`, that returns the matching lines. Call it from a test and from the agent, and confirm the arity check rejects a one argument call.

3. **A write tool with an allow list.** Add a `write-file` tool that refuses any path outside a configured base directory. Test both an allowed and a rejected path. Explain in one sentence why the allow list matters even when you trust the model.

4. **Cap the output.** Large tool results crowd the transcript and cost tokens. Add a limit that truncates each result to a fixed number of characters and appends a marker when it cuts. Run the agent on a large file and compare behavior with and without the cap.

5. **JSON instead of lines.** Add a second parser that reads a reply shaped like `{"action":"read-file","args":["a.txt"]}` and wire the loop to accept either shape. Compare how often a small local model produces valid JSON against how often it produces a clean `CALL:` line.

6. **Record the run.** Have the loop return the full transcript, not just the final answer, and print it in the demo. Use it to see how many rounds a task takes and what the model saw at each step.
