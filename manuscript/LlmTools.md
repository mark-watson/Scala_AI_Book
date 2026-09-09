# LLM Tool Use Without a Framework

The AgentScope chapter drives tools through a full agent SDK. At times that is more than you need. This chapter builds a lean tool loop in pure Scala 3: a registry of named tools, three safe builtins, and a loop that lets any chat model call them through plain text. No SDK, no JSON schema dance. The model writes `CALL:` lines, the code runs them, the results flow back in.

All code is in `source-code/llm-tools`.

## A Registry in Thirty Lines

Each tool has a name, a one line brief for the prompt, named string params, and a run function from args to text. The registry checks arity and fails loud on unknown names:

```scala
final case class ToolDefinition(
    name: String,
    description: String,
    parameters: List[String],
    run: List[String] => String
)

def execute(name: String, args: List[String]): String =
  val tool = tools.getOrElse(name, throw new NoSuchElementException(s"Tool $name not found."))
  if args.size != tool.parameters.size then
    throw new IllegalArgumentException(...)
  tool.run(args)
```

Strings in, strings out keeps tools uniform: files, clocks, and web calls all fit. Parse rich types at the edge if a tool needs them.

## Three Safe Builtins

`BuiltinTools` ports the Lisp originals: `current-time` prints an ISO stamp, `list-files` lists a dir while hiding dot files, `read-file` reads text or raises on lost paths. All three are read only. That is on purpose: an agent that can only read cannot wipe your disk while you test prompts. Add writes later behind an allow list of paths.

## The CALL and FINAL Loop

The loop prints the tool list as a prompt block, sends it with the task to local Ollama, and reads the reply for two line shapes. `CALL: name | arg` runs a tool. `FINAL: text` ends the run with the answer. Tool outputs join the transcript each round, up to five steps:

```scala
def parseReply(reply: String): (List[(String, List[String])], Option[String])
def run(task: String, model: String = "mistral", maxSteps: Int = 5): String
```

Text beats JSON here for one reason: small local models print clean lines far more often than clean JSON. When a call fails, the loop feeds back `ERROR:` text and the model can retry with fixed args, the same shape the book's coding agent uses.

Run the demo and the checks:

```bash
cd source-code/llm-tools
scala-cli run . --main-class llmtools.toolsDemo
scala-cli run . --main-class llmtools.toolsTest
```

The tests run offline on a temp dir: registry shape, arity errors, ISO stamp form, hidden file skip, read round trip, and reply parse. The live loop needs Ollama and skips clean when it is down.
