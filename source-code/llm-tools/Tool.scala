//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package llmtools

// Tool contract and registry. Port of cl-llm-agent/tools.lisp:
// each tool has a name, a short description for the model prompt,
// named string params, and a function from args to text output.
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
