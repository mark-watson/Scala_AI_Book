# Autonomous Agents with AgentScope

Autonomous agents are programs that can perceive their environment, make decisions, and execute actions to achieve specific goals. In modern AI, multi-agent frameworks enable orchestrating multiple LLM agents that collaborate and use external tools.

In this chapter, we explore how to build autonomous agents in Scala 3 using the **AgentScope** framework, equipping agents with custom tools to list files, read directories, and query weather reports.

All code is in `source-code/AgentScope_gemini`.

## From Text Generator to Agent

The three previous chapters called an LLM and read back text. An **agent** is the next step: an LLM placed inside a loop, given memory and tools, and pointed at a goal it pursues over several steps without a human directing each one. The model still only predicts text, but wrapping it in a control loop and connecting it to real actions turns that text into behavior.

The dominant design for this is the **ReAct** pattern (short for Reason + Act), introduced by Yao and colleagues in 2022. The agent runs a cycle:

1. **Reason**: the model thinks about the goal and what it needs next.
2. **Act**: if it needs information or an effect in the world, it calls a **tool**.
3. **Observe**: the framework runs the tool and feeds the result back to the model.

The loop repeats, each observation informing the next thought, until the model decides it has enough to answer. This interleaving of reasoning and acting is what lets an agent break a vague request into concrete steps and adapt when a step returns something unexpected. The `ReActAgent` we build implements exactly this cycle.

Agents solve the fundamental limit of a raw LLM: a language model on its own cannot *do* anything. It cannot read today's news, open a file, run a query, or call an API; it can only generate plausible text. Tools remove that wall. This is the culmination of the whole book's arc through modern AI: the Gemini chapter **grounded** a model in live search, the Knowledge Graph Navigator **retrieved** structured facts, and here an agent **acts**, choosing and running real operations to accomplish a task.

## Project Configuration

AgentScope integrates the official Google GenAI Java SDK, and because Scala runs on the JVM we use these Java libraries directly. We declare our dependencies at the top of **AgentScope_gemini/GeminiConfig.scala** and define a helper object to build the chat model:

```scala
//> using scala 3.8.3
//> using dep io.agentscope:agentscope:1.0.12
//> using dep com.google.genai:google-genai:1.59.0
//> using dep org.slf4j:slf4j-simple:2.0.18

package agentscope_gemini

import io.agentscope.core.model.GeminiChatModel

object GeminiConfig:
  val MODEL_NAME = "gemini-2.5-flash"

  def requireApiKey(): String =
    sys.env.get("GOOGLE_API_KEY") match
      case Some(key) if key.trim.nonEmpty => key.trim
      case _ =>
        System.err.println("ERROR: GOOGLE_API_KEY environment variable is not set.")
        sys.exit(1)
        ""

  def createModel(): GeminiChatModel =
    GeminiChatModel.builder()
      .apiKey(requireApiKey())
      .modelName(MODEL_NAME)
      .build()
```

Unlike the hand-rolled REST clients of the last three chapters, here we lean on a framework. AgentScope supplies the ReAct loop, the tool-calling machinery, and message handling, so we do not have to build them. It uses Gemini as the underlying model, the same `gemini-2.5-flash` from the Gemini chapter, now driving an agent rather than answering a single prompt.

## Creating a Conversational Agent

A basic conversational agent is defined by a system prompt (which establishes its persona and constraints) and a backing LLM. The system prompt plays the same role as the `system` message from the OpenAI chapter: it sets the rules the agent follows for the whole session. In **AgentScope_gemini/Main.scala**, we initialize a `ReActAgent` and send a message:

```scala
package agentscope_gemini

import io.agentscope.core.ReActAgent
import io.agentscope.core.message.Msg

@main def agentScopeDemo(): Unit =
  val model = GeminiConfig.createModel()

  val agent = ReActAgent.builder()
    .name("Assistant")
    .sysPrompt("You are a helpful AI assistant.")
    .model(model)
    .build()

  val response = agent.call(
    Msg.builder()
      .textContent("Hello! Tell me a fun fact about Scala programming.")
      .build()
  ).block()

  println("=== Agent Response ===")
  println(response.getTextContent())
```

With no tools registered, this agent behaves much like a plain chat call, reasoning and replying in one step. Note the `.block()` at the end: AgentScope exposes an asynchronous, reactive API that returns a publisher of the eventual result, and `block()` waits for it to complete. That asynchronous design matters once an agent makes several tool calls, since each is a separate slow operation the framework can manage without stalling the whole program.

## Equipping Agents with Custom Tools

The true power of autonomous agents lies in **Tool Use** (also called function calling). It is worth understanding the mechanism, because it is how a text model reaches into the real world. The framework describes each available tool to the model as a **schema**: a name, a plain-English description of what the tool does, and the names and types of its parameters. When the model decides a tool is needed, it does not run it; instead it emits a structured request naming the tool and its arguments. The framework executes the real function, captures the result, and returns it to the model as an observation. The model reads that observation and continues. The description text is not documentation for humans, it is the exact information the model uses to decide when and how to call the tool, so writing good descriptions is the real work of building an agent.

In AgentScope, tools are defined by annotating class methods with `@Tool` and parameters with `@ToolParam`, which generate that schema automatically from the method signature:

```scala
class WeatherService:
  @Tool(description = "Get the current weather for a specified city")
  def getWeather(
    @ToolParam(name = "city", description = "The name of the city") city: String
  ): String =
    s"$city weather: Sunny, 25°C"

class FileService:
  @Tool(description = "List the files and sub-directories inside a directory")
  def listDir(
    @ToolParam(name = "path", description = "Absolute or relative path of the directory") path: String
  ): String =
    // Filesystem lookup logic
    ...

  @Tool(description = "Read the first N lines of a text file")
  def readFile(
    @ToolParam(name = "path", description = "Path to the file to read") path: String,
    @ToolParam(name = "max_lines", description = "Maximum number of lines to return") maxLines: Int
  ): String =
    // Read file logic
    ...
```

The `WeatherService` returns a hard-coded stub, a stand-in for a real weather API, while `FileService` performs genuine work through Java's `java.nio.file` APIs to list directories and read files. This mix is deliberate: it shows that a tool is just an ordinary method, whether it fakes data or touches the real filesystem. We register these services inside a `Toolkit` and attach it to the agent builder in **AgentScope_gemini/ToolUseExample.scala**:

```scala
@main def toolUseDemo(): Unit =
  val model = GeminiConfig.createModel()

  val toolkit = Toolkit()
  toolkit.registerTool(WeatherService())
  toolkit.registerTool(FileService())

  val agent = ReActAgent.builder()
    .name("AssistantAgent")
    .sysPrompt("You are a helpful assistant with access to weather data and the local filesystem.")
    .model(model)
    .toolkit(toolkit)
    .build()
```

The only difference from the basic agent is `.toolkit(toolkit)`. Registering the toolkit hands the model the schemas for all three methods, and from that point the agent can choose to call any of them.

## Running the Agent Demo

Run the basic conversational agent using:

```bash
scala-cli run . --main-class agentscope_gemini.agentScopeDemo
```

Run the tool-use demo using:

```bash
scala-cli run . --main-class agentscope_gemini.toolUseDemo
```

During the tool-use execution, the agent receives two queries. The first query requests weather data, prompting the agent to invoke the `getWeather` tool. The second query is a complex multi-step filesystem operation:

`List the directory "..." and for every .md file you find there, display its name followed by its first 10 lines.`

This second query is where the ReAct loop earns its keep. No single tool answers it; the agent has to **plan** and decompose the request into a sequence of tool calls, using the result of one to drive the next:
1. It calls `listDir` on the current working directory.
2. It parses the resulting file list.
3. For each `.md` file discovered, it calls `readFile` with `max_lines = 10`.
4. It consolidates the contents into the final response:

```text
=== Weather Query ===
The weather in Tokyo is Sunny, 25°C. In Paris, the weather is also Sunny, 25°C.

=== Filesystem Query ===
I found the following Markdown files in the directory:
1. README.md:
   # AgentScope Gemini Demo
   This project demonstrates how to build conversational and tool-using agents using AgentScope and Gemini.
   ...
```

The weather answer shows the model calling one tool twice, once for Tokyo and once for Paris, and merging the results into a single sentence. The filesystem answer shows genuine autonomy: from one plain-English instruction the agent worked out that it must first list the directory, then loop over the results and read each Markdown file, then assemble everything into a report. We wrote no orchestration code for those steps. We supplied three tools and a goal, and the ReAct loop supplied the plan.

This is a fitting place to end the book. We began with classical search, where we wrote every step of the algorithm by hand. We end with an agent that writes its own plan of action and carries it out using tools we provide. The techniques across these chapters, search, learning, knowledge representation, and language models, are the building blocks, and Scala's blend of functional clarity and JVM performance makes it a strong language for assembling them into real AI systems.
