# Autonomous Agents with AgentScope

Autonomous agents are programs that can perceive their environment, make decisions, and execute actions to achieve specific goals. In modern AI, multi-agent frameworks enable orchestrating multiple LLM agents that collaborate and use external tools.

In this chapter, we explore how to build autonomous agents in Scala 3 using the **AgentScope** framework, equipping agents with custom tools to list files, read directories, and query weather reports.

All code is in `source-code/AgentScope_gemini`.

## Project Configuration

AgentScope integrates the official Google GenAI Java SDK. We declare our dependencies at the top of **AgentScope_gemini/GeminiConfig.scala** and define a helper object to build the chat model:

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

## Creating a Conversational Agent

A basic conversational agent is defined by a system prompt (which establishes its persona and constraints) and a backing LLM. In **AgentScope_gemini/Main.scala**, we initialize a `ReActAgent` and send a message:

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

## Equipping Agents with Custom Tools

The true power of autonomous agents lies in **Tool Use** (function calling). When an agent receives a prompt requiring information it does not possess, it can choose to execute a registered tool and inspect the results before formulating its final response.

In AgentScope, tools are defined by annotating class methods with `@Tool` and parameters with `@ToolParam`:

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

We register these services inside a `Toolkit` and attach it to the agent builder in **AgentScope_gemini/ToolUseExample.scala**:

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

The agent automatically plans and executes its steps:
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
