// Copyright 2025-2026 Mark Watson. All rights reserved.

package agentscope_gemini

import io.agentscope.core.ReActAgent
import io.agentscope.core.message.Msg
import io.agentscope.core.tool.{Tool, ToolParam, Toolkit}

import java.nio.file.{Files, Path}
import java.util.Comparator
import java.util.stream.Collectors
import scala.jdk.CollectionConverters.*

class WeatherService:
  @Tool(description = "Get the current weather for a specified city")
  def getWeather(
    @ToolParam(name = "city", description = "The name of the city") city: String
  ): String =
    s"$city weather: Sunny, 25°C"

class FileService:
  @Tool(description = "List the files and sub-directories inside a directory")
  def listDir(
    @ToolParam(name = "path", description = "Absolute or relative path of the directory to list") path: String
  ): String =
    try
      val entries = Files.list(Path.of(path))
      try
        val listing = entries
          .sorted(Comparator.naturalOrder[Path]())
          .map[String](p =>
            (if Files.isDirectory(p) then "[DIR]  " else "[FILE] ") + p.getFileName)
          .collect(Collectors.joining("\n"))
        if listing.isEmpty then "(empty directory)" else listing
      finally entries.close()
    catch
      case e: Exception => s"Error listing directory: ${e.getMessage}"

  @Tool(description = "Read the first N lines of a text file")
  def readFile(
    @ToolParam(name = "path", description = "Path to the file to read") path: String,
    @ToolParam(name = "max_lines", description = "Maximum number of lines to return (default 10)") maxLines: Int
  ): String =
    try
      val lines = Files.readAllLines(Path.of(path))
      val limit = if maxLines <= 0 then 10 else maxLines
      lines.asScala.take(limit).mkString("\n")
    catch
      case e: Exception => s"Error reading file: ${e.getMessage}"

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

  // Query 1: weather
  val weatherResponse = agent.call(
    Msg.builder()
      .textContent("What is the weather like in Tokyo and Paris?")
      .build()
  ).block()

  println("=== Weather Query ===")
  println(weatherResponse.getTextContent())

  // Query 2: filesystem
  val cwd = System.getProperty("user.dir")
  val fsResponse = agent.call(
    Msg.builder()
      .textContent(
        s"""List the directory "$cwd" and for every .md file you find there, display its name followed by its first 10 lines.""")
      .build()
  ).block()

  println("\n=== Filesystem Query ===")
  println(fsResponse.getTextContent())
