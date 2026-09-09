//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package llmtools

import java.nio.file.{Files, Paths}
import java.time.{ZoneId, ZonedDateTime}
import java.time.format.DateTimeFormatter

// Builtin tools. Port of llm-tools/current-time.lisp,
// llm-tools/files-in-current-directory.lisp, and the file helpers
// in cl-llm-agent/tools.lisp.
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
