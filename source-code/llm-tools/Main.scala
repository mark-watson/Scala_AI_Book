//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package llmtools

@main def toolsDemo(): Unit =
  BuiltinTools.registerAll()
  println("Registered tools:")
  println(ToolRegistry.promptBlock)
  println("\n--- current-time ---")
  println(ToolRegistry.execute("current-time", Nil))
  println("\n--- list-files . ---")
  println(ToolRegistry.execute("list-files", List(".")))
  val task = "What time is it and what files sit here?"
  println(s"\n--- agent run (needs local Ollama): $task ---")
  try println("FINAL: " + AgentLoop.run(task))
  catch case e: Exception => println(s"[skip live loop] ${e.getMessage}")
