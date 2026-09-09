//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package llmtools

import java.nio.file.Files

// Offline checks. Run with: scala-cli run . --main-class llmtools.toolsTest
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
