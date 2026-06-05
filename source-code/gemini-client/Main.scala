// Copyright 2025-2026 Mark Watson. All rights reserved.

package gemini_client

@main def geminiDemo(): Unit =
  println("=" * 50)
  println("Gemini LLM Client Demo")
  println("=" * 50)

  val prompt1 = "Briefly explain the concept of pattern matching in Scala."
  println(s"\nPrompt: $prompt1\n")
  try
    val completion1 = GeminiClient.getCompletion(prompt1)
    println(s"Response:\n$completion1")
  catch
    case e: Exception =>
      println(s"Error getting completion: ${e.getMessage}")

  val prompt2 = "What are the latest news updates on space exploration from this week?"
  println(s"\nPrompt with Google Search Grounding: $prompt2\n")
  try
    val completion2 = GeminiClient.getCompletionWithSearch(prompt2)
    println(s"Response:\n$completion2")
  catch
    case e: Exception =>
      println(s"Error getting grounded completion: ${e.getMessage}")
