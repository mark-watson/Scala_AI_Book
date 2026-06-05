// Copyright 2025-2026 Mark Watson. All rights reserved.

package openai_client

@main def openAIDemo(): Unit =
  println("=" * 50)
  println("OpenAI LLM Client Demo")
  println("=" * 50)

  val prompt = "Translate the following English text to French: 'Hello, how are you?'"
  println(s"\nPrompt: $prompt\n")
  try
    val completion = OpenAIClient.getCompletion(prompt)
    println(s"Response:\n$completion")
  catch
    case e: Exception =>
      println(s"Error getting completion: ${e.getMessage}")
