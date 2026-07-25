// Copyright 2025-2026 Mark Watson. All rights reserved.

package ollama_client

@main def ollamaDemo(args: String*): Unit =
  println("=" * 50)
  println("Ollama Local LLM Client Demo")
  println("=" * 50)

  val model = if args.nonEmpty then args(0) else "gemma4:12b-it-qat"
  val prompt = "Translate the following English text to French: 'Hello, how are you?'"
  
  println(s"Using model: $model")
  println(s"Prompt: $prompt\n")
  
  try
    println("Sending request to local Ollama (ensure it is running via 'ollama run')...")
    val completion = OllamaClient.getCompletion(prompt, model = model)
    println(s"\nResponse:\n$completion")
  catch
    case e: Exception =>
      println(s"\nError getting completion: ${e.getMessage}")
      println("Please verify that Ollama is running and that the model is pulled ('ollama pull gemma4:12b').")
