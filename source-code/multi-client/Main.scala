//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package multiclient

@main def clientsDemo(): Unit =
  val prompt = "Mary is 30 and Bob is 25. Who is older? Answer in one short sentence."
  println("Payload shapes (no keys needed):")
  println("Anthropic: " + ModelClients.anthropic.buildPayload("claude-haiku-4-5", prompt, 64).render().take(160))
  println("Mistral:   " + ModelClients.mistral.buildPayload("mistral-small-latest", prompt, 64).render().take(160))
  println("HF:        " + ModelClients.huggingface("gpt2").buildPayload("gpt2", prompt, 64).render().take(160))
  println("\nLive calls (need API keys, skipped when missing):")
  val models = Map("anthropic" -> "claude-haiku-4-5", "mistral" -> "mistral-small-latest",
    "groq" -> "llama-3.3-70b-versatile", "moonshot" -> "kimi-k2-0711-preview", "perplexity" -> "sonar")
  ModelClients.all.foreach { client =>
    val label = client.name
    try println(s"$label: " + ModelClients.send(client, models(label), prompt, 64))
    catch case e: Exception => println(s"$label: [skip] ${e.getMessage}")
  }
