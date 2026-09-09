//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package multiclient

// Offline checks on payload shapes and reply parsers.
// Run with: scala-cli run . --main-class multiclient.clientsTest
@main def clientsTest(): Unit =
  // OpenAI-shape payload carries model, prompt, and token cap.
  val mistralBody = ModelClients.mistral.buildPayload("m1", "hi?", 64)
  assert(mistralBody("model").str == "m1", "model field")
  assert(mistralBody("messages")(0)("content").str == "hi?", "prompt field")
  assert(mistralBody("max_tokens").num == 64, "token cap field")

  // Anthropic payload uses max_tokens plus a messages array.
  val anthBody = ModelClients.anthropic.buildPayload("c1", "hi?", 32)
  assert(anthBody("max_tokens").num == 32 && anthBody("messages")(0)("role").str == "user", "anthropic shape")

  // Groq, Moonshot, and Perplexity share the OpenAI shape.
  List(ModelClients.groq, ModelClients.moonshot, ModelClients.perplexity).foreach { c =>
    assert(c.buildPayload("m", "hi?", 8)("messages")(0)("content").str == "hi?", s"${c.name} shape")
  }

  // Reply parsers pull text from sample server JSON.
  val chatJson = """{"choices":[{"message":{"content":"Mary is older."}}]}"""
  assert(ModelClients.parseOpenAiChat(chatJson) == "Mary is older.", "chat parse")
  val anthJson = """{"content":[{"text":"Mary is older."}]}"""
  assert(ModelClients.parseAnthropic(anthJson) == "Mary is older.", "anthropic parse")
  val hfArr = """[{"generated_text":"once upon a time"}]"""
  val hfObj = """{"generated_text":"once upon a time"}"""
  assert(ModelClients.parseHf(hfArr) == "once upon a time", "hf array parse")
  assert(ModelClients.parseHf(hfObj) == "once upon a time", "hf object parse")

  // Headers: Anthropic uses x-api-key plus version, others use Bearer.
  assert(ModelClients.anthropic.headers("K")("x-api-key") == "K", "anthropic key header")
  assert(ModelClients.anthropic.headers("K")("anthropic-version") == "2023-06-01", "anthropic version header")
  assert(ModelClients.mistral.headers("K")("Authorization") == "Bearer K", "bearer header")

  // Live send without a key raises a clear error, not a network call.
  // Uses a bogus env name so ambient keys in the shell cannot mask it.
  val noKey = ModelClients.mistral.copy(apiKeyEnv = "MISSING_ENV_VAR_FOR_TEST_XYZ")
  try { ModelClients.send(noKey, "m", "hi?", 8); assert(false, "missing key must raise") }
  catch case e: IllegalStateException => assert(e.getMessage.contains("MISSING_ENV_VAR_FOR_TEST_XYZ"), "error names the env var")

  println("All client tests passed.")
