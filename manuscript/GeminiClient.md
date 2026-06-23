# Integrating Google Gemini

Large Language Models (LLMs) have transformed artificial intelligence by providing natural language processing, coding assistance, and reasoning capabilities. Google's Gemini models are highly capable models accessible via a developer-friendly REST API.

In this chapter, we implement a Google Gemini REST API client in Scala 3 using the lightweight `requests` HTTP library and the `ujson` library for JSON serialization. We also show how to enable **Google Search Grounding** to let Gemini query Google Search to answer real-time questions.

All code is in `source-code/gemini-client`.

## Project Setup and Dependencies

Instead of using a heavy SDK, we build our client directly on top of Gemini's HTTP POST endpoints. We declare our dependencies at the top of **gemini-client/GeminiClient.scala** using `scala-cli` directive syntax:

```scala
//> using scala 3.6.4
//> using dep com.lihaoyi::requests:0.9.3
//> using dep com.lihaoyi::ujson:4.4.3
```

## Implementing the Gemini Client

The client connects to the `generativelanguage.googleapis.com` API. It reads the API key from the `GOOGLE_API_KEY` environment variable.

We define two completion methods: `getCompletion` for standard prompts, and `getCompletionWithSearch` which instructs the model to ground its response using live Google Search results:

```scala
package gemini_client

import java.io.IOException

object GeminiClient:
  private val DEFAULT_MODEL = "gemini-2.5-flash"
  private val API_HOST = "generativelanguage.googleapis.com"

  def getCompletion(prompt: String, model: String = DEFAULT_MODEL): String =
    val apiKey = getApiKey()
    val url = s"https://$API_HOST/v1beta/models/$model:generateContent?key=$apiKey"

    val payload = ujson.Obj(
      "contents" -> ujson.Arr(
        ujson.Obj(
          "parts" -> ujson.Arr(
            ujson.Obj("text" -> prompt)
          )
        )
      )
    ).render()

    val response = requests.post(
      url = url,
      headers = Map("Content-Type" -> "application/json"),
      data = payload
    )

    if response.statusCode != 200 then
      throw IOException(s"Gemini request failed: ${response.text()}")

    parseResponse(response.text())

  def getCompletionWithSearch(prompt: String, model: String = DEFAULT_MODEL): String =
    val apiKey = getApiKey()
    val url = s"https://$API_HOST/v1beta/models/$model:generateContent?key=$apiKey"

    val payload = ujson.Obj(
      "contents" -> ujson.Arr(
        ujson.Obj(
          "parts" -> ujson.Arr(
            ujson.Obj("text" -> prompt)
          )
        )
      ),
      "tools" -> ujson.Arr(
        ujson.Obj("google_search" -> ujson.Obj()) // Enable Google Search Grounding
      )
    ).render()

    val response = requests.post(
      url = url,
      headers = Map("Content-Type" -> "application/json"),
      data = payload
    )

    if response.statusCode != 200 then
      throw IOException(s"Gemini request failed: ${response.text()}")

    parseResponse(response.text())

  private def getApiKey(): String =
    val key = System.getenv("GOOGLE_API_KEY")
    if key == null || key.trim.isEmpty then
      throw IOException("GOOGLE_API_KEY environment variable is not set.")
    key.trim

  private def parseResponse(jsonStr: String): String =
    val data = ujson.read(jsonStr)
    data("candidates")(0)("content")("parts")(0)("text").str
```

## Running the Demos

In **gemini-client/Main.scala**, we run a standard prompt explaining Scala concepts, and a real-time news query grounded in Google Search:

```scala
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
    case e: Exception => println(s"Error: ${e.getMessage}")

  val prompt2 = "What are the latest news updates on space exploration from this week?"
  println(s"\nPrompt with Google Search Grounding: $prompt2\n")
  try
    val completion2 = GeminiClient.getCompletionWithSearch(prompt2)
    println(s"Response:\n$completion2")
  catch
    case e: Exception => println(s"Error: ${e.getMessage}")
```

Export your API key and launch the project:

```bash
export GOOGLE_API_KEY="your-api-key-here"
scala-cli run .
```

This will run the demo, printing Gemini's responses:

```text
==================================================
Gemini LLM Client Demo
==================================================

Prompt: Briefly explain the concept of pattern matching in Scala.

Response:
Pattern matching in Scala is a powerful mechanism for checking a value against a pattern. It's like a type-safe and more expressive version of the `switch` statement found in Java or C++. 
You use the `match` keyword followed by one or more `case` clauses. It can match constants, variable patterns, case classes, types, and extract values using extractor objects...

Prompt with Google Search Grounding: What are the latest news updates on space exploration from this week?

Response:
According to recent reports on space exploration this week:
1. NASA's James Webb Space Telescope detected a new super-Earth atmosphere...
2. SpaceX successfully launched its 11th Starlink batch of the month and prepared for the next Starship flight test...
```
