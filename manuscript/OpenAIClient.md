# Integrating OpenAI

OpenAI is a pioneer in LLM development, hosting models such as GPT-4o and GPT-4o-mini. These models are accessible via a standardized JSON API.

In this chapter, we implement an OpenAI client in Scala 3 using the `requests` and `ujson` libraries, demonstrating how to construct payload objects and authenticate requests using headers.

All code is in `source-code/openai-client`.

## The OpenAI REST Client

In **openai-client/OpenAIClient.scala**, we target the `/v1/chat/completions` endpoint. The API key is read from the `OPENAI_API_KEY` environment variable and passed in the HTTP request as a `Bearer` token in the `Authorization` header:

```scala
//> using scala 3.6.4
//> using dep com.lihaoyi::requests:0.9.3
//> using dep com.lihaoyi::ujson:4.4.3

package openai_client

import java.io.IOException

object OpenAIClient:
  private val DEFAULT_MODEL = "gpt-4o-mini"
  private val API_URL = "https://api.openai.com/v1/chat/completions"

  def getCompletion(prompt: String, model: String = DEFAULT_MODEL): String =
    val apiKey = getApiKey()

    val payload = ujson.Obj(
      "model" -> model,
      "messages" -> ujson.Arr(
        ujson.Obj(
          "role" -> "user",
          "content" -> prompt
        )
      )
    ).render()

    val response = requests.post(
      url = API_URL,
      headers = Map(
        "Content-Type" -> "application/json",
        "Authorization" -> s"Bearer $apiKey"
      ),
      data = payload
    )

    if response.statusCode != 200 then
      throw IOException(s"OpenAI request failed: ${response.text()}")

    parseResponse(response.text())

  private def getApiKey(): String =
    val key = System.getenv("OPENAI_API_KEY")
    if key == null || key.trim.isEmpty then
      throw IOException("OPENAI_API_KEY environment variable is not set.")
    key.trim

  private def parseResponse(jsonStr: String): String =
    val data = ujson.read(jsonStr)
    data("choices")(0)("message")("content").str
```

## Running the OpenAI Demo

The demo program in **openai-client/Main.scala** uses the OpenAI client to translate a sentence from English to French:

```scala
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
    case e: Exception => println(s"Error: ${e.getMessage}")
```

Export your API key and run the example:

```bash
export OPENAI_API_KEY="your-openai-api-key"
scala-cli run .
```

This compiles and runs the project, printing the response from OpenAI:

```text
==================================================
OpenAI LLM Client Demo
==================================================

Prompt: Translate the following English text to French: 'Hello, how are you?'

Response:
Bonjour, comment ça va ?
```
