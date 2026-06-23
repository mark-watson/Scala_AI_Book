# Local LLMs with Ollama

While cloud APIs (like Google Gemini and OpenAI) are powerful, running models locally offers complete data privacy, offline capabilities, and zero API costs. **Ollama** is a popular open-source tool that allows developers to run large language models locally on their own hardware.

In this chapter, we implement a Scala 3 client to query a local Ollama instance over its REST API.

All code is in `source-code/ollama-client`.

## The Ollama REST Client

Ollama exposes a REST API on `http://localhost:11434`. We query the `/api/generate` endpoint. Because local models run on consumer hardware, model inference can take time. We configure a generous read timeout of 3 minutes (`180000` milliseconds) inside **ollama-client/OllamaClient.scala**:

```scala
//> using scala 3.6.4
//> using dep com.lihaoyi::requests:0.9.3
//> using dep com.lihaoyi::ujson:4.4.3

package ollama_client

import java.io.IOException

object OllamaClient:
  private val DEFAULT_MODEL = "mistral"
  private val DEFAULT_BASE_URL = "http://localhost:11434"

  def getCompletion(
      prompt: String,
      model: String = DEFAULT_MODEL,
      baseUrl: String = DEFAULT_BASE_URL
  ): String =
    val url = s"$baseUrl/api/generate"

    val payload = ujson.Obj(
      "model" -> model,
      "prompt" -> prompt,
      "stream" -> false
    ).render()

    val response = requests.post(
      url = url,
      headers = Map("Content-Type" -> "application/json"),
      data = payload,
      readTimeout = 180000 // 3 minutes timeout to allow for local model startup and inference
    )

    if response.statusCode != 200 then
      throw IOException(s"Ollama request failed: ${response.text()}")

    parseResponse(response.text())

  private def parseResponse(jsonStr: String): String =
    val data = ujson.read(jsonStr)
    data("response").str
```

## Running the Ollama Demo

In **ollama-client/Main.scala**, we take the model name as an optional command-line argument (defaulting to `gemma4:12b`) and request a translation:

```scala
package ollama_client

@main def ollamaDemo(args: String*): Unit =
  println("=" * 50)
  println("Ollama Local LLM Client Demo")
  println("=" * 50)

  val model = if args.nonEmpty then args(0) else "gemma4:12b"
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
```

To run this demo, make sure Ollama is installed and running locally, pull the model, and execute the Scala client:

```bash
# In your terminal
ollama pull gemma4:12b

# Run the Scala program
scala-cli run .
```

This outputs the translation completed by your local GPU/CPU:

```text
==================================================
Ollama Local LLM Client Demo
==================================================
Using model: gemma4:12b
Prompt: Translate the following English text to French: 'Hello, how are you?'

Sending request to local Ollama (ensure it is running via 'ollama run')...

Response:
Bonjour, comment allez-vous ?
```
