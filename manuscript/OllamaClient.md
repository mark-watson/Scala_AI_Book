# Local LLMs with Ollama

While cloud APIs (like Google Gemini and OpenAI) are powerful, running models locally offers complete data privacy, offline capabilities, and zero API costs. **Ollama** is a popular open-source tool that allows developers to run large language models locally on their own hardware.

In this chapter, we implement a Scala 3 client to query a local Ollama instance over its REST API.

All code is in `source-code/ollama-client`.

## Why Run a Model Locally

The last two chapters sent every prompt to a company's servers. Running the model on your own machine changes the trade-offs in ways worth understanding before you write the code.

The case **for** local inference is strong. Your data never leaves the machine, which matters for medical records, legal documents, or proprietary source code you cannot send to a third party. There is no per-token bill, so you can process millions of words for the cost of electricity. It works with no network, and it frees you from rate limits and from a provider changing or retiring a model underneath you. The case **against** is equally real: you are limited to models small enough to fit your hardware, inference is slower than a data center's, especially without a GPU, and the very largest, most capable models remain cloud-only.

Local inference is possible at all because of **open-weight** models. Where GPT-4o and Gemini are reachable only through an API, models such as Mistral, Llama, and Gemma publish their trained weights for anyone to download and run. ("Open weight" is a weaker claim than "open source": the weights are free to use, but the training data and code usually are not.) The second enabler is **quantization**. A model's weights are trained at high numeric precision, but you can round them to fewer bits, often 4, which shrinks the model's memory footprint several-fold and speeds up inference, at a small cost in quality. Quantization is what lets a model with billions of parameters run on a laptop, and Ollama distributes models in the quantized GGUF format designed for exactly this.

## The Ollama REST Client

Ollama runs as a background server that downloads models, loads them into memory, and exposes a REST API on `http://localhost:11434`. We query the `/api/generate` endpoint. Because local models run on consumer hardware, model inference can take time, and the first request must also load the model into memory. We configure a generous read timeout of 3 minutes (`180000` milliseconds) inside **ollama-client/OllamaClient.scala**:

```scala
//> using scala 3.6.4
//> using dep com.lihaoyi::requests:0.9.3
//> using dep com.lihaoyi::ujson:4.4.3

package ollama_client

import java.io.IOException

object OllamaClient:
  private val DEFAULT_MODEL = "gemma4:12b"
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

The client is the simplest of the three. It needs no API key, because the server runs on your own machine, and it points at `localhost` instead of a remote host. The payload names the model, the prompt, and one important flag: `stream`. Recall that an LLM generates text one token at a time. With streaming enabled, the server sends each token the instant it is produced, which lets a chat interface show the answer appearing word by word. We set `stream` to `false` so the server instead buffers the whole generation and returns it as a single JSON object, which is simpler to parse in a batch program like ours. We then read the completed text from the `response` field. The long read timeout reflects the reality of local inference: the first call may spend a minute loading a multi-gigabyte model from disk before it generates a single token.

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

Taking the model name as an argument makes the client easy to point at whatever model you have pulled, so you can compare a small fast model against a larger, slower, more capable one on the same prompt. To run this demo, make sure Ollama is installed and running locally, pull the model, and execute the Scala client:

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

This is the same translation task the OpenAI chapter ran, producing an equally correct result, but no data left the machine and no request crossed the network. That is the whole point of local inference. The three LLM chapters give you the full spectrum: a frontier cloud model with live grounding, the standard chat API that most providers share, and a private model on your own hardware. Which you choose is an engineering decision about capability, cost, privacy, and control, and knowing all three lets you make it deliberately.
