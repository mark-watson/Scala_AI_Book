# Integrating Google Gemini

Large Language Models (LLMs) have transformed artificial intelligence by providing natural language processing, coding assistance, and reasoning capabilities. Google's Gemini models are highly capable models accessible via a developer-friendly REST API.

In this chapter, we implement a Google Gemini REST API client in Scala 3 using the lightweight `requests` HTTP library and the `ujson` library for JSON serialization. We also show how to enable **Google Search Grounding** to let Gemini query Google Search to answer real-time questions.

All code is in `source-code/gemini-client`.

## How Large Language Models Work

To use an LLM well it helps to know what one is. Modern LLMs are **transformer** networks, the architecture introduced by Vaswani and colleagues in the 2017 paper "Attention Is All You Need". The transformer's core mechanism is **self-attention**, which lets the model weigh how much each word in the input relates to every other word, so it can track meaning across long passages. This is a direct descendant of the small neural network we built by hand earlier in the book, scaled up by many orders of magnitude and trained on a large fraction of the public internet.

An LLM is trained on one deceptively simple task: predict the next **token**. A token is a chunk of text, usually a common word or a word fragment, produced by the kind of subword tokenizer mentioned in the NLP chapter. Given a sequence of tokens, the model outputs a probability distribution over the next token. To generate text it works **autoregressively**: it predicts one token, appends it to the input, predicts the next, and repeats. A setting called **temperature** controls how the next token is drawn from the distribution: near zero the model almost always takes the most likely token and answers deterministically, while higher values sample more freely and produce more varied, creative text.

Two more ideas explain the behavior you will see. The amount of text a model can consider at once is its **context window**, measured in tokens; everything the model knows about your request must fit inside it. And a raw model trained only to predict text is not yet helpful, so providers add **instruction tuning** and reinforcement learning from human feedback to teach it to follow instructions and answer politely. The Gemini model we call has already been through all of this; our job is only to send it tokens and read back what it generates.

## Project Setup and Dependencies

Instead of using a heavy SDK, we build our client directly on top of Gemini's HTTP POST endpoints. This keeps the moving parts visible: an LLM API, at bottom, is just an HTTP endpoint that accepts a JSON request and returns a JSON response. We declare our dependencies at the top of **gemini-client/GeminiClient.scala** using `scala-cli` directive syntax:

```scala
//> using scala 3.6.4
//> using dep com.lihaoyi::requests:0.9.3
//> using dep com.lihaoyi::ujson:4.4.3
```

`requests` is a small Scala HTTP client and `ujson` is a small JSON library, both from the lihaoyi ecosystem. Together they are all we need to talk to any REST-based LLM, which is why the OpenAI and Ollama clients in the next two chapters reuse the same two dependencies.

## Implementing the Gemini Client

The client connects to the `generativelanguage.googleapis.com` API. It reads the API key from the `GOOGLE_API_KEY` environment variable, which is the standard way to keep secrets out of source code.

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

The request body shows the shape of Gemini's API. A request is a list of `contents`, each holding a list of `parts`, and each part carries a piece of `text`. This nesting looks heavy for a single text prompt, but it exists because Gemini is **multimodal**: a part can hold an image or audio clip as easily as text, so the same structure serves every input type. The response mirrors it. The model returns a list of `candidates` (alternative completions), and we reach into the first candidate's content to pull out the generated text with `data("candidates")(0)("content")("parts")(0)("text").str`. Note also that the call is **stateless**: the API remembers nothing between requests, so a multi-turn conversation must resend the whole history each time. We check the HTTP status and raise an `IOException` on anything but 200, because API calls fail for real reasons: a bad key, a rate limit, or a network drop.

### Grounding with Google Search

The second method adds a `tools` array containing `google_search`, and this small change addresses one of the biggest weaknesses of LLMs. A model's knowledge is frozen at its training **cutoff** and it can state false things fluently, a failure called **hallucination**. **Grounding** fixes both: the model runs a live Google Search, reads the results, and bases its answer on them instead of on memory alone. This is the same retrieve-then-generate idea behind Retrieval-Augmented Generation (RAG) and the same pattern as the Knowledge Graph Navigator, where we fetched facts from DBpedia before answering. Turning it on is as simple as declaring the tool; Gemini decides when to search and folds the results into its reply.

## Running the Demos

In **gemini-client/Main.scala**, we run a standard prompt explaining Scala concepts, and a real-time news query grounded in Google Search. The contrast between the two calls is the point: the first tests the model's trained knowledge, the second tests its ability to fetch current facts:

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

The two responses come from the same model through the same code, yet they draw on different sources. The first answer about pattern matching comes straight from the model's trained weights, knowledge baked in during pretraining. The second answer about this week's launches could not possibly live in the weights, since the events happened after training; it comes from the live search results the grounding tool fetched. That difference, between what a model knows and what it can look up, is the single most important idea for building reliable LLM applications, and the rest of the LLM chapters build on it.
