# More Hosted Models: Anthropic, Mistral, Groq, Moonshot, Perplexity, Hugging Face

The book so far covers Gemini, OpenAI, and local Ollama. Real work spreads wider: Claude for long docs, Mistral for cheap EU hosting, Groq for fast open weights, Moonshot Kimi for long Chinese context, Perplexity for search backed answers, Hugging Face for niche classifiers. This chapter adds all six as thin REST clients in one file, in the same shape as the book's OpenAI client.

All code is in `source-code/multi-client`.

## One Shape, Six Endpoints

Four of the six speak the OpenAI chat shape: model, messages, max tokens. One builder serves them all, so a fix lands once:

```scala
def openAiShapePayload(model: String, prompt: String, maxTokens: Int): ujson.Value =
  ujson.Obj(
    "model" -> model,
    "max_tokens" -> maxTokens,
    "messages" -> ujson.Arr(ujson.Obj("role" -> "user", "content" -> prompt))
  )
```

Anthropic differs in headers (`x-api-key` plus a version stamp) and in reply shape (`content[0].text`). Hugging Face differs more: POST `{inputs}` to a per model URL and read `generated_text`. Each client is one `ClientConfig` with URL, key name, headers, builder, and parser. `send` reads the key from env, posts, checks status, and parses:

```scala
final case class ClientConfig(
    name: String, url: String, apiKeyEnv: String,
    headers: String => Map[String, String],
    buildPayload: (String, String, Int) => ujson.Value,
    parseReply: String => String
)
```

## Keys Stay in Env

Each client names its key var: `ANTHROPIC_API_KEY`, `MISTRAL_API_KEY`, `GROQ_API_KEY`, `MOONSHOT_API_KEY`, `PERPLEXITY_API_KEY`, `HF_API_KEY`. A missing key raises before any socket opens, and the message names the var to set. The demo tries each client in turn and skips the ones with no key, so one command shows all six states.

## Parsers You Can Test Offline

Reply parsers are pure functions on strings, which makes them the easiest code in this book to test. The suite feeds canned server JSON for the chat shape, the Anthropic shape, and both Hugging Face shapes (bare object and one item array). If a vendor drifts its format next year, paste a fresh reply into the test and fix the parser with proof in hand.

Run the demo and the checks:

```bash
cd source-code/multi-client
scala-cli run . --main-class multiclient.clientsDemo
scala-cli run . --main-class multiclient.clientsTest
```

Payload shapes print with no keys. Live calls need keys and cost real credit, so the demo keeps max tokens at 64.
