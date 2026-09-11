# More Hosted Models: Anthropic, Mistral, Groq, Moonshot, Perplexity, Hugging Face

The book so far covers Gemini, OpenAI, and local Ollama. Real work spreads wider: Claude for long docs, Mistral for cheap EU hosting, Groq for fast open weights, Moonshot Kimi for long Chinese context, Perplexity for search backed answers, Hugging Face for niche classifiers. This chapter adds all six as thin REST clients in one file, in the same shape as the book's OpenAI client.

All code is in `source-code/multi-client`.

## One Record for Six Vendors

Each client is one `ClientConfig`: a name, a URL, the env var holding its key, a header builder, a payload builder, and a reply parser. Behavior varies per vendor, so each slot is a function:

```scala
final case class ClientConfig(
    name: String,
    url: String,
    apiKeyEnv: String,
    headers: String => Map[String, String],
    buildPayload: (String, String, Int) => ujson.Value,
    parseReply: String => String
)
```

Four of the six speak the OpenAI chat shape: model, messages, max tokens. One builder serves them all, so any future code fix lands in just one place:

```scala
def openAiShapePayload(model: String, prompt: String, maxTokens: Int): ujson.Value =
  ujson.Obj(
    "model" -> model,
    "max_tokens" -> maxTokens,
    "messages" -> ujson.Arr(ujson.Obj("role" -> "user", "content" -> prompt))
  )

def parseOpenAiChat(json: String): String =
  ujson.read(json)("choices")(0)("message")("content").str
```

## The Odd Two: Anthropic and Hugging Face

Anthropic differs in headers (`x-api-key` plus a version stamp) and in reply shape (`content[0].text`). Hugging Face differs more: POST `{inputs}` to a per model URL and read `generated_text`, which arrives bare or wrapped in a one item array:

```scala
def anthropicPayload(model: String, prompt: String, maxTokens: Int): ujson.Value =
  ujson.Obj(
    "model" -> model,
    "max_tokens" -> maxTokens,
    "messages" -> ujson.Arr(ujson.Obj("role" -> "user", "content" -> prompt))
  )

def parseAnthropic(json: String): String =
  ujson.read(json)("content")(0)("text").str

def hfPayload(prompt: String): ujson.Value =
  ujson.Obj("inputs" -> prompt)

def parseHf(json: String): String =
  val data = ujson.read(json)
  if data.isInstanceOf[ujson.Arr] then data(0)("generated_text").str
  else data("generated_text").str
```

The six configs then read as a table. Four share the Bearer header helper and the OpenAI shape; Anthropic and Hugging Face plug their own slots:

```scala
val anthropic: ClientConfig = ClientConfig(
  "anthropic", "https://api.anthropic.com/v1/messages", "ANTHROPIC_API_KEY",
  key => Map("Content-Type" -> "application/json", "x-api-key" -> key, "anthropic-version" -> "2023-06-01"),
  anthropicPayload, parseAnthropic
)
val mistral: ClientConfig = ClientConfig(
  "mistral", "https://api.mistral.ai/v1/chat/completions", "MISTRAL_API_KEY",
  bearer, (m, p, t) => openAiShapePayload(m, p, t), parseOpenAiChat
)
val groq: ClientConfig = ClientConfig(
  "groq", "https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY",
  bearer, (m, p, t) => openAiShapePayload(m, p, t), parseOpenAiChat
)
val moonshot: ClientConfig = ClientConfig(
  "moonshot", "https://api.moonshot.ai/v1/chat/completions", "MOONSHOT_API_KEY",
  bearer, (m, p, t) => openAiShapePayload(m, p, t), parseOpenAiChat
)
val perplexity: ClientConfig = ClientConfig(
  "perplexity", "https://api.perplexity.ai/chat/completions", "PERPLEXITY_API_KEY",
  bearer, (m, p, t) => openAiShapePayload(m, p, t), parseOpenAiChat
)

def huggingface(model: String): ClientConfig = ClientConfig(
  "huggingface", s"https://api-inference.huggingface.co/models/$model", "HF_API_KEY",
  bearer, (m, p, t) => hfPayload(p), parseHf
)
```

## One Send for All

`send` reads the key from env, posts the built payload, checks status, and parses. A missing key raises before any socket opens, and the message names the var to set:

```scala
def send(client: ClientConfig, model: String, prompt: String, maxTokens: Int = 256): String =
  val key = sys.env.getOrElse(client.apiKeyEnv, throw new IllegalStateException(s"Set ${client.apiKeyEnv} first."))
  val response = requests.post(client.url, headers = client.headers(key),
    data = client.buildPayload(model, prompt, maxTokens).render(), readTimeout = 120000)
  if response.statusCode != 200 then
    throw new java.io.IOException(s"${client.name} failed (HTTP ${response.statusCode}): ${response.text().take(200)}")
  client.parseReply(response.text())
```

## Demo and Its Output

The demo in **multi-client/Main.scala** prints payload shapes first, which needs no keys, then tries each live client and skips the ones with no key:

```scala
@main def clientsDemo(): Unit =
  val prompt = "Mary is 30 and Bob is 25. Who is older? Answer in one short sentence."
  println("Payload shapes (no keys needed):")
  println("Anthropic: " + ModelClients.anthropic.buildPayload("claude-haiku-4-5", prompt, 64).render().take(160))
  println("Mistral:   " + ModelClients.mistral.buildPayload("mistral-small-latest", prompt, 64).render().take(160))
  println("HF:        " + ModelClients.huggingface("gpt2").buildPayload("gpt2", prompt, 64).render().take(160))
  println("\nLive calls (need API keys, skipped when missing):")
  ...
  ModelClients.all.foreach { client =>
    try println(s"$label: " + ModelClients.send(client, models(label), prompt, 64))
    catch case e: Exception => println(s"$label: [skip] ${e.getMessage}")
  }
```

With no keys set it prints:

```
Payload shapes (no keys needed):
Anthropic: {"model":"claude-haiku-4-5","max_tokens":64,"messages":[{"role":"user","content":"Mary is 30 and Bob is 25. Who is older? Answer in one short sentence."}]}
Mistral:   {"model":"mistral-small-latest","max_tokens":64,"messages":[{"role":"user","content":"Mary is 30 and Bob is 25. Who is older? Answer in one short sentence."}]}
HF:        {"inputs":"Mary is 30 and Bob is 25. Who is older? Answer in one short sentence."}

Live calls (need API keys, skipped when missing):
anthropic: [skip] Set ANTHROPIC_API_KEY first.
mistral: [skip] Set MISTRAL_API_KEY first.
groq: [skip] Set GROQ_API_KEY first.
moonshot: [skip] Set MOONSHOT_API_KEY first.
perplexity: [skip] Set PERPLEXITY_API_KEY first.
```

Set one or more of `ANTHROPIC_API_KEY`, `MISTRAL_API_KEY`, `GROQ_API_KEY`, `MOONSHOT_API_KEY`, `PERPLEXITY_API_KEY`, `HF_API_KEY` and the matching rows turn into live one sentence answers. The demo keeps max tokens at 64, so each call costs pocket change.

## Parsers You Can Test Offline

Reply parsers are pure functions on strings, which makes them the easiest code in this book to test. The suite feeds canned server JSON for the chat shape, the Anthropic shape, and both Hugging Face shapes. It also checks the header maps and the missing key error with a bogus var name, so ambient keys in the shell cannot mask the check. If a vendor drifts its format next year, paste a fresh reply into the test and fix the parser with proof in hand:

```scala
val chatJson = """{"choices":[{"message":{"content":"Mary is older."}}]}"""
assert(ModelClients.parseOpenAiChat(chatJson) == "Mary is older.", "chat parse")
val anthJson = """{"content":[{"text":"Mary is older."}]}"""
assert(ModelClients.parseAnthropic(anthJson) == "Mary is older.", "anthropic parse")
```

Run the demo and the checks:

```bash
cd source-code/multi-client
scala-cli run . --main-class multiclient.clientsDemo
scala-cli run . --main-class multiclient.clientsTest
```

The checks need no keys and end with:

```
All client tests passed.
```
