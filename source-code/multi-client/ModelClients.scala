//> using scala 3.6.4
//> using dep com.lihaoyi::requests:0.9.3
//> using dep com.lihaoyi::ujson:4.4.3
// Copyright 2026 Mark Watson. All rights reserved.

package multiclient

// Extra hosted model clients, after anthropic, mistral, groq,
// moonshot_kimi_k2, perplexity, agent_grok, and huggingface in the
// Loving Common Lisp repo. Payload builders stay pure so tests run
// offline; only send() touches the net, and only with a key set.
final case class ClientConfig(
    name: String,
    url: String,
    apiKeyEnv: String,
    headers: String => Map[String, String],
    buildPayload: (String, String, Int) => ujson.Value,
    parseReply: String => String
)

object ModelClients:
  def openAiShapePayload(model: String, prompt: String, maxTokens: Int): ujson.Value =
    ujson.Obj(
      "model" -> model,
      "max_tokens" -> maxTokens,
      "messages" -> ujson.Arr(ujson.Obj("role" -> "user", "content" -> prompt))
    )

  def parseOpenAiChat(json: String): String =
    ujson.read(json)("choices")(0)("message")("content").str

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

  private def bearer(key: String): Map[String, String] =
    Map("Content-Type" -> "application/json", "Authorization" -> s"Bearer $key")

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

  val all: List[ClientConfig] = List(anthropic, mistral, groq, moonshot, perplexity)

  // Live call. Throws when the key env var is missing.
  def send(client: ClientConfig, model: String, prompt: String, maxTokens: Int = 256): String =
    val key = sys.env.getOrElse(client.apiKeyEnv, throw new IllegalStateException(s"Set ${client.apiKeyEnv} first."))
    val response = requests.post(client.url, headers = client.headers(key),
      data = client.buildPayload(model, prompt, maxTokens).render(), readTimeout = 120000)
    if response.statusCode != 200 then
      throw new java.io.IOException(s"${client.name} failed (HTTP ${response.statusCode}): ${response.text().take(200)}")
    client.parseReply(response.text())
