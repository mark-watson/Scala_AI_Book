// Copyright 2025-2026 Mark Watson. All rights reserved.
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
      throw IOException(s"OpenAI API request failed (HTTP ${response.statusCode}): ${response.text()}")

    parseResponse(response.text())

  private def getApiKey(): String =
    val key = System.getenv("OPENAI_API_KEY")
    if key == null || key.trim.isEmpty then
      throw IOException("OPENAI_API_KEY environment variable is not set. Please export it before running.")
    key.trim

  private def parseResponse(jsonStr: String): String =
    val data = ujson.read(jsonStr)
    data("choices")(0)("message")("content").str
