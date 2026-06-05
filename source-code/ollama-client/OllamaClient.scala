// Copyright 2025-2026 Mark Watson. All rights reserved.
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
      readTimeout = 180000 // 3 minutes timeout matching Java version
    )

    if response.statusCode != 200 then
      throw IOException(s"Ollama API request failed (HTTP ${response.statusCode}): ${response.text()}")

    parseResponse(response.text())

  private def parseResponse(jsonStr: String): String =
    val data = ujson.read(jsonStr)
    data("response").str
