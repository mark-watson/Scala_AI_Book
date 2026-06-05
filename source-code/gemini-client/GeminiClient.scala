// Copyright 2025-2026 Mark Watson. All rights reserved.
//> using scala 3.6.4
//> using dep com.lihaoyi::requests:0.9.3
//> using dep com.lihaoyi::ujson:4.4.3


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
      throw IOException(s"Gemini API request failed (HTTP ${response.statusCode}): ${response.text()}")

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
        ujson.Obj("google_search" -> ujson.Obj())
      )
    ).render()

    val response = requests.post(
      url = url,
      headers = Map("Content-Type" -> "application/json"),
      data = payload
    )

    if response.statusCode != 200 then
      throw IOException(s"Gemini API request failed (HTTP ${response.statusCode}): ${response.text()}")

    parseResponse(response.text())

  private def getApiKey(): String =
    val key = System.getenv("GOOGLE_API_KEY")
    if key == null || key.trim.isEmpty then
      throw IOException("GOOGLE_API_KEY environment variable is not set. Please export it before running.")
    key.trim

  private def parseResponse(jsonStr: String): String =
    val data = ujson.read(jsonStr)
    data("candidates")(0)("content")("parts")(0)("text").str
