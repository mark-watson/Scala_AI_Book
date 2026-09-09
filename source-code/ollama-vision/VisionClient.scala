//> using scala 3.6.4
//> using dep com.lihaoyi::requests:0.9.3
//> using dep com.lihaoyi::ujson:4.4.3
// Copyright 2026 Mark Watson. All rights reserved.

package ollamavision

import java.nio.file.{Files, Paths}
import java.util.Base64

// Vision calls to local Ollama models. Port of
// ollama_images/describe-image.lisp: base64-encode image files,
// send them with a prompt to /api/chat, read back the text.
object VisionClient:
  val defaultModel = sys.env.getOrElse("OLLAMA_MODEL", "qwen3-vl:2b")
  val defaultHost = sys.env.getOrElse("OLLAMA_HOST", "http://localhost:11434/api/chat")

  def encodeImage(path: String): String =
    val p = Paths.get(path)
    if !Files.isRegularFile(p) then throw new IllegalArgumentException(s"Image file not found: $path")
    Base64.getEncoder.encodeToString(Files.readAllBytes(p))

  def chatPayload(model: String, prompt: String, imagesBase64: List[String]): ujson.Value =
    ujson.Obj(
      "model" -> model,
      "stream" -> false,
      "messages" -> ujson.Arr(
        ujson.Obj(
          "role" -> "user",
          "content" -> prompt,
          "images" -> ujson.Arr.from(imagesBase64.map(ujson.Str(_)))
        )
      )
    )

  def parseReply(json: String): String =
    if json == null || json.trim.isEmpty then
      throw new IllegalArgumentException("Empty response from Ollama, is the server running?")
    val data = ujson.read(json)
    data.obj.get("error") match
      case Some(err) => throw new java.io.IOException(s"Ollama API error: $err")
      case None => data("message")("content").str

  def imageToText(
      paths: List[String],
      prompt: String,
      model: String = defaultModel,
      host: String = defaultHost
  ): String =
    val encoded = paths.map(encodeImage)
    val response = requests.post(host,
      headers = Map("Content-Type" -> "application/json"),
      data = chatPayload(model, prompt, encoded).render(),
      readTimeout = 180000)
    if response.statusCode != 200 then
      throw new java.io.IOException(s"Ollama vision failed (HTTP ${response.statusCode}).")
    parseReply(response.text())

  def describeImageSimple(path: String): String =
    imageToText(List(path), "What is in this image?")
