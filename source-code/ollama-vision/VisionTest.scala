//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package ollamavision

import java.nio.file.Files
import java.util.Base64

// Offline checks. Run with: scala-cli run . --main-class ollamavision.visionTest
@main def visionTest(): Unit =
  // A 1x1 red PNG as base64: decode it to a temp file for encode tests.
  val tinyPngBase64 =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
  val img = Files.createTempFile("vision-test", ".png")
  Files.write(img, Base64.getDecoder.decode(tinyPngBase64))

  // encodeImage round trips the bytes and rejects missing files.
  assert(VisionClient.encodeImage(img.toString) == tinyPngBase64, "base64 round trip")
  try { VisionClient.encodeImage(img.toString + ".missing"); assert(false, "missing image must raise") }
  catch case _: IllegalArgumentException => ()

  // Payload carries model, prompt, and the images array.
  val payload = VisionClient.chatPayload("m", "read this", List("AAA", "BBB"))
  assert(payload("model").str == "m", "model field")
  assert(payload("messages")(0)("content").str == "read this", "prompt field")
  assert(payload("messages")(0)("images").arr.map(_.str).toList == List("AAA", "BBB"), "images field")
  assert(payload("stream").bool == false, "stream must be false")

  // Parser reads message content and surfaces problems.
  val ok = """{"message":{"role":"assistant","content":"A red dot."}}"""
  assert(VisionClient.parseReply(ok) == "A red dot.", "content parse")
  try { VisionClient.parseReply(""); assert(false, "empty reply must raise") }
  catch case _: IllegalArgumentException => ()
  try { VisionClient.parseReply("""{"error":"model not found"}"""); assert(false, "error reply must raise") }
  catch case _: java.io.IOException => ()

  println("All vision tests passed.")
