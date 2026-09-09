//> using scala 3.6.4
// Copyright 2026 Mark Watson. All rights reserved.

package ollamavision

@main def visionDemo(): Unit =
  println("Payload shape for one image plus prompt:")
  println(VisionClient.chatPayload("qwen3-vl:2b", "Print the plain text in this image.", List("<base64-bytes>")).render().take(200))
  val args = sys.env.get("VISION_IMAGE")
  args match
    case Some(path) =>
      println("\n--- live call ---")
      try println(VisionClient.imageToText(List(path), "Print out the plain text in this image."))
      catch case e: Exception => println(s"[live call failed] ${e.getMessage}")
    case None =>
      println("\nSet VISION_IMAGE=/path/to/pic.png to run a live call against local Ollama.")
