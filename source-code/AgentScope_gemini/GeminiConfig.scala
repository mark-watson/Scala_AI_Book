// Copyright 2025-2026 Mark Watson. All rights reserved.
//> using scala 3.8.3
//> using dep io.agentscope:agentscope:2.0.3
//> using dep com.google.genai:google-genai:1.70.0
//> using dep org.slf4j:slf4j-simple:2.0.19

package agentscope_gemini

import io.agentscope.extensions.model.gemini.GeminiChatModel

object GeminiConfig:
  val MODEL_NAME = "gemini-2.5-flash"

  def requireApiKey(): String =
    sys.env.get("GOOGLE_API_KEY") match
      case Some(key) if key.trim.nonEmpty => key.trim
      case _ =>
        System.err.println("ERROR: GEMINI_API_KEY environment variable is not set.")
        sys.exit(1)
        "" // unreachable

  def createModel(): GeminiChatModel =
    GeminiChatModel.builder()
      .apiKey(requireApiKey())
      .modelName(MODEL_NAME)
      .build()
